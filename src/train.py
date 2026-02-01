# src/train.py
import os, time, argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

from src.data import BinTokenWindowDataset
from src.model import TinyTransformerLM

# +1 vocab slot for a real [MASK] token
VOCAB_SIZE = 50258
MASK_ID = 50257


def ar_loss(logits, y):
    # Standard next-token cross entropy
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        y.view(-1)
    )


def mdm_step(model, x, device):
    """
    Masked Diffusion LM step:
    - sample mask ratio r
    - mask tokens with prob r
    - predict original tokens at masked positions
    - normalize by expected masked count (paper-style)
    """
    x = x.to(device)
    targets = x

    # sample r in paper-reasonable range (avoid nearly-full masking)
    r = torch.empty((), device=device).uniform_(0.15, 0.5)

    # mask = (torch.rand_like(x.float()) < r)

    # # guarantee at least one masked token
    # if not mask.any():
    #     mask.view(-1)[0] = True

    mask = (torch.rand_like(x.float()) < r)   # (B,T)
    need = ~mask.any(dim=1)                   # (B,)
    if need.any():
        mask[need, 0] = True

    corrupted = x.masked_fill(mask, MASK_ID)
    logits = model(corrupted)

    B, T = x.shape
    denom = r * (B * T)  # expected masked tokens

    ce_sum = F.cross_entropy(
        logits[mask],
        targets[mask],
        reduction="sum"
    )

    loss = ce_sum / denom
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ar", "mdm"], required=True)
    ap.add_argument("--train_bin", required=True)
    ap.add_argument("--val_bin", required=True)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--offset_tokens", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=None, help="cap train/val to this many tokens (fixed small corpus)")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--eval_batches", type=int, default=50)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()

    ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
    else:
        rank, world_size = 0, 1
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    is_main = rank == 0
    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)
        wandb.init(
            project="diffusion-vs-ar",
            name=f"{args.mode}_tok{args.max_tokens}_steps{args.steps}",
            config=vars(args),
        )

    causal = (args.mode == "ar")
    model = TinyTransformerLM(
        vocab_size=VOCAB_SIZE,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        causal=causal,
    ).to(device)
    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    use_amp = device.startswith("cuda")
    autocast_dtype = torch.bfloat16

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )

    if is_main:
        wandb.log({"hparams/lr": args.lr})

    train_ds = BinTokenWindowDataset(
        args.train_bin, seq_len=args.seq_len, seed=0, rank=rank,
        offset_tokens=args.offset_tokens, max_tokens=args.max_tokens
    )
    val_ds = BinTokenWindowDataset(
        args.val_bin, seq_len=args.seq_len, seed=123, rank=rank,
        offset_tokens=args.offset_tokens, max_tokens=args.max_tokens
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True,
    )

    train_it = iter(train_dl)

    def eval_loss():
        model.eval()
        losses = []
        if args.mode == "mdm":
            rng_state = torch.get_rng_state()
            torch.manual_seed(12345)
        with torch.no_grad():
            it = iter(val_dl)
            for _ in range(args.eval_batches):
                x, y = next(it)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_amp):
                    if args.mode == "ar":
                        logits = model(x)
                        loss = ar_loss(logits, y)
                    else:
                        # Same objective as training: sample r ~ U(0.01, 0.99), random masks,
                        # masked-only CE with 1/r normalization (apples-to-apples with AR val_loss)
                        # r = torch.empty((), device=device).uniform_(0.01, 0.99)
                        r = torch.tensor(0.3, device=device)
                        mask = (torch.rand_like(x.float()) < r)
                        need = ~mask.any(dim=1)
                        if need.any():
                            mask[need, 0] = True
                        corrupted = x.masked_fill(mask, MASK_ID)
                        logits = model(corrupted)

                        B, T = x.shape
                        denom = r * (B * T)

                        ce_sum = F.cross_entropy(
                            logits[mask],
                            x[mask],
                            reduction="sum"
                        )
                        loss = ce_sum / denom

                losses.append(loss.item())

        if args.mode == "mdm":
            torch.set_rng_state(rng_state)
        model.train()
        local_sum = sum(losses)
        local_count = len(losses)
        if ddp and dist.is_initialized():
            t_sum = torch.tensor([local_sum], device=device, dtype=torch.float32)
            t_count = torch.tensor([local_count], device=device, dtype=torch.float32)
            dist.all_reduce(t_sum)
            dist.all_reduce(t_count)
            return (t_sum / t_count).item()
        return local_sum / local_count

    t0 = time.time()

    for step in range(1, args.steps + 1):
        x, y = next(train_it)
        opt.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_amp):
            if args.mode == "ar":
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                loss = ar_loss(logits, y)
            else:
                loss = mdm_step(model, x, device)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if is_main and step % 20 == 0:
            toks = args.batch_size * args.seq_len * world_size
            it_s = step / max(time.time() - t0, 1e-6)
            print(
                f"step {step:6d} | loss {loss.item():.4f} "
                f"| tok/step {toks} | it/s {it_s:.2f}"
            )
            wandb.log({
                "train/loss": loss.item(),
                "train/step": step,
                "train/tokens_per_step": toks,
                "train/tokens_seen": step * toks,
            })

        if step % args.eval_every == 0:
            vl = eval_loss()
            if is_main:
                toks = args.batch_size * args.seq_len * world_size
                print(f"[eval] step {step:6d} | val_loss {vl:.4f}")
                wandb.log({
                    "val/loss": vl,
                    "val/step": step,
                    "val/tokens_seen": step * toks,
                })


    ckpt = os.path.join(args.out_dir, f"{args.mode}_final.pt")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
        state = model.module.state_dict() if ddp else model.state_dict()
        torch.save({"args": vars(args), "state_dict": state}, ckpt)
        print("saved", ckpt)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    if is_main:
        wandb.finish()


if __name__ == "__main__":
    main()
