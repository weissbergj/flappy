# src/train.py
import os, time, argparse, math
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
EVAL_R = 0.3


def ar_loss(logits, y):
    # Standard next-token cross entropy
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        y.view(-1)
    )


def mdm_step(model, x, device):
    """
    Masked Diffusion LM step (paper-aligned):
    - sample r ~ U(0,1) (clamped to avoid degenerate 0/1)
    - mask tokens with prob r
    - loss ~= (1/r) * mean CE over masked tokens
    """
    x = x.to(device, non_blocking=True)
    targets = x

    r = torch.rand((), device=device).clamp(0.01, 0.99)

    mask = (torch.rand_like(x.float()) < r)   # (B,T)
    need = ~mask.any(dim=1)
    if need.any():
        mask[need, 0] = True

    corrupted = x.masked_fill(mask, MASK_ID)
    logits = model(corrupted)

    ce_mean = F.cross_entropy(
        logits[mask],
        targets[mask],
        reduction="mean"
    )

    loss = ce_mean / r
    masked_frac = mask.float().mean()
    return loss, masked_frac, r


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

    # --- after model is on device (and after DDP wrap) ---
    raw_model = model.module if isinstance(model, DDP) else model
    n_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    if is_main:
        wandb.log({"hparams/n_params": n_params})

    use_amp = device.startswith("cuda")
    autocast_dtype = torch.bfloat16

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )

    warmup_steps = max(1, int(0.02 * args.steps))   # 2% warmup
    min_lr = args.lr * 0.1                          # decay to 10% of base

    def lr_scale(step: int):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, args.steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (min_lr / args.lr) + cosine * (1.0 - (min_lr / args.lr))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_scale)

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
        masked_fracs = [] if args.mode == "mdm" else None

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
                        masked_frac_batch = None
                    else:
                        # FIXED eval corruption rate (stable)
                        r = x.new_tensor(EVAL_R)
                        mask = (torch.rand_like(x.float()) < r)
                        need = ~mask.any(dim=1)
                        if need.any():
                            mask[need, 0] = True

                        corrupted = x.masked_fill(mask, MASK_ID)
                        logits = model(corrupted)

                        # IMPORTANT: report UN-SCALED masked CE for eval (no /r)
                        loss = F.cross_entropy(
                            logits[mask],
                            x[mask],
                            reduction="mean"
                        )
                        masked_frac_batch = mask.float().mean().item()

                losses.append(loss.item())
                if args.mode == "mdm":
                    masked_fracs.append(masked_frac_batch)

        model.train()

        local_sum = sum(losses)
        local_count = len(losses)

        if ddp and dist.is_initialized():
            t_sum = torch.tensor([local_sum], device=device, dtype=torch.float32)
            t_count = torch.tensor([local_count], device=device, dtype=torch.float32)
            dist.all_reduce(t_sum)
            dist.all_reduce(t_count)
            val_loss = (t_sum / t_count).item()
        else:
            val_loss = local_sum / local_count

        if args.mode == "mdm":
            local_mf_sum = sum(masked_fracs)
            if ddp and dist.is_initialized():
                t_mf_sum = torch.tensor([local_mf_sum], device=device, dtype=torch.float32)
                t_mf_cnt = torch.tensor([local_count], device=device, dtype=torch.float32)
                dist.all_reduce(t_mf_sum)
                dist.all_reduce(t_mf_cnt)
                val_masked_frac = (t_mf_sum / t_mf_cnt).item()
            else:
                val_masked_frac = local_mf_sum / local_count
            return val_loss, val_masked_frac

        return val_loss, None

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
                loss, masked_frac, r_mdm = mdm_step(model, x, device)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        if is_main and step % 20 == 0:
            toks = args.batch_size * args.seq_len * world_size
            it_s = step / max(time.time() - t0, 1e-6)
            print(
                f"step {step:6d} | loss {loss.item():.4f} "
                f"| tok/step {toks} | it/s {it_s:.2f}"
            )
            tokens_seen = step * toks
            flops_dense = 6 * n_params * tokens_seen
            flops_pred = flops_dense * (masked_frac.item() if args.mode == "mdm" else 1.0)
            log_dict = {
                "train/loss": loss.item(),
                "train/step": step,
                "train/tokens_per_step": toks,
                "train/tokens_seen": tokens_seen,
                "train/flops_dense": flops_dense,
                "train/flops_pred": flops_pred,
                "train/lr": scheduler.get_last_lr()[0],
            }
            if args.mode == "mdm":
                log_dict["train/masked_frac"] = masked_frac.item()
                log_dict["train/r"] = r_mdm.item()
            wandb.log(log_dict)

        if step % args.eval_every == 0:
            vl, val_masked_frac = eval_loss()
            if is_main:
                toks = args.batch_size * args.seq_len * world_size
                tokens_seen = step * toks
                flops_dense = 6 * n_params * tokens_seen
                flops_pred = flops_dense * (val_masked_frac if val_masked_frac is not None else 1.0)
                print(f"[eval] step {step:6d} | val_loss {vl:.4f}")
                eval_log = {
                    "val/loss": vl,
                    "val/step": step,
                    "val/tokens_seen": tokens_seen,
                    "val/flops_dense": flops_dense,
                    "val/flops_pred": flops_pred,
                }
                if val_masked_frac is not None:
                    eval_log["val/masked_frac"] = val_masked_frac
                wandb.log(eval_log)


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
