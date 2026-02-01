import os, time, math, argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

from src.data.window_dataset import BinTokenWindowDataset
from src.models.ar_transformer import ARTransformerLM
from src.utils_dist import ddp_setup, ddp_cleanup

VOCAB_SIZE = 50257

def ar_loss(logits, y):
    # logits (B,T,V), y (B,T)
    return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_bin", required=True)
    ap.add_argument("--val_bin", required=True)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=64)     # per GPU
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--eval_batches", type=int, default=50)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    is_ddp, rank, local_rank, world_size, device = ddp_setup()
    is_main = (rank == 0)

    model = ARTransformerLM(
        vocab_size=VOCAB_SIZE,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    ).to(device)

    if args.compile and torch.cuda.is_available():
        model = torch.compile(model)

    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.wd)

    train_ds = BinTokenWindowDataset(args.train_bin, seq_len=args.seq_len, seed=0 + rank)
    val_ds   = BinTokenWindowDataset(args.val_bin,   seq_len=args.seq_len, seed=123 + rank)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_ddp else None
    val_sampler   = DistributedSampler(val_ds, shuffle=False) if is_ddp else None

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, shuffle=(train_sampler is None),
                          num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True, drop_last=True)

    train_it = iter(train_dl)

    def eval_loss():
        model.eval()
        losses = []
        with torch.no_grad():
            it = iter(val_dl)
            for _ in range(args.eval_batches):
                x, y = next(it)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    logits = model(x)
                    loss = ar_loss(logits, y)
                losses.append(loss.item())
        model.train()
        return float(sum(losses) / len(losses))

    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        if is_ddp and train_sampler is not None and step % 1000 == 1:
            train_sampler.set_epoch(step)  # cheap reshuffle

        x, y = next(train_it)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            logits = model(x)
            loss = ar_loss(logits, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if is_main and step % 20 == 0:
            toks = args.batch_size * args.seq_len * world_size
            it_s = step / max(time.time() - t0, 1e-6)
            print(f"step {step:6d} | loss {loss.item():.4f} | tok/step {toks} | it/s {it_s:.2f}")

        if step % args.eval_every == 0:
            if is_main:
                vl = eval_loss()
                print(f"[eval] step {step:6d} | val_loss {vl:.4f}")

    if is_main:
        ckpt = os.path.join(args.out_dir, f"ar_final.pt")
        sd = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        torch.save({"args": vars(args), "state_dict": sd}, ckpt)
        print("saved", ckpt)

    ddp_cleanup()

if __name__ == "__main__":
    main()
