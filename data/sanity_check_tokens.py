#!/usr/bin/env python3
import argparse, os, random, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2TokenizerFast, GPT2Config, GPT2LMHeadModel

"""
python3 sanity_check_tokens.py   --train-bin gpt2tok_c4_100_text_document.bin   --train-idx gpt2tok_c4_100_text_document.idx   --val-bin gpt2tok_c4_val_text_document.bin   --val-idx gpt2tok_c4_val_text_document.idx   --seq-len 512

this tells us .bin is a flat gpt-2 token stream (uint16) with correct shift
"""

def human_bytes(n):
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024: return f"{n:.2f}{unit}"
        n /= 1024
    return f"{n:.2f}PB"

def try_load_flat_bin(bin_path, vocab_size, try_dtypes=("int32","uint16","int16","uint32")):
    size = os.path.getsize(bin_path)
    for dt in try_dtypes:
        dtype = np.dtype(dt)
        if size % dtype.itemsize != 0:
            continue
        arr = np.memmap(bin_path, mode="r", dtype=dtype)
        # quick stats from a slice
        sl = np.array(arr[:50000], dtype=np.int64)
        mn, mx = int(sl.min()), int(sl.max())
        frac_in_vocab = float(((0 <= sl) & (sl < vocab_size)).mean())
        # heuristics: should mostly be in range
        if frac_in_vocab > 0.98 and mx < vocab_size + 10_000 and mn >= 0:
            return arr, dtype
    raise RuntimeError("Could not find a plausible dtype for token IDs. Try setting --vocab-size correctly or inspect the .bin format.")

def decode_preview(tok, ids, n=200):
    text = tok.decode([int(x) for x in ids[:n]])
    # shorten whitespace for display
    return text.replace("\n", "\\n")[:800]

def check_files(train_bin, train_idx, val_bin, val_idx):
    print("== File check ==")
    for p in [train_bin, train_idx, val_bin, val_idx]:
        exists = os.path.exists(p)
        print(f"{p}: {'OK' if exists else 'MISSING'}")
        if exists:
            print(f"  size: {human_bytes(os.path.getsize(p))}")
    print()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-bin", required=True)
    ap.add_argument("--train-idx", required=True)
    ap.add_argument("--val-bin", required=True)
    ap.add_argument("--val-idx", required=True)
    ap.add_argument("--vocab-size", type=int, default=50257)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--overfit-steps", type=int, default=300)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    check_files(args.train_bin, args.train_idx, args.val_bin, args.val_idx)

    print("== Load flat .bin as token stream (ignoring .idx) ==")
    train_tokens, train_dtype = try_load_flat_bin(args.train_bin, args.vocab_size)
    val_tokens, val_dtype = try_load_flat_bin(args.val_bin, args.vocab_size)
    print(f"train dtype: {train_dtype}, tokens: {len(train_tokens):,}")
    print(f"val   dtype: {val_dtype}, tokens: {len(val_tokens):,}")
    print()

    print("== Range / plausibility check ==")
    sample = np.array(train_tokens[:200000], dtype=np.int64)
    print(f"min={sample.min()} max={sample.max()} in_vocab={( (0<=sample)&(sample<args.vocab_size) ).mean():.4f}")
    print()

    print("== Decode check (GPT-2 tokenizer) ==")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    print("train preview:", decode_preview(tok, sample, n=200))
    print()

    print("== Window + shift check ==")
    i = random.randint(0, len(train_tokens) - (args.seq_len + 2))
    x = np.array(train_tokens[i:i+args.seq_len], dtype=np.int64)
    y = np.array(train_tokens[i+1:i+args.seq_len+1], dtype=np.int64)
    assert np.all(y[:-1] == x[1:]), "Shift check failed: y is not x shifted by 1."
    print("Shift check: OK")
    print("x preview:", decode_preview(tok, x[:150], n=150))
    print("y preview:", decode_preview(tok, y[:150], n=150))
    print()

    print("== Quick overfit-on-one-batch test ==")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # Tiny GPT2-ish model (small, fast)
    cfg = GPT2Config(
        vocab_size=args.vocab_size,
        n_positions=args.seq_len,
        n_ctx=args.seq_len,
        n_embd=384,
        n_layer=6,
        n_head=6,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    model = GPT2LMHeadModel(cfg).to(device)
    model.train()

    # fixed batch: consecutive windows
    starts = [random.randint(0, len(train_tokens) - (args.seq_len + 2)) for _ in range(args.micro_batch)]
    xb = np.stack([np.array(train_tokens[s:s+args.seq_len], dtype=np.int64) for s in starts], axis=0)
    yb = np.stack([np.array(train_tokens[s+1:s+args.seq_len+1], dtype=np.int64) for s in starts], axis=0)
    xb = torch.tensor(xb, device=device)
    yb = torch.tensor(yb, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    for t in range(1, args.overfit_steps + 1):
        opt.zero_grad(set_to_none=True)
        out = model(input_ids=xb)
        logits = out.logits  # [B, T, V]
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
        loss.backward()
        opt.step()
        if t == 1 or t % 50 == 0:
            print(f"step {t:4d} | loss {loss.item():.4f}")

    print("\nIf loss dropped a lot (e.g. ~10 -> ~3-6), data+labels are very likely correct.")

if __name__ == "__main__":
    main()
