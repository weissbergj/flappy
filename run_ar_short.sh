#!/usr/bin/env bash
set -euo pipefail

# --- paths (edit if needed) ---
TRAIN_BIN="data/gpt2tok_c4_100_text_document.bin"
VAL_BIN="data/gpt2tok_c4_val_text_document.bin"

# --- short sanity run ---
python -m src.train \
  --mode ar \
  --train_bin "$TRAIN_BIN" \
  --val_bin   "$VAL_BIN" \
  --seq_len 512 \
  --batch_size 64 \
  --steps 800 \
  --eval_every 200 \
  --eval_batches 20 \
  --d_model 512 --n_layers 8 --n_heads 8 \
  --lr 3e-4 \
  --out_dir outputs
