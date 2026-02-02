#!/usr/bin/env bash
set -euo pipefail

# Paths (override via env if needed)
TRAIN_BIN="${TRAIN_BIN:-/home/ubuntu/Jared/attempt/data/gpt2tok_c4_100_text_document.bin}"
VAL_BIN="${VAL_BIN:-/home/ubuntu/Jared/attempt/data/gpt2tok_c4_val_text_document.bin}"

# Hardware / training
NPROC="${NPROC:-8}"
SEQ_LEN="${SEQ_LEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS="${STEPS:-20000}"
OFFSET_TOKENS="${OFFSET_TOKENS:-0}"
TOKENS="${TOKENS:-25000000}"   # fixed data regime
OUT_ROOT="${OUT_ROOT:-outputs}"

# -------- AR ordering sweep --------
for N in 6; do
  out_dir="${OUT_ROOT}/ar_perm_N${N}_100M"
  mkdir -p "$out_dir"

  torchrun --standalone --nproc_per_node="$NPROC" -m src.train_permute \
    --mode ar \
    --num_orders "$N" \
    --train_bin "$TRAIN_BIN" \
    --val_bin "$VAL_BIN" \
    --seq_len "$SEQ_LEN" \
    --batch_size "$BATCH_SIZE" \
    --steps "$STEPS" \
    --offset_tokens "$OFFSET_TOKENS" \
    --max_tokens "$TOKENS" \
    --out_dir "$out_dir" \
    2>&1 | tee "${out_dir}/train.log"
done