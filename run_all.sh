#!/usr/bin/env bash
set -euo pipefail

# Set these (or override via env vars when running)
TRAIN_BIN="${TRAIN_BIN:-/home/ubuntu/Jared/attempt/data/gpt2tok_c4_100_text_document.bin}"
VAL_BIN="${VAL_BIN:-/home/ubuntu/Jared/attempt/data/gpt2tok_c4_val_text_document.bin}"

NPROC="${NPROC:-8}"
SEQ_LEN="${SEQ_LEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS="${STEPS:-40000}"
OFFSET_TOKENS="${OFFSET_TOKENS:-0}"
OUT_ROOT="${OUT_ROOT:-outputs}"

for mode in ar mdm; do
  for tokens in 25000000 50000000 100000000; do
    tag="$((tokens/1000000))M"
    out_dir="${OUT_ROOT}/${mode}_${tag}_update"
    mkdir -p "$out_dir"

    torchrun --standalone --nproc_per_node="$NPROC" -m src.train \
      --mode "$mode" \
      --train_bin "$TRAIN_BIN" \
      --val_bin "$VAL_BIN" \
      --seq_len "$SEQ_LEN" \
      --batch_size "$BATCH_SIZE" \
      --steps "$STEPS" \
      --offset_tokens "$OFFSET_TOKENS" \
      --max_tokens "$tokens" \
      --out_dir "$out_dir" \
      2>&1 | tee "${out_dir}/train.log"
  done
done
