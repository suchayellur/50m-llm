#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-runs/final/final}"
OUTPUT_DIR="${2:-eval_results}"

mkdir -p "$OUTPUT_DIR"
lm_eval \
  --model hf \
  --model_args "pretrained=$MODEL_DIR,trust_remote_code=True" \
  --tasks hellaswag,arc_easy,piqa,winogrande,wikitext \
  --device cuda:0 \
  --batch_size auto \
  --output_path "$OUTPUT_DIR"
