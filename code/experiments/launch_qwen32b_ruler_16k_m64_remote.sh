#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/jhl/model_inference/llm_infer
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
exec /home/jhl/anaconda3/envs/qwen-opt/bin/python -u run_qwen32b_ruler_multigpu.py \
  --runner-dir "$ROOT" \
  --model-path /data/share/Qwen3-32B \
  --data-dir "$ROOT/ruler_data_8b_16k" \
  --output-dir "$ROOT/results/qwen32b_ruler_16k_m64_first100" \
  --length 16384 --m 64 --source-samples 500 --samples 100 "$@"
