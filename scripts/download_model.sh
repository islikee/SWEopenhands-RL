#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${SKYRL_MODEL_PATH:-Qwen/Qwen2.5-Coder-7B-Instruct}"
LOCAL_DIR="${SKYRL_MODEL_LOCAL_DIR:-}"

echo "model=$MODEL_PATH"
echo "local_dir=${LOCAL_DIR:-<huggingface-cache>}"

if [[ -n "$LOCAL_DIR" ]]; then
  huggingface-cli download "$MODEL_PATH" --local-dir "$LOCAL_DIR"
else
  huggingface-cli download "$MODEL_PATH"
fi
