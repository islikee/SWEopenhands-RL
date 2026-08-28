#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SKYRL_ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python3}"
command -v "$PYTHON_BIN" >/dev/null || {
  echo "Missing Python. Run scripts/setup_cloud.sh first." >&2
  exit 1
}

eval "$("$PYTHON_BIN" scripts/cloud_deploy.py env --shell)"
"$PYTHON_BIN" scripts/cloud_deploy.py summary

export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

if [[ "${SKYRL_HF_DOWNLOADER:-xet-tool}" == "xet-tool" ]]; then
  command -v uv >/dev/null || {
    echo "Missing uv. Run scripts/setup_cloud.sh first." >&2
    exit 1
  }
  echo "Downloading model with an isolated Hugging Face downloader environment."
  uv run --no-project --with "huggingface_hub[hf_xet]>=0.32.0" \
    python "$ROOT_DIR/scripts/cloud_deploy.py" download-model
else
  echo "Downloading model with the frozen project environment."
  uv run --isolated --directory "$ROOT_DIR" --frozen \
    python scripts/cloud_deploy.py download-model
fi
