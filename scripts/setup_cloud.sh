#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "setup_cloud.sh is Linux-only. Run local pytest smoke on Windows instead." >&2
  exit 1
fi

ROOT_DIR="${SKYRL_ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python3}"
command -v "$PYTHON_BIN" >/dev/null || {
  echo "Missing Python. Install Python 3.12 before setup." >&2
  exit 1
}
command -v git >/dev/null || {
  echo "Missing git." >&2
  exit 1
}
command -v docker >/dev/null || {
  echo "Missing docker. Install Docker before running cloud smoke." >&2
  exit 1
}
command -v nvidia-smi >/dev/null || {
  echo "Missing nvidia-smi. Use a Linux GPU host with NVIDIA drivers." >&2
  exit 1
}

GPU_COUNT="$(nvidia-smi -L | grep -c '^GPU ' || true)"
if [[ "$GPU_COUNT" -lt 1 ]]; then
  echo "No visible NVIDIA GPUs." >&2
  exit 1
fi

if ! command -v uv >/dev/null; then
  echo "Installing uv into the current user environment."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
command -v uv >/dev/null || {
  echo "uv installation did not put uv on PATH." >&2
  exit 1
}

eval "$("$PYTHON_BIN" scripts/cloud_deploy.py env --shell)"
python_for_helper="$PYTHON_BIN"

echo "Repository: $ROOT_DIR"
git rev-parse HEAD
git submodule status
git status --short
echo "GPU count: $GPU_COUNT"
df -h "$SKYRL_DATA_ROOT" "$ROOT_DIR" || true

"$python_for_helper" scripts/cloud_deploy.py summary

export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

echo "Syncing frozen uv.lock environment."
uv sync --frozen

echo "Verifying frozen dependency versions."
uv run --isolated --directory "$ROOT_DIR" --frozen python scripts/cloud_deploy.py check-deps

if [[ "${SKYRL_SKIP_DATASET_DOWNLOAD:-0}" != "1" ]]; then
  echo "Preparing smoke parquet data under SKYRL_DATA_PATH."
  uv run --isolated --directory "$ROOT_DIR" --frozen python scripts/cloud_deploy.py prepare-dataset
else
  echo "Skipping dataset download because SKYRL_SKIP_DATASET_DOWNLOAD=1."
fi

echo "setup_cloud.sh complete. Next: bash scripts/check_docker.sh"
