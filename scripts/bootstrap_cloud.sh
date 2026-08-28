#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SKYRL_ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

bash scripts/setup_cloud.sh
bash scripts/check_docker.sh
bash scripts/download_model.sh

echo "Bootstrap complete. Start training explicitly with: bash scripts/run_cloud_smoke.sh"
