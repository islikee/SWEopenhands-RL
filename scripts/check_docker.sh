#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SKYRL_ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python3}"
command -v "$PYTHON_BIN" >/dev/null || {
  echo "Missing Python. Run scripts/setup_cloud.sh first." >&2
  exit 1
}
command -v docker >/dev/null || {
  echo "Missing docker." >&2
  exit 1
}

eval "$("$PYTHON_BIN" scripts/cloud_deploy.py env --shell)"

if docker version >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  echo "Docker daemon: OK"
else
  echo "Docker daemon: FAIL"
  echo "Check that Docker is installed, the daemon is running, and this user can access it." >&2
  exit 1
fi

if docker run --rm --pull=missing hello-world >/dev/null; then
  echo "Container smoke: OK"
else
  echo "Container smoke: FAIL" >&2
  exit 1
fi

if SMOKE_IMAGES="$("$PYTHON_BIN" scripts/cloud_deploy.py docker-images)"; then
  while IFS= read -r SMOKE_IMAGE; do
    [[ -n "$SMOKE_IMAGE" ]] || continue
    echo "Smoke image: $SMOKE_IMAGE"
    if docker image inspect "$SMOKE_IMAGE" >/dev/null 2>&1; then
      echo "Image local: yes"
    else
      echo "Image local: no"
      if [[ "${SKYRL_PULL_SMOKE_IMAGE:-1}" == "1" ]]; then
        docker pull "$SMOKE_IMAGE"
      else
        echo "Skipping smoke image pull because SKYRL_PULL_SMOKE_IMAGE=0."
      fi
    fi
  done <<< "$SMOKE_IMAGES"
else
  echo "Smoke image: unavailable until dataset exists"
  echo "Run scripts/setup_cloud.sh or provide SKYRL_SMOKE_DATA_PATH with train.parquet." >&2
  exit 1
fi

echo "Disk usage:"
docker system df || true
df -h "$SKYRL_DATA_ROOT" "$ROOT_DIR" || true
