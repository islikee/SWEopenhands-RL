#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SKYRL_ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

mkdir -p outputs logs checkpoints

echo "root=$ROOT_DIR"
echo "python=${PYTHON:-python}"
echo "This script intentionally does not embed credentials."
echo "Install the frozen SkyRL-v0 environment on a Linux/CUDA host before cloud smoke."
echo "Do not run full uv sync on Windows local smoke machines."
