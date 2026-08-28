#!/usr/bin/env bash
set -euo pipefail

docker version >/dev/null
docker info >/dev/null
df -h .
echo "Docker and disk checks passed."
