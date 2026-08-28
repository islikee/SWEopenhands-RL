#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SKYRL_ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"
mkdir -p outputs logs checkpoints

echo "Running local Python smoke tests only."
"${PYTHON:-python}" -m pytest \
  tests/reward_manager/test_swebench_test_informed.py \
  tests/reward_manager/test_reward_manager_registration.py \
  tests/workers/test_lora_training_and_sync.py \
  tests/workers/test_lora_fsdp_wiring_static.py \
  tests/trainer/test_grpo_and_mask_local.py \
  tests/agentic/test_codeact_reward_fields_static.py \
  tests/agentic/test_runtime_instance_payload.py \
  tests/agentic/test_result_normalization.py \
  tests/config/test_local_cloud_smoke_files.py \
  tests/docs/test_baseline_freeze_doc.py \
  -q
