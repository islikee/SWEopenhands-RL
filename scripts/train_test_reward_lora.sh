#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SKYRL_ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_PATH="${SKYRL_DATA_PATH:?Set SKYRL_DATA_PATH to the SWE-Gym parquet directory}"
CKPT_PATH="${SKYRL_CKPT_PATH:-$ROOT_DIR/checkpoints/test_reward_lora}"
MODEL_PATH="${SKYRL_MODEL_PATH:-Qwen/Qwen2.5-Coder-7B-Instruct}"

cd "$ROOT_DIR"
mkdir -p outputs logs "$CKPT_PATH"
echo "reward_manager=swebench_test_informed"
echo "model=$MODEL_PATH"

bash examples/sky/run_skyrl_agent_oh7b_s1.sh \
  data.train_files="[\"$DATA_PATH/train.parquet\"]" \
  data.val_files="[\"$DATA_PATH/validation.parquet\"]" \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.training_mode=lora \
  reward_model.reward_manager=swebench_test_informed \
  trainer.default_local_dir="$CKPT_PATH" \
  "$@"
