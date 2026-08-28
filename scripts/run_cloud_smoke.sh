#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SKYRL_ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_PATH="${SKYRL_DATA_PATH:?Set SKYRL_DATA_PATH to the SWE-Gym parquet directory}"
CKPT_PATH="${SKYRL_CKPT_PATH:-$ROOT_DIR/checkpoints/cloud_smoke}"
MODEL_PATH="${SKYRL_MODEL_PATH:-Qwen/Qwen2.5-Coder-7B-Instruct}"
GPU_COUNT="${SKYRL_GPUS_PER_NODE:-4}"
MAX_PARALLEL_AGENTS="${SKYRL_MAX_PARALLEL_AGENTS:-$GPU_COUNT}"

cd "$ROOT_DIR"
mkdir -p outputs logs "$CKPT_PATH"

export SKYRL_REQUIRE_ROLLOUT_WEIGHT_SYNC=1
export SKYRL_REQUIRE_ROLLOUT_WEIGHT_CHANGE_AFTER_FIRST_SYNC=1

echo "model=$MODEL_PATH"
echo "data=$DATA_PATH"
echo "checkpoint=$CKPT_PATH"
echo "rollout_weight_sync_required=true"

PYTHONUNBUFFERED=1 uv run --isolated --directory . --frozen --env-file .env -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="[\"$DATA_PATH/train.parquet\"]" \
  data.val_files="[\"$DATA_PATH/validation.parquet\"]" \
  data.train_batch_size="$GPU_COUNT" \
  data.max_prompt_length=8192 \
  data.max_response_length=1024 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.training_mode=lora \
  actor_rollout_ref.actor.ppo_mini_batch_size="$GPU_COUNT" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.masking=True \
  actor_rollout_ref.rollout.name=async \
  actor_rollout_ref.rollout.n_trajectories=2 \
  actor_rollout_ref.rollout.max_iterations=3 \
  actor_rollout_ref.rollout.max_parallel_agents="$MAX_PARALLEL_AGENTS" \
  actor_rollout_ref.rollout.max_eval_parallel_agents="$MAX_PARALLEL_AGENTS" \
  +actor_rollout_ref.rollout.max_starting_message_length=12000 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.enable_memory_saver=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  reward_model.reward_manager=swebench_test_informed \
  trainer.total_training_steps=2 \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node="$GPU_COUNT" \
  trainer.save_freq=1 \
  trainer.logger='["console"]' \
  trainer.default_local_dir="$CKPT_PATH" \
  "$@"
