#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SKYRL_ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
command -v "$PYTHON_BIN" >/dev/null || {
  echo "Missing Python. Run scripts/setup_cloud.sh first." >&2
  exit 1
}

eval "$("$PYTHON_BIN" scripts/cloud_deploy.py env --shell)"
if [[ -f .env ]]; then
  EXISTING_SANDBOX_REMOTE_RUNTIME_API_URL="${SANDBOX_REMOTE_RUNTIME_API_URL:-}"
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
  if [[ "${SANDBOX_REMOTE_RUNTIME_API_URL:-}" == "<insert_remote_sandbox_url>" && -n "$EXISTING_SANDBOX_REMOTE_RUNTIME_API_URL" ]]; then
    export SANDBOX_REMOTE_RUNTIME_API_URL="$EXISTING_SANDBOX_REMOTE_RUNTIME_API_URL"
  fi
fi

GPU_COUNT="${SKYRL_GPUS_PER_NODE:-4}"
MAX_PARALLEL_AGENTS="${SKYRL_MAX_PARALLEL_AGENTS:-$GPU_COUNT}"
CKPT_PATH="${SKYRL_CKPT_PATH:-$OUTPUT_DIR/checkpoints/stage1_32x16_lora_nokl}"
MODEL_PATH="${SKYRL_MODEL_PATH:-/data/skyrl/models/NovaSky-AI/SWE-Gym-OpenHands-7B-Agent}"
DATA_PATH="${SKYRL_STAGE1_DATA_PATH:-$DATASET_DIR/skyrl-v0-80-stage1-32x16}"

mkdir -p "$OUTPUT_DIR" "$CKPT_PATH"

export SKYRL_OPENHANDS_RUNTIME=docker
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=1

"$PYTHON_BIN" scripts/prepare_stage1_split.py --output "$DATA_PATH"

echo "Running stage1 32x16 LoRA no-KL training."
echo "model=$MODEL_PATH"
echo "dataset=$DATA_PATH"
echo "train_tasks=32"
echo "validation_tasks=16"
echo "batch=4"
echo "n_trajectories=8"
echo "total_training_steps=8"
echo "ppo_epochs=1"
echo "temperature=0.5"
echo "lr=1e-6"
echo "lora=on"
echo "kl=off"
echo "max_iterations=22"
echo "agent_max_prompt_length=20000"
echo "validation_n=1"
echo "validation_temperature=0"
echo "validation_steps=0,8"

PYTHONUNBUFFERED=1 "$PYTHON_BIN" \
  -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="[\"$DATA_PATH/train.parquet\"]" \
  data.val_files="[\"$DATA_PATH/validation.parquet\"]" \
  data.prompt_key=prompt \
  data.train_batch_size=4 \
  data.max_prompt_length=8192 \
  data.max_response_length=1024 \
  data.dataloader_num_workers=0 \
  data.shuffle=False \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.training_mode=lora \
  actor_rollout_ref.model.lora_rank="${SKYRL_LORA_RANK:-16}" \
  actor_rollout_ref.model.lora_alpha="${SKYRL_LORA_ALPHA:-32}" \
  actor_rollout_ref.model.lora_dropout="${SKYRL_LORA_DROPOUT:-0.0}" \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.masking=True \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.name=async \
  actor_rollout_ref.rollout.task_type=swegym \
  actor_rollout_ref.rollout.n_trajectories=8 \
  actor_rollout_ref.rollout.max_iterations=22 \
  actor_rollout_ref.rollout.agent_max_prompt_length=20000 \
  actor_rollout_ref.rollout.max_parallel_agents="$MAX_PARALLEL_AGENTS" \
  actor_rollout_ref.rollout.max_eval_parallel_agents="$MAX_PARALLEL_AGENTS" \
  actor_rollout_ref.rollout.max_starting_message_length=10000 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.enable_memory_saver=True \
  actor_rollout_ref.rollout.max_total_tokens=32768 \
  actor_rollout_ref.rollout.max_prefill_tokens=18432 \
  actor_rollout_ref.rollout.disable_cuda_graph=True \
  actor_rollout_ref.rollout.temperature=0.5 \
  actor_rollout_ref.rollout.sampling_params.temperature=0.5 \
  actor_rollout_ref.rollout.sampling_params.top_p=0.95 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  reward_model.reward_manager=swebench_test_informed \
  trainer.total_epochs=1 \
  trainer.total_training_steps=8 \
  trainer.val_before_train=True \
  trainer.test_freq=8 \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node="$GPU_COUNT" \
  trainer.save_freq=-1 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${WANDB_PROJECT:-skyrl-swegym-stage1}" \
  trainer.experiment_name="${WANDB_NAME:-oh7b_stage1_32x16_lora_nokl}" \
  trainer.default_local_dir="$CKPT_PATH" \
  "$@"

echo "STAGE1 32x16 LORA NOKL TRAINING FINISHED"
