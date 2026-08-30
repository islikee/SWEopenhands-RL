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
CKPT_PATH="${SKYRL_CKPT_PATH:-$OUTPUT_DIR/checkpoints/real_pilot}"
MODEL_PATH="${SKYRL_MODEL_PATH:-$SKYRL_MODEL_LOCAL_DIR}"
DATA_PATH="${SKYRL_SMOKE_DATA_PATH}"

if [[ "$MODEL_PATH" == "Qwen/Qwen2.5-Coder-7B-Instruct" ]]; then
  MODEL_PATH="$SKYRL_MODEL_LOCAL_DIR"
fi

mkdir -p "$OUTPUT_DIR" "$CKPT_PATH"

export SKYRL_OPENHANDS_RUNTIME=docker
export SKYRL_REQUIRE_ROLLOUT_WEIGHT_SYNC=1
export SKYRL_REQUIRE_ROLLOUT_WEIGHT_CHANGE_AFTER_FIRST_SYNC=1
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=1

echo "Running realistic training pilot preflight."
"$PYTHON_BIN" scripts/cloud_deploy.py preflight-run

COMMIT_SHA="$(git rev-parse HEAD)"
echo "commit=$COMMIT_SHA"
echo "model path=$MODEL_PATH"
echo "smoke dataset path=$DATA_PATH"
echo "output path=$OUTPUT_DIR"
echo "checkpoint=$CKPT_PATH"
echo "training_mode=lora"
echo "reward_manager=swebench_test_informed"
echo "GPU count=$GPU_COUNT"
echo "train batch=$GPU_COUNT"
echo "ppo mini batch=$GPU_COUNT"
echo "entropy_coeff=0.0"
echo "actor param offload=true"
echo "actor optimizer offload=true"
echo "ref param offload=true"
echo "prompt_key=problem_statement"
echo "dataloader_num_workers=0"
echo "n_trajectories=8"
echo "max_iterations=25"
echo "max prompt=8192"
echo "max response=1024"
echo "max starting message=12000"
echo "rollout tensor parallel size=2"
echo "rollout max total tokens=32768"
echo "rollout max prefill tokens=18432"
echo "rollout disable cuda graph=true"
echo "rollout weight exchange size=500000000"
echo "sglang disable tp memory imbalance check=true"
echo "rollout_weight_sync_required=true"
echo "rollout_weight_change_after_first_sync_required=true"
echo "force_nonzero_advantage_if_all_zero=false"

PYTHONUNBUFFERED=1 "$PYTHON_BIN" \
  -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="[\"$DATA_PATH/train.parquet\"]" \
  data.val_files="[\"$DATA_PATH/validation.parquet\"]" \
  data.prompt_key=problem_statement \
  data.train_batch_size="$GPU_COUNT" \
  data.max_prompt_length=8192 \
  data.max_response_length=1024 \
  data.dataloader_num_workers=0 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.training_mode=lora \
  actor_rollout_ref.exchange_size=500000000 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$GPU_COUNT" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.masking=True \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=async \
  actor_rollout_ref.rollout.n_trajectories=8 \
  actor_rollout_ref.rollout.max_iterations=25 \
  actor_rollout_ref.rollout.max_parallel_agents="$MAX_PARALLEL_AGENTS" \
  actor_rollout_ref.rollout.max_eval_parallel_agents="$MAX_PARALLEL_AGENTS" \
  +actor_rollout_ref.rollout.max_starting_message_length=12000 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.enable_memory_saver=True \
  actor_rollout_ref.rollout.max_total_tokens=32768 \
  actor_rollout_ref.rollout.max_prefill_tokens=18432 \
  actor_rollout_ref.rollout.disable_cuda_graph=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  reward_model.reward_manager=swebench_test_informed \
  +smoke.require_nonzero_lora_delta=False \
  +smoke.require_base_unchanged=True \
  +smoke.force_nonzero_advantage_if_all_zero=False \
  trainer.total_training_steps=1 \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node="$GPU_COUNT" \
  trainer.save_freq=-1 \
  trainer.logger='["console"]' \
  trainer.default_local_dir="$CKPT_PATH" \
  "$@"

echo "REALISTIC TRAINING PILOT FINISHED"
