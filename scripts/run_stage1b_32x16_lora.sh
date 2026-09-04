#!/usr/bin/env bash
set -euo pipefail

# Stage1B keeps the Stage1 32x16 baseline name while training from the
# derived 64-task candidate pool.  The implementation and all overrides live
# in the executable launcher below so there is one source of truth.
# Contract: stage1b_candidates.parquet, validation.parquet, n_trajectories=8,
# target_informative_groups=4, max_candidate_groups_per_update=8,
# max_iterations=15, agent_max_prompt_length=32768,
# reward_manager=swebench_stage1b, algorithm.use_kl_in_reward=False.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_stage1b_64_lora_nokl.sh" "$@"
