from pathlib import Path

import yaml


def test_smoke_configs_exist_and_define_training_modes_and_rollout_sync_gate():
    for config_path in [
        Path("configs/local_smoke.yaml"),
        Path("configs/cloud_smoke.yaml"),
        Path("configs/cloud_train.yaml"),
    ]:
        assert config_path.exists(), f"missing {config_path}"

    local = Path("configs/local_smoke.yaml").read_text()
    cloud = Path("configs/cloud_smoke.yaml").read_text()

    assert "training_mode: lora" in cloud
    assert "n_trajectories: 2" in local
    assert "total_training_steps: 2" in cloud
    assert "rollout_weight_sync_required: true" in cloud
    assert "rollout_weight_change_after_first_sync_required: true" in cloud


def test_cloud_smoke_allocates_at_least_one_agent_slot_per_default_dp_rank():
    cloud = yaml.safe_load(Path("configs/cloud_smoke.yaml").read_text())
    rollout = cloud["actor_rollout_ref"]["rollout"]

    assert rollout["max_parallel_agents"] >= 4
    assert rollout["max_eval_parallel_agents"] >= 4


def test_cloud_smoke_enables_sglang_memory_saver_for_hybrid_fsdp_rollout():
    cloud = yaml.safe_load(Path("configs/cloud_smoke.yaml").read_text())

    assert cloud["actor_rollout_ref"]["rollout"]["enable_memory_saver"] is True
    assert "actor_rollout_ref.rollout.enable_memory_saver=True" in Path(
        "scripts/run_cloud_smoke.sh"
    ).read_text()


def test_cloud_smoke_uses_explicit_a800_rollout_token_pool_and_tp_overrides():
    cloud = yaml.safe_load(Path("configs/cloud_smoke.yaml").read_text())
    rollout = cloud["actor_rollout_ref"]["rollout"]
    script = Path("scripts/run_cloud_smoke.sh").read_text()

    assert cloud["actor_rollout_ref"]["exchange_size"] == 500000000
    assert rollout["tensor_model_parallel_size"] == 2
    assert rollout["max_total_tokens"] == 32768
    assert rollout["max_prefill_tokens"] == 18432
    assert rollout["disable_cuda_graph"] is True
    assert "actor_rollout_ref.exchange_size=500000000" in script
    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=2" in script
    assert "actor_rollout_ref.rollout.max_total_tokens=32768" in script
    assert "actor_rollout_ref.rollout.max_prefill_tokens=18432" in script
    assert "actor_rollout_ref.rollout.disable_cuda_graph=True" in script


def test_cloud_smoke_disables_sglang_tp_memory_balance_guard_for_colocated_a800():
    script = Path("scripts/run_cloud_smoke.sh").read_text()

    assert "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=1" in script


def test_cloud_smoke_declares_async_rollout_starting_message_limit():
    cloud = yaml.safe_load(Path("configs/cloud_smoke.yaml").read_text())

    assert cloud["actor_rollout_ref"]["rollout"]["max_starting_message_length"] > 0
    assert "+actor_rollout_ref.rollout.max_starting_message_length=" in Path(
        "scripts/run_cloud_smoke.sh"
    ).read_text()


def test_cloud_smoke_sets_frozen_trainer_batch_and_microbatch_requirements():
    script = Path("scripts/run_cloud_smoke.sh").read_text()

    assert 'GPU_COUNT="${SKYRL_GPUS_PER_NODE:-4}"' in script
    assert 'PYTHON_BIN="${PYTHON:-$ROOT_DIR/.venv/bin/python}"' in script
    assert "uv run --isolated" not in script
    assert "SKYRL_OPENHANDS_RUNTIME=docker" in script
    assert "data.prompt_key=problem_statement" in script
    assert "data.dataloader_num_workers=0" in script
    assert 'data.train_batch_size="$GPU_COUNT"' in script
    assert 'trainer.n_gpus_per_node="$GPU_COUNT"' in script
    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=2" in script
    assert "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1" in script
    assert "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1" in script


def test_local_smoke_override_is_valid_for_frozen_single_gpu_validator():
    local = yaml.safe_load(Path("configs/local_smoke.yaml").read_text())
    actor = local["actor_rollout_ref"]

    assert local["data"]["train_batch_size"] >= actor["actor"]["ppo_mini_batch_size"]
    assert local["trainer"]["n_gpus_per_node"] == 1
    assert actor["rollout"]["max_starting_message_length"] > 0
    assert actor["ref"]["log_prob_micro_batch_size_per_gpu"] == 1
    assert actor["rollout"]["log_prob_micro_batch_size_per_gpu"] == 1


def test_cloud_smoke_override_is_valid_for_frozen_batch_validator():
    cloud = yaml.safe_load(Path("configs/cloud_smoke.yaml").read_text())
    actor = cloud["actor_rollout_ref"]
    trainer = cloud["trainer"]

    assert cloud["data"]["train_batch_size"] >= actor["actor"]["ppo_mini_batch_size"]
    assert cloud["data"]["train_batch_size"] % trainer["n_gpus_per_node"] == 0
    assert actor["ref"]["log_prob_micro_batch_size_per_gpu"] == 1
    assert actor["rollout"]["log_prob_micro_batch_size_per_gpu"] == 1


def test_cloud_smoke_disables_checkpoint_saves_for_two_step_engineering_smoke():
    cloud = yaml.safe_load(Path("configs/cloud_smoke.yaml").read_text())
    script = Path("scripts/run_cloud_smoke.sh").read_text()

    assert cloud["trainer"]["save_freq"] == -1
    assert "trainer.save_freq=-1" in script
    assert "trainer.save_freq=1" not in script


def test_zero_advantage_fallback_is_not_enabled_in_smoke_or_production_training():
    cloud_smoke = yaml.safe_load(Path("configs/cloud_smoke.yaml").read_text())
    script = Path("scripts/run_cloud_smoke.sh").read_text()
    cloud_train = yaml.safe_load(Path("configs/cloud_train.yaml").read_text())

    assert "force_nonzero_advantage_if_all_zero" not in cloud_smoke.get("smoke", {})
    assert "force_nonzero_advantage_if_all_zero" not in cloud_train.get("smoke", {})
    assert "force_nonzero_advantage_if_all_zero" not in script


def test_cloud_train_defaults_are_valid_for_frozen_validator():
    cloud_train = yaml.safe_load(Path("configs/cloud_train.yaml").read_text())

    assert cloud_train["data"]["train_batch_size"] == "${oc.env:SKYRL_TRAIN_BATCH_SIZE,4}"
    assert cloud_train["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] == (
        "${oc.env:SKYRL_PPO_MINI_BATCH_SIZE,4}"
    )
    assert cloud_train["actor_rollout_ref"]["ref"]["log_prob_micro_batch_size_per_gpu"] == 1
    assert cloud_train["actor_rollout_ref"]["rollout"]["log_prob_micro_batch_size_per_gpu"] == 1


def test_cloud_scripts_exist_and_avoid_embedding_tokens():
    for script_path in [
        Path("scripts/setup_cloud.sh"),
        Path("scripts/check_docker.sh"),
        Path("scripts/download_model.sh"),
        Path("scripts/run_local_smoke.sh"),
        Path("scripts/run_cloud_smoke.sh"),
        Path("scripts/train_binary_lora.sh"),
        Path("scripts/train_test_reward_lora.sh"),
    ]:
        assert script_path.exists(), f"missing {script_path}"
        text = script_path.read_text()
        assert "set -euo pipefail" in text
        assert "API_KEY=" not in text
        assert "TOKEN=" not in text

    cloud_smoke = Path("scripts/run_cloud_smoke.sh").read_text()
    assert "SKYRL_REQUIRE_ROLLOUT_WEIGHT_SYNC=1" in cloud_smoke
    assert "SKYRL_REQUIRE_ROLLOUT_WEIGHT_CHANGE_AFTER_FIRST_SYNC=1" in cloud_smoke
    assert "trainer.total_training_steps=2" in cloud_smoke
    assert "actor_rollout_ref.rollout.n_trajectories=2" in cloud_smoke
