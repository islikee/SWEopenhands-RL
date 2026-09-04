from pathlib import Path
import importlib.util

import pandas as pd
import yaml


MANIFEST_PATH = Path("configs/stage1_32x16_tasks.yaml")
TRAIN_SCRIPT_PATH = Path("scripts/run_stage1_32x16_lora.sh")
STAGE1B_TRAIN_SCRIPT_PATH = Path("scripts/run_stage1b_64_lora_nokl.sh")
PREPARE_SCRIPT_PATH = Path("scripts/prepare_stage1_split.py")

_SPEC = importlib.util.spec_from_file_location("prepare_stage1_split", PREPARE_SCRIPT_PATH)
prepare_stage1_split = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(prepare_stage1_split)


def test_stage1_task_manifest_has_non_overlapping_train_and_validation_tasks():
    manifest = yaml.safe_load(MANIFEST_PATH.read_text())

    universe = manifest["stage1_universe"]
    train = manifest["train"]
    validation = manifest["validation"]

    assert len(universe) == 80
    assert len(train) == 32
    assert len(validation) == 16
    assert len(set(universe)) == 80
    assert len(set(train)) == 32
    assert len(set(validation)) == 16
    assert set(train).isdisjoint(set(validation))
    assert set(train).issubset(set(universe))
    assert set(validation).issubset(set(universe))


def test_stage1_prepare_script_declares_non_overlap_guard_and_output_files():
    source = PREPARE_SCRIPT_PATH.read_text()

    assert "set(train_ids).isdisjoint(validation_ids)" in source
    assert "train.parquet" in source
    assert "validation.parquet" in source
    assert "train_instance_ids.txt" in source
    assert "validation_instance_ids.txt" in source
    assert "stage1b_candidates.parquet" in source
    assert "stage1b_candidate_instance_ids.txt" in source


def _instance_ids(frame: pd.DataFrame) -> list[str]:
    return [str(instance["instance_id"]) for instance in frame["instance"].tolist()]


def test_stage1_prepare_script_writes_non_overlapping_train_and_validation_parquets(tmp_path):
    output_dir = prepare_stage1_split.prepare_split(MANIFEST_PATH, None, tmp_path / "stage1")

    train = pd.read_parquet(output_dir / "train.parquet")
    validation = pd.read_parquet(output_dir / "validation.parquet")
    train_ids = _instance_ids(train)
    validation_ids = _instance_ids(validation)

    assert len(train_ids) == 32
    assert len(validation_ids) == 16
    assert set(train_ids).isdisjoint(validation_ids)
    assert train_ids == (output_dir / "train_instance_ids.txt").read_text().splitlines()
    assert validation_ids == (output_dir / "validation_instance_ids.txt").read_text().splitlines()

    candidates = pd.read_parquet(output_dir / "stage1b_candidates.parquet")
    candidate_ids = _instance_ids(candidates)
    manifest = yaml.safe_load(MANIFEST_PATH.read_text())
    universe = set(manifest["stage1_universe"])
    old_train = set(manifest["train"])
    validation_ids_set = set(validation_ids)
    unused = universe - old_train - validation_ids_set
    expected_candidates = old_train | unused

    assert len(unused) == 32
    assert len(candidate_ids) == 64
    assert set(candidate_ids) == expected_candidates
    assert set(candidate_ids).isdisjoint(validation_ids_set)
    assert set(candidate_ids) | validation_ids_set == universe
    assert candidate_ids == (output_dir / "stage1b_candidate_instance_ids.txt").read_text().splitlines()


def test_stage1_training_script_matches_base_lora_nokl_experiment_contract():
    script = TRAIN_SCRIPT_PATH.read_text()

    assert "SWE-Gym-OpenHands-7B-Agent" in script
    assert "skyrl-v0-80-stage1-32x16" in script
    assert "data.train_batch_size=4" in script
    assert "actor_rollout_ref.actor.ppo_mini_batch_size=4" in script
    assert "actor_rollout_ref.rollout.n_trajectories=8" in script
    assert "trainer.total_training_steps=8" in script
    assert "actor_rollout_ref.actor.ppo_epochs=1" in script
    assert "actor_rollout_ref.actor.optim.lr=1e-6" in script
    assert "actor_rollout_ref.model.training_mode=lora" in script
    assert "actor_rollout_ref.actor.use_kl_loss=False" in script
    assert "algorithm.use_kl_in_reward=False" in script
    assert "actor_rollout_ref.rollout.max_iterations=22" in script
    assert "+actor_rollout_ref.rollout.agent_max_prompt_length=20000" in script
    assert "+actor_rollout_ref.rollout.max_starting_message_length=10000" in script
    assert "actor_rollout_ref.rollout.temperature=0.5" in script
    assert "actor_rollout_ref.rollout.sampling_params.temperature=0.5" in script
    assert "actor_rollout_ref.rollout.val_kwargs.n=1" in script
    assert "actor_rollout_ref.rollout.val_kwargs.temperature=0" in script
    assert "trainer.val_before_train=True" in script
    assert "trainer.test_freq=8" in script
    assert "trainer.save_freq=8" in script
    assert 'trainer.logger=\'["console","wandb"]\'' in script
    assert 'WANDB_API_KEY:-}" == "<wandb_api_key>"' in script
    assert "actor_rollout_ref.rollout.log_messages_dir=" in script
    assert "+trainer.rollout_log_dir=" in script
    assert 'GPU_COUNT="${SKYRL_GPUS_PER_NODE:-$(detect_visible_gpu_count)}"' in script
    assert 'GPU_COUNT="${SKYRL_GPUS_PER_NODE:-4}"' not in script


def test_stage1b_training_script_defaults_to_four_parallel_agents():
    script = STAGE1B_TRAIN_SCRIPT_PATH.read_text()

    assert 'MAX_PARALLEL_AGENTS="${SKYRL_MAX_PARALLEL_AGENTS:-4}"' in script
    assert 'MAX_EVAL_PARALLEL_AGENTS="${SKYRL_MAX_EVAL_PARALLEL_AGENTS:-$MAX_PARALLEL_AGENTS}"' in script
    assert 'actor_rollout_ref.rollout.max_parallel_agents="$MAX_PARALLEL_AGENTS"' in script
    assert 'actor_rollout_ref.rollout.max_eval_parallel_agents="$MAX_EVAL_PARALLEL_AGENTS"' in script


def test_stage1b_training_script_detects_gpu_count_unless_explicitly_overridden():
    script = STAGE1B_TRAIN_SCRIPT_PATH.read_text()

    assert 'detect_visible_gpu_count()' in script
    assert 'GPU_COUNT="${SKYRL_GPUS_PER_NODE:-$(detect_visible_gpu_count)}"' in script
    assert 'trainer.n_gpus_per_node="$GPU_COUNT"' in script
    assert 'GPU_COUNT="${SKYRL_GPUS_PER_NODE:-4}"' not in script
