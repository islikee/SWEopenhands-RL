from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.stage1b_sampling import Stage1BSampler


class _Dataset:
    def __init__(self, task_ids):
        self.task_ids = task_ids

    def __getitem__(self, index):
        return {
            "input_ids": torch.ones(4, dtype=torch.long) * index,
            "attention_mask": torch.ones(4, dtype=torch.long),
            "position_ids": torch.arange(4, dtype=torch.long),
            "instance": {"instance_id": self.task_ids[index]},
        }


class _Rollout:
    def generate_sequences(self, gen_batch):
        task_id = gen_batch.non_tensor_batch["instance"][0]["instance_id"]
        value = 0.0 if task_id.startswith("flat") else 0.5
        size = 8
        return DataProto(
            batch=TensorDict(
                {
                    "input_ids": torch.ones((size, 4), dtype=torch.long),
                    "responses": torch.ones((size, 3), dtype=torch.long),
                    "attention_mask": torch.ones((size, 7), dtype=torch.long),
                    "position_ids": torch.arange(7, dtype=torch.long).repeat(size, 1),
                },
                batch_size=(size,),
            ),
            non_tensor_batch={
                "reward_valid": np.asarray([True] * size, dtype=object),
                "resolved": np.asarray([False] * size, dtype=object),
                "finish_reason": np.asarray([None] * size, dtype=object),
                "infra_error": np.asarray([False] * size, dtype=object),
                "git_patch": np.asarray(["patch"] * size, dtype=object),
            },
        )


class _TwoGpuRollout:
    world_size = 2

    def __init__(self):
        self.prompt_count = None

    def generate_sequences(self, gen_batch):
        self.prompt_count = len(gen_batch)
        assert len(gen_batch) == self.world_size
        size = 16
        return DataProto(
            batch=TensorDict(
                {
                    "input_ids": torch.ones((size, 4), dtype=torch.long),
                    "responses": torch.ones((size, 3), dtype=torch.long),
                    "attention_mask": torch.ones((size, 7), dtype=torch.long),
                    "position_ids": torch.arange(7, dtype=torch.long).repeat(size, 1),
                },
                batch_size=(size,),
            ),
            non_tensor_batch={
                "reward_valid": np.asarray([True] * size, dtype=object),
                "resolved": np.asarray([False] * size, dtype=object),
                "finish_reason": np.asarray([None] * size, dtype=object),
                "infra_error": np.asarray([False] * size, dtype=object),
                "git_patch": np.asarray(["patch"] * size, dtype=object),
            },
        )


def test_stage1b_single_candidate_rollout_is_padded_for_two_gpu_worker_group():
    trainer = object.__new__(RayPPOTrainer)
    trainer.stage1b_dataset_indices = {"informative-0": 0}
    trainer.stage1b_reward_epsilon = 1e-6
    trainer.train_dataset = _Dataset(["informative-0"])
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(task_type="swegym", n_trajectories=8)
        )
    )
    rollout = _TwoGpuRollout()
    trainer.actor_wg = rollout
    trainer.rollout_wg = rollout
    trainer.global_steps = 1

    def reward_fn(data):
        assert len(data) == 8
        scores = torch.zeros((8, 3), dtype=torch.float32)
        scores[:, -1] = torch.linspace(0.0, 0.7, steps=8)
        return {"all": scores}, {"reward_v2": 0.35}

    trainer.reward_fn = reward_fn

    group = trainer._stage1b_candidate_group("informative-0", {})

    assert rollout.prompt_count == 2
    assert len(group["payload"]) == 8
    assert len(group["trajectories"]) == 8
    assert group["reward_valid"] == [True] * 8


def test_trainer_candidate_batch_replaces_rejected_groups_and_returns_4x8():
    trainer = object.__new__(RayPPOTrainer)
    trainer.stage1b_sampler = Stage1BSampler(
        ["flat-0", "flat-1", "flat-2", "flat-3", "informative-0", "informative-1", "informative-2", "informative-3"],
        target_groups=4,
        max_candidates=8,
        seed=4,
    )
    trainer.stage1b_dataset_indices = {
        task_id: index for index, task_id in enumerate(trainer.stage1b_sampler.candidate_ids)
    }
    trainer.stage1b_reward_epsilon = 1e-6
    trainer.train_dataset = _Dataset(trainer.stage1b_sampler.candidate_ids)
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(task_type="swegym", n_trajectories=8)
        )
    )
    trainer.actor_wg = _Rollout()
    trainer.rollout_wg = trainer.actor_wg
    trainer.global_steps = 1

    def reward_fn(data):
        task_id = data.non_tensor_batch["uid"][0]
        if task_id.startswith("flat"):
            values = [0.0] * 8
            metric_value = -1.0
        else:
            values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
            metric_value = float(task_id.rsplit("-", 1)[1])
        scores = torch.zeros((8, 3), dtype=torch.float32)
        scores[:, -1] = torch.tensor(values)
        return {"all": scores}, {"reward_v2": metric_value}

    trainer.reward_fn = reward_fn
    batch, metrics = trainer._stage1b_candidate_batch(1, {})

    assert batch is not None
    assert len(batch) == 32
    assert metrics["train/accepted_group_count"] == 4
    assert metrics["train/candidate_group_count"] >= 4
    assert metrics["train/underfilled_informative_batch"] == 0
    assert batch.non_tensor_batch["uid"].tolist().count("flat-0") == 0
    assert batch.batch["loss_mask"].sum().item() == 32 * 3
    assert batch.meta_info["stage1b_reward_metrics"]["reward_v2"] == pytest.approx(1.5)
