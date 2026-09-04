import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
from verl.trainer.ppo.metric_utils import compute_data_metrics
from verl.trainer.ppo.stage1b_sampling import apply_stage1b_training_masks, effective_train_tokens


def test_grpo_normalizes_only_valid_trajectories_in_each_task_group():
    rewards = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [0.0, 0.9], [0.0, 0.2], [0.0, 0.2]],
        dtype=torch.float32,
    )
    response_mask = torch.ones_like(rewards)
    valid = torch.tensor([True, True, False, True, True])
    index = np.asarray(["task-a", "task-a", "task-a", "task-b", "task-b"], dtype=object)

    advantages, returns = compute_grpo_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=index,
        sample_valid_mask=valid,
    )

    assert torch.equal(advantages, returns)
    assert torch.equal(advantages[2], torch.zeros(2))
    assert not torch.allclose(advantages[0], torch.zeros(2))
    assert torch.equal(advantages[3], torch.zeros(2))
    assert torch.equal(advantages[4], torch.zeros(2))


def test_effective_train_tokens_counts_only_final_loss_mask_positions():
    final_loss_mask = torch.tensor(
        [[1, 1, 0, 0], [0, 1, 0, 1], [0, 0, 0, 0]], dtype=torch.bool
    )

    assert effective_train_tokens(final_loss_mask) == 4


def test_stage1b_masks_rejected_and_invalid_trajectories_from_actor_loss():
    data = DataProto(
        batch=TensorDict(
            {
                "responses": torch.ones((2, 3), dtype=torch.long),
                "attention_mask": torch.ones((2, 5), dtype=torch.long),
                "loss_mask": torch.ones((2, 3), dtype=torch.bool),
            },
            batch_size=(2,),
        ),
        non_tensor_batch={
            "reward_valid": np.asarray([True, False], dtype=object),
            "accepted_group": np.asarray([True, False], dtype=object),
        },
    )

    apply_stage1b_training_masks(data)

    assert data.batch["response_mask"].sum().item() == 3
    assert data.batch["loss_mask"].sum().item() == 3


def test_stage1b_data_metrics_exclude_reward_invalid_trajectory_scores():
    data = DataProto(
        batch=TensorDict(
            {
                "responses": torch.ones((3, 2), dtype=torch.long),
                "attention_mask": torch.ones((3, 4), dtype=torch.long),
                "loss_mask": torch.ones((3, 2), dtype=torch.bool),
                "token_level_scores": torch.tensor(
                    [[0.0, 0.2], [0.0, 0.9], [0.0, 0.0]], dtype=torch.float32
                ),
                "token_level_rewards": torch.tensor(
                    [[0.0, 0.2], [0.0, 0.9], [0.0, 0.0]], dtype=torch.float32
                ),
                "advantages": torch.ones((3, 2), dtype=torch.float32),
                "returns": torch.ones((3, 2), dtype=torch.float32),
            },
            batch_size=(3,),
        ),
        non_tensor_batch={
            "reward_valid": np.asarray([True, True, False], dtype=object),
            "accepted_group": np.asarray([True, True, True], dtype=object),
        },
    )
    apply_stage1b_training_masks(data)

    metrics = compute_data_metrics(data, use_critic=False)

    assert metrics["critic/score/mean"] == pytest.approx(0.55)
    assert metrics["critic/score/min"] == pytest.approx(0.2)
    assert metrics["critic/score/max"] == pytest.approx(0.9)
    assert metrics["critic/rewards/mean"] == pytest.approx(0.55)
