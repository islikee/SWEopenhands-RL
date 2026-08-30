from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.ray_trainer import apply_smoke_nonzero_advantage_fallback


def _batch(advantages: torch.Tensor, response_mask: torch.Tensor) -> DataProto:
    rewards = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0]],
        dtype=torch.float32,
    )
    return DataProto(
        batch=TensorDict(
            {
                "advantages": advantages.clone(),
                "returns": advantages.clone(),
                "response_mask": response_mask.clone(),
                "token_level_rewards": rewards.clone(),
            },
            batch_size=(2,),
        ),
        non_tensor_batch={"uid": np.array(["task-a", "task-a"], dtype=object)},
    )


def test_disabled_smoke_fallback_leaves_all_zero_advantages_unchanged():
    data = _batch(
        advantages=torch.zeros((2, 4), dtype=torch.float32),
        response_mask=torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.long),
    )

    result = apply_smoke_nonzero_advantage_fallback(data, enabled=False)

    assert result["applied"] is False
    assert torch.equal(data.batch["advantages"], torch.zeros((2, 4), dtype=torch.float32))


def test_enabled_smoke_fallback_leaves_existing_nonzero_advantage_unchanged():
    original = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [0.0, -0.25, 0.0, 0.0]],
        dtype=torch.float32,
    )
    data = _batch(
        advantages=original,
        response_mask=torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.long),
    )

    result = apply_smoke_nonzero_advantage_fallback(data, enabled=True)

    assert result["applied"] is False
    assert torch.equal(data.batch["advantages"], original)


def test_enabled_smoke_fallback_injects_first_valid_trajectory_deterministically():
    data = _batch(
        advantages=torch.zeros((2, 4), dtype=torch.float32),
        response_mask=torch.tensor([[0, 0, 0, 0], [1, 1, 0, 0]], dtype=torch.long),
    )

    result = apply_smoke_nonzero_advantage_fallback(data, enabled=True)

    assert result == {
        "applied": True,
        "before_max_abs": 0.0,
        "after_max_abs": 1.0,
        "trajectory_index": 1,
        "valid_token_count": 2,
    }
    assert torch.equal(
        data.batch["advantages"],
        torch.tensor(
            [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
    )


def test_enabled_smoke_fallback_does_not_modify_padding_or_rewards():
    data = _batch(
        advantages=torch.zeros((2, 4), dtype=torch.float32),
        response_mask=torch.tensor([[1, 0, 1, 0], [1, 1, 0, 0]], dtype=torch.long),
    )
    original_rewards = data.batch["token_level_rewards"].clone()

    apply_smoke_nonzero_advantage_fallback(data, enabled=True)

    assert torch.equal(
        data.batch["advantages"],
        torch.tensor(
            [[1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(data.batch["token_level_rewards"], original_rewards)
