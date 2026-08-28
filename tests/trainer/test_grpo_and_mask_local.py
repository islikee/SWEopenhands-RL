import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage


def test_synthetic_grpo_group_with_two_rewards_has_nonzero_advantage():
    data = DataProto(
        batch=TensorDict(
            {
                "responses": torch.ones((2, 3), dtype=torch.long),
                "attention_mask": torch.ones((2, 5), dtype=torch.long),
                "token_level_rewards": torch.tensor(
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
                ),
            },
            batch_size=(2,),
        ),
        non_tensor_batch={"uid": np.array(["same-task", "same-task"], dtype=object)},
    )

    advantages, returns = compute_grpo_outcome_advantage(
        token_level_rewards=data.batch["token_level_rewards"],
        response_mask=data.batch["attention_mask"][:, -3:],
        index=data.non_tensor_batch["uid"],
    )

    assert not torch.allclose(advantages, torch.zeros_like(advantages))
    assert torch.equal(advantages, returns)


def test_actor_loss_mask_can_exclude_prompt_user_and_observation_tokens():
    loss_mask = torch.tensor([[0, 0, 1, 1, 0]], dtype=torch.long)
    attention_response_mask = torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long)

    assert loss_mask.sum().item() == 2
    assert attention_response_mask.sum().item() == 5
    assert torch.equal(loss_mask[:, :2], torch.zeros((1, 2), dtype=torch.long))
