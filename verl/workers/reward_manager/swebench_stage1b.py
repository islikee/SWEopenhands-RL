from __future__ import annotations

import statistics
from typing import Any

import torch

from verl import DataProto

from .swebench_report import binary_reward_from_resolved, reward_v2_from_facts


def _value(values: dict[str, Any], key: str, index: int) -> Any:
    return values[key][index]


class Stage1BSWEBenchRewardManager:
    """Reward manager for Stage1B validity and dense reward semantics."""

    __test__ = False

    def __init__(self, tokenizer, num_examine, config, compute_score=None) -> None:
        self.data_source = "SWE-Gym/SWE-Gym"
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.config = config

    def verify(self, data: DataProto):
        fields = data.non_tensor_batch
        size = len(data.batch["responses"])
        valid = fields.get("reward_valid", [True] * size)
        scores: list[float] = []
        binary_scores: list[float] = []
        invalid_count = 0
        for index in range(size):
            if not bool(valid[index]):
                invalid_count += 1
                scores.append(0.0)
                binary_scores.append(0.0)
                continue
            if not fields.get("git_patch", [None] * size)[index]:
                scores.append(0.0)
                binary_scores.append(0.0)
                continue
            facts = {
                key: _value(fields, key, index)
                for key in (
                    "resolved",
                    "ftp_passed",
                    "ftp_total",
                    "ftp_failed",
                    "ptp_passed",
                    "ptp_total",
                    "ptp_failed",
                )
            }
            scores.append(reward_v2_from_facts(facts))
            binary_scores.append(binary_reward_from_resolved(facts["resolved"]))

        score_tensor = torch.tensor(scores, dtype=torch.float32, device=data.batch["responses"].device)
        data.batch["acc"] = score_tensor
        data.batch["binary_acc"] = torch.tensor(
            binary_scores, dtype=torch.float32, device=data.batch["responses"].device
        )
        reward_metrics: dict[str, Any] = {
            "reward_invalid_count": invalid_count,
            "reward_v2": float(score_tensor.mean().item()),
            "all": float(score_tensor.mean().item()),
        }
        if "ability" in fields:
            for ability in set(fields["ability"]):
                ability_scores = [scores[i] for i in range(size) if fields["ability"][i] == ability]
                reward_metrics[str(ability)] = statistics.mean(ability_scores)
        return scores, binary_scores, reward_metrics

    def __call__(self, data: DataProto, return_dict: bool = False):
        scores, binary_scores, reward_metrics = self.verify(data)
        response_length = data.batch["responses"].shape[-1]
        valid_response_length = data.batch["attention_mask"][:, -response_length:].sum(-1)
        verifier_reward = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        binary_reward = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        for index in range(len(scores)):
            reward_index = valid_response_length[index] - 1
            verifier_reward[index, reward_index] = scores[index]
            binary_reward[index, reward_index] = binary_scores[index]

        reward_tensor = verifier_reward * float(self.config.verifier.reward_coef)
        reward_tensor_dict = {
            "gt_scores": verifier_reward,
            "binary_scores": binary_reward,
            "test_informed_scores": verifier_reward.clone(),
            "all": reward_tensor,
        }
        reward_metrics["reward_all"] = float(reward_tensor.sum(dim=-1).mean().item())
        if return_dict:
            patches = data.non_tensor_batch.get("git_patch", [""] * len(scores))
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {
                    "binary_reward": binary_scores,
                    "test_informed_reward": scores,
                    "verifier": scores,
                    "reward_valid": [bool(value) for value in data.non_tensor_batch.get("reward_valid", [True] * len(scores))],
                    "pred": [patch if isinstance(patch, str) else "" for patch in patches],
                },
            }
        return reward_tensor_dict, reward_metrics
