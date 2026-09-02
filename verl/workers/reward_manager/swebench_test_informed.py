import statistics
from typing import Any

import torch
from verl import DataProto

from .swebench_report import (
    binary_reward_from_resolved,
    test_informed_reward_from_facts,
)


def _getattr_or_get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if hasattr(obj, "get"):
        return obj.get(name, default)
    return getattr(obj, name, default)


class TestInformedSWEBenchRewardManager:
    """Map normalized SWE-Bench test facts to a scalar verifier reward."""

    __test__ = False

    def __init__(self, tokenizer, num_examine, config, compute_score=None) -> None:
        self.data_source = "SWE-Gym/SWE-Gym"
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.config = config
        reward_model_cfg = _getattr_or_get(config, "reward_model")
        shaped_cfg = _getattr_or_get(reward_model_cfg, "test_informed")
        self.target_weight = float(_getattr_or_get(shaped_cfg, "target_weight", 0.8))
        self.regression_weight = float(_getattr_or_get(shaped_cfg, "regression_weight", 0.2))

    def verify(self, data: DataProto):
        resolved = data.non_tensor_batch["resolved"]
        scores = []
        binary_scores = []
        for i, is_resolved in enumerate(resolved):
            facts = {
                "resolved": is_resolved,
                "target_tests_total": data.non_tensor_batch["target_tests_total"][i],
                "target_tests_passed": data.non_tensor_batch["target_tests_passed"][i],
                "target_tests_failed": data.non_tensor_batch["target_tests_failed"][i],
                "regression_tests_total": data.non_tensor_batch["regression_tests_total"][i],
                "regression_tests_passed": data.non_tensor_batch["regression_tests_passed"][i],
                "regression_tests_failed": data.non_tensor_batch["regression_tests_failed"][i],
                "evaluation_error": data.non_tensor_batch["evaluation_error"][i],
                "evaluation_timeout": data.non_tensor_batch["evaluation_timeout"][i],
            }
            scores.append(
                test_informed_reward_from_facts(
                    facts,
                    target_weight=self.target_weight,
                    regression_weight=self.regression_weight,
                )
            )
            binary_scores.append(binary_reward_from_resolved(is_resolved))

        reward_metrics = {}
        data.batch["acc"] = torch.tensor(
            scores, dtype=torch.float32, device=data.batch["responses"].device
        )
        data.batch["binary_acc"] = torch.tensor(
            binary_scores, dtype=torch.float32, device=data.batch["responses"].device
        )
        if "ability" in data.non_tensor_batch:
            for ability in list(set(data.non_tensor_batch["ability"])):
                ability_scores = [
                    data.batch["acc"][i].item()
                    for i in range(len(data.batch["acc"]))
                    if data.non_tensor_batch["ability"][i] == ability
                ]
                reward_metrics[f"{ability}"] = statistics.mean(ability_scores)
        reward_metrics["binary_reward"] = data.batch["binary_acc"].mean().item()
        reward_metrics["test_informed_reward"] = data.batch["acc"].mean().item()
        reward_metrics["all"] = data.batch["acc"].mean().item()
        return scores, binary_scores, reward_metrics

    def __call__(self, data: DataProto, return_dict: bool = False):
        reward_tensor_dict = {}
        reward_metrics = {}
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        verifier_reward = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        binary_reward = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        response_ids = data.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data.batch["attention_mask"][:, -response_length:].sum(-1)

        verifier_score, binary_score, verifier_metrics = self.verify(data)
        reward_metrics.update(verifier_metrics)
        for i in range(verifier_reward.shape[0]):
            reward_index = valid_response_length[i] - 1
            verifier_reward[i, reward_index] = verifier_score[i]
            binary_reward[i, reward_index] = binary_score[i]

        reward_tensor_dict["gt_scores"] = verifier_reward
        reward_tensor_dict["binary_scores"] = binary_reward
        reward_tensor_dict["test_informed_scores"] = verifier_reward.clone()

        if "rm_scores" in data.batch.keys():
            reward_tensor_dict["rm_scores"] = data.batch["rm_scores"]
            reward_metrics["reward_model"] = data.batch["rm_scores"].sum(dim=1).mean().item()
            if self.config.reward_model.rm_coef != 0:
                reward_tensor += self.config.reward_model.rm_coef * reward_tensor_dict["rm_scores"]

        if self.config.verifier.reward_coef != 0:
            reward_metrics["verifier"] = reward_tensor_dict["gt_scores"].sum(dim=1).mean().item()
            reward_tensor += self.config.verifier.reward_coef * reward_tensor_dict["gt_scores"]

        reward_tensor_dict["all"] = reward_tensor
        reward_metrics["reward_all"] = reward_tensor.sum(dim=-1).mean(dim=0).item()
        if return_dict:
            patches = data.non_tensor_batch.get("git_patch", [None] * len(verifier_score))
            reward_extra_info = {
                "binary_reward": [float(score) for score in binary_score],
                "test_informed_reward": [float(score) for score in verifier_score],
                "verifier": [float(score) for score in verifier_score],
                "pred": [patch if isinstance(patch, str) else "" for patch in patches],
            }
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        return reward_tensor_dict, reward_metrics
