from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.workers.reward_manager.swebench import SWEBenchRewardManager
from verl.workers.reward_manager.swebench_report import (
    FAILURE_FLAGS,
    classify_trajectory_failure,
    normalize_evaluation_report,
    test_informed_reward_from_facts,
    trajectory_reward_fields,
)
from verl.trainer.ppo.metric_utils import process_validation_metrics
from verl.workers.reward_manager.swebench_test_informed import (
    TestInformedSWEBenchRewardManager,
)


def _config(target_weight=0.8, regression_weight=0.2):
    return SimpleNamespace(
        verifier=SimpleNamespace(reward_coef=1.0),
        reward_model=SimpleNamespace(
            rm_coef=0.0,
            test_informed=SimpleNamespace(
                target_weight=target_weight,
                regression_weight=regression_weight,
            ),
        ),
    )


def _data(non_tensors):
    batch_size = len(next(iter(non_tensors.values())))
    responses = torch.ones((batch_size, 3), dtype=torch.long)
    attention_mask = torch.ones((batch_size, 5), dtype=torch.long)
    return DataProto(
        batch=TensorDict(
            {"responses": responses, "attention_mask": attention_mask},
            batch_size=(batch_size,),
        ),
        non_tensor_batch={k: np.array(v, dtype=object) for k, v in non_tensors.items()},
    )


def test_binary_swebench_reward_manager_keeps_resolved_zero_one_behavior():
    data = _data(
        {
            "resolved": [False, True, False],
            "error": [None, None, "RuntimeError: Agent reached maximum iteration in headless mode"],
            "finish": [False, True, False],
            "ability": ["coding", "coding", "coding"],
        }
    )

    reward_dict, metrics = SWEBenchRewardManager(
        tokenizer=None, num_examine=0, config=_config()
    )(data)

    assert reward_dict["gt_scores"].sum(dim=1).tolist() == [0.0, 1.0, 0.0]
    assert reward_dict["all"].sum(dim=1).tolist() == [0.0, 1.0, 0.0]
    assert metrics["all"] == pytest.approx(1.0 / 3.0)


def test_normalize_evaluation_report_maps_fail_to_pass_and_pass_to_pass_counts():
    facts = normalize_evaluation_report(
        {
            "resolved": False,
            "tests_status": {
                "FAIL_TO_PASS": {
                    "success": ["test_target_a"],
                    "failure": ["test_target_b", "test_target_c"],
                },
                "PASS_TO_PASS": {
                    "success": ["test_regression_a", "test_regression_b"],
                    "failure": ["test_regression_c"],
                },
            },
        }
    )

    assert facts == {
        "resolved": False,
        "target_tests_total": 3,
        "target_tests_passed": 1,
        "target_tests_failed": 2,
        "regression_tests_total": 3,
        "regression_tests_passed": 2,
        "regression_tests_failed": 1,
        "evaluation_error": None,
        "evaluation_timeout": False,
    }


def test_test_informed_reward_handles_partial_target_and_regression_penalty():
    facts = {
        "resolved": False,
        "target_tests_total": 4,
        "target_tests_passed": 2,
        "target_tests_failed": 2,
        "regression_tests_total": 2,
        "regression_tests_passed": 1,
        "regression_tests_failed": 1,
        "evaluation_error": None,
        "evaluation_timeout": False,
    }

    assert test_informed_reward_from_facts(facts, target_weight=0.8, regression_weight=0.2) == 0.45


def test_test_informed_reward_resolved_is_one_and_evaluation_failure_is_zero():
    resolved = {
        "resolved": True,
        "target_tests_total": 1,
        "target_tests_passed": 0,
        "target_tests_failed": 1,
        "regression_tests_total": 1,
        "regression_tests_passed": 0,
        "regression_tests_failed": 1,
        "evaluation_error": None,
        "evaluation_timeout": False,
    }
    errored = {**resolved, "resolved": False, "evaluation_error": "grader crashed"}
    timeout = {**resolved, "resolved": False, "evaluation_timeout": True}

    assert test_informed_reward_from_facts(resolved) == 1.0
    assert test_informed_reward_from_facts(errored) == 0.0
    assert test_informed_reward_from_facts(timeout) == 0.0


def test_test_informed_reward_falls_back_to_binary_without_target_tests():
    facts = {
        "resolved": False,
        "target_tests_total": 0,
        "target_tests_passed": 0,
        "target_tests_failed": 0,
        "regression_tests_total": 3,
        "regression_tests_passed": 3,
        "regression_tests_failed": 0,
        "evaluation_error": None,
        "evaluation_timeout": False,
    }

    assert test_informed_reward_from_facts(facts) == 0.0
    assert test_informed_reward_from_facts({**facts, "resolved": True}) == 1.0


def test_failure_classification_records_multiple_flags_and_primary_outcome():
    assert "target_tests_failed" in FAILURE_FLAGS
    primary, flags = classify_trajectory_failure(
        {
            "resolved": False,
            "git_patch": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n",
            "error": "RuntimeError: Agent reached maximum iteration in headless mode",
            "target_tests_failed": 2,
            "regression_tests_failed": 1,
            "evaluation_timeout": False,
        }
    )

    assert primary == "agent_max_turn"
    assert flags == ["agent_max_turn", "target_tests_failed", "regression_tests_failed"]


def test_trajectory_reward_fields_are_normalized_primitives_for_dataproto():
    fields = trajectory_reward_fields(
        result={
            "resolved": False,
            "git_patch": "",
            "error": "Evaluation timed out after 1200 seconds",
        },
        evaluation_report=None,
        evaluation_error="Evaluation timed out after 1200 seconds",
    )

    assert fields["evaluation_timeout"] is True
    assert fields["outcome_primary"] == "test_timeout"
    assert fields["failure_flags"] == ["test_timeout", "no_patch"]
    for key in [
        "resolved",
        "target_tests_total",
        "target_tests_passed",
        "target_tests_failed",
        "regression_tests_total",
        "regression_tests_passed",
        "regression_tests_failed",
        "evaluation_error",
        "evaluation_timeout",
        "outcome_primary",
        "failure_flags",
        "binary_reward",
        "test_informed_reward",
    ]:
        assert key in fields


def test_test_informed_reward_manager_uses_primitive_fact_fields():
    data = _data(
        {
            "resolved": [False, False, True],
            "target_tests_total": [4, 4, 1],
            "target_tests_passed": [2, 4, 0],
            "target_tests_failed": [2, 0, 1],
            "regression_tests_total": [2, 2, 1],
            "regression_tests_passed": [1, 0, 0],
            "regression_tests_failed": [1, 2, 1],
            "evaluation_error": [None, None, None],
            "evaluation_timeout": [False, False, False],
            "error": [None, None, None],
            "finish": [True, True, True],
            "ability": ["coding", "coding", "coding"],
        }
    )

    reward_dict, metrics = TestInformedSWEBenchRewardManager(
        tokenizer=None, num_examine=0, config=_config()
    )(data)

    assert reward_dict["gt_scores"].sum(dim=1).tolist() == pytest.approx([0.45, 0.8, 1.0])
    assert reward_dict["binary_scores"].sum(dim=1).tolist() == pytest.approx([0.0, 0.0, 1.0])
    assert reward_dict["test_informed_scores"].sum(dim=1).tolist() == pytest.approx([0.45, 0.8, 1.0])
    assert metrics["all"] == pytest.approx(torch.tensor([0.45, 0.8, 1.0]).mean().item())


def test_test_informed_reward_manager_supports_validation_return_dict():
    data = _data(
        {
            "resolved": [False, True],
            "target_tests_total": [4, 1],
            "target_tests_passed": [2, 1],
            "target_tests_failed": [2, 0],
            "regression_tests_total": [2, 1],
            "regression_tests_passed": [2, 1],
            "regression_tests_failed": [0, 0],
            "evaluation_error": [None, None],
            "evaluation_timeout": [False, False],
            "git_patch": ["diff --git a/a.py b/a.py", "diff --git a/b.py b/b.py"],
            "ability": ["coding", "coding"],
        }
    )

    result = TestInformedSWEBenchRewardManager(
        tokenizer=None, num_examine=0, config=_config()
    )(data, return_dict=True)

    assert sorted(result) == ["reward_extra_info", "reward_tensor"]
    assert result["reward_tensor"].sum(dim=1).tolist() == pytest.approx([0.5, 1.0])
    assert result["reward_extra_info"]["test_informed_reward"] == pytest.approx([0.5, 1.0])
    assert result["reward_extra_info"]["binary_reward"] == pytest.approx([0.0, 1.0])
    assert result["reward_extra_info"]["pred"] == [
        "diff --git a/a.py b/a.py",
        "diff --git a/b.py b/b.py",
    ]

    metrics = process_validation_metrics(
        np.array(["swe-gym", "swe-gym"], dtype=object),
        ["prompt-a", "prompt-b"],
        {"final_reward": [0.5, 1.0], **result["reward_extra_info"]},
    )

    assert metrics["swe-gym"]["test_informed_reward"]["mean@1"] == pytest.approx(0.75)
