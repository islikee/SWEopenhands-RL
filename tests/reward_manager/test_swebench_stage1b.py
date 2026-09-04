from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.workers.reward_manager.swebench_report import (
    normalize_stage1b_report,
    reward_v2_from_facts,
)
from verl.workers.reward_manager.swebench_stage1b import Stage1BSWEBenchRewardManager


def _config():
    return SimpleNamespace(
        verifier=SimpleNamespace(reward_coef=1.0),
        reward_model=SimpleNamespace(rm_coef=0.0),
    )


def _data(non_tensors):
    size = len(next(iter(non_tensors.values())))
    return DataProto(
        batch=TensorDict(
            {
                "responses": torch.ones((size, 3), dtype=torch.long),
                "attention_mask": torch.ones((size, 5), dtype=torch.long),
            },
            batch_size=(size,),
        ),
        non_tensor_batch={key: np.asarray(value, dtype=object) for key, value in non_tensors.items()},
    )


def _facts(**overrides):
    facts = {
        "resolved": False,
        "ftp_passed": 1,
        "ftp_total": 2,
        "ptp_passed": 2,
        "ptp_total": 2,
        "ptp_failed": 0,
    }
    facts.update(overrides)
    return facts


def test_reward_v2_returns_one_for_resolved_report():
    assert reward_v2_from_facts(_facts(resolved=True)) == 1.0


def test_reward_v2_uses_nine_tenths_when_no_regression_failed():
    assert reward_v2_from_facts(_facts(ftp_passed=1, ftp_total=2, ptp_failed=0)) == pytest.approx(0.45)
    assert reward_v2_from_facts(_facts(ftp_passed=1, ftp_total=1, ptp_failed=0)) == pytest.approx(0.9)


def test_reward_v2_penalizes_regression_using_preservation_fraction():
    assert reward_v2_from_facts(
        _facts(ftp_passed=1, ftp_total=1, ptp_passed=628, ptp_total=629, ptp_failed=1)
    ) == pytest.approx(0.6 + 0.1 * 628 / 629)
    assert reward_v2_from_facts(
        _facts(ftp_passed=1, ftp_total=2, ptp_passed=4, ptp_total=5, ptp_failed=1)
    ) == pytest.approx(0.34)


def test_stage1b_report_normalization_requires_target_and_regression_suites():
    facts = normalize_stage1b_report(
        {
            "resolved": False,
            "tests_status": {
                "FAIL_TO_PASS": {"success": ["target_a"], "failure": ["target_b"]},
                "PASS_TO_PASS": {"success": ["reg_a"], "failure": ["reg_b"]},
            },
        }
    )

    assert facts == {
        "resolved": False,
        "ftp_passed": 1,
        "ftp_total": 2,
        "ftp_failed": 1,
        "ptp_passed": 1,
        "ptp_total": 2,
        "ptp_failed": 1,
    }
    with pytest.raises(ValueError, match="FAIL_TO_PASS"):
        normalize_stage1b_report({"resolved": False, "tests_status": {"PASS_TO_PASS": {}}})


def test_stage1b_reward_manager_excludes_reward_invalid_trajectory():
    data = _data(
        {
            "reward_valid": [True, False],
            "resolved": [False, False],
            "ftp_passed": [1, 1],
            "ftp_total": [2, 2],
            "ftp_failed": [1, 1],
            "ptp_passed": [2, 2],
            "ptp_total": [2, 2],
            "ptp_failed": [0, 0],
            "evaluation_error": [None, "infrastructure failed"],
            "git_patch": ["patch-a", "patch-b"],
        }
    )

    reward_dict, metrics = Stage1BSWEBenchRewardManager(
        tokenizer=None, num_examine=0, config=_config()
    )(data)

    assert reward_dict["gt_scores"].sum(dim=1).tolist() == pytest.approx([0.45, 0.0])
    assert reward_dict["all"].sum(dim=1).tolist() == pytest.approx([0.45, 0.0])
    assert metrics["reward_invalid_count"] == 1
    assert metrics["reward_v2"] == pytest.approx(0.45)
    assert metrics["all"] == pytest.approx(0.45)
    assert metrics["reward_all"] == pytest.approx(0.45)
