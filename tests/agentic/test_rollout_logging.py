from __future__ import annotations

import json
from pathlib import Path

from verl.workers.agentic.rollout_logging import (
    aggregate_rollout_records,
    format_rollout_summary,
    make_rollout_record,
    write_trajectory_trace,
    write_rollout_jsonl,
)


def test_make_rollout_record_keeps_compact_facts_and_omits_heavy_transcript_payload():
    result = {
        "instance_id": "getmoto__moto-4950",
        "trajectory_id": 7,
        "finish_reason": "context_limit",
        "truncated": True,
        "turns": 21,
        "context_tokens": 20031,
        "context_limit": 20000,
        "action_counts": {"FileReadAction": 7, "CmdRunAction": 5},
        "git_patch": "diff --git a/a.py b/a.py\n" + "x" * 500,
        "patch_size": 526,
        "resolved": False,
        "evaluation_error": None,
        "target_tests_passed": 0,
        "target_tests_total": 1,
        "regression_tests_passed": 6,
        "regression_tests_total": 6,
        "binary_reward": 0.0,
        "test_informed_reward": 0.8,
        "selected_training_reward": 0.8,
        "messages": [{"role": "assistant", "content": "large transcript"}],
        "state": object(),
    }

    record = make_rollout_record(
        result,
        step=3,
        phase="train",
        trace_path="outputs/rollouts/run/step_0003/getmoto__moto-4950_traj7.json",
    )

    assert record["step"] == 3
    assert record["phase"] == "train"
    assert record["instance_id"] == "getmoto__moto-4950"
    assert record["trajectory"] == 7
    assert record["finish_reason"] == "context_limit"
    assert record["truncated"] is True
    assert record["context"] == {"tokens": 20031, "limit": 20000}
    assert record["patch"] == {"exists": True, "size": 526}
    assert record["eval"]["status"] == "not_resolved"
    assert record["eval"]["fail_to_pass"] == "0/1"
    assert record["eval"]["pass_to_pass"] == "6/6"
    assert record["reward"]["test_informed"] == 0.8
    assert record["trace_path"].endswith("getmoto__moto-4950_traj7.json")
    assert "messages" not in record
    assert "git_patch" not in record
    assert "state" not in record


def test_make_rollout_record_reports_resolved_before_eval_error_when_tests_passed():
    record = make_rollout_record(
        {
            "instance_id": "conan-io__conan-13403",
            "trajectory_id": 0,
            "finish_reason": "unknown",
            "turns": 22,
            "resolved": True,
            "evaluation_error": "RuntimeError: Agent reached maximum iteration in headless mode",
            "target_tests_passed": 1,
            "target_tests_total": 1,
            "regression_tests_passed": 0,
            "regression_tests_total": 0,
            "binary_reward": 1.0,
            "test_informed_reward": 1.0,
            "selected_training_reward": 1.0,
        },
        step=0,
        phase="validation",
    )

    assert record["eval"]["status"] == "resolved"
    assert record["eval"]["error"] == "RuntimeError: Agent reached maximum iteration in headless mode"


def test_rollout_aggregation_and_summary_are_short_but_informative():
    records = [
        {
            "step": 8,
            "phase": "train",
            "instance_id": "a",
            "trajectory": 0,
            "finish_reason": "model_finish",
            "truncated": False,
            "patch": {"exists": True, "size": 10},
            "eval": {"status": "resolved"},
            "reward": {"all": 1.0, "test_informed": 1.0},
        },
        {
            "step": 8,
            "phase": "train",
            "instance_id": "b",
            "trajectory": 1,
            "finish_reason": "context_limit",
            "truncated": True,
            "patch": {"exists": False, "size": 0},
            "eval": {"status": "eval_error"},
            "reward": {"all": 0.0, "test_informed": 0.0},
        },
    ]

    aggregate = aggregate_rollout_records(records)
    summary = format_rollout_summary(step=8, phase="train", records=records, aggregate=aggregate)

    assert aggregate["rollout/trajectory_count"] == 2
    assert aggregate["rollout/truncated_count"] == 1
    assert aggregate["rollout/truncated_rate"] == 0.5
    assert aggregate["rollout/eval_error_count"] == 1
    assert aggregate["rollout/patch_exists_rate"] == 0.5
    assert aggregate["rollout/reward_mean"] == 0.5
    assert "STEP 8 TRAIN ROLLOUT SUMMARY" in summary
    assert "truncated=1/2" in summary
    assert "eval_error=1" in summary
    assert "a | 0 | model_finish" in summary
    assert "b | 1 | context_limit" in summary


def test_write_rollout_jsonl_appends_one_machine_readable_record_per_line(tmp_path):
    path = tmp_path / "rollout_summary.jsonl"
    records = [
        {"step": 0, "phase": "validation", "instance_id": "task-a", "trajectory": 0},
        {"step": 0, "phase": "validation", "instance_id": "task-b", "trajectory": 0},
    ]

    write_rollout_jsonl(path, records)
    write_rollout_jsonl(path, records[:1])

    lines = path.read_text().splitlines()
    assert len(lines) == 3
    assert [json.loads(line)["instance_id"] for line in lines] == ["task-a", "task-b", "task-a"]


def test_write_trajectory_trace_keeps_full_debug_payload_under_step_directory(tmp_path):
    result = {
        "instance_id": "getmoto__moto-4950",
        "trajectory_id": 7,
        "messages": [{"role": "assistant", "content": "full transcript"}],
        "git_patch": "diff --git a/a.py b/a.py\n",
        "state": object(),
    }

    trace_path = write_trajectory_trace(tmp_path, step=3, result=result)

    assert trace_path is not None
    assert trace_path.endswith("step_0003/getmoto__moto-4950_traj7.json")
    payload = json.loads(Path(trace_path).read_text())
    assert payload["messages"][0]["content"] == "full transcript"
    assert payload["git_patch"].startswith("diff --git")
    assert "state" in payload
