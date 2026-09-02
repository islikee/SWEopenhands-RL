from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


HEAVY_RESULT_KEYS = {"messages", "state", "git_patch", "evaluation_report"}


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _to_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _patch_size(result: dict[str, Any]) -> int:
    if "patch_size" in result:
        return _to_int(result.get("patch_size"))
    patch = result.get("git_patch")
    return len(patch.strip()) if isinstance(patch, str) else 0


def _eval_status(result: dict[str, Any]) -> str:
    if bool(result.get("resolved", False)):
        return "resolved"
    if result.get("evaluation_error") or result.get("eval_error"):
        return "eval_error"
    return "not_resolved"


def _ratio(passed: Any, total: Any) -> str:
    return f"{_to_int(passed)}/{_to_int(total)}"


def make_rollout_record(
    result: dict[str, Any],
    *,
    step: int,
    phase: str,
    trace_path: str | None = None,
) -> dict[str, Any]:
    patch_size = _patch_size(result)
    context_tokens = result.get("context_tokens")
    context_limit = result.get("context_limit")
    record = {
        "step": int(step),
        "phase": phase,
        "instance_id": str(result.get("instance_id", "")),
        "trajectory": _to_int(result.get("trajectory_id")),
        "finish_reason": result.get("finish_reason") or "unknown",
        "truncated": bool(result.get("truncated", False)),
        "turns": _to_int(result.get("turns")),
        "context": {
            "tokens": _to_int(context_tokens) if context_tokens is not None else None,
            "limit": _to_int(context_limit) if context_limit is not None else None,
        },
        "action_counts": dict(result.get("action_counts") or {}),
        "patch": {
            "exists": bool(result.get("git_patch_present", result.get("git_patch"))),
            "size": patch_size,
        },
        "eval": {
            "status": _eval_status(result),
            "resolved": bool(result.get("resolved", False)),
            "error": result.get("evaluation_error") or result.get("eval_error"),
            "timeout": bool(result.get("evaluation_timeout", False)),
            "fail_to_pass": _ratio(
                result.get("target_tests_passed"),
                result.get("target_tests_total"),
            ),
            "pass_to_pass": _ratio(
                result.get("regression_tests_passed"),
                result.get("regression_tests_total"),
            ),
        },
        "reward": {
            "binary": _to_float(result.get("binary_reward")),
            "test_informed": _to_float(result.get("test_informed_reward")),
            "all": _to_float(
                result.get("selected_training_reward"),
                _to_float(result.get("test_informed_reward")),
            ),
        },
        "wall_time_sec": (
            _to_float(result.get("wall_time_sec")) if result.get("wall_time_sec") is not None else None
        ),
    }
    if trace_path:
        record["trace_path"] = trace_path
    return record


def aggregate_rollout_records(records: Iterable[dict[str, Any]]) -> dict[str, float]:
    rows = list(records)
    count = len(rows)
    finish_counts = Counter(str(row.get("finish_reason") or "unknown") for row in rows)
    eval_counts = Counter(str((row.get("eval") or {}).get("status") or "unknown") for row in rows)
    reward_values = [
        _to_float((row.get("reward") or {}).get("all"))
        for row in rows
        if (row.get("reward") or {}).get("all") is not None
    ]
    patch_count = sum(1 for row in rows if (row.get("patch") or {}).get("exists"))
    truncated_count = sum(1 for row in rows if row.get("truncated"))
    resolved_count = sum(1 for row in rows if (row.get("eval") or {}).get("status") == "resolved")
    nonzero_reward_count = sum(1 for value in reward_values if value != 0.0)

    aggregate: dict[str, float] = {
        "rollout/trajectory_count": float(count),
        "rollout/truncated_count": float(truncated_count),
        "rollout/truncated_rate": truncated_count / count if count else 0.0,
        "rollout/patch_exists_count": float(patch_count),
        "rollout/patch_exists_rate": patch_count / count if count else 0.0,
        "rollout/resolved_count": float(resolved_count),
        "rollout/resolved_rate": resolved_count / count if count else 0.0,
        "rollout/eval_error_count": float(eval_counts.get("eval_error", 0)),
        "rollout/reward_mean": sum(reward_values) / len(reward_values) if reward_values else 0.0,
        "rollout/reward_nonzero_count": float(nonzero_reward_count),
    }
    for reason, value in finish_counts.items():
        aggregate[f"rollout/finish_reason/{reason}"] = float(value)
    for status, value in eval_counts.items():
        aggregate[f"rollout/eval_status/{status}"] = float(value)
    return aggregate


def format_rollout_summary(
    *,
    step: int,
    phase: str,
    records: list[dict[str, Any]],
    aggregate: dict[str, float],
) -> str:
    count = int(aggregate.get("rollout/trajectory_count", 0))
    truncated = int(aggregate.get("rollout/truncated_count", 0))
    eval_error = int(aggregate.get("rollout/eval_error_count", 0))
    resolved = int(aggregate.get("rollout/resolved_count", 0))
    patch_exists = int(aggregate.get("rollout/patch_exists_count", 0))
    lines = [
        f"STEP {step} {phase.upper()} ROLLOUT SUMMARY",
        (
            f"trajectories={count} truncated={truncated}/{count} "
            f"resolved={resolved} eval_error={eval_error} "
            f"patch_exists={patch_exists}/{count} "
            f"reward_mean={aggregate.get('rollout/reward_mean', 0.0):.4f}"
        ),
        "task | traj | finish | trunc | turns | ctx | reward | eval | ftp | ptp | trace",
    ]
    for row in records:
        context = row.get("context") or {}
        reward = row.get("reward") or {}
        eval_info = row.get("eval") or {}
        patch = row.get("patch") or {}
        ctx_tokens = context.get("tokens")
        ctx_limit = context.get("limit")
        ctx = "-" if ctx_tokens is None or ctx_limit is None else f"{ctx_tokens}/{ctx_limit}"
        lines.append(
            " | ".join(
                [
                    str(row.get("instance_id", "")),
                    str(row.get("trajectory", "")),
                    str(row.get("finish_reason", "")),
                    "yes" if row.get("truncated") else "no",
                    str(row.get("turns", 0)),
                    ctx,
                    f"{_to_float(reward.get('all')):.4f}",
                    str(eval_info.get("status", "")),
                    str(eval_info.get("fail_to_pass", "")),
                    str(eval_info.get("pass_to_pass", "")),
                    str(row.get("trace_path", "-") or "-"),
                ]
            )
        )
        if not patch.get("exists") and eval_info.get("status") != "eval_error":
            lines[-1] += " | no_patch"
    return "\n".join(lines)


def write_rollout_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def write_trajectory_trace(
    root_dir: str | Path | None,
    *,
    step: int | None,
    result: dict[str, Any],
) -> str | None:
    if not root_dir:
        return None
    instance_id = str(result.get("instance_id", "unknown"))
    trajectory_id = _to_int(result.get("trajectory_id"))
    step_name = "step_unknown" if step is None else f"step_{int(step):04d}"
    trace_dir = Path(root_dir) / step_name
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"{instance_id}_traj{trajectory_id}.json"
    trace_path.write_text(
        json.dumps(result, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return str(trace_path)
