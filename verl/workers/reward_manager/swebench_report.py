from __future__ import annotations

from typing import Any


FAIL_TO_PASS = "FAIL_TO_PASS"
PASS_TO_PASS = "PASS_TO_PASS"

FAILURE_FLAGS = (
    "success_repair",
    "target_tests_failed",
    "regression_tests_failed",
    "no_patch",
    "empty_modification",
    "agent_max_turn",
    "agent_crash",
    "environment_crash",
    "test_timeout",
    "illegal_action",
    "repeated_loop",
)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def _count_tests(report: dict[str, Any], suite_name: str) -> tuple[int, int, int]:
    suite = (report.get("tests_status") or {}).get(suite_name) or {}
    success = suite.get("success") or []
    failure = suite.get("failure") or []
    passed = len(success)
    failed = len(failure)
    return passed + failed, passed, failed


def normalize_evaluation_report(
    report: dict[str, Any] | None,
    evaluation_error: str | None = None,
    evaluation_timeout: bool | None = None,
) -> dict[str, Any]:
    report = report or {}
    target_total, target_passed, target_failed = _count_tests(report, FAIL_TO_PASS)
    regression_total, regression_passed, regression_failed = _count_tests(report, PASS_TO_PASS)

    error = evaluation_error
    if error is None:
        error = report.get("evaluation_error") or report.get("eval_error")
    timeout = evaluation_timeout
    if timeout is None:
        timeout = _as_bool(report.get("evaluation_timeout") or report.get("test_timeout"))
        if error:
            timeout = timeout or "timed out" in str(error).lower()

    return {
        "resolved": _as_bool(report.get("resolved", False)),
        "target_tests_total": int(target_total),
        "target_tests_passed": int(target_passed),
        "target_tests_failed": int(target_failed),
        "regression_tests_total": int(regression_total),
        "regression_tests_passed": int(regression_passed),
        "regression_tests_failed": int(regression_failed),
        "evaluation_error": error,
        "evaluation_timeout": bool(timeout),
    }


def binary_reward_from_resolved(resolved: Any) -> float:
    return 1.0 if _as_bool(resolved) else 0.0


def normalize_stage1b_report(report: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize a complete SWE report for the Stage1B reward contract."""
    if not isinstance(report, dict):
        raise ValueError("Stage1B evaluator report must be a dictionary")
    tests_status = report.get("tests_status")
    if not isinstance(tests_status, dict):
        raise ValueError("Stage1B evaluator report is missing tests_status")

    normalized: dict[str, Any] = {"resolved": _as_bool(report.get("resolved", False))}
    for source_name, prefix in ((FAIL_TO_PASS, "ftp"), (PASS_TO_PASS, "ptp")):
        suite = tests_status.get(source_name)
        if not isinstance(suite, dict) or "success" not in suite or "failure" not in suite:
            raise ValueError(f"Stage1B evaluator report is missing {source_name}")
        success = suite["success"]
        failure = suite["failure"]
        if not isinstance(success, (list, tuple)) or not isinstance(failure, (list, tuple)):
            raise ValueError(f"Stage1B {source_name} success/failure must be lists")
        normalized[f"{prefix}_passed"] = len(success)
        normalized[f"{prefix}_total"] = len(success) + len(failure)
        normalized[f"{prefix}_failed"] = len(failure)

    if normalized["ftp_total"] <= 0:
        raise ValueError("Stage1B FAIL_TO_PASS denominator must be positive")
    if normalized["ptp_failed"] > 0 and normalized["ptp_total"] <= 0:
        raise ValueError("Stage1B PASS_TO_PASS denominator must be positive when regression fails")
    return normalized


def reward_v2_from_facts(facts: dict[str, Any]) -> float:
    """Compute the Stage1B dense regression-aware reward."""
    required = ("resolved", "ftp_passed", "ftp_total", "ptp_passed", "ptp_total", "ptp_failed")
    missing = [key for key in required if key not in facts]
    if missing:
        raise ValueError(f"Stage1B reward facts missing: {', '.join(missing)}")

    if _as_bool(facts["resolved"]):
        return 1.0
    ftp_total = int(facts["ftp_total"])
    if ftp_total <= 0:
        raise ValueError("Stage1B FAIL_TO_PASS denominator must be positive")
    target = int(facts["ftp_passed"]) / ftp_total
    if int(facts["ptp_failed"]) == 0:
        return float(0.9 * target)

    ptp_total = int(facts["ptp_total"])
    if ptp_total <= 0:
        raise ValueError("Stage1B PASS_TO_PASS denominator must be positive when regression fails")
    preservation = int(facts["ptp_passed"]) / ptp_total
    return float(target * (0.6 + 0.1 * preservation))


def test_informed_reward_from_facts(
    facts: dict[str, Any],
    target_weight: float = 0.8,
    regression_weight: float = 0.2,
) -> float:
    if abs((target_weight + regression_weight) - 1.0) > 1e-6:
        raise ValueError(
            f"target_weight + regression_weight must be 1.0, got {target_weight + regression_weight}"
        )
    if _as_bool(facts.get("resolved", False)):
        return 1.0
    if facts.get("evaluation_error") or _as_bool(facts.get("evaluation_timeout", False)):
        return 0.0

    target_total = int(facts.get("target_tests_total") or 0)
    if target_total <= 0:
        return binary_reward_from_resolved(facts.get("resolved", False))

    target_fraction = float(facts.get("target_tests_passed") or 0) / target_total
    regression_total = int(facts.get("regression_tests_total") or 0)
    if regression_total <= 0:
        regression_fraction = 1.0
    else:
        regression_fraction = float(facts.get("regression_tests_passed") or 0) / regression_total

    reward = target_fraction * (target_weight + regression_weight * regression_fraction)
    return max(0.0, min(1.0, float(reward)))


test_informed_reward_from_facts.__test__ = False


def _patch_size(patch: Any) -> int:
    return len(patch.strip()) if isinstance(patch, str) else 0


def classify_trajectory_failure(result: dict[str, Any]) -> tuple[str, list[str]]:
    resolved = _as_bool(result.get("resolved", False))
    if resolved:
        return "success_repair", ["success_repair"]

    flags: list[str] = []
    error = result.get("evaluation_error") or result.get("eval_error") or result.get("error")
    error_text = str(error or "")
    error_lower = error_text.lower()
    patch = result.get("git_patch")
    patch_size = int(result.get("patch_size") or _patch_size(patch))

    if not patch:
        flags.append("no_patch")
    elif patch_size == 0:
        flags.append("empty_modification")

    if _as_bool(result.get("evaluation_timeout", False)) or "timed out" in error_lower:
        flags.append("test_timeout")
    if "maximum iteration" in error_lower or "max iteration" in error_lower:
        flags.append("agent_max_turn")
    if "stuck in a loop" in error_lower or "agentstuckinlooperror" in error_lower:
        flags.append("repeated_loop")
    if any(
        token in error_lower
        for token in [
            "malformed action",
            "no action",
            "functioncallvalidation",
            "functioncallnotexists",
            "illegal action",
        ]
    ):
        flags.append("illegal_action")
    if any(token in error_lower for token in ["runtime", "sandbox", "docker", "container", "connect"]):
        if "agent reached maximum iteration" not in error_lower:
            flags.append("environment_crash")
    if error and not any(
        flag in flags
        for flag in [
            "agent_max_turn",
            "repeated_loop",
            "illegal_action",
            "environment_crash",
            "test_timeout",
        ]
    ):
        flags.append("agent_crash")

    if int(result.get("target_tests_failed") or 0) > 0:
        flags.append("target_tests_failed")
    if int(result.get("regression_tests_failed") or 0) > 0:
        flags.append("regression_tests_failed")

    priority = (
        "test_timeout",
        "agent_max_turn",
        "repeated_loop",
        "illegal_action",
        "environment_crash",
        "agent_crash",
        "no_patch",
        "empty_modification",
        "target_tests_failed",
        "regression_tests_failed",
    )
    unique_flags = [flag for flag in FAILURE_FLAGS if flag in set(flags)]
    primary = next((flag for flag in priority if flag in unique_flags), "target_tests_failed")
    flags = [primary] + [flag for flag in unique_flags if flag != primary]
    return primary, flags


def trajectory_reward_fields(
    result: dict[str, Any],
    evaluation_report: dict[str, Any] | None = None,
    evaluation_error: str | None = None,
    target_weight: float = 0.8,
    regression_weight: float = 0.2,
) -> dict[str, Any]:
    error = evaluation_error
    if error is None:
        error = result.get("eval_error") or result.get("evaluation_error") or result.get("error")
    facts = normalize_evaluation_report(evaluation_report, evaluation_error=error)
    if "resolved" in result:
        facts["resolved"] = _as_bool(result.get("resolved"))
    enriched = {**result, **facts, "patch_size": _patch_size(result.get("git_patch"))}
    outcome_primary, failure_flags = classify_trajectory_failure(enriched)
    binary_reward = binary_reward_from_resolved(facts["resolved"])
    test_reward = test_informed_reward_from_facts(
        facts, target_weight=target_weight, regression_weight=regression_weight
    )
    return {
        **facts,
        "outcome_primary": outcome_primary,
        "failure_flags": failure_flags,
        "git_patch_present": bool(result.get("git_patch")),
        "patch_size": enriched["patch_size"],
        "binary_reward": binary_reward,
        "test_informed_reward": test_reward,
        "selected_training_reward": None,
    }
