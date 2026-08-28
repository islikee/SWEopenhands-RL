from pathlib import Path


def test_codeact_dataproto_carries_normalized_reward_primitives_not_raw_report_only():
    source = Path("verl/workers/agentic/codeact.py").read_text()

    assert "trajectory_reward_fields" in source
    assert "evaluation_report" in source
    for key in [
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
    ]:
        assert f"'{key}'" in source or f'"{key}"' in source
