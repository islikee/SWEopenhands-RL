from verl.workers.agentic.evaluator_validity import (
    EvaluatorInfrastructureError,
    ModelEvaluationError,
    evaluate_patch_with_retry,
)


def test_infrastructure_failure_retries_same_patch_without_rerollout():
    calls = []

    def evaluate_once(patch):
        calls.append(patch)
        if len(calls) == 1:
            raise EvaluatorInfrastructureError("docker temporarily unavailable")
        return {"resolved": True}

    outcome = evaluate_patch_with_retry("same-patch", evaluate_once, max_retries=1)

    assert calls == ["same-patch", "same-patch"]
    assert outcome.reward_valid is True
    assert outcome.evaluation_report == {"resolved": True}
    assert outcome.attempt_count == 2
    assert outcome.retry_count == 1
    assert outcome.retry_succeeded is True
    assert outcome.infra_error is True


def test_repeated_infrastructure_failure_is_invalid_without_fake_reward():
    calls = []

    def evaluate_once(patch):
        calls.append(patch)
        raise EvaluatorInfrastructureError("network timeout")

    outcome = evaluate_patch_with_retry("same-patch", evaluate_once, max_retries=1)

    assert calls == ["same-patch", "same-patch"]
    assert outcome.reward_valid is False
    assert outcome.reward is None
    assert outcome.retry_succeeded is False


def test_missing_patch_is_a_valid_model_negative_without_evaluator_call():
    called = False

    def evaluate_once(_patch):
        nonlocal called
        called = True
        return {"resolved": True}

    outcome = evaluate_patch_with_retry("", evaluate_once, max_retries=1)

    assert called is False
    assert outcome.reward_valid is True
    assert outcome.reward == 0.0
    assert outcome.attempt_count == 0


def test_model_evaluation_failure_is_valid_zero_without_retry():
    calls = []

    def evaluate_once(patch):
        calls.append(patch)
        raise ModelEvaluationError("patch does not apply")

    outcome = evaluate_patch_with_retry("bad-patch", evaluate_once, max_retries=1)

    assert calls == ["bad-patch"]
    assert outcome.reward_valid is True
    assert outcome.reward == 0.0
    assert outcome.retry_count == 0


def test_finish_reason_does_not_change_valid_report_outcome():
    for finish_reason in ("context_limit", "unknown"):
        outcome = evaluate_patch_with_retry(
            "patch",
            lambda _patch: {"resolved": True},
            finish_reason=finish_reason,
        )
        assert outcome.reward_valid is True


def test_infra_retry_prepares_same_evaluator_workspace():
    calls = []
    resets = []

    def evaluate_once(patch):
        calls.append(patch)
        if len(calls) == 1:
            raise EvaluatorInfrastructureError("docker daemon unavailable")
        return {"resolved": False}

    outcome = evaluate_patch_with_retry(
        "same-patch",
        evaluate_once,
        prepare_retry=lambda: resets.append(True),
    )

    assert calls == ["same-patch", "same-patch"]
    assert resets == [True]
    assert outcome.reward_valid is True
    assert outcome.retry_succeeded is True
