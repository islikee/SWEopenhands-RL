from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


class EvaluatorInfrastructureError(RuntimeError):
    """A transient evaluator/runtime failure that is safe to retry."""


class ModelEvaluationError(RuntimeError):
    """A patch/model outcome that should become a valid zero reward."""


@dataclass(frozen=True)
class EvaluationOutcome:
    reward_valid: bool
    reward: float | None
    evaluation_report: dict[str, Any] | None
    attempt_count: int
    retry_count: int
    retry_succeeded: bool
    infra_error: bool
    finish_reason: str | None = None
    error: str | None = None


def is_evaluator_infrastructure_error(error: BaseException) -> bool:
    if isinstance(error, EvaluatorInfrastructureError):
        return True
    if getattr(error, "infrastructure", False):
        return True
    text = f"{type(error).__name__}: {error}".lower()
    return any(
        marker in text
        for marker in ("docker", "network", "ssl", "connection", "container", "evaluator timeout")
    )


def evaluate_patch_with_retry(
    patch: str | None,
    evaluate_once: Callable[[str], dict[str, Any]],
    max_retries: int = 1,
    finish_reason: str | None = None,
    prepare_retry: Callable[[], None] | None = None,
) -> EvaluationOutcome:
    """Evaluate one patch, retrying only classified infrastructure failures."""
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    if not isinstance(patch, str) or not patch.strip():
        return EvaluationOutcome(
            reward_valid=True,
            reward=0.0,
            evaluation_report=None,
            attempt_count=0,
            retry_count=0,
            retry_succeeded=False,
            infra_error=False,
            finish_reason=finish_reason,
        )

    infra_error = False
    last_error: str | None = None
    for attempt in range(max_retries + 1):
        try:
            report = evaluate_once(patch)
        except Exception as error:
            last_error = str(error)
            if not is_evaluator_infrastructure_error(error):
                return EvaluationOutcome(
                    reward_valid=True,
                    reward=0.0,
                    evaluation_report=None,
                    attempt_count=attempt + 1,
                    retry_count=attempt,
                    retry_succeeded=False,
                    infra_error=False,
                    finish_reason=finish_reason,
                    error=last_error,
                )
            infra_error = True
            if attempt < max_retries:
                if prepare_retry is not None:
                    prepare_retry()
                continue
            return EvaluationOutcome(
                reward_valid=False,
                reward=None,
                evaluation_report=None,
                attempt_count=attempt + 1,
                retry_count=attempt,
                retry_succeeded=False,
                infra_error=True,
                finish_reason=finish_reason,
                error=last_error,
            )

        return EvaluationOutcome(
            reward_valid=True,
            reward=None,
            evaluation_report=report,
            attempt_count=attempt + 1,
            retry_count=attempt,
            retry_succeeded=infra_error and attempt > 0,
            infra_error=infra_error,
            finish_reason=finish_reason,
        )

    raise RuntimeError("unreachable evaluator retry state")
