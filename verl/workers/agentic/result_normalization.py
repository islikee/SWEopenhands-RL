from __future__ import annotations

from typing import Any


def _copy_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [message.copy() for message in messages]


def fill_empty_trajectory_messages(
    results: list[dict[str, Any]],
    fallback_messages: list[dict[str, Any]] | None = None,
) -> None:
    valid_messages = next(
        (result.get("messages") for result in results if result.get("messages")),
        None,
    )
    replacement_messages = valid_messages or fallback_messages
    if not replacement_messages:
        return

    for result in results:
        if not result.get("messages"):
            result["messages"] = _copy_messages(replacement_messages)
            result["reward_valid"] = False
            result["infra_error"] = True
            result.setdefault(
                "evaluation_error",
                result.get("error") or "empty trajectory messages",
            )
            if "empty trajectory messages" not in str(result["evaluation_error"]):
                result["evaluation_error"] = (
                    f"{result['evaluation_error']} (empty trajectory messages)"
                )
