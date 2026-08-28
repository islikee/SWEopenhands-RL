from __future__ import annotations

from typing import Any


def fill_empty_trajectory_messages(results: list[dict[str, Any]]) -> None:
    valid_messages = next(
        (result.get("messages") for result in results if result.get("messages")),
        None,
    )
    if not valid_messages:
        return

    for result in results:
        if not result.get("messages"):
            result["messages"] = valid_messages.copy()
