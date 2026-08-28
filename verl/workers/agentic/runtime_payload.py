from __future__ import annotations

from typing import Any


_RUNTIME_INSTANCE_FIELDS = ("instance_id", "repo", "version")


def runtime_instance_record(instance: Any) -> dict[str, Any]:
    source = instance if isinstance(instance, dict) else instance.to_dict()
    return {key: source[key] for key in _RUNTIME_INSTANCE_FIELDS if key in source}
