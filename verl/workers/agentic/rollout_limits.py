from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SGLangTokenPoolLimits:
    max_total_tokens: int
    max_prefill_tokens: int


def _config_get(config: Any, name: str, default: Any = None) -> Any:
    if hasattr(config, "get"):
        return config.get(name, default)
    return getattr(config, name, default)


def _optional_int(value: Any, fallback: int) -> int:
    if value is None:
        return fallback
    return int(value)


def resolve_sglang_token_pool_limits(config: Any) -> SGLangTokenPoolLimits:
    total_len = int(config.prompt_length) + int(config.response_length)
    return SGLangTokenPoolLimits(
        max_total_tokens=_optional_int(_config_get(config, "max_total_tokens"), 60 * total_len),
        max_prefill_tokens=_optional_int(_config_get(config, "max_prefill_tokens"), 2 * total_len),
    )
