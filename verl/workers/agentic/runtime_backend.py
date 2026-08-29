import os
from typing import Any, Mapping


def openhands_runtime_backend(env: Mapping[str, str] | None = None) -> str:
    if env is None:
        env = os.environ
    return env.get("SKYRL_OPENHANDS_RUNTIME", env.get("RUNTIME", "remote")).strip().lower()


def openhands_runtime_requires_remote_api(runtime: str) -> bool:
    return runtime == "remote"


def prepare_sandbox_for_runtime(sandbox_config: Any, runtime: str) -> None:
    if not openhands_runtime_requires_remote_api(runtime):
        sandbox_config.api_key = None
        sandbox_config.remote_runtime_api_url = None
