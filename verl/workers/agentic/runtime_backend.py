import os
from functools import lru_cache
from typing import Any, Mapping


def openhands_runtime_backend(env: Mapping[str, str] | None = None) -> str:
    if env is None:
        env = os.environ
    return env.get("SKYRL_OPENHANDS_RUNTIME", env.get("RUNTIME", "remote")).strip().lower()


def openhands_runtime_requires_remote_api(runtime: str) -> bool:
    return runtime == "remote"


def _env_flag_enabled(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = env.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", "disabled"}


def _rootfs_layers(image: Any) -> list[str]:
    return list(image.attrs.get("RootFS", {}).get("Layers", []))


def _repo_tags(image: Any) -> list[str]:
    tags = image.attrs.get("RepoTags") or []
    return [str(tag) for tag in tags]


def _layers_start_with(layers: list[str], prefix: list[str]) -> bool:
    return bool(prefix) and layers[: len(prefix)] == prefix


def _find_prewarmed_runtime_image_for_base_image(
    base_image: str,
    docker_client: Any,
    runtime_repo: str,
) -> str | None:
    try:
        base_layers = _rootfs_layers(docker_client.images.get(base_image))
    except Exception:
        return None

    try:
        runtime_images = docker_client.images.list(name=runtime_repo)
    except Exception:
        return None

    matching_tags: list[str] = []
    for image in runtime_images:
        runtime_layers = _rootfs_layers(image)
        if not _layers_start_with(runtime_layers, base_layers):
            continue
        for tag in _repo_tags(image):
            if tag.startswith(runtime_repo + ":"):
                matching_tags.append(tag)

    return matching_tags[0] if matching_tags else None


@lru_cache(maxsize=256)
def _find_prewarmed_runtime_image_for_base_image_cached(
    base_image: str,
    runtime_repo: str,
) -> str | None:
    try:
        import docker

        docker_client = docker.from_env()
    except Exception:
        return None
    return _find_prewarmed_runtime_image_for_base_image(
        base_image,
        docker_client,
        runtime_repo,
    )


def find_prewarmed_runtime_image_for_base_image(
    base_image: str,
    docker_client: Any | None = None,
    runtime_repo: str = "ghcr.io/all-hands-ai/runtime",
) -> str | None:
    if docker_client is not None:
        return _find_prewarmed_runtime_image_for_base_image(
            base_image,
            docker_client,
            runtime_repo,
        )
    return _find_prewarmed_runtime_image_for_base_image_cached(
        base_image,
        runtime_repo,
    )


def prepare_sandbox_for_runtime(
    sandbox_config: Any,
    runtime: str,
    env: Mapping[str, str] | None = None,
    docker_client: Any | None = None,
) -> None:
    if env is None:
        env = os.environ
    if not openhands_runtime_requires_remote_api(runtime):
        sandbox_config.api_key = None
        sandbox_config.remote_runtime_api_url = None
    if runtime != "docker":
        return
    if getattr(sandbox_config, "runtime_container_image", None):
        return
    if not _env_flag_enabled(env, "SKYRL_REUSE_PREWARMED_RUNTIME", True):
        return

    base_image = getattr(sandbox_config, "base_container_image", None)
    if not base_image:
        return
    runtime_repo = env.get("OH_RUNTIME_RUNTIME_IMAGE_REPO", "ghcr.io/all-hands-ai/runtime")
    prewarmed_image = find_prewarmed_runtime_image_for_base_image(
        base_image,
        docker_client=docker_client,
        runtime_repo=runtime_repo,
    )
    if prewarmed_image:
        sandbox_config.runtime_container_image = prewarmed_image
        print(f"Reusing prewarmed OpenHands runtime image: {prewarmed_image}")
