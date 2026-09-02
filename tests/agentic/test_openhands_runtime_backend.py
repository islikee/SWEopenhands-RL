from verl.workers.agentic.runtime_backend import (
    find_prewarmed_runtime_image_for_base_image,
    openhands_runtime_backend,
    openhands_runtime_requires_remote_api,
    prepare_sandbox_for_runtime,
)


class _Sandbox:
    api_key = "secret"
    remote_runtime_api_url = "https://runtime.eval.all-hands.dev"
    base_container_image = "xingyaoww/sweb.eval.x86_64.getmoto_s_moto-4950:latest"
    runtime_container_image = None


def test_openhands_runtime_backend_preserves_frozen_remote_default():
    assert openhands_runtime_backend({}) == "remote"


def test_openhands_runtime_backend_accepts_cloud_docker_override():
    assert openhands_runtime_backend({"SKYRL_OPENHANDS_RUNTIME": "docker"}) == "docker"
    assert openhands_runtime_requires_remote_api("docker") is False
    assert openhands_runtime_requires_remote_api("remote") is True


def test_prepare_sandbox_for_docker_runtime_drops_remote_credentials():
    sandbox = _Sandbox()

    prepare_sandbox_for_runtime(sandbox, "docker")

    assert sandbox.api_key is None
    assert sandbox.remote_runtime_api_url is None


class _FakeImage:
    def __init__(self, tags, layers):
        self.attrs = {
            "RepoTags": tags,
            "RootFS": {"Layers": layers},
        }


class _FakeImages:
    def __init__(self):
        self.base = _FakeImage(["task:latest"], ["a", "b"])
        self.runtime = _FakeImage(
            ["ghcr.io/all-hands-ai/runtime:oh_v0.25.0_cached"],
            ["a", "b", "c", "d"],
        )
        self.unrelated = _FakeImage(
            ["ghcr.io/all-hands-ai/runtime:oh_v0.25.0_other"],
            ["x", "y", "z"],
        )

    def get(self, image_name):
        assert image_name == "task:latest"
        return self.base

    def list(self, name=None):
        assert name == "ghcr.io/all-hands-ai/runtime"
        return [self.unrelated, self.runtime]


class _FakeDockerClient:
    images = _FakeImages()


def test_find_prewarmed_runtime_image_matches_exact_base_layer_prefix():
    image = find_prewarmed_runtime_image_for_base_image(
        "task:latest",
        docker_client=_FakeDockerClient(),
    )

    assert image == "ghcr.io/all-hands-ai/runtime:oh_v0.25.0_cached"


def test_prepare_sandbox_for_docker_runtime_reuses_prewarmed_runtime(monkeypatch):
    sandbox = _Sandbox()
    sandbox.base_container_image = "task:latest"

    monkeypatch.setenv("SKYRL_REUSE_PREWARMED_RUNTIME", "1")
    prepare_sandbox_for_runtime(sandbox, "docker", docker_client=_FakeDockerClient())

    assert sandbox.runtime_container_image == (
        "ghcr.io/all-hands-ai/runtime:oh_v0.25.0_cached"
    )


def test_reward_evaluator_uses_selected_docker_runtime(monkeypatch):
    import verl.utils.reward_score.openhands_swebench as reward_swebench

    monkeypatch.setenv("SKYRL_OPENHANDS_RUNTIME", "docker")
    monkeypatch.setattr(
        reward_swebench,
        "get_instance_resource_factor",
        lambda dataset_name, instance_id: 1,
    )

    config = reward_swebench.get_config(
        "SWE-Gym/SWE-Gym",
        "getmoto__moto-7365",
    )

    assert config.runtime == "docker"
    assert config.sandbox.api_key is None
    assert config.sandbox.remote_runtime_api_url is None
