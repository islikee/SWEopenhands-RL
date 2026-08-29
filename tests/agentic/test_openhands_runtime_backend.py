from verl.workers.agentic.runtime_backend import (
    openhands_runtime_backend,
    openhands_runtime_requires_remote_api,
    prepare_sandbox_for_runtime,
)


class _Sandbox:
    api_key = "secret"
    remote_runtime_api_url = "https://runtime.eval.all-hands.dev"


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
