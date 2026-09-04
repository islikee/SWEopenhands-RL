import importlib.util
from pathlib import Path


_SPEC = importlib.util.spec_from_file_location("cloud_deploy", Path("scripts/cloud_deploy.py"))
cloud_deploy = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(cloud_deploy)


def test_expected_gpu_count_defaults_to_visible_gpu_count(monkeypatch):
    monkeypatch.delenv("SKYRL_GPUS_PER_NODE", raising=False)
    monkeypatch.setattr(cloud_deploy, "gpu_count", lambda: 2)

    assert cloud_deploy.expected_gpu_count() == 2


def test_expected_gpu_count_allows_explicit_override(monkeypatch):
    monkeypatch.setenv("SKYRL_GPUS_PER_NODE", "4")
    monkeypatch.setattr(cloud_deploy, "gpu_count", lambda: 2)

    assert cloud_deploy.expected_gpu_count() == 4
