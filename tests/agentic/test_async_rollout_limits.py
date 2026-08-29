from types import SimpleNamespace

import pytest

from verl.workers.agentic import async_rollout
from verl.workers.agentic.rollout_limits import resolve_sglang_token_pool_limits


class _Config(SimpleNamespace):
    def get(self, name, default=None):
        return getattr(self, name, default)


class _FakeDeviceMesh:
    def get_local_rank(self, mesh_dim):
        assert mesh_dim == 1
        return 0

    def size(self, mesh_dim):
        return 2 if mesh_dim == 1 else 1

    def get_group(self, mesh_dim):
        assert mesh_dim == 1
        return object()


class _FakeEngine:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def release_memory_occupation(self):
        pass


def test_async_rollout_uses_frozen_token_pool_defaults_when_unset():
    config = _Config(prompt_length=8192, response_length=1024)

    limits = resolve_sglang_token_pool_limits(config)

    assert limits.max_total_tokens == 60 * (8192 + 1024)
    assert limits.max_prefill_tokens == 2 * (8192 + 1024)


def test_async_rollout_accepts_explicit_sglang_token_pool_limits():
    config = _Config(
        prompt_length=8192,
        response_length=1024,
        max_total_tokens=32768,
        max_prefill_tokens=18432,
    )

    limits = resolve_sglang_token_pool_limits(config)

    assert limits.max_total_tokens == 32768
    assert limits.max_prefill_tokens == 18432


def test_async_rollout_passes_disable_cuda_graph_to_sglang_engine(monkeypatch):
    captured = {}

    def fake_engine(**kwargs):
        captured.update(kwargs)
        return _FakeEngine(**kwargs)

    def fake_all_gather_object(output, input_obj, group=None):
        output[:] = ["0", "1"]

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(async_rollout.sgl, "Engine", fake_engine)
    monkeypatch.setattr(async_rollout.torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(async_rollout.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        async_rollout.torch.distributed,
        "all_gather_object",
        fake_all_gather_object,
    )

    config = _Config(
        prompt_length=8192,
        response_length=1024,
        max_total_tokens=32768,
        max_prefill_tokens=18432,
        disable_cuda_graph=True,
        dtype="bfloat16",
        enable_memory_saver=True,
        gpu_memory_utilization=0.55,
        task_type="swegym",
        sampling_params={},
    )

    rollout = async_rollout.AsyncRollout(
        "/data/skyrl/models/Qwen/Qwen2.5-Coder-7B-Instruct",
        config,
        _FakeDeviceMesh(),
    )

    assert rollout.engine is not None
    assert captured["disable_cuda_graph"] is True
    assert captured["max_total_tokens"] == 32768
    assert captured["max_prefill_tokens"] == 18432
    assert captured["tp_size"] == 2
    assert captured["mem_fraction_static"] == pytest.approx(0.55)
