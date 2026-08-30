from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from verl.workers.lora_utils import (
    RolloutWeightSyncFingerprintAccumulator,
    build_effective_rollout_weight_payload,
    check_rollout_weight_payload_changed_after_first_sync,
    configure_lora_training,
    get_trainable_parameter_stats,
    iter_effective_rollout_weight_payload,
    iter_trainable_parameters,
    save_lora_checkpoint,
    load_lora_checkpoint,
    sync_rollout_weight_payload,
    lora_sensitive_payload_names,
)

def test_lora_effective_payload_resolves_nested_fsdp_wrapped_module_names():
    class FakeLoraTarget(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.active_adapters = ["default"]
            self.scaling = {"default": 1.0}
            self.fan_in_fan_out = False

    class FakeFSDPWrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self._fsdp_wrapped_module = module

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            layer = torch.nn.Module()
            layer.q_proj = FakeLoraTarget()
            self.layer = FakeFSDPWrapper(layer)

    model = FakeModel()

    state_dict = {
        "layer.q_proj.base_layer.weight": torch.zeros(2, 2),
        "layer.q_proj.lora_A.default.weight": torch.eye(2),
        "layer.q_proj.lora_B.default.weight": torch.full((2, 2), 0.25),
    }

    sensitive_names = lora_sensitive_payload_names(
        model,
        state_dict=state_dict,
    )

    payload = dict(
        iter_effective_rollout_weight_payload(
            model,
            state_dict=state_dict,
        )
    )

    assert sensitive_names == {"layer.q_proj.weight"}

    assert torch.allclose(
        payload["layer.q_proj.weight"],
        torch.full((2, 2), 0.25),
    )
def _tiny_gpt2():
    transformers = pytest.importorskip("transformers")
    config = transformers.GPT2Config(
        n_embd=16,
        n_layer=1,
        n_head=2,
        n_positions=8,
        n_ctx=8,
        vocab_size=20,
        bos_token_id=0,
        eos_token_id=1,
    )
    return transformers.GPT2LMHeadModel(config)


def _tiny_qwen2():
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        tie_word_embeddings=False,
    )
    return transformers.Qwen2ForCausalLM(config)


def _qwen2_lora_cfg():
    return SimpleNamespace(
        training_mode="lora",
        lora_rank=2,
        lora_alpha=4,
        lora_dropout=0.0,
        lora_target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def _lora_cfg():
    return SimpleNamespace(
        training_mode="lora",
        lora_rank=2,
        lora_alpha=4,
        lora_dropout=0.0,
        lora_target_modules=["c_attn"],
    )


def test_configure_lora_training_freezes_base_and_keeps_only_adapters_trainable():
    pytest.importorskip("peft")
    model = configure_lora_training(_tiny_gpt2(), _lora_cfg())

    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    frozen_base = [
        name
        for name, param in model.named_parameters()
        if not param.requires_grad and "lora_" not in name
    ]

    assert trainable
    assert all("lora_" in name for name in trainable)
    assert frozen_base
    assert get_trainable_parameter_stats(model)["trainable"] < get_trainable_parameter_stats(model)["total"]


def test_full_training_mode_keeps_original_parameter_path_trainable():
    model = _tiny_gpt2()
    configured = configure_lora_training(model, SimpleNamespace(training_mode="full"))

    assert configured is model
    assert all(param.requires_grad for param in configured.parameters())


def test_optimizer_step_changes_lora_but_not_frozen_base():
    pytest.importorskip("peft")
    torch.manual_seed(1)
    model = configure_lora_training(_tiny_gpt2(), _lora_cfg())
    optimizer = torch.optim.AdamW(iter_trainable_parameters(model), lr=0.5)
    base_before = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if "base_layer.weight" in name
    }
    lora_before = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if "lora_" in name
    }

    input_ids = torch.tensor([[1, 2, 3, 4]])
    loss = model(input_ids=input_ids, labels=input_ids).loss
    loss.backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    optimizer.step()

    for name, before in base_before.items():
        assert torch.equal(dict(model.named_parameters())[name], before)
    assert any(
        not torch.equal(dict(model.named_parameters())[name], before)
        for name, before in lora_before.items()
    )


def test_lora_checkpoint_round_trip_preserves_adapter_outputs(tmp_path: Path):
    pytest.importorskip("peft")
    torch.manual_seed(2)
    model = configure_lora_training(_tiny_gpt2(), _lora_cfg())
    optimizer = torch.optim.AdamW(iter_trainable_parameters(model), lr=0.5)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    model(input_ids=input_ids, labels=input_ids).loss.backward()
    optimizer.step()
    model.eval()
    expected = model(input_ids=input_ids).logits.detach()

    save_lora_checkpoint(model, tmp_path)
    torch.manual_seed(2)
    reloaded = configure_lora_training(_tiny_gpt2(), _lora_cfg())
    load_lora_checkpoint(reloaded, tmp_path)
    reloaded.eval()

    assert torch.allclose(reloaded(input_ids=input_ids).logits, expected, atol=1e-5)


def test_effective_rollout_payload_matches_base_names_shapes_and_changes_after_lora_step():
    pytest.importorskip("peft")
    torch.manual_seed(3)
    model = configure_lora_training(_tiny_gpt2(), _lora_cfg())
    optimizer = torch.optim.AdamW(iter_trainable_parameters(model), lr=0.5)

    payload_before = dict(build_effective_rollout_weight_payload(model))
    input_ids = torch.tensor([[1, 2, 3, 4]])
    model(input_ids=input_ids, labels=input_ids).loss.backward()
    optimizer.step()
    payload_after = dict(build_effective_rollout_weight_payload(model))

    target_name = "transformer.h.0.attn.c_attn.weight"
    assert target_name in payload_after
    assert "base_model.model.transformer.h.0.attn.c_attn.lora_A.default.weight" not in payload_after
    assert "base_model.model.transformer.h.0.attn.c_attn.base_layer.weight" not in payload_after
    assert payload_after[target_name].shape == (16, 48)
    assert any(
        not torch.equal(payload_after[name], payload_before[name])
        for name in payload_after
        if name in payload_before
    )


def test_iter_effective_rollout_payload_matches_eager_payload_and_can_stage_to_cpu_bf16():
    pytest.importorskip("peft")
    torch.manual_seed(6)
    model = configure_lora_training(_tiny_gpt2(), _lora_cfg())

    eager_payload = build_effective_rollout_weight_payload(model)
    iter_payload = list(
        iter_effective_rollout_weight_payload(
            model,
            target_device=torch.device("cpu"),
            target_dtype=torch.bfloat16,
        )
    )

    assert [name for name, _ in iter_payload] == [name for name, _ in eager_payload]
    assert all(tensor.device.type == "cpu" for _, tensor in iter_payload)
    assert all(tensor.dtype == torch.bfloat16 for _, tensor in iter_payload)
    assert torch.allclose(
        dict(iter_payload)["transformer.h.0.attn.c_attn.weight"].float(),
        dict(eager_payload)["transformer.h.0.attn.c_attn.weight"].float(),
        atol=1e-2,
        rtol=1e-2,
    )


def test_iter_effective_rollout_payload_can_preserve_small_lora_delta_precision_for_smoke_gate():
    pytest.importorskip("peft")
    torch.manual_seed(7)
    model = configure_lora_training(_tiny_gpt2(), _lora_cfg())
    target_name = "transformer.h.0.attn.c_attn.weight"

    before_bf16 = dict(
        iter_effective_rollout_weight_payload(
            model,
            target_device=torch.device("cpu"),
            target_dtype=torch.bfloat16,
        )
    )[target_name]
    before_precise = dict(
        iter_effective_rollout_weight_payload(
            model,
            target_device=torch.device("cpu"),
            target_dtype=torch.bfloat16,
            preserve_lora_delta_precision=True,
        )
    )[target_name]

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.add_(1e-9)

    after_bf16 = dict(
        iter_effective_rollout_weight_payload(
            model,
            target_device=torch.device("cpu"),
            target_dtype=torch.bfloat16,
        )
    )[target_name]
    after_precise = dict(
        iter_effective_rollout_weight_payload(
            model,
            target_device=torch.device("cpu"),
            target_dtype=torch.bfloat16,
            preserve_lora_delta_precision=True,
        )
    )[target_name]

    assert torch.equal(after_bf16, before_bf16)
    assert before_precise.dtype == torch.float32
    assert after_precise.dtype == torch.float32
    assert not torch.equal(after_precise, before_precise)


def test_iter_effective_rollout_payload_can_drain_collectives_without_emitting_payload():
    class FakeDTensor:
        def __init__(self, tensor):
            self.tensor = tensor
            self.full_tensor_calls = 0

        def full_tensor(self):
            self.full_tensor_calls += 1
            return self.tensor

    class FakeModel:
        def named_modules(self):
            return []

    fake_weight = FakeDTensor(torch.ones(2, 2))
    payload = list(
        iter_effective_rollout_weight_payload(
            FakeModel(),
            state_dict={"base_model.model.layers.0.weight": fake_weight},
            emit_payload=False,
        )
    )

    assert payload == []
    assert fake_weight.full_tensor_calls == 1


def test_effective_rollout_payload_matches_peft_forward_for_qwen2_target_modules():
    pytest.importorskip("peft")
    torch.manual_seed(5)
    model = configure_lora_training(_tiny_qwen2(), _qwen2_lora_cfg())
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.add_(0.05)
    model.eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    expected = model(input_ids=input_ids).logits.detach()

    plain = _tiny_qwen2()
    missing, unexpected = plain.load_state_dict(
        dict(build_effective_rollout_weight_payload(model)), strict=False
    )
    plain.eval()

    assert missing == []
    assert unexpected == []
    actual = plain(input_ids=input_ids).logits.detach()
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


class _MockInferenceEngine:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    def update_weights_from_tensor(self, payload, **kwargs):
        self.calls.append((list(payload), kwargs))
        return self.result


def test_sync_rollout_weight_payload_calls_real_update_interface_with_effective_weights():
    pytest.importorskip("peft")
    torch.manual_seed(4)
    model = configure_lora_training(_tiny_gpt2(), _lora_cfg())
    payload = build_effective_rollout_weight_payload(model)
    engine = _MockInferenceEngine()

    summary = sync_rollout_weight_payload(engine, payload)

    assert summary["tensor_count"] == len(payload)
    assert engine.calls
    sent_payload, kwargs = engine.calls[0]
    assert kwargs == {}
    assert dict(sent_payload)["transformer.h.0.attn.c_attn.weight"].shape == (16, 48)
    assert all(".lora_" not in name and ".base_layer." not in name for name, _ in sent_payload)


def test_required_rollout_weight_sync_fails_on_negative_ack(monkeypatch):
    monkeypatch.setenv("SKYRL_REQUIRE_ROLLOUT_WEIGHT_SYNC", "1")
    engine = _MockInferenceEngine(result=False)

    with pytest.raises(RuntimeError, match="update_weights_from_tensor returned"):
        sync_rollout_weight_payload(engine, [("model.layers.0.weight", torch.ones(2, 2))])


def test_required_rollout_weight_sync_accepts_frozen_verl_engine_no_ack(monkeypatch):
    monkeypatch.setenv("SKYRL_REQUIRE_ROLLOUT_WEIGHT_SYNC", "1")
    engine = _MockInferenceEngine(result=None)

    summary = sync_rollout_weight_payload(
        engine, [("model.layers.0.weight", torch.ones(2, 2))]
    )

    assert summary["tensor_count"] == 1


@pytest.mark.parametrize(
    "result",
    [
        (False, "SGLang rejected the update"),
        {"success": False, "message": "SGLang rejected the update"},
    ],
)
def test_required_rollout_weight_sync_fails_on_structured_negative_ack(monkeypatch, result):
    monkeypatch.setenv("SKYRL_REQUIRE_ROLLOUT_WEIGHT_SYNC", "1")
    engine = _MockInferenceEngine(result=result)

    with pytest.raises(RuntimeError, match="update_weights_from_tensor returned"):
        sync_rollout_weight_payload(engine, [("model.layers.0.weight", torch.ones(2, 2))])


def test_rollout_weight_payload_rejects_adapter_keys():
    engine = _MockInferenceEngine()

    with pytest.raises(RuntimeError, match="PEFT adapter"):
        sync_rollout_weight_payload(
            engine,
            [("base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight", torch.ones(2, 2))],
        )


def test_required_rollout_weight_sync_detects_unchanged_effective_payload(monkeypatch):
    monkeypatch.setenv("SKYRL_REQUIRE_ROLLOUT_WEIGHT_CHANGE_AFTER_FIRST_SYNC", "1")
    engine = _MockInferenceEngine()
    payload = [("model.layers.0.weight", torch.ones(2, 2))]

    sync_rollout_weight_payload(engine, payload)
    with pytest.raises(RuntimeError, match="payload did not change"):
        sync_rollout_weight_payload(engine, payload)


def test_required_rollout_weight_sync_accepts_changed_effective_payload(monkeypatch):
    monkeypatch.setenv("SKYRL_REQUIRE_ROLLOUT_WEIGHT_CHANGE_AFTER_FIRST_SYNC", "1")
    engine = _MockInferenceEngine()

    sync_rollout_weight_payload(engine, [("model.layers.0.weight", torch.ones(2, 2))])
    summary = sync_rollout_weight_payload(engine, [("model.layers.0.weight", torch.full((2, 2), 2.0))])

    assert summary["parameter_count"] == 4
    assert len(engine.calls) == 2


def test_chunked_rollout_weight_change_gate_compares_complete_lora_sensitive_sync(monkeypatch):
    monkeypatch.setenv("SKYRL_REQUIRE_ROLLOUT_WEIGHT_CHANGE_AFTER_FIRST_SYNC", "1")
    engine = _MockInferenceEngine()

    initial = RolloutWeightSyncFingerprintAccumulator(include_names={"model.layers.0.lora_target.weight"})
    initial.extend([("model.layers.0.static.weight", torch.ones(2, 2))])
    initial.extend([("model.layers.0.lora_target.weight", torch.ones(2, 2))])
    check_rollout_weight_payload_changed_after_first_sync(engine, initial.fingerprint())

    updated = RolloutWeightSyncFingerprintAccumulator(include_names={"model.layers.0.lora_target.weight"})
    updated.extend([("model.layers.0.static.weight", torch.ones(2, 2))])
    updated.extend([("model.layers.0.lora_target.weight", torch.full((2, 2), 2.0))])
    check_rollout_weight_payload_changed_after_first_sync(engine, updated.fingerprint())


def test_rollout_weight_change_gate_accepts_global_lora_sensitive_change(monkeypatch):
    monkeypatch.setenv("SKYRL_REQUIRE_ROLLOUT_WEIGHT_CHANGE_AFTER_FIRST_SYNC", "1")
    engine = _MockInferenceEngine()
    fingerprint = (("model.layers.0.lora_target.weight", (2, 2), "torch.float32", 4.0, 4.0, 4.0),)
    check_rollout_weight_payload_changed_after_first_sync(engine, fingerprint)

    def global_change_seen(value, device=None, process_group=None):
        assert value is False
        assert process_group == "dp"
        return True

    monkeypatch.setattr("verl.workers.lora_utils.distributed_any", global_change_seen)
    check_rollout_weight_payload_changed_after_first_sync(engine, fingerprint, process_group="dp")
