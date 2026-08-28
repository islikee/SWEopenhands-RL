from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from verl.workers.lora_utils import (
    build_effective_rollout_weight_payload,
    configure_lora_training,
    get_trainable_parameter_stats,
    iter_trainable_parameters,
    save_lora_checkpoint,
    load_lora_checkpoint,
    sync_rollout_weight_payload,
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
