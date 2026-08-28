from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import torch


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(name, default)
    return getattr(config, name, default)


def _unwrap_module(module):
    if hasattr(module, "_fsdp_wrapped_module"):
        return module._fsdp_wrapped_module
    return module


def resolve_lora_target_modules(model, configured_targets=None) -> list[str]:
    if configured_targets and configured_targets != "auto":
        if isinstance(configured_targets, str):
            return [item.strip() for item in configured_targets.split(",") if item.strip()]
        return list(configured_targets)

    common_names = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    module_leaf_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    targets = [name for name in common_names if name in module_leaf_names]
    if not targets:
        raise ValueError(
            "Could not infer LoRA target modules from model; set lora_target_modules explicitly."
        )
    return targets


def configure_lora_training(model, config):
    training_mode = _config_value(config, "training_mode", "full")
    if training_mode == "full":
        return model
    if training_mode != "lora":
        raise ValueError(f"Unsupported training_mode={training_mode!r}")

    from peft import LoraConfig, TaskType, get_peft_model

    targets = resolve_lora_target_modules(model, _config_value(config, "lora_target_modules"))
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(_config_value(config, "lora_rank", 16)),
        lora_alpha=int(_config_value(config, "lora_alpha", 32)),
        lora_dropout=float(_config_value(config, "lora_dropout", 0.0)),
        target_modules=targets,
    )
    return get_peft_model(model, lora_config)


def iter_trainable_parameters(model) -> Iterable[torch.nn.Parameter]:
    return (parameter for parameter in model.parameters() if parameter.requires_grad)


def get_trainable_parameter_stats(model) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"trainable": trainable, "total": total}


def save_lora_checkpoint(model, path: str | Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if not hasattr(model, "peft_config"):
        raise ValueError("save_lora_checkpoint requires a PEFT LoRA model")
    model.save_pretrained(str(path), safe_serialization=True)


def load_lora_checkpoint(model, path: str | Path) -> None:
    if not hasattr(model, "peft_config"):
        raise ValueError("load_lora_checkpoint requires a PEFT LoRA model")
    from peft import set_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights

    state_dict = load_peft_weights(str(path), device="cpu")
    set_peft_model_state_dict(model, state_dict, adapter_name="default")


def _materialize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if hasattr(tensor, "full_tensor"):
        tensor = tensor.full_tensor()
    return tensor.detach().clone()


def _normalize_peft_key(name: str) -> str:
    if name.startswith("base_model.model."):
        name = name[len("base_model.model.") :]
    return name.replace(".base_layer.", ".")


def _active_adapters(module) -> list[str]:
    adapters = getattr(module, "active_adapters", None)
    if adapters:
        return list(adapters)
    active_adapter = getattr(module, "active_adapter", None)
    if active_adapter:
        return [active_adapter]
    return ["default"]


def _delta_from_state_dict(prefix: str, module, state_dict: dict[str, torch.Tensor]) -> torch.Tensor | None:
    deltas = []
    for adapter in _active_adapters(module):
        a_key = f"{prefix}.lora_A.{adapter}.weight"
        b_key = f"{prefix}.lora_B.{adapter}.weight"
        if a_key not in state_dict or b_key not in state_dict:
            continue
        a = _materialize_tensor(state_dict[a_key]).float()
        b = _materialize_tensor(state_dict[b_key]).float()
        delta = b @ a
        if getattr(module, "fan_in_fan_out", False):
            delta = delta.T
        scaling = getattr(module, "scaling", {}).get(adapter, 1.0)
        deltas.append(delta * scaling)
    if not deltas:
        return None
    return sum(deltas)


def build_effective_rollout_weight_payload(
    model,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> list[tuple[str, torch.Tensor]]:
    unwrapped = _unwrap_module(model)
    state_dict = state_dict or unwrapped.state_dict()
    modules = dict(unwrapped.named_modules())
    has_lora = any(".lora_A." in name or ".lora_B." in name for name in state_dict)
    payload: list[tuple[str, torch.Tensor]] = []

    for name, tensor in state_dict.items():
        if ".lora_" in name or ".lora_embedding_" in name:
            continue
        materialized = _materialize_tensor(tensor)
        if has_lora and name.endswith(".base_layer.weight"):
            prefix = name[: -len(".base_layer.weight")]
            module = modules.get(prefix)
            if module is not None:
                delta = _delta_from_state_dict(prefix, module, state_dict)
                if delta is not None:
                    materialized = materialized.float() + delta.to(materialized.device)
                    materialized = materialized.to(dtype=tensor.dtype)
        payload.append((_normalize_peft_key(name), materialized))

    return payload


def has_lora_adapters(model) -> bool:
    unwrapped = _unwrap_module(model)
    return hasattr(unwrapped, "peft_config") or any("lora_" in name for name, _ in unwrapped.named_parameters())


def rollout_weight_sync_required() -> bool:
    return os.getenv("SKYRL_REQUIRE_ROLLOUT_WEIGHT_SYNC", "").lower() in {"1", "true", "yes", "on"}


def validate_rollout_weight_payload(payload: list[tuple[str, torch.Tensor]]) -> dict[str, int]:
    if not payload:
        raise RuntimeError("rollout weight synchronization payload is empty")

    tensor_count = 0
    parameter_count = 0
    for name, tensor in payload:
        if ".lora_" in name or ".lora_embedding_" in name or ".base_layer." in name:
            raise RuntimeError(f"rollout payload contains PEFT adapter/base-layer key: {name}")
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"rollout payload item {name} is not a tensor")
        tensor_count += 1
        parameter_count += tensor.numel()

    return {"tensor_count": tensor_count, "parameter_count": parameter_count}


_MISSING = object()
_SYNC_FINGERPRINTS: dict[int, tuple] = {}


def _rollout_update_succeeded(result: Any) -> bool | None:
    # Frozen sglang's VerlEngine wrapper has no return statement on success;
    # the raw async Engine returns an explicit (success, message) tuple.
    if result is None:
        return True
    if isinstance(result, bool):
        return result
    if isinstance(result, (tuple, list)) and result:
        return bool(result[0])
    if isinstance(result, dict) and "success" in result:
        return bool(result["success"])
    return None


def sync_rollout_weight_payload(inference_engine, payload, *, load_format=_MISSING) -> dict[str, int]:
    payload = list(payload)
    summary = validate_rollout_weight_payload(payload)

    if load_format is _MISSING:
        result = inference_engine.update_weights_from_tensor(payload)
    else:
        result = inference_engine.update_weights_from_tensor(payload, load_format=load_format)

    if rollout_weight_sync_required():
        succeeded = _rollout_update_succeeded(result)
        if succeeded is not True:
            raise RuntimeError(
                "rollout weight synchronization was required but update_weights_from_tensor "
                f"returned {result!r}"
            )

    if rollout_weight_change_required():
        _check_payload_changed_after_first_sync(inference_engine, payload)

    return summary


def rollout_weight_change_required() -> bool:
    return os.getenv("SKYRL_REQUIRE_ROLLOUT_WEIGHT_CHANGE_AFTER_FIRST_SYNC", "").lower() in {"1", "true", "yes", "on"}


def _payload_fingerprint(payload: list[tuple[str, torch.Tensor]]) -> tuple:
    fingerprint = []
    with torch.no_grad():
        for name, tensor in payload:
            materialized = tensor.full_tensor() if hasattr(tensor, "full_tensor") else tensor.detach()
            materialized = materialized.float()
            fingerprint.append(
                (
                    name,
                    tuple(materialized.shape),
                    str(tensor.dtype),
                    float(materialized.sum().item()),
                    float(materialized.abs().sum().item()),
                    float((materialized * materialized).sum().item()),
                )
            )
    return tuple(fingerprint)


def _check_payload_changed_after_first_sync(inference_engine, payload: list[tuple[str, torch.Tensor]]) -> None:
    engine_key = id(inference_engine)
    fingerprint = _payload_fingerprint(payload)
    previous = _SYNC_FINGERPRINTS.get(engine_key)
    _SYNC_FINGERPRINTS[engine_key] = fingerprint
    if previous is not None and fingerprint == previous:
        raise RuntimeError(
            "rollout weight synchronization payload did not change after the previous sync; "
            "cloud_smoke requires an updated effective policy to be sent to the rollout engine"
        )
