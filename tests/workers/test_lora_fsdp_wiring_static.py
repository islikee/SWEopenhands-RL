from pathlib import Path


def test_fsdp_actor_build_uses_lora_training_mode_without_changing_full_mode_default():
    source = Path("verl/workers/fsdp_workers.py").read_text()

    assert "configure_lora_training" in source
    assert "training_mode" in source
    assert "use_orig_params" in source
    assert "iter_trainable_parameters" in source


def test_sglang_sharding_manager_builds_effective_lora_payload_for_update_weights():
    source = Path("verl/workers/agentic/fsdp_sgl.py").read_text()

    assert "build_effective_rollout_weight_payload" in source
    assert "has_lora_adapters" in source
    assert "update_weights_from_tensor" in source
