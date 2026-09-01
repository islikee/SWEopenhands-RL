from pathlib import Path


def test_trainer_does_not_define_or_call_zero_advantage_fallback():
    source = Path("verl/trainer/ppo/ray_trainer.py").read_text()

    assert "apply_smoke_nonzero_advantage_fallback" not in source
    assert "force_nonzero_advantage_if_all_zero" not in source
