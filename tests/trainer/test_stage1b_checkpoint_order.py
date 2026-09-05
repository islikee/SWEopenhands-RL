from types import SimpleNamespace

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def test_stage1b_saves_checkpoint_before_validation_when_both_are_due():
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(save_freq=1, test_freq=1))
    trainer.global_steps = 8
    trainer.val_reward_fn = object()

    calls = []
    trainer._save_checkpoint = lambda: calls.append("save_checkpoint")
    trainer._validate = lambda: calls.append("validate") or {"validation/metric": 1.0}

    metrics = {}
    val_metrics = trainer._stage1b_save_checkpoint_then_validate(
        metrics,
        timing_raw={},
        is_last_step=True,
    )

    assert calls == ["save_checkpoint", "validate"]
    assert val_metrics == {"validation/metric": 1.0}
    assert metrics["validation/metric"] == 1.0
