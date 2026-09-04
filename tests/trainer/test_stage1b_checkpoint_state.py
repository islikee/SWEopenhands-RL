import random

import numpy as np
import torch

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.stage1b_sampling import Stage1BSampler


def test_stage1b_checkpoint_roundtrip_restores_sampler_and_rng(tmp_path):
    trainer = object.__new__(RayPPOTrainer)
    trainer.stage1b_enabled = True
    trainer.stage1b_sampler = Stage1BSampler([f"task-{i}" for i in range(8)], seed=11)
    trainer.stage1b_sampler.begin_update(3)
    trainer.stage1b_sampler.next_candidate()
    trainer.config = type("Config", (), {"trainer": {"default_local_dir": str(tmp_path)}})()
    trainer.global_steps = 3
    trainer.train_dataloader = type("Loader", (), {"state_dict": lambda self: {}})()
    trainer.actor_rollout_wg = type("Workers", (), {
        "save_checkpoint": lambda *args, **kwargs: None,
    })()
    trainer.actor_wg = type("Actor", (), {})()
    trainer.use_critic = False

    state = trainer._stage1b_state_dict()
    restored = Stage1BSampler([f"task-{i}" for i in range(8)], seed=99)
    restored_trainer = object.__new__(RayPPOTrainer)
    restored_trainer.stage1b_enabled = True
    restored_trainer.stage1b_sampler = restored
    restored_trainer._load_stage1b_state_dict(state)
    assert restored.state_dict() == state["sampler"]
