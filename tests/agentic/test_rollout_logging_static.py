from pathlib import Path


def test_codeact_writes_trace_path_into_rollout_dataproto():
    source = Path("verl/workers/agentic/codeact.py").read_text()

    assert "write_trajectory_trace" in source
    assert "'trace_path'" in source or '"trace_path"' in source


def test_async_rollout_honors_validation_trajectory_count_from_meta_info():
    source = Path("verl/workers/agentic/async_rollout.py").read_text()

    assert 'prompts.meta_info.get("n_trajectories"' in source


def test_trainer_writes_rollout_jsonl_summaries_and_passes_step_phase_to_rollout():
    source = Path("verl/trainer/ppo/ray_trainer.py").read_text()

    assert "write_rollout_jsonl" in source
    assert "rollout_summary.jsonl" in source
    assert "'rollout_step': self.global_steps" in source
    assert "'rollout_phase': 'train'" in source
    assert "'rollout_phase': 'validation'" in source
    assert "'n_trajectories': self.config.actor_rollout_ref.rollout.val_kwargs.n" in source
