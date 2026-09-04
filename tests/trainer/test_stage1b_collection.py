from verl.trainer.ppo.stage1b_sampling import (
    Stage1BSampler,
    collect_informative_groups,
)


def _trajectory(reward, valid=True):
    return {
        "reward": reward,
        "reward_valid": valid,
        "resolved": reward == 1.0,
    }


def test_collection_accepts_only_informative_groups_and_stops_at_target():
    sampler = Stage1BSampler(
        [f"informative-{i}" for i in range(4)] + [f"flat-{i}" for i in range(4)],
        target_groups=4,
        max_candidates=8,
        seed=7,
    )

    def get_group(task_id):
        if task_id.startswith("informative"):
            rewards = [0.0, 0.5, 1.0, 0.25, 0.0, 0.3, 0.48, 1.0]
        else:
            rewards = [0.0] * 8
        return {
            "rewards": rewards,
            "reward_valid": [True] * 8,
            "trajectories": [_trajectory(rewards[0])] * 8,
            "payload": task_id,
        }

    result = collect_informative_groups(sampler, get_group, step=1)

    assert set(result.accepted_groups) == {f"informative-{i}" for i in range(4)}
    assert result.informative_groups == 4
    assert result.attempted_groups <= 8
    assert not result.underfilled


def test_collection_reports_underfilled_without_an_optimizer_batch():
    sampler = Stage1BSampler(
        [f"flat-{i}" for i in range(10)],
        target_groups=4,
        max_candidates=8,
        seed=3,
    )

    result = collect_informative_groups(
        sampler,
        lambda _: {
            "rewards": [0.0] * 8,
            "reward_valid": [True] * 8,
            "trajectories": [_trajectory(0.0)] * 8,
        },
        step=1,
    )

    assert result.accepted_groups == ()
    assert result.attempted_groups == 8
    assert result.zero_variance_groups == 8
    assert result.underfilled
