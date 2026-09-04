import math

import pytest

from verl.trainer.ppo.stage1b_sampling import (
    GroupDecision,
    Stage1BSampler,
    classify_group,
)


def test_dense_reward_spread_is_informative_even_without_resolved_success():
    decision = classify_group(
        [0.0, 0.3, 0.48, 1.0],
        [True, True, True, True],
    )

    assert isinstance(decision, GroupDecision)
    assert decision.status == "informative"
    assert decision.valid_reward_count == 4
    assert decision.reward_std == pytest.approx(0.3634212432)


@pytest.mark.parametrize("rewards", [[0.48] * 8, [1.0] * 8, [0.0] * 8])
def test_equal_valid_rewards_are_zero_variance(rewards):
    decision = classify_group(rewards, [True] * 8)

    assert decision.status == "zero_variance"
    assert decision.reward_std == 0.0
    assert decision.all_success is (rewards[0] == 1.0)
    assert decision.all_failure is (rewards[0] == 0.0)


@pytest.mark.parametrize("validity", [[False] * 8, [True] + [False] * 7])
def test_groups_with_fewer_than_two_valid_rewards_are_insufficient(validity):
    decision = classify_group([0.0] * 8, validity)

    assert decision.status == "insufficient_valid"
    assert decision.valid_reward_count == sum(validity)


def test_invalid_rewards_do_not_change_group_variance():
    decision = classify_group(
        [0.2, 0.2, 1.0, 0.0],
        [True, False, True, True],
    )

    assert decision.status == "informative"
    assert decision.valid_reward_count == 3
    assert decision.reward_std == pytest.approx(math.sqrt(0.56 / 3))


def test_sampler_does_not_repeat_a_candidate_within_one_update_and_stops_at_cap():
    sampler = Stage1BSampler(["a", "b", "c", "d"], target_groups=2, max_candidates=3, seed=7)
    sampler.begin_update(step=4)

    attempted = [sampler.next_candidate() for _ in range(4)]

    assert attempted[:3] == list(dict.fromkeys(attempted[:3]))
    assert attempted[3] is None


def test_sampler_history_and_rng_round_trip():
    sampler = Stage1BSampler(["a", "b", "c"], seed=11)
    sampler.begin_update(step=9)
    task_id = sampler.next_candidate()
    decision = classify_group([0.0, 1.0], [True, True])
    sampler.record_group(
        task_id,
        decision,
        [
            {"reward": 0.0, "reward_valid": True, "resolved": False, "finish_reason": "model_finish", "infra_error": False},
            {"reward": 1.0, "reward_valid": True, "resolved": True, "finish_reason": "context_limit", "infra_error": True},
        ],
        step=9,
    )
    state = sampler.state_dict()

    restored = Stage1BSampler(["a", "b", "c"], seed=999)
    restored.load_state_dict(state)

    assert restored.state_dict() == state
    assert restored.history[task_id].num_groups_seen == 1
    assert restored.history[task_id].num_trajectories == 2
    assert restored.history[task_id].resolved_rate == pytest.approx(0.5)
    assert restored.history[task_id].reward_mean == pytest.approx(0.5)
    assert restored.history[task_id].context_limit_rate == pytest.approx(0.5)
    assert restored.history[task_id].infra_error_rate == pytest.approx(0.5)
    assert restored.history[task_id].last_seen_step == 9
