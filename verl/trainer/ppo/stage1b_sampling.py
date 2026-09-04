from __future__ import annotations

import random
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

import torch


@dataclass(frozen=True)
class GroupDecision:
    status: str
    valid_reward_count: int
    reward_std: float
    all_success: bool
    all_failure: bool


@dataclass(frozen=True)
class Stage1BCollectionResult:
    """Result of one bounded candidate-collection round."""

    accepted_groups: tuple[Any, ...]
    attempted_groups: int
    informative_groups: int
    zero_variance_groups: int
    insufficient_valid_groups: int
    target_groups: int

    @property
    def underfilled(self) -> bool:
        return self.informative_groups < self.target_groups


def collect_informative_groups(
    sampler: "Stage1BSampler",
    get_group: Callable[[str], Mapping[str, Any]],
    step: int,
) -> Stage1BCollectionResult:
    """Collect candidate groups until the sampler target or candidate cap.

    ``get_group`` owns rollout/evaluation and returns ``rewards``,
    ``reward_valid``, ``trajectories`` and an optional ``payload``.  Keeping
    this coordinator free of Ray/DataProto details makes the sampling policy
    deterministic and directly testable.
    """
    sampler.begin_update(step)
    accepted = []
    while (task_id := sampler.next_candidate()) is not None:
        group = get_group(task_id)
        decision = classify_group(
            group["rewards"],
            group["reward_valid"],
            epsilon=group.get("reward_epsilon", 1e-6),
        )
        sampler.record_group(
            task_id,
            decision,
            group.get("trajectories", ()),
            step,
        )
        if decision.status == "informative":
            accepted.append(group.get("payload", group))

    return Stage1BCollectionResult(
        accepted_groups=tuple(accepted),
        attempted_groups=sampler.candidate_groups_attempted,
        informative_groups=sampler.accepted_informative_groups,
        zero_variance_groups=sampler.zero_variance_groups,
        insufficient_valid_groups=sampler.insufficient_valid_groups,
        target_groups=sampler.target_groups,
    )


def effective_train_tokens(loss_mask: torch.Tensor) -> int:
    return int(loss_mask.to(dtype=torch.bool).sum().item())


def apply_stage1b_training_masks(data):
    """Apply acceptance and evaluator-validity masks to a training DataProto."""
    batch_size = data.batch["responses"].shape[0]
    response_length = data.batch["responses"].shape[1]
    device = data.batch["responses"].device
    valid = torch.as_tensor(
        list(data.non_tensor_batch.get("reward_valid", [True] * batch_size)),
        device=device,
        dtype=torch.bool,
    )
    accepted = torch.as_tensor(
        list(data.non_tensor_batch.get("accepted_group", [True] * batch_size)),
        device=device,
        dtype=torch.bool,
    )
    row_mask = valid & accepted
    response_mask = data.batch["attention_mask"][:, -response_length:].bool()
    data.batch["response_mask"] = response_mask & row_mask.unsqueeze(-1)
    existing_loss_mask = data.batch.get("loss_mask", response_mask)
    data.batch["loss_mask"] = existing_loss_mask.bool() & row_mask.unsqueeze(-1)
    return data


def classify_group(
    rewards: Sequence[float | None],
    reward_valid: Sequence[bool],
    epsilon: float = 1e-6,
) -> GroupDecision:
    if len(rewards) != len(reward_valid):
        raise ValueError("rewards and reward_valid must have the same length")

    valid_rewards = [
        float(reward)
        for reward, valid in zip(rewards, reward_valid)
        if valid and reward is not None
    ]
    valid_count = len(valid_rewards)
    if valid_count < 2:
        return GroupDecision(
            status="insufficient_valid",
            valid_reward_count=valid_count,
            reward_std=0.0,
            all_success=False,
            all_failure=False,
        )

    reward_std = float(statistics.pstdev(valid_rewards))
    status = (
        "informative"
        if max(valid_rewards) - min(valid_rewards) > epsilon
        else "zero_variance"
    )
    return GroupDecision(
        status=status,
        valid_reward_count=valid_count,
        reward_std=reward_std,
        all_success=all(reward == 1.0 for reward in valid_rewards),
        all_failure=all(reward == 0.0 for reward in valid_rewards),
    )


@dataclass
class Stage1BTaskHistory:
    num_groups_seen: int = 0
    num_trajectories: int = 0
    valid_trajectory_count: int = 0
    resolved_count: int = 0
    reward_sum: float = 0.0
    reward_sumsq: float = 0.0
    informative_group_count: int = 0
    context_limit_count: int = 0
    infra_error_count: int = 0
    last_seen_step: int | None = None

    @property
    def resolved_rate(self) -> float:
        if self.valid_trajectory_count == 0:
            return 0.0
        return self.resolved_count / self.valid_trajectory_count

    @property
    def reward_mean(self) -> float:
        if self.valid_trajectory_count == 0:
            return 0.0
        return self.reward_sum / self.valid_trajectory_count

    @property
    def reward_std(self) -> float:
        if self.valid_trajectory_count == 0:
            return 0.0
        mean = self.reward_mean
        return (max(0.0, self.reward_sumsq / self.valid_trajectory_count - mean**2)) ** 0.5

    @property
    def context_limit_rate(self) -> float:
        if self.num_trajectories == 0:
            return 0.0
        return self.context_limit_count / self.num_trajectories

    @property
    def infra_error_rate(self) -> float:
        if self.num_trajectories == 0:
            return 0.0
        return self.infra_error_count / self.num_trajectories

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "Stage1BTaskHistory":
        return cls(**state)


class Stage1BSampler:
    STATE_VERSION = 1

    def __init__(
        self,
        candidate_ids: Sequence[str],
        target_groups: int = 4,
        max_candidates: int = 8,
        seed: int = 0,
    ) -> None:
        self.candidate_ids = list(candidate_ids)
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("candidate_ids must be unique")
        if not self.candidate_ids:
            raise ValueError("candidate_ids must not be empty")
        if target_groups < 1 or max_candidates < target_groups:
            raise ValueError("sampling limits must be positive and cap >= target")

        self.target_groups = int(target_groups)
        self.max_candidates = int(max_candidates)
        self.rng = random.Random(seed)
        self.history = {task_id: Stage1BTaskHistory() for task_id in self.candidate_ids}
        self.update_step: int | None = None
        self._candidate_order: list[str] = []
        self._cursor = 0
        self._attempted: list[str] = []
        self.accepted_informative_groups = 0
        self.zero_variance_groups = 0
        self.insufficient_valid_groups = 0
        self.group_reward_stds: list[float] = []

    @property
    def candidate_groups_attempted(self) -> int:
        return len(self._attempted)

    @property
    def attempted_ids(self) -> tuple[str, ...]:
        return tuple(self._attempted)

    def begin_update(self, step: int) -> None:
        self.update_step = int(step)
        self._candidate_order = self.rng.sample(self.candidate_ids, len(self.candidate_ids))
        self._cursor = 0
        self._attempted = []
        self.accepted_informative_groups = 0
        self.zero_variance_groups = 0
        self.insufficient_valid_groups = 0
        self.group_reward_stds = []

    def next_candidate(self) -> str | None:
        if self.accepted_informative_groups >= self.target_groups:
            return None
        if len(self._attempted) >= self.max_candidates or self._cursor >= len(self._candidate_order):
            return None
        task_id = self._candidate_order[self._cursor]
        self._cursor += 1
        self._attempted.append(task_id)
        return task_id

    def record_group(
        self,
        task_id: str,
        decision: GroupDecision,
        trajectories: Sequence[dict[str, Any]],
        step: int,
    ) -> None:
        if task_id not in self.history:
            raise KeyError(f"unknown candidate task: {task_id}")
        record = self.history[task_id]
        record.num_groups_seen += 1
        record.num_trajectories += len(trajectories)
        record.last_seen_step = int(step)
        if decision.status == "informative":
            self.accepted_informative_groups += 1
            record.informative_group_count += 1
            self.group_reward_stds.append(decision.reward_std)
        elif decision.status == "zero_variance":
            self.zero_variance_groups += 1
        elif decision.status == "insufficient_valid":
            self.insufficient_valid_groups += 1
        else:
            raise ValueError(f"unknown group status: {decision.status}")

        for trajectory in trajectories:
            if trajectory.get("finish_reason") == "context_limit":
                record.context_limit_count += 1
            if trajectory.get("infra_error", False):
                record.infra_error_count += 1
            if not trajectory.get("reward_valid", False):
                continue
            reward = trajectory.get("reward")
            if reward is None:
                continue
            reward = float(reward)
            record.valid_trajectory_count += 1
            record.reward_sum += reward
            record.reward_sumsq += reward * reward
            if trajectory.get("resolved", False):
                record.resolved_count += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.STATE_VERSION,
            "candidate_ids": list(self.candidate_ids),
            "target_groups": self.target_groups,
            "max_candidates": self.max_candidates,
            "rng_state": self.rng.getstate(),
            "update_step": self.update_step,
            "candidate_order": list(self._candidate_order),
            "cursor": self._cursor,
            "attempted": list(self._attempted),
            "accepted_informative_groups": self.accepted_informative_groups,
            "zero_variance_groups": self.zero_variance_groups,
            "insufficient_valid_groups": self.insufficient_valid_groups,
            "group_reward_stds": list(self.group_reward_stds),
            "history": {task_id: item.state_dict() for task_id, item in self.history.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("version") != self.STATE_VERSION:
            raise ValueError(f"unsupported Stage1B sampler state: {state.get('version')}")
        if list(state["candidate_ids"]) != self.candidate_ids:
            raise ValueError("checkpoint candidate pool does not match current candidate pool")
        if int(state["target_groups"]) != self.target_groups or int(state["max_candidates"]) != self.max_candidates:
            raise ValueError("checkpoint sampling limits do not match current configuration")
        self.rng.setstate(state["rng_state"])
        self.update_step = state["update_step"]
        self._candidate_order = list(state["candidate_order"])
        self._cursor = int(state["cursor"])
        self._attempted = list(state["attempted"])
        self.accepted_informative_groups = int(state["accepted_informative_groups"])
        self.zero_variance_groups = int(state["zero_variance_groups"])
        self.insufficient_valid_groups = int(state["insufficient_valid_groups"])
        self.group_reward_stds = [float(value) for value in state["group_reward_stds"]]
        self.history = {
            task_id: Stage1BTaskHistory.from_state_dict(item)
            for task_id, item in state["history"].items()
        }
