# Stage1B SkyRL-LoRA-noKL Dynamic Sampling Design

## Goal and non-goals

Stage1B changes only the explicitly requested parts of the existing Stage1
SkyRL-LoRA-noKL experiment:

- sample from 64 non-validation candidate tasks and accept only informative
  dense-reward groups;
- distinguish evaluator infrastructure validity from rollout finish reason;
- use the regression-aware reward v2 formula;
- mask invalid trajectories before GRPO normalization and actor loss;
- expose the requested training diagnostics;
- use approximately 32k OpenHands agent context and 15 iterations; and
- persist and restore Stage1B sampler/history state alongside the existing
  checkpoint state.

The baseline's model, LoRA settings, optimizer, PPO/GRPO settings, KL
settings, temperature, validation split, and resource settings remain
unchanged unless explicitly listed above. History-biased sampling is not part
of this phase.

## Existing execution chain

The actual Stage1 path is:

```text
verl.trainer.main_ppo
  -> RayPPOTrainer.fit
  -> AsyncRollout.generate_sequences
  -> CodeActAgentGroup.run
  -> OpenHands rollout and SWE evaluator
  -> TestInformedSWEBenchRewardManager
  -> compute_advantage / GRPO
  -> FSDP actor update_policy
  -> checkpoint and Tracking logger
```

The repository does not contain a usable SkyRL DAPO/Polaris training sampler.
`SkyRL-OpenHands` is the OpenHands submodule rather than a training sampler.
Stage1B therefore adds a small coordinator around the existing trainer path,
with pure sampling and group-classification helpers kept independently
testable.

## Data split and configuration

The existing manifest remains the source of truth for the 80-task universe,
the 32-task original train split, and the 16-task validation split. The
preparation script derives:

```python
universe = set(stage1_universe)
old_train = set(train)
validation = set(validation)
unused = universe - old_train - validation
stage1b_candidates = old_train | unused
```

Startup validation must assert:

```python
len(universe) == 80
len(old_train) == 32
len(validation) == 16
len(unused) == 32
len(stage1b_candidates) == 64
stage1b_candidates.isdisjoint(validation)
stage1b_candidates | validation == universe
```

The preparation script writes a candidate parquet derived from these sets and
an ID file for auditability; no second hand-maintained task list is added.
The Stage1B launcher uses the candidate parquet for training and the existing
validation parquet for fixed validation. It has independent output,
checkpoint, experiment, and W&B names and starts from the configured raw SFT
checkpoint/model path rather than a Stage1 step-8 checkpoint.

The logical Stage1B sampling values are:

```text
target informative groups = 4
trajectories per task = 8
maximum candidate groups per update = 8
informative reward epsilon = 1e-6
```

They are represented through the existing Hydra configuration tree or its
minimal extension, without creating a duplicate configuration system.

## Dynamic sampling coordinator

For every trainer update, the coordinator chooses distinct task IDs uniformly
from the 64-task candidate pool. A task already attempted in the current
candidate search cannot be selected again. Each selected task is materialized
as a one-task `DataProto` and passed through the existing rollout with
`n_trajectories=8`; no new rollout implementation is introduced.

After rollout/evaluation/reward, the group is classified using only valid
dense rewards:

```python
valid_rewards = [t.reward for t in group if t.reward_valid]

if len(valid_rewards) < 2:
    status = "insufficient_valid"
elif max(valid_rewards) - min(valid_rewards) > 1e-6:
    status = "informative"
else:
    status = "zero_variance"
```

Only informative groups are concatenated into the actor training batch. The
search stops at four accepted groups or eight attempted groups. If fewer than
four informative groups are found, no actor update or optimizer step occurs;
the round is logged as underfilled and the next sampling round starts without
reusing the current round's candidate IDs.

The sampler owns a per-task history initialized to zero for Stage1B. It
records `num_groups_seen`, `num_trajectories`, `resolved_rate`, `reward_mean`,
`reward_std`, `informative_group_count`, `context_limit_rate`,
`infra_error_rate`, and `last_seen_step`. These statistics are collected but
do not affect Phase 1 sampling probabilities.

## Evaluator validity and retry

Rollout `finish_reason` is diagnostic only. In particular, `context_limit`,
`max_iterations`, and `unknown` do not invalidate a trajectory when a patch
and a valid evaluator report exist. The evaluator is still run after the
agent reaches its iteration/context limit if a patch was produced.

Each trajectory carries `reward_valid`, the normalized report, evaluator
attempt/retry counters, retry success, and infrastructure-error metadata.

- A valid report means `reward_valid=True`, independent of finish reason.
- A missing patch is a valid model negative: `reward_valid=True`, reward 0,
  and no evaluator retry.
- A patch/evaluation failure attributable to the model (for example an
  unapplicable patch) is a valid negative reward rather than infrastructure
  invalidity.
- A confirmed evaluator infrastructure/transient failure retries the same
  patch at most once, without rerunning OpenHands.
- A retry that yields a valid report makes the trajectory valid while still
  counting the encountered infrastructure error.
- Two infrastructure failures leave `reward_valid=False`; no artificial
  reward 0 is created.

The existing SWE report's `tests_status.FAIL_TO_PASS` and
`tests_status.PASS_TO_PASS` arrays are normalized to the required
`ftp_passed`, `ftp_total`, `ptp_passed`, `ptp_total`, `ptp_failed`, and
`resolved` fields. For valid reports, reward v2 is exactly:

```python
if resolved:
    reward = 1.0
else:
    target = ftp_passed / ftp_total
    if ptp_failed == 0:
        reward = 0.9 * target
    else:
        preservation = ptp_passed / ptp_total
        reward = target * (0.6 + 0.1 * preservation)
```

Missing report data is invalid rather than silently converted to zeros. If
the SWE-Gym contract guarantees `ftp_total > 0`, that contract is asserted
or validated at the normalization boundary. The old Stage1 reward manager is
left unchanged; only the Stage1B launcher selects the new path.

## GRPO and actor masks

The accepted batch retains invalid trajectories for auditability, but carries
an explicit per-trajectory `reward_valid` mask. GRPO advantage computation is
extended to:

1. group scores by task ID using only valid trajectories;
2. normalize only those valid scores;
3. emit zero advantage for invalid trajectories; and
4. broadcast the result only over valid response positions.

The actor's final loss mask is the intersection of the existing assistant/
action mask, accepted-group membership, and `reward_valid`. Rejected groups
never reach advantage computation. No fallback advantage is enabled.

The implementation must retain the current GRPO grouping semantics for valid
samples and must test that zero-variance, insufficient-valid, and invalid
trajectories cannot affect normalization or loss.

## Context and rollout behavior

The active OpenHands context knob is `agent_max_prompt_length`, enforced by
`OnlineCodeActAgent.step()` after tokenizing the complete conversation. It is
set to approximately 32768 tokens for Stage1B, while `max_iterations` is set
to 15. `max_starting_message_length`, `data.max_prompt_length`, and
`data.max_response_length` retain their existing trainer-side meanings and
are not changed to impersonate the agent context budget. The SGLang
`max_total_tokens`/prefill settings remain engine-pool controls.

## Metrics

The trainer emits the requested aggregate metrics. Counts for
`train/valid_trajectory_count` and `train/effective_train_tokens` refer only
to the final accepted batch. `effective_train_tokens` is the sum of the
actual actor loss mask after all masks are applied. Advantage statistics use
the same final training mask as existing advantage statistics.

Gradient norms are measured in the actor worker after backward and AMP
unscale (if applicable): pre-clip immediately before clipping, and post-clip
immediately after clipping and before `optimizer.step()`. Underfilled rounds
record `train/optimizer_step_performed=0` and do not fabricate zero gradient
norms.

Required keys are:

```text
actor/grad_norm_pre_clip
actor/grad_norm_post_clip
advantages/std
train/valid_trajectory_count
train/informative_group_count
train/zero_variance_group_count
train/effective_train_tokens
train/candidate_group_count
train/accepted_group_count
train/insufficient_valid_group_count
train/all_success_group_count
train/all_failure_group_count
train/group_reward_std_mean
train/underfilled_informative_batch
train/optimizer_step_performed
eval/infra_error_count
eval/retry_attempt_count
eval/retry_success_count
eval/reward_invalid_count
rollout/context_limit_rate
rollout/max_iteration_rate
```

## Checkpoint and resume

The existing actor checkpoint remains responsible for LoRA parameters,
optimizer state, scheduler state, and extra state. Stage1B adds a
`stage1b_state.pt` beside `data.pt` in every global-step checkpoint. It
contains a versioned sampler state, task history, candidate pool, current
sampler state, and RNG state needed for reproducible continuation.

On resume, trainer global step and existing actor/dataloader state are loaded
as before, then Stage1B state is restored. History is never reset and
`last_seen_step` remains aligned with the restored trainer step. A missing
Stage1B state is permitted only for a fresh Stage1B run; it must not silently
import Stage1 history.

## Verification strategy

CPU-verifiable tests cover:

- split cardinality/disjointness/coverage;
- informative, zero-variance, all-success, all-failure, and
  insufficient-valid group classification;
- candidate replacement and within-round uniqueness;
- evaluator retry/no-retry behavior and finish-reason independence;
- every reward-v2 numeric case and invalid report behavior;
- GRPO normalization and actor masking;
- effective training token counting;
- distinct pre/post clipping norms; and
- checkpoint round-trip for actor-adjacent Stage1B sampler/history state.

After CPU tests pass, the GPU environment must run one real Stage1B update
through candidate sampling, eight trajectories per candidate, evaluator/retry,
reward v2, replacement, four accepted groups, GRPO, LoRA backward, clipping,
optimizer step, W&B logging, checkpoint save, and checkpoint resume. If the
candidate cap produces an underfilled round, that round is accepted only as
an underfilled test and must not be reported as a successful optimizer update.
