# Stage1B Dynamic Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved Stage1B SkyRL-LoRA-noKL training path with 64-task uniform dynamic sampling, valid dense reward filtering, masked GRPO, diagnostics, and resumable sampler state.

**Architecture:** Keep Stage1 unchanged and add a Stage1B-only coordinator around `RayPPOTrainer.fit()`. Pure task-split, group-classification, sampler-history, reward, mask, and gradient-norm helpers remain independently testable; the existing OpenHands rollout/evaluator, FSDP actor, logger, and checkpoint manager remain the runtime integration points.

**Tech Stack:** Python 3.12, PyTorch, Hydra/OmegaConf, Ray, DataProto/TensorDict, OpenHands/SWE-Gym, pytest, W&B Tracking.

**Spec:** `docs/superpowers/specs/2026-09-04-stage1b-dynamic-sampling-design.md`

## Global Constraints

- Stage1B candidates are exactly 64 tasks: 80 universe, 32 old train, 32 unused, 16 validation, candidate/validation overlap 0.
- Every candidate group attempts exactly 8 trajectories; each update accepts exactly 4 informative groups or performs no optimizer step after 8 candidates.
- Informative means at least two valid dense rewards and `max(rewards) - min(rewards) > 1e-6`.
- Invalid evaluator results are never converted into reward 0 or included in variance, advantage normalization, or actor loss.
- Evaluator retry is at most one retry of the same patch and never reruns OpenHands.
- `finish_reason=context_limit` and `finish_reason=unknown` do not invalidate a trajectory with a valid patch/report.
- Stage1B reward v2 is the exact resolved/FTP/PTP formula in the approved spec; Stage1's old reward path remains unchanged.
- `FALLBACK_ADVANTAGE=false`, KL loss and KL-in-reward remain disabled, and existing LoRA/LR/WD/PPO/temperature settings remain unchanged.
- The active OpenHands context limit is `agent_max_prompt_length≈32768`; `max_iterations=15`; trainer response length is not changed to impersonate context.
- Existing user changes in `scripts/run_stage1_32x16_lora.sh` and `tests/config/test_stage1_32x16_training.py` must be preserved.

## File Map

- Modify `scripts/prepare_stage1_split.py`: derive and validate Stage1B candidates; write candidate parquet/IDs.
- Modify `tests/config/test_stage1_32x16_training.py`: test candidate artifact and preserve baseline launcher contract.
- Create `verl/trainer/ppo/stage1b_sampling.py`: pure group classifier, uniform no-replacement sampler, task history, state serialization, and aggregate metrics.
- Create `tests/trainer/test_stage1b_sampling.py`: split-independent sampler/group/history tests.
- Modify `verl/workers/reward_manager/swebench_report.py`: add strict report normalization and reward-v2 helper without changing the old reward helper.
- Create `verl/workers/reward_manager/swebench_stage1b.py`: Stage1B reward manager that consumes actual rollout validity fields and calls reward v2.
- Modify `verl/workers/reward_manager/__init__.py` and `verl/trainer/main_ppo.py`: register the Stage1B reward manager.
- Modify `verl/workers/agentic/codeact.py`: carry validity/attempt metadata, run evaluator after context/iteration finish when a patch exists, and retry only infrastructure failures.
- Modify `verl/workers/agentic/rollout_logging.py`: include the new compact validity/retry diagnostics.
- Create `tests/agentic/test_stage1b_evaluator_validity.py`: test retry/no-retry and finish-reason semantics without starting Docker.
- Create `tests/reward_manager/test_swebench_stage1b.py`: test reward v2 and report validity through the registered manager.
- Modify `verl/trainer/ppo/core_algos.py` and `verl/trainer/ppo/ray_trainer.py`: valid-sample GRPO normalization and Stage1B coordinator integration.
- Modify `verl/workers/actor/dp_actor.py`: report distinct gradient norms and final effective loss tokens.
- Modify `verl/trainer/ppo/metric_utils.py`: use the final training mask for advantage diagnostics safely.
- Create `tests/trainer/test_stage1b_grpo_masks.py`: test accepted/rejected/invalid group masking and effective token count.
- Create `tests/workers/test_stage1b_gradient_metrics.py`: test pre/post clipping measurements.
- Modify `verl/trainer/config/ppo_trainer.yaml`: add disabled-by-default Stage1B config subtree.
- Create `scripts/run_stage1b_32x16_lora.sh`: independent Stage1B launcher/output/checkpoint/W&B identity and approved overrides.
- Create `tests/config/test_stage1b_training.py`: test Stage1B launcher/config contract.
- Modify `verl/trainer/ppo/ray_trainer.py`: save/load `stage1b_state.pt` beside existing state.
- Create `tests/trainer/test_stage1b_checkpoint_state.py`: test sampler/history checkpoint round-trip.

### Task 1: Candidate split artifacts

**Files:**
- Modify: `scripts/prepare_stage1_split.py`
- Modify: `tests/config/test_stage1_32x16_training.py`

**Interfaces:**
- Produce `stage1b_candidates.parquet`, `stage1b_candidate_instance_ids.txt`, and the existing train/validation artifacts from one manifest.
- Keep `prepare_split(manifest_path, source_path, output_path)` as the public entry point.

- [ ] **Step 1: Write the failing split tests**

Add assertions that `prepare_split()` writes a 64-row candidate parquet whose IDs equal `set(stage1_universe) - set(validation)`, and that all seven cardinality/coverage assertions are enforced.

- [ ] **Step 2: Run the split tests to verify failure**

Run: `pytest tests/config/test_stage1_32x16_training.py -q`

Expected: FAIL because the candidate parquet and candidate ID file do not yet exist.

- [ ] **Step 3: Implement derived candidate output**

Compute `unused = universe - old_train - validation`, `candidate_ids = old_train | unused`, assert the exact cardinalities/disjointness/coverage, select candidate rows from the existing source dataframe, and write the candidate parquet/ID file. Do not add a hand-maintained list.

- [ ] **Step 4: Run the split tests to verify green**

Run: `pytest tests/config/test_stage1_32x16_training.py -q`

Expected: PASS with the pre-existing baseline contract tests still passing.

- [ ] **Step 5: Commit the split artifact change**

Run: `git add scripts/prepare_stage1_split.py tests/config/test_stage1_32x16_training.py && git commit -m "feat: derive Stage1B candidate split"`

### Task 2: Pure Stage1B sampler and group state

**Files:**
- Create: `verl/trainer/ppo/stage1b_sampling.py`
- Create: `tests/trainer/test_stage1b_sampling.py`

**Interfaces:**
- `classify_group(rewards: Sequence[float | None], reward_valid: Sequence[bool], epsilon: float = 1e-6) -> GroupDecision` returns `status`, valid reward count, reward std, and all-success/all-failure diagnostics.
- `Stage1BSampler(candidate_ids: Sequence[str], target_groups: int = 4, max_candidates: int = 8, seed: int = 0)` exposes `begin_update(step)`, `next_candidate()`, `record_group(task_id, decision, trajectory_records, step)`, `state_dict()`, and `load_state_dict(state)`.
- `Stage1BTaskHistory` stores the nine required fields and serializes JSON-safe primitives.

- [ ] **Step 1: Write failing group and sampler tests**

Cover `[0.0, 0.3, 0.48, 1.0]` informative, eight equal rewards including all success/all failure zero variance, zero/one valid reward insufficient, invalid rewards excluded from range, no duplicate candidate in an update, cap at eight, and state/history round-trip.

- [ ] **Step 2: Run focused tests to verify failure**

Run: `pytest tests/trainer/test_stage1b_sampling.py -q`

Expected: FAIL because `stage1b_sampling` does not exist.

- [ ] **Step 3: Implement minimal pure sampler/state**

Use a dedicated `random.Random` instance, shuffle/sample candidate IDs without replacement per `begin_update`, update history only from valid trajectories for rates/means, and preserve RNG state in `state_dict()`.

- [ ] **Step 4: Run focused tests to verify green**

Run: `pytest tests/trainer/test_stage1b_sampling.py -q`

Expected: PASS.

- [ ] **Step 5: Commit sampler/state**

Run: `git add verl/trainer/ppo/stage1b_sampling.py tests/trainer/test_stage1b_sampling.py && git commit -m "feat: add Stage1B sampler state"`

### Task 3: Strict report normalization and Stage1B reward manager

**Files:**
- Modify: `verl/workers/reward_manager/swebench_report.py`
- Create: `verl/workers/reward_manager/swebench_stage1b.py`
- Modify: `verl/workers/reward_manager/__init__.py`
- Modify: `verl/trainer/main_ppo.py`
- Create: `tests/reward_manager/test_swebench_stage1b.py`

**Interfaces:**
- Add `normalize_stage1b_report(report) -> dict[str, Any]` with strict `ftp_*`, `ptp_*`, `resolved` fields and invalid-report errors.
- Add `reward_v2_from_facts(facts) -> float` implementing the exact approved formula.
- `Stage1BSWEBenchRewardManager.__call__(data: DataProto, return_dict: bool = False)` consumes `reward_valid` and normalized trajectory fields and emits the existing reward tensor shape plus Stage1B reward metadata.

- [ ] **Step 1: Write failing reward-v2 and validity tests**

Test resolved `1.0`, target `0.5` with no regression `0.45`, target `1.0` with no regression `0.9`, `628/629` preservation approximately `0.69984`, target `0.5`/preservation `0.8` equals `0.34`, and incomplete reports raise invalid rather than fill zeros. Test invalid trajectories get no training reward contribution.

- [ ] **Step 2: Run focused tests to verify failure**

Run: `pytest tests/reward_manager/test_swebench_stage1b.py -q`

Expected: FAIL because the Stage1B normalization/helper/manager are absent.

- [ ] **Step 3: Implement strict normalization and reward manager**

Map existing `tests_status.FAIL_TO_PASS/PASS_TO_PASS` arrays to `ftp_*`/`ptp_*`, retain the old `test_informed_reward_from_facts` unchanged for Stage1, and use `reward_valid` to leave invalid rows at a zero tensor with explicit metadata rather than treating them as valid zeros.

- [ ] **Step 4: Run reward tests and existing reward tests**

Run: `pytest tests/reward_manager/test_swebench_stage1b.py tests/reward_manager/test_swebench_test_informed.py -q`

Expected: PASS, including unchanged Stage1 reward tests.

- [ ] **Step 5: Commit reward path**

Run: `git add verl/workers/reward_manager/swebench_report.py verl/workers/reward_manager/swebench_stage1b.py verl/workers/reward_manager/__init__.py verl/trainer/main_ppo.py tests/reward_manager/test_swebench_stage1b.py && git commit -m "feat: add Stage1B valid reward v2"`

### Task 4: Evaluator validity and same-patch retry

**Files:**
- Modify: `verl/workers/agentic/codeact.py`
- Modify: `verl/workers/agentic/rollout_logging.py`
- Create: `tests/agentic/test_stage1b_evaluator_validity.py`

**Interfaces:**
- Add a small evaluator outcome helper in `codeact.py` or a focused helper module: `evaluate_patch_with_retry(evaluate_once, patch, max_retries=1) -> EvaluationOutcome`.
- Outcome includes `reward_valid`, `evaluation_report`, `attempt_count`, `retry_count`, `retry_succeeded`, and `infra_error`.

- [ ] **Step 1: Write failing evaluator tests**

Use a callback counter to prove infra-failure-then-success calls evaluator twice and rollout once; infra-failure-twice returns invalid without reward 0; no patch calls evaluator zero times and returns valid zero; valid reports remain valid for `context_limit` and `unknown` finish reasons.

- [ ] **Step 2: Run focused tests to verify failure**

Run: `pytest tests/agentic/test_stage1b_evaluator_validity.py -q`

Expected: FAIL because the retry outcome helper and validity fields do not exist.

- [ ] **Step 3: Implement evaluator outcome behavior**

Refactor the existing `_evaluate_agent` path so patch absence is a valid model negative, valid reports are accepted independent of finish reason, only explicitly classified runtime/network/Docker/transient evaluator failures retry the same patch, and final infrastructure failure sets `reward_valid=False` without changing the rollout result.

- [ ] **Step 4: Run evaluator tests and static field tests**

Run: `pytest tests/agentic/test_stage1b_evaluator_validity.py tests/agentic/test_codeact_reward_fields_static.py -q`

Expected: PASS.

- [ ] **Step 5: Commit evaluator validity**

Run: `git add verl/workers/agentic/codeact.py verl/workers/agentic/rollout_logging.py tests/agentic/test_stage1b_evaluator_validity.py && git commit -m "feat: separate evaluator validity from finish reason"`

### Task 5: Valid-aware GRPO and effective loss mask

**Files:**
- Modify: `verl/trainer/ppo/core_algos.py`
- Modify: `verl/trainer/ppo/ray_trainer.py`
- Modify: `verl/workers/actor/dp_actor.py`
- Modify: `verl/trainer/ppo/metric_utils.py`
- Create: `tests/trainer/test_stage1b_grpo_masks.py`

**Interfaces:**
- Extend `compute_grpo_outcome_advantage(..., sample_valid_mask: torch.Tensor | None = None)` so group statistics use only valid samples and invalid outputs are zero.
- Add `apply_stage1b_training_masks(data: DataProto) -> DataProto` to intersect `loss_mask`/`response_mask` with acceptance and validity.
- Add `effective_train_tokens(loss_mask: torch.Tensor) -> int`.
- Actor update metrics include distinct `actor/grad_norm_pre_clip`, `actor/grad_norm_post_clip`, and `train/effective_train_tokens`.

- [ ] **Step 1: Write failing mask/metric tests**

Construct two task groups with one rejected group and one invalid trajectory in an accepted group; assert only valid accepted rewards determine advantages, invalid/rejected positions are zero, final loss-mask sum is the effective token count, and a clipped mock gradient yields different pre/post norms.

- [ ] **Step 2: Run focused tests to verify failure**

Run: `pytest tests/trainer/test_stage1b_grpo_masks.py tests/workers/test_stage1b_gradient_metrics.py -q`

Expected: FAIL because the valid-mask argument/helpers/metrics do not exist.

- [ ] **Step 3: Implement valid-aware advantage and actor metrics**

Filter each task's score list by `sample_valid_mask` before computing mean/std, multiply output by valid response masks, intersect actor `loss_mask` with validity, count the final mask, and compute post-clip norm from parameters after clipping rather than reusing the clipping function return value.

- [ ] **Step 4: Run focused and existing GRPO tests**

Run: `pytest tests/trainer/test_stage1b_grpo_masks.py tests/workers/test_stage1b_gradient_metrics.py tests/trainer/test_grpo_and_mask_local.py tests/trainer/test_no_advantage_fallback.py -q`

Expected: PASS.

- [ ] **Step 5: Commit masking and actor metrics**

Run: `git add verl/trainer/ppo/core_algos.py verl/trainer/ppo/ray_trainer.py verl/workers/actor/dp_actor.py verl/trainer/ppo/metric_utils.py tests/trainer/test_stage1b_grpo_masks.py tests/workers/test_stage1b_gradient_metrics.py && git commit -m "feat: mask invalid Stage1B GRPO samples"`

### Task 6: Trainer coordinator and resumable Stage1B state

**Files:**
- Create: `verl/trainer/ppo/stage1b_sampling.py` (extend Task 2 integration methods)
- Modify: `verl/trainer/ppo/ray_trainer.py`
- Modify: `verl/trainer/config/ppo_trainer.yaml`
- Create: `tests/trainer/test_stage1b_checkpoint_state.py`

**Interfaces:**
- `RayPPOTrainer._stage1b_candidate_batch(step) -> tuple[DataProto | None, dict[str, Any]]` repeatedly creates one-task generation batches, calls existing rollout/reward flow, accepts only informative groups, and returns `None` when underfilled.
- `RayPPOTrainer._save_stage1b_state(global_step_folder)` and `_load_stage1b_state(global_step_folder)` persist/restore sampler/history state.

- [ ] **Step 1: Write failing coordinator/checkpoint tests**

Mock candidate generation/reward callbacks to assert replacement stops at four informative groups or eight attempts, underfilled returns no training batch, and a saved state restores RNG/history/last-seen step exactly.

- [ ] **Step 2: Run focused tests to verify failure**

Run: `pytest tests/trainer/test_stage1b_checkpoint_state.py tests/trainer/test_stage1b_sampling.py -q`

Expected: FAIL because the trainer methods and state file do not exist.

- [ ] **Step 3: Implement Stage1B trainer flow**

Initialize the sampler from the 64-row candidate dataset, materialize one-row `DataProto` values with existing `collate_fn`, pass `n_trajectories=8`, run reward before old/ref log probabilities and advantage, concatenate accepted groups only, skip actor/optimizer work on underfilled rounds, and emit every required aggregate metric.

- [ ] **Step 4: Add checkpoint persistence and run focused tests**

Save versioned `stage1b_state.pt` beside `data.pt` after existing actor checkpoint save; load it after actor/dataloader restore; then run `pytest tests/trainer/test_stage1b_checkpoint_state.py tests/trainer/test_stage1b_sampling.py -q`.

Expected: PASS with history, sampler RNG, candidate pool, and `last_seen_step` restored.

- [ ] **Step 5: Commit trainer/checkpoint integration**

Run: `git add verl/trainer/ppo/ray_trainer.py verl/trainer/config/ppo_trainer.yaml tests/trainer/test_stage1b_checkpoint_state.py && git commit -m "feat: integrate Stage1B dynamic updates and resume state"`

### Task 7: Independent launcher and configuration contract

**Files:**
- Create: `scripts/run_stage1b_32x16_lora.sh`
- Create: `tests/config/test_stage1b_training.py`

**Interfaces:**
- Launcher calls `scripts/prepare_stage1_split.py`, uses `stage1b_candidates.parquet` for train and validation parquet for val, selects the registered Stage1B reward manager, enables Stage1B config, uses independent `stage1b` output/checkpoint/W&B names, starts from the raw SFT model path, sets `max_iterations=15`, and sets `agent_max_prompt_length=32768`.

- [ ] **Step 1: Write failing launcher tests**

Assert the launcher contains candidate/validation paths, Stage1B manager/config, target 4/cap 8/n 8, context 32768, max iterations 15, independent names, no KL/fallback, and unchanged baseline hyperparameter strings.

- [ ] **Step 2: Run launcher tests to verify failure**

Run: `pytest tests/config/test_stage1b_training.py -q`

Expected: FAIL because the Stage1B launcher does not exist.

- [ ] **Step 3: Implement the independent launcher/config**

Copy only the required baseline launcher settings, replace the train artifact and reward/config overrides, preserve the existing GPU parallelism defaults, and avoid changing the Stage1 launcher.

- [ ] **Step 4: Run configuration and baseline contract tests**

Run: `pytest tests/config/test_stage1b_training.py tests/config/test_stage1_32x16_training.py -q`

Expected: PASS.

- [ ] **Step 5: Commit launcher/config**

Run: `git add scripts/run_stage1b_32x16_lora.sh tests/config/test_stage1b_training.py && git commit -m "feat: add Stage1B training launcher"`

### Task 8: Full CPU verification and GPU handoff

**Files:**
- Modify only files required by failing tests from Tasks 1-7.

- [ ] **Step 1: Run the complete CPU test selection**

Run: `pytest tests/config tests/reward_manager tests/trainer tests/agentic tests/workers -q`

Expected: all selected tests pass; report any unrelated pre-existing failures separately.

- [ ] **Step 2: Run syntax/import checks**

Run: `python -m compileall -q verl scripts`

Expected: exit code 0.

- [ ] **Step 3: Run a CPU synthetic Stage1B update integration test**

Run: `pytest tests/trainer/test_stage1b_sampling.py tests/trainer/test_stage1b_grpo_masks.py tests/agentic/test_stage1b_evaluator_validity.py -q`

Expected: candidate replacement, validity, masking, and metrics all pass without GPU/Docker.

- [ ] **Step 4: Verify final diff and preserve user changes**

Run: `git status --short && git diff -- scripts/run_stage1_32x16_lora.sh tests/config/test_stage1_32x16_training.py`

Expected: the two pre-existing user changes remain intact; Stage1B files are the only new implementation changes.

- [ ] **Step 5: Prepare the GPU command without claiming execution**

Use: `WANDB_MODE=online SKYRL_GPUS_PER_NODE=4 bash scripts/run_stage1b_32x16_lora.sh`

Expected after GPU provisioning: one complete informative update, W&B metrics, checkpoint save, and a separate resume invocation using `trainer.resume_mode=resume_path`.
