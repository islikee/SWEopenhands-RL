# Baseline Freeze

This project is frozen on SkyRL-v0 and must not be upgraded to a newer SkyRL
mainline while implementing test-informed rewards.

## Provenance

- Upstream: `https://github.com/NovaSky-AI/SkyRL`
- Project remote: `https://github.com/islikee/SWEopenhands-RL`
- Frozen SkyRL commit: `a0d50c482436af7fac8caffa4533616a78431d66`
- Current submodule `SkyRL-OpenHands`: `6f3ba323d8ad7d17891c1d171abc9eaf8baae252`
- `pyproject.toml` points `swegym` to `https://github.com/SWE-Gym/SWE-Bench-Package.git`
- `uv.lock` pins `swegym` to `SWE-Bench-Package.git#16dd480cce9b27bf111a362d280881c6def5d2a7`
- `uv.lock` pins `swebench` to `SWE-Bench-Fork.git#242429c188fcfd06aad13fce9a54d450470bf0ac`
- `uv.lock` pins `sglang==0.4.6.post1`, `peft==0.15.1`, and `transformers==4.51.1`
- Local development environment observed on Windows: `torch==2.5.1+cpu`,
  `transformers==5.16.1`, `peft==0.20.0`; this is not the official full
  SkyRL-v0 reproduction environment.

## Evaluator Schema Evidence

The frozen dependency source was checked directly from
`SWE-Gym/SWE-Bench-Package.git` at
`16dd480cce9b27bf111a362d280881c6def5d2a7`.

In `swegym/harness/grading.py`, `get_eval_report(test_spec, prediction,
log_path, include_tests_status=True)` builds `eval_ref` from
`test_spec.FAIL_TO_PASS` and `test_spec.PASS_TO_PASS`, calls
`get_eval_tests_report(eval_sm, eval_ref)`, sets top-level `resolved=True`
only for full resolution, and when `include_tests_status` is true stores that
test report under `report_map[instance_id]["tests_status"]`.

The historical `tests_status` structure is:

```python
{
    "FAIL_TO_PASS": {"success": [...], "failure": [...]},
    "PASS_TO_PASS": {"success": [...], "failure": [...]},
}
```

`FAIL_TO_PASS` is the target repair-test suite. `PASS_TO_PASS` is the
regression-maintenance suite.

## Frozen Call Paths

- Baseline binary reward path: `verl/workers/reward_manager/swebench.py`
- New shaped reward path: `verl/workers/reward_manager/swebench_test_informed.py`
- Reward manager registration: `verl/trainer/main_ppo.py`
- OpenHands rollout path: `verl/workers/agentic/codeact.py`
- Evaluator call path: `codeact.py` calls
  `swegym.harness.grading.get_eval_report(..., include_tests_status=True)`
- GRPO launch example: `examples/sky/run_skyrl_agent_oh7b_s1.sh` uses
  `algorithm.adv_estimator=grpo`
- Actor mask path: `codeact.py` writes `loss_mask`; `verl/workers/actor/dp_actor.py`
  uses it when `actor_rollout_ref.actor.masking=True`
- Checkpoint path: `verl/trainer/ppo/ray_trainer.py` calls worker
  `save_checkpoint`; `verl/utils/checkpoint/fsdp_checkpoint_manager.py` saves
  sharded model, optimizer, and extra state
- Rollout weight sync path: `verl/workers/fsdp_workers.py` enters
  `verl/workers/agentic/fsdp_sgl.py::FSDPSGLShardingManager`, which calls
  SGLang `update_weights_from_tensor`

## Reward Data Flow

OpenHands rollout generates a trajectory and git patch. The original SWE-Gym
evaluator runs once and returns the raw report. `codeact.py` stores the raw
report for debug logs, normalizes primitive fields into `DataProto`, and does
not expose test facts to the Agent prompt.

The primitive training fields are `resolved`, `target_tests_total`,
`target_tests_passed`, `target_tests_failed`, `regression_tests_total`,
`regression_tests_passed`, `regression_tests_failed`, `evaluation_error`,
`evaluation_timeout`, `outcome_primary`, and `failure_flags`.

## Test-Informed Reward

Binary reward remains `1.0` if `resolved=True`, else `0.0`.

Test-informed reward uses:

```text
if resolved: 1.0
elif evaluation_error or evaluation_timeout: 0.0
elif target_tests_total == 0: binary_reward
else:
  target_fraction = target_tests_passed / target_tests_total
  regression_fraction = 1.0 if regression_tests_total == 0 else regression_tests_passed / regression_tests_total
  reward = target_fraction * (target_weight + regression_weight * regression_fraction)
```

Defaults are `target_weight=0.8` and `regression_weight=0.2`; they must sum to
`1.0`.

## LoRA and Rollout Sync

`training_mode=full` keeps the original actor construction, FSDP
`use_orig_params=False`, and optimizer parameter path.

`training_mode=lora` wraps the actor with PEFT LoRA, freezes base parameters,
keeps adapter parameters trainable, sets FSDP `use_orig_params=True`, and gives
the optimizer only trainable parameters.

SGLang `update_weights_from_tensor` expects effective model weight names and
tensors, not PEFT adapter keys. For LoRA actors, rollout sync builds an
effective payload by adding each LoRA delta to its base layer weight and
normalizing PEFT names such as
`base_model.model.<path>.base_layer.weight` back to `<path>.weight`. Adapter
keys such as `lora_A` and `lora_B` are not sent to SGLang.

Real CUDA/SGLang synchronization remains a hard `cloud_smoke` gate. The cloud
script runs two training steps and exports `SKYRL_REQUIRE_ROLLOUT_WEIGHT_SYNC=1`
and `SKYRL_REQUIRE_ROLLOUT_WEIGHT_CHANGE_AFTER_FIRST_SYNC=1`; under that gate
the sharding manager validates a non-empty effective weight payload, rejects
PEFT adapter/base-layer keys, calls SGLang `update_weights_from_tensor`, fails
the run if the raw async SGLang update interface returns a negative
acknowledgement, and fails the second synchronization if the effective payload
is unchanged from the previous rollout sync. The frozen `VerlEngine` wrapper
returns `None` after its successful update path, so `None` is accepted for that
wrapper; the raw async engine used by `cloud_smoke` returns an explicit
`(success, message)` result and is hard-gated on `success=True`.
