# Cloud Setup

This branch is meant to be cloned onto a fresh Linux GPU host and moved into
cloud smoke with as little manual state as possible. The cloud commit must match
the commit that was tested and pushed locally.

## Clone

Recommended shallow clone:

```bash
git clone \
  --depth 1 \
  --single-branch \
  --branch reward-shaping-local \
  --recurse-submodules \
  --shallow-submodules \
  https://github.com/islikee/SWEopenhands-RL.git
cd SWEopenhands-RL
```

If the frozen submodule cannot be checked out with shallow submodules, use:

```bash
git clone \
  --depth 1 \
  --single-branch \
  --branch reward-shaping-local \
  https://github.com/islikee/SWEopenhands-RL.git
cd SWEopenhands-RL
git submodule update --init --recursive
```

Immediately record the exact cloud version:

```bash
git rev-parse HEAD
git submodule status
git status
```

## First Run

```bash
bash scripts/setup_cloud.sh
bash scripts/check_docker.sh
bash scripts/download_model.sh
bash scripts/run_cloud_smoke.sh
```

Or prepare everything except training:

```bash
bash scripts/bootstrap_cloud.sh
bash scripts/run_cloud_smoke.sh
```

All setup scripts are intended to be repeatable. They create missing
directories, reuse existing uv/Hugging Face/Docker caches, and do not delete
existing environments or large files.

## Data Root

Set `SKYRL_DATA_ROOT` when the machine has a preferred data disk:

```bash
export SKYRL_DATA_ROOT=/data/skyrl
```

If it is not set, the scripts try `/data`, `/root/autodl-tmp`, and `/workspace`,
then fall back to `$HOME/skyrl-data`.

The resolved layout is:

```text
SKYRL_DATA_ROOT        root for all cloud state
UV_CACHE_DIR           $SKYRL_DATA_ROOT/cache/uv
HF_HOME                $SKYRL_DATA_ROOT/cache/huggingface
XDG_CACHE_HOME         $SKYRL_DATA_ROOT/cache
MODEL_DIR              $SKYRL_DATA_ROOT/models
DATASET_DIR            $SKYRL_DATA_ROOT/datasets
OUTPUT_DIR             $SKYRL_DATA_ROOT/outputs
SKYRL_MODEL_LOCAL_DIR  $MODEL_DIR/Qwen/Qwen2.5-Coder-7B-Instruct
SKYRL_SMOKE_DATA_PATH  $DATASET_DIR/swegym-smoke
SKYRL_DATA_PATH        $DATASET_DIR/swegym
```

The model directory is the largest item. Hugging Face cache and uv wheels are
also reusable across rented machines if the provider lets you remount the same
data disk. Docker images can be reused through the Docker image store; the
scripts never run Docker prune.

## Frozen Versions

`setup_cloud.sh` uses `uv sync --frozen` and then checks:

```text
peft==0.15.1
transformers==4.51.1
sglang==0.4.6.post1
huggingface-hub==0.30.2
swegym from SWE-Bench-Package.git#16dd480cce9b27bf111a362d280881c6def5d2a7
swebench from SWE-Bench-Fork.git#242429c188fcfd06aad13fce9a54d450470bf0ac
```

The project lock also pins `huggingface-hub==0.30.2`. That frozen version is
kept for training. `download_model.sh` defaults to an isolated uv tool
environment with `huggingface_hub[hf_xet]` so model download can use the newer
Xet path without changing the training environment. Set
`SKYRL_HF_DOWNLOADER=frozen` to force the locked downloader.

## Hugging Face

The fixed model is:

```text
Qwen/Qwen2.5-Coder-7B-Instruct
```

Provide credentials through the environment only when needed:

```bash
export HF_TOKEN='<your token>'
```

The scripts also set `HF_XET_HIGH_PERFORMANCE=1` unless you set it yourself.

## Dataset

Cloud smoke reads only the dedicated smoke dataset:

```text
$SKYRL_SMOKE_DATA_PATH/train.parquet
$SKYRL_SMOKE_DATA_PATH/validation.parquet
```

`setup_cloud.sh` prepares these from the Hugging Face dataset
`SWE-Gym/SWE-Gym` using the frozen project environment. By default it writes a
fixed four-instance smoke set under `$DATASET_DIR/swegym-smoke`:

```text
getmoto__moto-7365
getmoto__moto-6920
getmoto__moto-5876
getmoto__moto-5085
```

That count matches the default 4-GPU cloud smoke trainer configuration. Override
it with `SKYRL_SMOKE_INSTANCE_IDS` only when deliberately changing the smoke
workload. Set `SKYRL_SKIP_DATASET_DOWNLOAD=1` if you will place the smoke parquet
files yourself.

Full training should use the same parquet schema, usually under a separate
directory such as `$DATASET_DIR/swegym-full`, and run with `SKYRL_DATA_PATH`
pointed there. The smoke dataset is for infrastructure and end-to-end training
chain validation only; do not use it as a formal experiment dataset.

## Docker

`check_docker.sh` verifies:

```text
Docker daemon: OK/FAIL
Container smoke: OK/FAIL
Smoke image: docker.io/xingyaoww/sweb.eval.x86_64.<instance>
Image local: yes/no
Disk usage: docker system df
```

The smoke image names are read from every `instance_id` in
`$SKYRL_SMOKE_DATA_PATH/train.parquet` and follow the frozen OpenHands/SWE-Bench logic in
`codeact.py`: `sweb.eval.x86_64.` plus the instance id with `__` replaced by
`_s_`, using `EVAL_DOCKER_IMAGE_PREFIX` or `docker.io/xingyaoww/`. The script
checks every image that cloud smoke will actually use and pulls only missing
smoke images. Set
`SKYRL_PULL_SMOKE_IMAGE=0` to check without pulling.

## Cloud Smoke Gates

`run_cloud_smoke.sh` fails before training if any of these are missing:

```text
repo commit and submodule status
Linux GPU host with SKYRL_GPUS_PER_NODE visible GPUs
frozen dependency versions
SKYRL_MODEL_LOCAL_DIR model files
SKYRL_SMOKE_DATA_PATH train/validation parquet files
Docker daemon
all smoke task Docker images
writable OUTPUT_DIR
cloud_smoke/cloud_train config parse
```

It starts cloud smoke with:

```text
training_mode=lora
reward_manager=swebench_test_informed
n_trajectories=2
SKYRL_REQUIRE_ROLLOUT_WEIGHT_SYNC=1
SKYRL_REQUIRE_ROLLOUT_WEIGHT_CHANGE_AFTER_FIRST_SYNC=1
```

## Troubleshooting

Check these first:

```bash
git rev-parse HEAD
git submodule status
python --version
uv run --isolated --directory . --frozen python scripts/cloud_deploy.py check-deps
nvidia-smi
docker version
docker info
python scripts/cloud_deploy.py summary
python scripts/cloud_deploy.py dataset-status
python scripts/cloud_deploy.py docker-image
```
