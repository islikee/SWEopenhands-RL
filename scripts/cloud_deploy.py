#!/usr/bin/env python3
"""Small cloud deployment helpers for the frozen SkyRL-v0 checkout."""

from __future__ import annotations

import argparse
import importlib.metadata
import itertools
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None


MODEL_REPO = "Qwen/Qwen2.5-Coder-7B-Instruct"
DATASET_REPO = "SWE-Gym/SWE-Gym"
DEFAULT_DOCKER_IMAGE_PREFIX = "docker.io/xingyaoww/"
EXPECTED_VERSIONS = {
    "peft": "0.15.1",
    "transformers": "4.51.1",
    "sglang": "0.4.6.post1",
    "huggingface-hub": "0.30.2",
}


class DatasetStatus:
    def __init__(self, missing_files: list[str], first_instance_id: str | None):
        self.missing_files = missing_files
        self.first_instance_id = first_instance_id


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def _as_path(value: str | os.PathLike[str]) -> Path:
    return Path(value).expanduser().resolve()


def choose_data_root(
    env: dict[str, str] | None = None,
    candidates: list[Path] | None = None,
    fallback_home: Path | None = None,
) -> Path:
    env = env or os.environ
    if env.get("SKYRL_DATA_ROOT"):
        return _as_path(env["SKYRL_DATA_ROOT"])

    for candidate in candidates or [Path("/data"), Path("/root/autodl-tmp"), Path("/workspace")]:
        if candidate.exists():
            return (candidate / "skyrl").resolve()

    home = fallback_home or Path.home()
    return (home / "skyrl-data").expanduser().resolve()


def resolve_cloud_paths(
    env: dict[str, str] | None = None,
    candidates: list[Path] | None = None,
    fallback_home: Path | None = None,
) -> dict[str, Path]:
    env = env or os.environ
    root = choose_data_root(env=env, candidates=candidates, fallback_home=fallback_home)
    model_dir = _as_path(env.get("MODEL_DIR", root / "models"))
    dataset_dir = _as_path(env.get("DATASET_DIR", root / "datasets"))
    output_dir = _as_path(env.get("OUTPUT_DIR", root / "outputs"))
    return {
        "SKYRL_DATA_ROOT": root,
        "UV_CACHE_DIR": _as_path(env.get("UV_CACHE_DIR", root / "cache" / "uv")),
        "HF_HOME": _as_path(env.get("HF_HOME", root / "cache" / "huggingface")),
        "XDG_CACHE_HOME": _as_path(env.get("XDG_CACHE_HOME", root / "cache")),
        "MODEL_DIR": model_dir,
        "DATASET_DIR": dataset_dir,
        "OUTPUT_DIR": output_dir,
        "SKYRL_MODEL_LOCAL_DIR": _as_path(
            env.get("SKYRL_MODEL_LOCAL_DIR", model_dir / MODEL_REPO)
        ),
        "SKYRL_DATA_PATH": _as_path(env.get("SKYRL_DATA_PATH", dataset_dir / "swegym")),
    }


def ensure_cloud_dirs(paths: dict[str, Path]) -> None:
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)


def shell_exports(paths: dict[str, Path]) -> str:
    lines = []
    for key, value in paths.items():
        lines.append(f"export {key}={shlex.quote(str(value))}")
    lines.append(f"export SKYRL_MODEL_REPO={shlex.quote(MODEL_REPO)}")
    lines.append(f"export SKYRL_DATASET_REPO={shlex.quote(DATASET_REPO)}")
    return "\n".join(lines)


def missing_model_files(model_dir: Path) -> list[str]:
    missing: list[str] = []
    for name in ["config.json", "tokenizer_config.json"]:
        if not (model_dir / name).exists():
            missing.append(name)
    has_weights = (model_dir / "model.safetensors.index.json").exists() or any(
        model_dir.glob("*.safetensors")
    )
    if not has_weights:
        missing.append("model.safetensors or model.safetensors.index.json")
    return missing


def dataset_status(dataset_dir: Path) -> DatasetStatus:
    required = ["train.parquet", "validation.parquet"]
    missing = [name for name in required if not (dataset_dir / name).exists()]
    first_instance_id = None
    if not missing:
        try:
            import pandas as pd

            frame = pd.read_parquet(dataset_dir / "train.parquet", columns=["instance_id"])
            if not frame.empty:
                first_instance_id = str(frame.iloc[0]["instance_id"])
            else:
                missing.append("train.parquet has no rows")
        except Exception as exc:
            missing.append(f"cannot read train.parquet instance_id: {exc}")
    return DatasetStatus(missing, first_instance_id)


def instance_docker_image(instance_id: str, prefix: str | None = None) -> str:
    image_name = "sweb.eval.x86_64." + instance_id
    image_name = image_name.replace("__", "_s_")
    return ((prefix or os.environ.get("EVAL_DOCKER_IMAGE_PREFIX", DEFAULT_DOCKER_IMAGE_PREFIX)).rstrip("/") + "/" + image_name).lower()


def parse_uv_lock_versions(lock_path: Path) -> dict[str, str]:
    if tomllib is None:
        raise RuntimeError("Parsing uv.lock requires Python 3.11+ or the frozen uv environment.")
    data = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    versions: dict[str, str] = {}
    for package in data.get("package", []):
        name = package.get("name")
        if name not in {"peft", "transformers", "sglang", "huggingface-hub", "swegym", "swebench"}:
            continue
        source = package.get("source", {})
        if "git" in source:
            versions[name] = source["git"]
        else:
            versions[name] = package.get("version", "")
    return versions


def check_dependency_versions() -> list[str]:
    errors = []
    for dist_name, expected in EXPECTED_VERSIONS.items():
        try:
            actual = importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"{dist_name}: missing, expected {expected}")
            continue
        if actual != expected:
            errors.append(f"{dist_name}: {actual}, expected {expected}")
    return errors


def installed_dependency_versions() -> dict[str, str]:
    versions = {}
    for dist_name in EXPECTED_VERSIONS:
        try:
            versions[dist_name] = importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            versions[dist_name] = "missing"
    return versions


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def gpu_count() -> int:
    if not command_exists("nvidia-smi"):
        return 0
    result = run(["nvidia-smi", "-L"], check=False)
    return len([line for line in result.stdout.splitlines() if line.strip().startswith("GPU ")])


def prepare_dataset(dataset_dir: Path, rows: int) -> None:
    if not dataset_status(dataset_dir).missing_files:
        print(f"Dataset already prepared: {dataset_dir}")
        return

    from datasets import load_dataset
    import pandas as pd

    dataset_dir.mkdir(parents=True, exist_ok=True)
    if rows > 0:
        train_rows = list(itertools.islice(load_dataset(DATASET_REPO, split="train", streaming=True), rows))
        if not train_rows:
            raise SystemExit(f"{DATASET_REPO} train split returned no rows")
        pd.DataFrame(train_rows).to_parquet(dataset_dir / "train.parquet")
        pd.DataFrame(train_rows).to_parquet(dataset_dir / "validation.parquet")
        print(f"Dataset prepared: {dataset_dir}")
        return

    dataset_dict = load_dataset(DATASET_REPO)
    train = dataset_dict["train"]
    validation = dataset_dict.get("validation") or dataset_dict.get("test") or train
    train.to_parquet(str(dataset_dir / "train.parquet"))
    validation.to_parquet(str(dataset_dir / "validation.parquet"))
    print(f"Dataset prepared: {dataset_dir}")


def download_model(model_dir: Path) -> None:
    missing = missing_model_files(model_dir)
    if not missing and os.environ.get("SKYRL_FORCE_MODEL_DOWNLOAD", "0") != "1":
        print(f"Model already present: {model_dir}")
        return

    from huggingface_hub import snapshot_download

    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=os.environ.get("SKYRL_MODEL_REPO", MODEL_REPO),
        local_dir=str(model_dir),
        resume_download=True,
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
    )
    missing = missing_model_files(model_dir)
    if missing:
        raise SystemExit(f"Model download incomplete at {model_dir}: missing {', '.join(missing)}")
    print(f"Model path: {model_dir}")


def print_summary(paths: dict[str, Path]) -> None:
    for key in [
        "SKYRL_DATA_ROOT",
        "UV_CACHE_DIR",
        "HF_HOME",
        "XDG_CACHE_HOME",
        "MODEL_DIR",
        "DATASET_DIR",
        "OUTPUT_DIR",
        "SKYRL_MODEL_LOCAL_DIR",
        "SKYRL_DATA_PATH",
    ]:
        print(f"{key}={paths[key]}")


def preflight_run(repo_root: Path, paths: dict[str, Path]) -> None:
    errors: list[str] = []
    if sys.platform != "linux":
        errors.append("Linux is required for cloud smoke.")
    if gpu_count() < int(os.environ.get("SKYRL_GPUS_PER_NODE", "4")):
        errors.append("Not enough visible GPUs for SKYRL_GPUS_PER_NODE.")
    for command in ["git", "docker", "uv"]:
        if not command_exists(command):
            errors.append(f"Missing command: {command}")
    if command_exists("docker") and run(["docker", "info"], check=False).returncode != 0:
        errors.append("Docker daemon is not available.")
    errors.extend(check_dependency_versions())
    model_missing = missing_model_files(paths["SKYRL_MODEL_LOCAL_DIR"])
    if model_missing:
        errors.append(
            f"Model path incomplete: {paths['SKYRL_MODEL_LOCAL_DIR']} missing {', '.join(model_missing)}"
        )
    status = dataset_status(paths["SKYRL_DATA_PATH"])
    if status.missing_files:
        errors.append(
            f"Dataset path incomplete: {paths['SKYRL_DATA_PATH']} missing {', '.join(status.missing_files)}"
        )
    elif status.first_instance_id:
        image = instance_docker_image(status.first_instance_id)
        if command_exists("docker") and run(["docker", "image", "inspect", image], check=False).returncode != 0:
            errors.append(f"Smoke task Docker image missing locally: {image}")
    try:
        paths["OUTPUT_DIR"].mkdir(parents=True, exist_ok=True)
        probe = paths["OUTPUT_DIR"] / ".skyrl_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except Exception as exc:
        errors.append(f"Output path is not writable: {paths['OUTPUT_DIR']} ({exc})")
    try:
        run(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
        run(["git", "-C", str(repo_root), "submodule", "status"])
    except Exception as exc:
        errors.append(f"Git state check failed: {exc}")
    try:
        from omegaconf import OmegaConf

        OmegaConf.load(repo_root / "configs" / "cloud_smoke.yaml")
        OmegaConf.load(repo_root / "configs" / "cloud_train.yaml")
    except Exception as exc:
        errors.append(f"Config parse failed: {exc}")

    if errors:
        print("Cloud smoke preflight failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("env").add_argument("--shell", action="store_true")
    sub.add_parser("summary")
    sub.add_parser("check-deps")
    sub.add_parser("download-model")
    dataset_parser = sub.add_parser("prepare-dataset")
    dataset_parser.add_argument("--rows", type=int, default=int(os.environ.get("SKYRL_SMOKE_DATASET_ROWS", "4")))
    sub.add_parser("dataset-status")
    sub.add_parser("docker-image")
    sub.add_parser("preflight-run")
    args = parser.parse_args(argv)

    paths = resolve_cloud_paths()
    if args.command == "env":
        ensure_cloud_dirs(paths)
        if args.shell:
            print(shell_exports(paths))
        else:
            print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    elif args.command == "summary":
        ensure_cloud_dirs(paths)
        print_summary(paths)
    elif args.command == "check-deps":
        errors = check_dependency_versions()
        for name, version in installed_dependency_versions().items():
            print(f"{name}=={version}")
        try:
            lock_versions = parse_uv_lock_versions(repo_root_from_script() / "uv.lock")
            for name in ["swegym", "swebench"]:
                if name in lock_versions:
                    print(f"{name}={lock_versions[name]}")
        except Exception as exc:
            errors.append(f"uv.lock parse failed: {exc}")
        if errors:
            print("\n".join(errors))
            return 1
        print("Frozen dependency versions: OK")
    elif args.command == "download-model":
        download_model(paths["SKYRL_MODEL_LOCAL_DIR"])
    elif args.command == "prepare-dataset":
        prepare_dataset(paths["SKYRL_DATA_PATH"], args.rows)
    elif args.command == "dataset-status":
        status = dataset_status(paths["SKYRL_DATA_PATH"])
        print(f"dataset={paths['SKYRL_DATA_PATH']}")
        print(f"missing={','.join(status.missing_files) if status.missing_files else 'none'}")
        print(f"first_instance_id={status.first_instance_id or ''}")
    elif args.command == "docker-image":
        status = dataset_status(paths["SKYRL_DATA_PATH"])
        if status.missing_files or not status.first_instance_id:
            print("Smoke image: unavailable until dataset exists")
            return 1
        print(instance_docker_image(status.first_instance_id))
    elif args.command == "preflight-run":
        preflight_run(repo_root_from_script(), paths)
        print("Cloud smoke preflight: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
