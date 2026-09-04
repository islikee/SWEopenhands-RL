#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


DEFAULT_MANIFEST = Path("configs/stage1_32x16_tasks.yaml")


def _instance_id(row: pd.Series) -> str:
    if "instance_id" in row and pd.notna(row["instance_id"]):
        return str(row["instance_id"])
    instance = row.get("instance")
    if isinstance(instance, dict):
        return str(instance["instance_id"])
    raise KeyError("row does not contain instance_id or instance['instance_id']")


def _read_split(path: Path) -> pd.DataFrame:
    frames = []
    for name in ["train.parquet", "validation.parquet"]:
        parquet_path = path / name
        if parquet_path.exists():
            frames.append(pd.read_parquet(parquet_path))
    if not frames:
        raise FileNotFoundError(f"No train.parquet or validation.parquet under {path}")
    return pd.concat(frames, ignore_index=True)


def _select_rows(source: pd.DataFrame, ids: list[str], split_name: str) -> pd.DataFrame:
    rows_by_id: dict[str, dict[str, Any]] = {}
    for _, row in source.iterrows():
        task_id = _instance_id(row)
        if task_id not in rows_by_id:
            rows_by_id[task_id] = row.to_dict()

    missing = [task_id for task_id in ids if task_id not in rows_by_id]
    if missing:
        raise SystemExit(f"{split_name} ids missing from source dataset: {', '.join(missing)}")

    return pd.DataFrame([rows_by_id[task_id] for task_id in ids])


def _write_ids(path: Path, ids: list[str]) -> None:
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")


def prepare_split(manifest_path: Path, source_path: Path | None, output_path: Path | None) -> Path:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    train_ids = list(manifest["train"])
    validation_ids = list(manifest["validation"])
    universe_ids = list(manifest["stage1_universe"])
    universe = set(universe_ids)
    old_train = set(train_ids)
    validation = set(validation_ids)

    if len(universe) != 80 or len(universe_ids) != 80:
        raise SystemExit("stage1_universe must contain 80 unique task ids")
    if len(old_train) != 32 or len(train_ids) != 32:
        raise SystemExit("train split must contain 32 unique task ids")
    if len(validation) != 16 or len(validation_ids) != 16:
        raise SystemExit("validation split must contain 16 unique task ids")
    if not set(train_ids).isdisjoint(validation_ids):
        overlap = sorted(old_train.intersection(validation))
        raise SystemExit(f"train/validation task overlap: {', '.join(overlap)}")
    if not old_train.issubset(universe):
        raise SystemExit("train split contains ids outside stage1_universe")
    if not validation.issubset(universe):
        raise SystemExit("validation split contains ids outside stage1_universe")

    unused = universe - old_train - validation
    stage1b_candidates = old_train | unused
    assert len(universe) == 80
    assert len(old_train) == 32
    assert len(validation) == 16
    assert len(unused) == 32
    assert len(stage1b_candidates) == 64
    assert stage1b_candidates.isdisjoint(validation)
    assert stage1b_candidates | validation == universe
    candidate_ids = [task_id for task_id in universe_ids if task_id in stage1b_candidates]

    source_dir = source_path or Path(manifest["source_dataset"])
    output_dir = output_path or Path(manifest["output_dataset"])
    source = _read_split(source_dir)
    train = _select_rows(source, train_ids, "train")
    validation = _select_rows(source, validation_ids, "validation")
    candidates = _select_rows(source, candidate_ids, "stage1b_candidates")

    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(output_dir / "train.parquet", index=False)
    validation.to_parquet(output_dir / "validation.parquet", index=False)
    candidates.to_parquet(output_dir / "stage1b_candidates.parquet", index=False)
    _write_ids(output_dir / "train_instance_ids.txt", train_ids)
    _write_ids(output_dir / "validation_instance_ids.txt", validation_ids)
    _write_ids(output_dir / "stage1_universe_instance_ids.txt", universe_ids)
    _write_ids(output_dir / "stage1b_candidate_instance_ids.txt", candidate_ids)
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output_dir = prepare_split(args.manifest, args.source, args.output)
    print(f"Prepared stage1 split: {output_dir}")
    print("train=32 validation=16 candidates=64 overlap=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
