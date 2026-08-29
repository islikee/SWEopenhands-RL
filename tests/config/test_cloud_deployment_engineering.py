import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load_cloud_deploy():
    path = ROOT / "scripts" / "cloud_deploy.py"
    spec = importlib.util.spec_from_file_location("cloud_deploy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cloud_paths_preserve_explicit_env_and_create_cache_dirs(tmp_path):
    cloud_deploy = load_cloud_deploy()
    explicit_root = tmp_path / "explicit-data"
    explicit_hf = tmp_path / "custom-hf"
    env = {
        "SKYRL_DATA_ROOT": str(explicit_root),
        "HF_HOME": str(explicit_hf),
    }

    paths = cloud_deploy.resolve_cloud_paths(env=env, fallback_home=tmp_path)
    cloud_deploy.ensure_cloud_dirs(paths)

    assert paths["SKYRL_DATA_ROOT"] == explicit_root
    assert paths["HF_HOME"] == explicit_hf
    assert paths["UV_CACHE_DIR"] == explicit_root / "cache" / "uv"
    assert paths["MODEL_DIR"] == explicit_root / "models"
    assert (paths["UV_CACHE_DIR"] / "sentinel.txt").exists() is False
    for key in ["HF_HOME", "MODEL_DIR", "DATASET_DIR", "OUTPUT_DIR"]:
        assert paths[key].is_dir()


def test_cloud_paths_prefer_existing_large_disk_candidate(tmp_path):
    cloud_deploy = load_cloud_deploy()
    data = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data.mkdir()

    paths = cloud_deploy.resolve_cloud_paths(
        env={},
        candidates=[data, workspace],
        fallback_home=tmp_path / "home",
    )

    assert paths["SKYRL_DATA_ROOT"] == data / "skyrl"


def test_ensure_cloud_dirs_does_not_delete_existing_cache_or_model_files(tmp_path):
    cloud_deploy = load_cloud_deploy()
    paths = cloud_deploy.resolve_cloud_paths(
        env={"SKYRL_DATA_ROOT": str(tmp_path / "root")},
        fallback_home=tmp_path,
    )
    cache_file = paths["UV_CACHE_DIR"] / "wheel-cache.bin"
    model_file = paths["MODEL_DIR"] / "Qwen" / "Qwen2.5-Coder-7B-Instruct" / "config.json"
    cache_file.parent.mkdir(parents=True)
    model_file.parent.mkdir(parents=True)
    cache_file.write_text("keep", encoding="utf-8")
    model_file.write_text("keep", encoding="utf-8")

    cloud_deploy.ensure_cloud_dirs(paths)

    assert cache_file.read_text(encoding="utf-8") == "keep"
    assert model_file.read_text(encoding="utf-8") == "keep"


def test_model_completion_requires_config_tokenizer_and_weight(tmp_path):
    cloud_deploy = load_cloud_deploy()
    model_dir = tmp_path / "Qwen" / "Qwen2.5-Coder-7B-Instruct"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    missing = cloud_deploy.missing_model_files(model_dir)
    assert "model.safetensors or model.safetensors.index.json" in missing

    (model_dir / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    assert cloud_deploy.missing_model_files(model_dir) == []


def test_dataset_preflight_reads_first_instance_and_derives_smoke_image(tmp_path):
    pd = pytest.importorskip("pandas")
    cloud_deploy = load_cloud_deploy()
    dataset_dir = tmp_path / "swegym"
    dataset_dir.mkdir()
    pd.DataFrame([{"instance_id": "getmoto__moto-7365"}]).to_parquet(dataset_dir / "train.parquet")
    pd.DataFrame([{"instance_id": "getmoto__moto-7365"}]).to_parquet(dataset_dir / "validation.parquet")

    status = cloud_deploy.dataset_status(dataset_dir)

    assert status.missing_files == []
    assert status.first_instance_id == "getmoto__moto-7365"
    assert (
        cloud_deploy.instance_docker_image(status.first_instance_id)
        == "docker.io/xingyaoww/sweb.eval.x86_64.getmoto_s_moto-7365"
    )


def test_docker_preflight_images_match_all_cloud_smoke_instances(tmp_path):
    pd = pytest.importorskip("pandas")
    cloud_deploy = load_cloud_deploy()
    dataset_dir = tmp_path / "swegym-smoke"
    dataset_dir.mkdir()
    rows = [
        {"instance_id": "getmoto__moto-7365"},
        {"instance_id": "getmoto__moto-6920"},
        {"instance_id": "getmoto__moto-5876"},
        {"instance_id": "getmoto__moto-5085"},
    ]
    pd.DataFrame(rows).to_parquet(dataset_dir / "train.parquet")
    pd.DataFrame(rows).to_parquet(dataset_dir / "validation.parquet")

    status = cloud_deploy.dataset_status(dataset_dir)
    images = cloud_deploy.dataset_docker_images(dataset_dir)

    assert status.instance_ids == [row["instance_id"] for row in rows]
    assert images == [
        "docker.io/xingyaoww/sweb.eval.x86_64.getmoto_s_moto-7365",
        "docker.io/xingyaoww/sweb.eval.x86_64.getmoto_s_moto-6920",
        "docker.io/xingyaoww/sweb.eval.x86_64.getmoto_s_moto-5876",
        "docker.io/xingyaoww/sweb.eval.x86_64.getmoto_s_moto-5085",
    ]


def test_cloud_paths_separate_smoke_dataset_from_training_dataset(tmp_path):
    cloud_deploy = load_cloud_deploy()

    paths = cloud_deploy.resolve_cloud_paths(
        env={"SKYRL_DATA_ROOT": str(tmp_path / "root")},
        fallback_home=tmp_path,
    )

    assert paths["SKYRL_SMOKE_DATA_PATH"] == paths["DATASET_DIR"] / "swegym-smoke"
    assert paths["SKYRL_DATA_PATH"] == paths["DATASET_DIR"] / "swegym"
    assert paths["SKYRL_SMOKE_DATA_PATH"] != paths["SKYRL_DATA_PATH"]


def test_dataset_preflight_rejects_empty_train_parquet(tmp_path):
    pd = pytest.importorskip("pandas")
    cloud_deploy = load_cloud_deploy()
    dataset_dir = tmp_path / "swegym"
    dataset_dir.mkdir()
    pd.DataFrame({"instance_id": []}).to_parquet(dataset_dir / "train.parquet")
    pd.DataFrame([{"instance_id": "getmoto__moto-7365"}]).to_parquet(dataset_dir / "validation.parquet")

    status = cloud_deploy.dataset_status(dataset_dir)

    assert "train.parquet has no rows" in status.missing_files
    assert status.first_instance_id is None


def test_prepare_dataset_reuses_train_when_hf_dataset_has_no_validation_split(tmp_path, monkeypatch):
    pd = pytest.importorskip("pandas")
    cloud_deploy = load_cloud_deploy()
    dataset_dir = tmp_path / "swegym"

    class FakeSplit:
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __iter__(self):
            return iter(self.rows)

        def select(self, indices):
            return FakeSplit([self.rows[index] for index in indices])

        def to_parquet(self, path):
            pd.DataFrame(self.rows).to_parquet(path)

    calls = []

    def fake_load_dataset(_name, **kwargs):
        calls.append(kwargs)
        if kwargs.get("split") == "train":
            return FakeSplit([{"instance_id": "getmoto__moto-7365"}])
        return {"train": FakeSplit([{"instance_id": "getmoto__moto-7365"}])}

    monkeypatch.setitem(__import__("sys").modules, "datasets", type("D", (), {"load_dataset": fake_load_dataset}))

    cloud_deploy.prepare_dataset(dataset_dir, rows=1)

    assert (dataset_dir / "train.parquet").exists()
    assert (dataset_dir / "validation.parquet").exists()
    assert cloud_deploy.dataset_status(dataset_dir).first_instance_id == "getmoto__moto-7365"
    assert calls == [{"split": "train", "streaming": True}]


def test_prepare_smoke_dataset_uses_fixed_instance_ids_in_order(tmp_path, monkeypatch):
    pd = pytest.importorskip("pandas")
    cloud_deploy = load_cloud_deploy()
    dataset_dir = tmp_path / "swegym-smoke"

    rows = [
        {"instance_id": "not-used"},
        {"instance_id": "getmoto__moto-6920"},
        {"instance_id": "getmoto__moto-7365"},
        {"instance_id": "getmoto__moto-5085"},
        {"instance_id": "getmoto__moto-5876"},
    ]

    def fake_load_dataset(_name, **kwargs):
        assert kwargs == {"split": "train", "streaming": True}
        return iter(rows)

    monkeypatch.setitem(__import__("sys").modules, "datasets", type("D", (), {"load_dataset": fake_load_dataset}))

    cloud_deploy.prepare_dataset(
        dataset_dir,
        rows=4,
        instance_ids=[
            "getmoto__moto-7365",
            "getmoto__moto-6920",
            "getmoto__moto-5876",
            "getmoto__moto-5085",
        ],
    )

    train = pd.read_parquet(dataset_dir / "train.parquet")
    assert train["instance_id"].tolist() == [
        "getmoto__moto-7365",
        "getmoto__moto-6920",
        "getmoto__moto-5876",
        "getmoto__moto-5085",
    ]


def test_prepare_smoke_dataset_replaces_existing_non_matching_smoke_set(tmp_path, monkeypatch):
    pd = pytest.importorskip("pandas")
    cloud_deploy = load_cloud_deploy()
    dataset_dir = tmp_path / "swegym-smoke"
    dataset_dir.mkdir()
    pd.DataFrame([{"instance_id": "old__random-1"}]).to_parquet(dataset_dir / "train.parquet")
    pd.DataFrame([{"instance_id": "old__random-1"}]).to_parquet(dataset_dir / "validation.parquet")

    rows = [
        {"instance_id": "getmoto__moto-7365"},
        {"instance_id": "getmoto__moto-6920"},
    ]

    def fake_load_dataset(_name, **kwargs):
        assert kwargs == {"split": "train", "streaming": True}
        return iter(rows)

    monkeypatch.setitem(__import__("sys").modules, "datasets", type("D", (), {"load_dataset": fake_load_dataset}))

    cloud_deploy.prepare_dataset(
        dataset_dir,
        rows=2,
        instance_ids=["getmoto__moto-7365", "getmoto__moto-6920"],
    )

    assert pd.read_parquet(dataset_dir / "train.parquet")["instance_id"].tolist() == [
        "getmoto__moto-7365",
        "getmoto__moto-6920",
    ]


def test_uv_lock_version_parser_reports_frozen_cloud_versions():
    cloud_deploy = load_cloud_deploy()
    versions = cloud_deploy.parse_uv_lock_versions(ROOT / "uv.lock")

    assert versions["peft"] == "0.15.1"
    assert versions["transformers"] == "4.51.1"
    assert versions["sglang"] == "0.4.6.post1"
    assert versions["huggingface-hub"] == "0.30.2"
    assert versions["swegym"].endswith("SWE-Bench-Package.git#16dd480cce9b27bf111a362d280881c6def5d2a7")
    assert versions["swebench"].endswith("SWE-Bench-Fork.git#242429c188fcfd06aad13fce9a54d450470bf0ac")


def test_cloud_setup_doc_and_scripts_share_public_env_names():
    doc = (ROOT / "docs" / "cloud_setup.md").read_text(encoding="utf-8")
    script_text = "\n".join(
        (ROOT / "scripts" / name).read_text(encoding="utf-8")
        for name in [
            "setup_cloud.sh",
            "check_docker.sh",
            "download_model.sh",
            "run_cloud_smoke.sh",
            "bootstrap_cloud.sh",
            "cloud_deploy.py",
        ]
    )

    for name in [
        "SKYRL_DATA_ROOT",
        "UV_CACHE_DIR",
        "HF_HOME",
        "MODEL_DIR",
        "DATASET_DIR",
        "OUTPUT_DIR",
        "SKYRL_MODEL_LOCAL_DIR",
        "SKYRL_SMOKE_DATA_PATH",
        "SKYRL_DATA_PATH",
    ]:
        assert name in doc
        assert name in script_text


def test_cloud_shell_scripts_do_not_prune_or_delete_large_caches():
    for path in [
        ROOT / "scripts" / "setup_cloud.sh",
        ROOT / "scripts" / "check_docker.sh",
        ROOT / "scripts" / "download_model.sh",
        ROOT / "scripts" / "run_cloud_smoke.sh",
    ]:
        text = path.read_text(encoding="utf-8")
        assert "docker system prune" not in text
        assert "rm -rf" not in text
        assert "uv cache clean" not in text
        assert "huggingface-cli delete-cache" not in text
