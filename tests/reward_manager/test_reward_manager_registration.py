from pathlib import Path


def test_test_informed_reward_manager_is_importable_from_registry_package():
    from verl.workers.reward_manager import TestInformedSWEBenchRewardManager

    assert TestInformedSWEBenchRewardManager.__name__ == "TestInformedSWEBenchRewardManager"


def test_main_ppo_supports_swebench_test_informed_config_switch():
    source = Path("verl/trainer/main_ppo.py").read_text()

    assert "swebench_test_informed" in source
    assert "TestInformedSWEBenchRewardManager" in source
