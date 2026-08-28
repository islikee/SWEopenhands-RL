from verl.workers.agentic.runtime_payload import runtime_instance_record


def test_runtime_instance_record_allowlists_only_container_bootstrap_fields():
    record = runtime_instance_record(
        {
            "instance_id": "demo__repo-1",
            "repo": "demo/repo",
            "version": "1",
            "base_commit": "deadbeef",
            "patch": "gold patch",
            "test_patch": "gold test patch",
            "FAIL_TO_PASS": ["secret target test"],
            "PASS_TO_PASS": ["secret regression test"],
        }
    )

    assert record == {
        "instance_id": "demo__repo-1",
        "repo": "demo/repo",
        "version": "1",
    }
