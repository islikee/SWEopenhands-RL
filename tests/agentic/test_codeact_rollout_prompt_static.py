from pathlib import Path


CODEACT_SOURCE = Path("verl/workers/agentic/codeact.py")


def test_online_rollout_disables_github_microagent():
    source = CODEACT_SOURCE.read_text()

    assert "disabled_microagents=['github']" in source


def test_fake_user_response_pushes_small_source_patch():
    source = CODEACT_SOURCE.read_text()

    assert "edit the relevant non-test source file" in source
    assert "finish if you have already edited it" in source
    assert "Do not browse, explain, create tests" in source
    assert "create backup files" in source
