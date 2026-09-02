from pathlib import Path


CODEACT_SOURCE = Path("verl/workers/agentic/codeact.py")


def test_online_rollout_disables_github_microagent():
    source = CODEACT_SOURCE.read_text()

    assert "disabled_microagents=['github']" in source


def test_fake_user_response_uses_normal_auto_continue_prompt():
    source = CODEACT_SOURCE.read_text()

    assert "Please continue working on the task" in source
    assert "If you think you have solved the task" in source
    assert "You have very few turns left" not in source
    assert "finish if you have already edited it" not in source


def test_empty_action_does_not_silently_finish():
    source = CODEACT_SOURCE.read_text()

    assert "EMPTY_ACTION_RETRY" in source
    assert "FINISH_REASON=model_finish" in source
    assert "FINISH_REASON=context_limit" in source


def test_run_agent_initializes_state_before_controller_can_fail():
    source = CODEACT_SOURCE.read_text()

    assert "state = None\n        try:" in source
