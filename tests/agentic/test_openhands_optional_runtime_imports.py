def test_codeact_import_does_not_require_daytona_sdk():
    import openhands.agenthub.codeact_agent.function_calling  # noqa: F401


def test_async_rollout_import_does_not_require_mcp_sdk():
    from verl.workers.agentic.async_rollout import AsyncRollout

    assert AsyncRollout.__name__ == "AsyncRollout"


def test_online_codeact_agent_builds_tools_with_current_openhands_api():
    from verl.workers.agentic.codeact import OnlineCodeActAgent

    agent = OnlineCodeActAgent(
        instance_id="test__case-1",
        trajectory_id=0,
    )

    tool_names = {tool["function"]["name"] for tool in agent.tools}
    assert {"execute_bash", "finish", "str_replace_editor"}.issubset(tool_names)
