from openhands.events.action import CmdRunAction, FileReadAction
from openhands.runtime.impl.action_execution.action_execution_client import (
    ActionExecutionClient,
)
from openhands.runtime.impl.action_execution.action_execution_client import (
    _event_to_runtime_action_dict,
    _legacy_runtime_action_compat_enabled,
)


def test_legacy_runtime_action_compat_detects_oh_025_runtime_image():
    assert _legacy_runtime_action_compat_enabled(
        "ghcr.io/all-hands-ai/runtime:oh_v0.25.0_cached"
    )


def test_legacy_runtime_action_compat_preserves_modern_cmd_fields_by_default():
    action = CmdRunAction(command="echo healthy", is_static=True, cwd="/workspace")

    action_dict = _event_to_runtime_action_dict(action)

    assert action_dict["args"]["is_static"] is True
    assert action_dict["args"]["cwd"] == "/workspace"
    assert action_dict["args"]["hidden"] is False
    assert "confirmation_state" in action_dict["args"]


def test_legacy_runtime_action_compat_removes_cmd_fields_unknown_to_oh_025():
    action = CmdRunAction(command="echo healthy", is_static=True, cwd="/workspace")

    action_dict = _event_to_runtime_action_dict(action, legacy_runtime=True)

    assert action_dict["args"]["command"] == "cd /workspace && echo healthy"
    assert action_dict["args"]["is_input"] is False
    assert action_dict["args"]["blocking"] is False
    assert "is_static" not in action_dict["args"]
    assert "cwd" not in action_dict["args"]
    assert action_dict["args"]["hidden"] is False
    assert "confirmation_state" in action_dict["args"]


def test_file_read_action_is_sent_to_runtime_server_without_shell_tool_conversion():
    class _ConcreteActionExecutionClient(ActionExecutionClient):
        @property
        def action_execution_server_url(self):
            return "http://runtime"

        async def connect(self):
            return None

    client = object.__new__(_ConcreteActionExecutionClient)
    action = FileReadAction(path="/workspace/repo/file.py", view_range=[1, 3])
    sent = {}

    def fake_send_action_for_execution(runtime_action):
        sent["action"] = runtime_action
        return "observation"

    client.send_action_for_execution = fake_send_action_for_execution

    observation = ActionExecutionClient.read(client, action)

    assert observation == "observation"
    assert sent["action"] is action


def test_legacy_runtime_action_compat_removes_file_read_fields_unknown_to_oh_025():
    action = FileReadAction(
        path="/workspace/repo/file.py",
        view_range=[1, 3],
        concise=True,
    )

    modern_action_dict = _event_to_runtime_action_dict(action)
    legacy_action_dict = _event_to_runtime_action_dict(action, legacy_runtime=True)

    assert modern_action_dict["args"]["concise"] is True
    assert "concise" not in legacy_action_dict["args"]
    assert legacy_action_dict["args"]["view_range"] == [1, 3]
