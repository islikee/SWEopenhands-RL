import pandas as pd
import pytest

from openhands.events.observation import CmdOutputObservation

from verl.workers.agentic.utils import complete_runtime


def _cmd_obs(command: str, exit_code: int = 0, content: str = "") -> CmdOutputObservation:
    return CmdOutputObservation(
        content=content,
        command=command,
        metadata={
            "exit_code": exit_code,
            "pid": -1,
        },
    )


class _FakeRuntime:
    def __init__(self):
        self.actions = []
        self._responses = [
            _cmd_obs("cd /workspace/owner__repo__1.0", exit_code=-1),
            _cmd_obs("C-c"),
            _cmd_obs("cd /workspace/owner__repo__1.0"),
            _cmd_obs('git config --global core.pager ""'),
            _cmd_obs("git add -A"),
            _cmd_obs("git diff --no-color --cached abc123", content="diff --git a/file.py b/file.py"),
        ]

    def run_action(self, action):
        self.actions.append(action)
        if not self._responses:
            pytest.fail(f"unexpected runtime action: {action}")
        return self._responses.pop(0)


def test_complete_runtime_sends_ctrl_c_as_input_when_previous_command_is_running():
    runtime = _FakeRuntime()
    instance = pd.Series(
        {
            "repo": "owner/repo",
            "version": "1.0",
            "base_commit": "abc123",
        }
    )

    complete_runtime(runtime, instance)

    interrupt_actions = [action for action in runtime.actions if action.command == "C-c"]
    assert len(interrupt_actions) == 1
    assert interrupt_actions[0].is_input is True
