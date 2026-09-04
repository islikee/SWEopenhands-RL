from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto


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


def test_codeact_group_keys_agents_by_batch_slot_when_instance_ids_repeat(monkeypatch):
    from verl.workers.agentic import codeact

    class DummyAgent:
        def __init__(self, *, instance_id, trajectory_id, **_kwargs):
            self.instance_id = instance_id
            self.trajectory_id = trajectory_id
            self.instance_data = None
            self.max_iterations = None

        def close(self):
            pass

    monkeypatch.setattr(codeact, "OnlineCodeActAgent", DummyAgent)
    batch = DataProto(
        batch=TensorDict(
            {
                "input_ids": torch.ones((2, 4), dtype=torch.long),
                "attention_mask": torch.ones((2, 4), dtype=torch.long),
                "position_ids": torch.arange(4, dtype=torch.long).repeat(2, 1),
            },
            batch_size=(2,),
        ),
        non_tensor_batch={
            "instance": np.asarray(
                [
                    {"instance_id": "duplicate__case-1"},
                    {"instance_id": "duplicate__case-1"},
                ],
                dtype=object,
            )
        },
    )

    group = codeact.CodeActAgentGroup(
        batch=batch,
        num_trajectories=4,
        infer_engine=None,
        tokenizer=None,
    )

    assert set(group.agents) == {0, 1}
    assert group.agents[0][0] is not group.agents[1][0]
    assert group.agents[0][0].instance_id == "duplicate__case-1"
    assert group.agents[1][0].instance_id == "duplicate__case-1"
    assert group.agents[0][0].trajectory_id == 0
    assert group.agents[1][0].trajectory_id == 4
