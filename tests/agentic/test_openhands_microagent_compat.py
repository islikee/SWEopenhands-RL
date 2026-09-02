from openhands.microagent import load_microagents_from_dir


def test_task_microagents_are_ignored_by_current_loader(tmp_path):
    task_dir = tmp_path / "microagents" / "tasks"
    task_dir.mkdir(parents=True)
    (task_dir / "address_pr_comments.md").write_text(
        """---
name: address_pr_comments
type: task
version: 1.0.0
agent: CodeActAgent
---

Ignore this legacy task workflow.
""",
        encoding="utf-8",
    )

    repo_agents, knowledge_agents = load_microagents_from_dir(tmp_path / "microagents")

    assert repo_agents == {}
    assert knowledge_agents == {}
