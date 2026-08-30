import pandas as pd

from verl.workers.agentic.utils import get_instruction


def test_instruction_includes_available_test_context_and_small_patch_constraints():
    instance = pd.Series(
        {
            "repo": "owner/repo",
            "version": "1.0",
            "problem_statement": "Fix the tagged launch-template instance behavior.",
            "hints_text": "Look in the launch template model.",
            "FAIL_TO_PASS": ["tests/test_ec2.py::test_launch_template_tags"],
        }
    )

    instruction = get_instruction(instance)

    assert "tests/test_ec2.py::test_launch_template_tags" in instruction
    assert "Look in the launch template model." in instruction
    assert "Do not create or modify test files or reproduction scripts" in instruction
    assert "Do not create backup files such as `.bak` files" in instruction
    assert "do not copy or rewrite whole modules or classes" in instruction
