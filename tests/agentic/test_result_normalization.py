from verl.workers.agentic.result_normalization import fill_empty_trajectory_messages


def test_empty_trajectory_only_inherits_messages_not_evaluator_outcome():
    valid_messages = [{"role": "assistant", "content": "valid"}]
    results = [
        {"messages": valid_messages, "git_patch": "good patch", "resolved": True},
        {"messages": [], "git_patch": None, "resolved": False, "error": "init failed"},
    ]

    fill_empty_trajectory_messages(results)

    assert results[1]["messages"] == valid_messages
    assert results[1]["messages"] is not valid_messages
    assert results[1]["git_patch"] is None
    assert results[1]["resolved"] is False
    assert results[1]["error"] == "init failed"
