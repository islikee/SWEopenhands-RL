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


def test_empty_single_trajectory_uses_fallback_messages_and_marks_reward_invalid():
    fallback_messages = [
        {"role": "user", "content": "Fix the issue."},
        {"role": "assistant", "content": ""},
    ]
    results = [
        {
            "messages": [],
            "git_patch": None,
            "resolved": False,
            "error": "controller failed before final state",
        },
    ]

    fill_empty_trajectory_messages(results, fallback_messages=fallback_messages)

    assert results[0]["messages"] == fallback_messages
    assert results[0]["messages"] is not fallback_messages
    assert results[0]["messages"][0] is not fallback_messages[0]
    assert results[0]["reward_valid"] is False
    assert results[0]["infra_error"] is True
    assert "empty trajectory messages" in results[0]["evaluation_error"]
