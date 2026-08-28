from pathlib import Path


def test_baseline_freeze_doc_records_pinned_commits_and_evaluator_schema():
    doc = Path("docs/baseline_freeze.md")

    assert doc.exists()
    text = doc.read_text()
    assert "a0d50c482436af7fac8caffa4533616a78431d66" in text
    assert "6f3ba323d8ad7d17891c1d171abc9eaf8baae252" in text
    assert "SWE-Bench-Package.git#16dd480cce9b27bf111a362d280881c6def5d2a7" in text
    assert "tests_status" in text
    assert "FAIL_TO_PASS" in text
    assert "PASS_TO_PASS" in text
