import torch

from verl.utils.dataset import rl_dataset
from verl.utils.dataset.rl_dataset import RLHFDataset


class _Tokenizer:
    pad_token_id = 0

    def apply_chat_template(self, chat, add_generation_prompt=True, tokenize=False):
        assert chat == "issue only"
        return f"PROMPT: {chat}"

    def encode(self, prompt, add_special_tokens=False):
        assert prompt == "PROMPT: issue only"
        return [1, 2, 3]


def test_swegym_rows_preserve_instance_payload_when_prompt_key_is_problem_statement(tmp_path, monkeypatch):
    pd = __import__("pytest").importorskip("pandas")
    dataset_path = tmp_path / "train.parquet"
    pd.DataFrame(
        [
            {
                "instance_id": "demo__repo-1",
                "problem_statement": "issue only",
                "repo": "demo/repo",
                "base_commit": "abc123",
                "version": "1.0",
                "FAIL_TO_PASS": ["hidden target test"],
                "PASS_TO_PASS": ["hidden regression test"],
            }
        ]
    ).to_parquet(dataset_path)

    monkeypatch.setattr(
        rl_dataset.verl_F,
        "tokenize_and_postprocess_data",
        lambda **_: (torch.tensor([[1, 2, 3]]), torch.tensor([[1, 1, 1]])),
    )
    monkeypatch.setattr(
        rl_dataset,
        "compute_position_id_with_mask",
        lambda attention_mask: torch.zeros_like(attention_mask),
    )

    dataset = RLHFDataset(
        parquet_files=str(dataset_path),
        tokenizer=_Tokenizer(),
        prompt_key="problem_statement",
        max_prompt_length=128,
    )

    item = dataset[0]

    assert item["instance"]["instance_id"] == "demo__repo-1"
    assert item["instance"]["problem_statement"] == "issue only"
    assert item["instance"]["FAIL_TO_PASS"] == ["hidden target test"]
    assert item["raw_prompt_ids"] == [1, 2, 3]
