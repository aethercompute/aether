import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_script():
    path = Path(__file__).parents[2] / "scripts" / "prepare-sft-local.py"
    spec = importlib.util.spec_from_file_location("prepare_sft_local_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MESSAGES = [
    {"role": "user", "content": "one"},
    {"role": "assistant", "content": "two"},
    {"role": "user", "content": "three"},
    {"role": "assistant", "content": "four"},
]


def test_normalized_messages_cleans_content_and_rejects_malformed_values():
    script = load_script()
    sample = {
        "conversation": [
            {"role": "user", "content": "  question  "},
            {"role": "assistant", "content": "   "},
            {"role": "assistant", "content": " answer\n"},
        ]
    }
    assert script.normalized_messages(sample, "conversation") == [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]

    malformed = [
        {},
        {"conversation": "not a list"},
        {"conversation": ["not a mapping"]},
        {"conversation": [{"role": 1, "content": "text"}]},
        {"conversation": [{"role": "user", "content": None}]},
        {"conversation": [{"role": "user", "content": "   "}]},
    ]
    for value in malformed:
        assert script.normalized_messages(value, "conversation") is None


def test_common_prefix_len_handles_equal_mismatch_and_shorter_inputs():
    script = load_script()
    assert script.common_prefix_len([1, 2, 3], [1, 2, 3]) == 3
    assert script.common_prefix_len([1, 2, 9], [1, 2, 3]) == 2
    assert script.common_prefix_len([1, 2], [1, 2, 3]) == 2
    assert script.common_prefix_len([], [1, 2, 3]) == 0


def test_prompt_response_masking_truncation_and_padding():
    script = load_script()

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 9

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens
            if text == "question\n":
                return [1, 2]
            assert text == "question\nanswer"
            return [1, 2, 3, 4]

    padded = script.build_example(
        Tokenizer(),
        "question",
        "answer",
        SimpleNamespace(mode="prompt-response", sequence_length=6),
    )
    assert padded == (
        [1, 2, 3, 4, 9, 0],
        [-100, -100, 3, 4, 9, -100],
        5,
    )

    truncated = script.build_example(
        Tokenizer(),
        "question",
        "answer",
        SimpleNamespace(mode="prompt-response", sequence_length=4),
    )
    assert truncated == (
        [1, 2, 3, 4],
        [-100, -100, 3, 4],
        4,
    )


def test_message_tokens_supervises_all_tokenizer_identified_assistant_spans():
    script = load_script()

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["return_assistant_tokens_mask"]
            return {"input_ids": [1, 2, 3, 4], "assistant_masks": [0, 1, 0, 1]}

    assert script.message_tokens(Tokenizer(), MESSAGES) == (
        [1, 2, 3, 4],
        [-100, 2, -100, 4],
    )


def test_message_tokens_falls_back_to_last_assistant_span():
    script = load_script()

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            if kwargs.get("return_assistant_tokens_mask"):
                raise TypeError("unsupported")
            if kwargs["add_generation_prompt"]:
                return [1, 2, 3]
            return [1, 2, 3, 4]

    assert script.message_tokens(Tokenizer(), MESSAGES) == (
        [1, 2, 3, 4],
        [-100, -100, -100, 4],
    )


def test_build_messages_example_rejects_all_masked_after_truncation():
    script = load_script()

    class Tokenizer:
        pad_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["return_assistant_tokens_mask"]
            return {
                "input_ids": [1, 2, 3, 4],
                "assistant_masks": [0, 0, 0, 1],
            }

    args = SimpleNamespace(sequence_length=3)
    assert script.build_messages_example(Tokenizer(), MESSAGES, args) is None


def test_main_rotates_shards_records_actual_rows_and_removes_stale_output(
    monkeypatch, tmp_path
):
    script = load_script()

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 9
        chat_template = None

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens
            return [1, 2] if text.endswith("\n") else [1, 2, 3, 4]

    class Progress:
        def update(self, count):
            assert count == 1

        def close(self):
            pass

    args = SimpleNamespace(
        sequence_length=6,
        shard_size=2,
        output_dir=str(tmp_path),
        tokenizer="local-tokenizer",
        trust_remote_code=False,
        mode="prompt-response",
        dataset="local-dataset",
        subset=None,
        split="train",
        num_sequences=None,
        buffer_docs=0,
        seed=7,
        prompt_field="prompt",
        response_field="response",
        messages_field="messages",
        system_prompt=None,
    )
    samples = [
        {"prompt": f"question {index}", "response": f"answer {index}"}
        for index in range(5)
    ]
    stale = tmp_path / "train-99999.parquet"
    stale.write_bytes(b"stale")
    monkeypatch.setattr(script, "parse_args", lambda: args)
    monkeypatch.setattr(
        script.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: Tokenizer(),
    )
    monkeypatch.setattr(script, "iter_samples", lambda args: samples)
    monkeypatch.setattr(script, "tqdm", lambda **kwargs: Progress())

    script.main()

    shards = sorted(tmp_path.glob("train-*.parquet"))
    assert [path.name for path in shards] == [
        "train-00000.parquet",
        "train-00001.parquet",
        "train-00002.parquet",
    ]
    actual_rows = {
        path.name: script.pq.read_table(path).num_rows
        for path in shards
    }
    assert actual_rows == {
        "train-00000.parquet": 2,
        "train-00001.parquet": 2,
        "train-00002.parquet": 1,
    }
    metadata = json.loads((tmp_path / "subset_metadata.json").read_text())
    assert metadata["num_sequences"] == 5
    assert metadata["file_rows"] == actual_rows
    assert metadata["files"] == list(actual_rows)
    assert not stale.exists()


def test_main_rejects_zero_output_without_writing_artifacts(monkeypatch, tmp_path):
    script = load_script()

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 9
        chat_template = None

    class Progress:
        def update(self, count):
            raise AssertionError(f"unexpected progress update: {count}")

        def close(self):
            pass

    args = SimpleNamespace(
        sequence_length=6,
        shard_size=2,
        output_dir=str(tmp_path),
        tokenizer="local-tokenizer",
        trust_remote_code=False,
        mode="prompt-response",
        dataset="local-dataset",
        subset=None,
        split="train",
        num_sequences=None,
        buffer_docs=0,
        seed=7,
        prompt_field="prompt",
        response_field="response",
        messages_field="messages",
        system_prompt=None,
    )
    monkeypatch.setattr(script, "parse_args", lambda: args)
    monkeypatch.setattr(
        script.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: Tokenizer(),
    )
    monkeypatch.setattr(script, "iter_samples", lambda args: [{"prompt": "   "}])
    monkeypatch.setattr(script, "tqdm", lambda **kwargs: Progress())

    with pytest.raises(RuntimeError, match="No SFT examples were produced") as error:
        script.main()

    assert "missing prompt/response text: 1" in str(error.value)
    assert not (tmp_path / "subset_metadata.json").exists()
    assert list(tmp_path.glob("*.parquet")) == []
