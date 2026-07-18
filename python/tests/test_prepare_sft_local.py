import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


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
