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
