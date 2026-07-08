from types import SimpleNamespace

import pytest

from aether.vllm import rust_bridge


class FakeTokenizer:
    chat_template = None
    eos_token_id = 2
    eos_token = "</s>"


class FakeChatTemplateTokenizer(FakeTokenizer):
    chat_template = "fake-template"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return "|".join(f"{msg['role']}={msg['content']}" for msg in messages) + "|assistant="


class FakeEngine:
    model_name = "fake-model"
    tensor_parallel_size = 1

    def __init__(self, *, fail_on_add=False, outputs=None):
        self.fail_on_add = fail_on_add
        self.outputs = ["done"] if outputs is None else outputs
        self.prompts = []
        self.sampling_params = []
        self.shutdown_called = False

    def get_tokenizer(self):
        return FakeTokenizer()

    def add_request(self, prompt, sampling_params):
        if self.fail_on_add:
            raise RuntimeError("add failed")
        self.prompts.append(prompt)
        self.sampling_params.append(sampling_params)
        return "request-1"

    def has_unfinished_requests(self):
        return bool(self.outputs)

    def step(self):
        text = self.outputs.pop(0)
        return [
            SimpleNamespace(
                outputs=[SimpleNamespace(text=text, finish_reason="stop")]
            )
        ]

    def shutdown(self):
        self.shutdown_called = True


@pytest.fixture(autouse=True)
def clear_engines():
    with rust_bridge._engines_lock:
        rust_bridge._engines.clear()
    yield
    with rust_bridge._engines_lock:
        rust_bridge._engines.clear()


def test_run_inference_formats_messages_and_returns_output():
    engine = FakeEngine()
    with rust_bridge._engines_lock:
        rust_bridge._engines["engine-1"] = engine

    result = rust_bridge.run_inference(
        "engine-1",
        [
            {"role": "system", "content": "be precise"},
            {"role": "user", "content": "hello"},
        ],
        temperature=0.5,
        top_p=0.8,
        max_tokens=12,
    )

    assert result == {
        "status": "success",
        "request_id": "request-1",
        "generated_text": "done",
        "full_text": "System: be precise\n\nUser: hello\n\nAssistant: done",
    }
    assert engine.prompts == ["System: be precise\n\nUser: hello\n\nAssistant: "]
    assert engine.sampling_params == [
        {
            "temperature": 0.5,
            "top_p": 0.8,
            "max_tokens": 12,
            "stop_token_ids": [2],
            "stop": ["</s>"],
        }
    ]


def test_run_inference_uses_chat_template_when_available():
    class ChatTemplateEngine(FakeEngine):
        def get_tokenizer(self):
            return FakeChatTemplateTokenizer()

    engine = ChatTemplateEngine()
    with rust_bridge._engines_lock:
        rust_bridge._engines["engine-1"] = engine

    result = rust_bridge.run_inference(
        "engine-1",
        [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hello"},
        ],
    )

    assert result["status"] == "success"
    assert result["full_text"] == "system=be terse|user=hello|assistant=done"
    assert engine.prompts == ["system=be terse|user=hello|assistant="]


def test_run_inference_missing_engine_is_error():
    result = rust_bridge.run_inference("missing", [])

    assert result["status"] == "error"
    assert "not found" in result["error"]


def test_run_inference_reports_add_request_failure_without_unbound_request_id():
    with rust_bridge._engines_lock:
        rust_bridge._engines["engine-1"] = FakeEngine(fail_on_add=True)

    result = rust_bridge.run_inference("engine-1", [{"role": "user", "content": "hi"}])

    assert result["status"] == "error"
    assert result["request_id"] is None
    assert "add failed" in result["error"]


def test_shutdown_removes_engine_after_cleanup():
    engine = FakeEngine()
    with rust_bridge._engines_lock:
        rust_bridge._engines["engine-1"] = engine

    result = rust_bridge.shutdown_engine("engine-1")

    assert result == {"status": "success", "engine_id": "engine-1"}
    assert engine.shutdown_called
    assert rust_bridge.list_engines() == {"status": "success", "engine_ids": []}


def test_get_engine_stats_reports_registered_engine():
    with rust_bridge._engines_lock:
        rust_bridge._engines["engine-1"] = FakeEngine(outputs=[])

    result = rust_bridge.get_engine_stats("engine-1")

    assert result == {
        "status": "success",
        "engine_id": "engine-1",
        "model_name": "fake-model",
        "tensor_parallel_size": 1,
        "has_unfinished_requests": False,
    }


def test_list_engines_returns_snapshot_not_live_keys_view():
    with rust_bridge._engines_lock:
        rust_bridge._engines["engine-1"] = FakeEngine(outputs=[])

    result = rust_bridge.list_engines()

    assert result == {"status": "success", "engine_ids": ["engine-1"]}
    with rust_bridge._engines_lock:
        rust_bridge._engines["engine-2"] = FakeEngine(outputs=[])
    assert result == {"status": "success", "engine_ids": ["engine-1"]}
