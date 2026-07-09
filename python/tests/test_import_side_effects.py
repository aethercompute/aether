import builtins
import importlib.util
import sys
import types
from pathlib import Path

import pytest


def test_ext_import_does_not_swallow_keyboard_interrupt(monkeypatch):
    real_import = builtins.__import__

    def interrupting_import(name, *args, **kwargs):
        if name == "_aether_ext":
            raise KeyboardInterrupt
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", interrupting_import)

    ext_path = Path(__file__).parents[1] / "python" / "aether" / "_ext.py"
    spec = importlib.util.spec_from_file_location("aether._ext_under_test", ext_path)
    module = importlib.util.module_from_spec(spec)

    with pytest.raises(KeyboardInterrupt):
        spec.loader.exec_module(module)


def test_vllm_subprocess_import_does_not_set_multiprocessing_start_method(monkeypatch):
    import multiprocessing

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("set_start_method should only run from __main__")

    monkeypatch.setattr(multiprocessing, "set_start_method", fail_if_called)

    script_path = (
        Path(__file__).parents[1]
        / "python"
        / "aether"
        / "vllm"
        / "run_inference_subprocess.py"
    )
    spec = importlib.util.spec_from_file_location(
        "aether.vllm.run_inference_subprocess_under_test", script_path
    )
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)


def test_push_new_model_script_import_does_not_parse_args(monkeypatch):
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.LlamaConfig = type("LlamaConfig", (), {})
    fake_transformers.LlamaForCausalLM = type("LlamaForCausalLM", (), {})
    fake_transformers.AutoTokenizer = type("AutoTokenizer", (), {})
    fake_transformers.DeepseekV3Config = type("DeepseekV3Config", (), {})
    fake_transformers.DeepseekV3ForCausalLM = type("DeepseekV3ForCausalLM", (), {})

    fake_modeling_llama = types.ModuleType("transformers.models.llama.modeling_llama")
    fake_modeling_llama.LlamaDecoderLayer = type("LlamaDecoderLayer", (), {})

    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_torch.float64 = "float64"
    fake_torch.no_grad = lambda: None
    fake_torch.nn = types.SimpleNamespace(init=types.SimpleNamespace(ones_=lambda *_a: None))

    fake_huggingface_hub = types.ModuleType("huggingface_hub")
    fake_huggingface_hub.HfApi = type("HfApi", (), {})

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "transformers.models", types.ModuleType("transformers.models"))
    monkeypatch.setitem(
        sys.modules, "transformers.models.llama", types.ModuleType("transformers.models.llama")
    )
    monkeypatch.setitem(
        sys.modules, "transformers.models.llama.modeling_llama", fake_modeling_llama
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_huggingface_hub)

    import argparse

    real_parse_args = argparse.ArgumentParser.parse_args

    def fail_if_called(self):
        raise AssertionError("parse_args should only run from __main__")

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", fail_if_called)

    script_path = Path(__file__).parents[2] / "scripts" / "push-new-model-hf.py"
    spec = importlib.util.spec_from_file_location("push_new_model_hf_under_test", script_path)
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)
    assert callable(module.parse_args)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", real_parse_args)
    assert module.parse_args(["--dtype", "float16"]).dtype == fake_torch.float16
    assert module.parse_args(["--dtype", "float32"]).dtype == fake_torch.float32
