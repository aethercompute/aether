import builtins
import importlib.util
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
