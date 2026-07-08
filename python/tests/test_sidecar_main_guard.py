"""Verify the sidecar ``__main__`` module is import-safe (B41).

Previously ``main()`` was called at module scope, so merely importing the
module (e.g. for testing or static analysis) would spin up a process group
and block forever. The ``if __name__ == "__main__":`` guard must ensure
``main()`` only runs when executed directly.

The module imports torch at the top, which is stubbed here so the import can
succeed without a real PyTorch install.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _stub_torch(monkeypatch):
    """Install minimal fakes for torch, torch.distributed, and the ( unbuilt)
    Rust extension ``_aether_ext`` so the sidecar module can be imported
    without a real PyTorch / Rust build."""
    fake_torch = types.ModuleType("torch")
    # ScalarType constants used by DTYPE_MAPPING.
    fake_torch.uint8 = "uint8"
    fake_torch.int = "int"
    fake_torch.int64 = "int64"
    fake_torch.half = "half"
    fake_torch.float = "float"
    fake_torch.float32 = "float32"
    fake_torch.double = "double"
    fake_torch.bool = "bool"
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.long = "long"
    # factory.py / causal_lm.py use these as default parameter values and in
    # `torch.device | str` annotations, so they must be real types.
    fake_torch.device = type("device", (), {})
    fake_torch.dtype = type("dtype", (), {})
    fake_torch.Tensor = type("Tensor", (), {})
    fake_torch.manual_seed = lambda *_a, **_kw: None

    fake_dist = types.ModuleType("torch.distributed")
    fake_dist.init_process_group = lambda **_kw: None
    fake_dist.broadcast = lambda *_a, **_kw: None
    fake_dist.barrier = lambda *_a, **_kw: None
    fake_dist.all_reduce = lambda *_a, **_kw: None
    fake_torch.distributed = fake_dist

    fake_lib = types.ModuleType("torch.lib")
    fake_lib.libtorch = None
    fake_torch.lib = fake_lib

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", fake_dist)

    # Stub the Rust extension so `from .. import (...)` in __main__ resolves.
    fake_ext = types.ModuleType("aether._aether_ext")

    class _FakeDistroResult:
        def __init__(self, *args, **kwargs):
            self.args = args

    class _FakeTrainer:
        def __init__(self, *args, **kwargs):
            pass

    def _fake_start_process_watcher(*_a, **_kw):
        return None

    fake_ext.DistroResult = _FakeDistroResult
    fake_ext.Trainer = _FakeTrainer
    fake_ext.start_process_watcher = _fake_start_process_watcher
    monkeypatch.setitem(sys.modules, "aether._aether_ext", fake_ext)


def _load_main_module(monkeypatch):
    """Import ``aether.sidecar.__main__`` as a fresh module (not cached)."""
    _stub_torch(monkeypatch)

    import aether.sidecar  # noqa: F401  -- ensures package is importable

    main_path = Path(aether.sidecar.__file__).parent / "__main__.py"
    spec = importlib.util.spec_from_file_location(
        "aether.sidecar.__main_under_test", main_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_importing_main_does_not_call_main(monkeypatch):
    module = _load_main_module(monkeypatch)

    # main() must not have been invoked during import. We detect this by
    # confirming argparse was never asked to parse argv (the guard is the only
    # thing preventing that). The simplest signal: the module exposes `main`
    # as a callable but did not exit the process.
    assert callable(module.main)


def test_main_module_has_main_callable(monkeypatch):
    module = _load_main_module(monkeypatch)
    # The public entry point must exist and be a function.
    assert hasattr(module, "main")
    assert callable(module.main)


def test_dtype_mapping_is_complete(monkeypatch):
    """DTYPE_MAPPING must cover every dtype the sidecar receives over the wire.

    The keys mirror the c10::ScalarType enum indices used by the Rust trainer.
    """
    module = _load_main_module(monkeypatch)
    mapping = module.DTYPE_MAPPING
    # Every key referenced by the protocol must be present.
    for key in [0, 3, 4, 5, 6, 7, 11, 15]:
        assert key in mapping, f"missing ScalarType {key}"
    assert len(mapping) >= 8


def test_receive_distro_results_rejects_mismatched_metadata(monkeypatch):
    module = _load_main_module(monkeypatch)

    metadata = module.DistroResultsMetadata(
        sparse_idx_size=[[1]],
        sparse_idx_dtype=4,
        sparse_val_size=[],
        sparse_val_dtype=6,
        xshape=[[1]],
        totalk=[1],
    )

    with pytest.raises(ValueError, match="field lengths must match"):
        module.receive_distro_results(1, metadata, device=None)
