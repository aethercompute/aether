# Python Package

`python/` contains the Python-facing pieces of Aether: a PyO3 extension,
Python model backends, a distributed sidecar protocol, vLLM bridge helpers, and
tests that guard import-time behavior.

## Layout

- `src/lib.rs`: Rust package entry for the Python extension crate.
- `extension-impl/`: PyO3 implementation crate and embedded-Python initializer.
- `python/aether/`: importable Python package.
- `python/aether/models/`: CausalLM interface plus HF Transformers and Torchtitan backends.
- `python/aether/sidecar/`: TCPStore/process-group sidecar protocol and main loop.
- `python/aether/vllm/`: optional vLLM engine wrapper and Rust bridge registry.
- `tests/`: Python tests using stubs for heavy optional dependencies.
- `stub/torch/`: placeholder package used by `uv` because Nix supplies real Torch.

## Public Python API

The top-level `aether` package lazily exposes:

- Rust extension objects such as `Trainer`, `DistroResult`, and `start_process_watcher`.
- Model-source dataclasses and CausalLM factory helpers.
- Heavy dependencies only when the relevant module is imported.

## Model Backends

- `HfAuto`: Hugging Face Transformers backend.
- `Torchtitan`: Torchtitan-backed model implementation.

Both backends implement the shared causal LM interface and are selected by
`make_causal_lm()`.

## Sidecar

The sidecar is a Python worker used for distributed model operations. It
initializes a Torch process group/TCPStore, receives operation messages, and
handles training, optimization, extraction, BF16 truncation, forward passes, and
shutdown.

## vLLM Bridge

The vLLM modules are optional. The bridge keeps a thread-safe registry of Python
vLLM engines for Rust callers and exposes create, run, stats, list, and shutdown
operations.

## Commands

Run Python tests:

```sh
cd python
uv run --frozen --extra tests pytest
```

Run Rust tests that link the Python/Torch extension:

```sh
bash scripts/with-libtorch-env.sh cargo test -p aether-python-extension
bash scripts/with-libtorch-env.sh cargo test -p aether-python-extension-impl
```

## Runtime Notes

- Python is pinned to `==3.12.*` in `pyproject.toml`.
- `transformers==4.57.3` and `flash-linear-attention==0.4.0` are normal project dependencies.
- Torch, Torchtitan, FlashAttention, and Liger are intentionally overridden/stubbed for `uv`; Nix or the host environment provides them.
- `extension-impl` sets a default `TRITON_HOME=/tmp/aether-triton` before importing the Python package.
