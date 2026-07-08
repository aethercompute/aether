# aether-inference

Experimental inference protocol and optional vLLM bridge.

## Responsibilities

- Defines request/response protocol types for chat-style inference.
- Provides gossip/message wrappers for P2P inference routing.
- Optionally exposes a Python-backed vLLM engine bridge through PyO3.

## Important Types

- `ChatMessage`: role/content chat message.
- `InferenceRequest` and `InferenceResponse`: protocol payloads.
- `InferenceMessage` and `InferenceGossipMessage`: transport messages.
- `ModelSource`: model location/config reference.
- `InferenceNode`, `InferenceProtocol`, `INFERENCE_ALPN`: enabled by `vllm`.

## Features

- `vllm`: enables PyO3 and the Python vLLM bridge.
- `vllm-tests`: enables integration tests that exercise the vLLM bridge when available.

## Commands

```sh
cargo test -p aether-inference
cargo test -p aether-inference --features vllm-tests
```
