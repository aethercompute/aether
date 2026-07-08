# Shared Crates

`shared/` contains the reusable Rust crates that make up the Aether runtime.
These crates are used by one or more architecture packages under
`architectures/`.

## Crate Map

- [`core`](core/README.md): common identifiers, fixed-size containers, hashing, Merkle proofs, Bloom filters, shuffling, learning-rate schedules, and batch IDs.
- [`coordinator`](coordinator/README.md): coordinator state machine, model/run config, client membership, round progression, committee selection, and data assignment.
- [`client`](client/README.md): common client orchestration used by architecture-specific client binaries.
- [`data-provider`](data-provider/README.md): local, HTTP, weighted, preprocessed, and optional TCP training data providers.
- [`event-sourcing`](event-sourcing/README.md): structured event logging, file backends, tracing bridge, and timeline projections.
- [`eval`](eval/README.md): language-model evaluation harness and benchmark task implementations.
- [`inference`](inference/README.md): experimental inference protocol and optional vLLM bridge.
- [`metrics`](metrics/README.md): OpenTelemetry metrics and local TCP metrics serving.
- [`modeling`](modeling/README.md): Torch/tch model loading, training, optimization, sampling, and parallelism support.
- [`network`](network/README.md): Iroh-based networking, gossip, blobs, model sharing, and TCP utilities.
- [`tui`](tui/README.md): Ratatui helpers, render loops, widgets, and logging setup.
- [`watcher`](watcher/README.md): backend abstraction that feeds coordinator updates to clients and sends client updates back.
- [`test-support`](test-support/README.md): deterministic RNG and serialization helpers for tests.

## Common Commands

Run one shared crate:

```sh
cargo test -p aether-core
```

Run the shared crates as part of the full workspace test suite:

```sh
bash scripts/with-libtorch-env.sh cargo test --workspace
```

Crates that depend on Torch, Python, NCCL, or vLLM may need extra environment
setup. Use `scripts/with-libtorch-env.sh` when in doubt.
