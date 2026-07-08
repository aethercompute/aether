# aether-client

Architecture-independent training client orchestration.

The centralized client binary wraps this crate, but the long-running training
state machine and most CLI/config types live here.

## Responsibilities

- Parses common training arguments and environment variables.
- Prepares identities, checkpoint directories, logging, metrics, events, and uploads.
- Coordinates warmup, training, cooldown, witness, health-check, and eval states.
- Defines P2P broadcast protocol types for training results and checkpoints.
- Provides client TUI state for architecture-specific binaries.

## Important Types

- `Client`: main client runtime.
- `TrainArgs`: common training CLI arguments.
- `Broadcast`, `BroadcastType`, `TrainingResult`, `Finished`: P2P protocol messages.
- `RunInitConfig`, `CheckpointConfig`, `RoundState`, `UploadInfo`: runtime state/config.
- `ClientTUI`: terminal UI wrapper.

## Commands

```sh
cargo test -p aether-client
```

Optional features:

```sh
cargo test -p aether-client --features parallelism
cargo test -p aether-client --features python
```
