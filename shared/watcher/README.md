# aether-watcher

Coordinator backend abstraction and watcher loop used by clients.

## Responsibilities

- Polls a coordinator backend for state changes.
- Tracks coordinator state hashes to avoid redundant work.
- Sends client-side readiness, witness, health-check, and checkpoint updates back to the backend.
- Exposes coordinator state for TUI rendering.

## Important Types

- `Backend`: async trait implemented by architecture-specific backends.
- `BackendWatcher`: polling state machine around a backend.
- `OpportunisticData`: witness payload data.
- `CoordinatorTui`, `CoordinatorTuiState`, `TuiRunState`: coordinator UI state.

## Commands

```sh
cargo test -p aether-watcher
```
