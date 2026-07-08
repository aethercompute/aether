# aether-coordinator

In-memory coordinator state machine for distributed training runs.

## Responsibilities

- Tracks run state, rounds, clients, witnesses, commitments, and checkpoints.
- Validates whether clients can join, become ready, witness work, or advance a round.
- Selects committees and training data assignments.
- Holds model and training-run configuration used by servers and clients.

## Important Types

- `Coordinator`: full coordinator state and mutation API.
- `CoordinatorConfig`: run-level configuration.
- `RunState`, `Round`, `Client`, `Witness`: state-machine entities.
- `CommitteeSelection`, `CommitteeProof`, `Commitment`: committee and proof data.
- `Model`, `LLM`, `Checkpoint`, `LLMArchitecture`: model and checkpoint config.

## Commands

```sh
cargo test -p aether-coordinator
```

Most behavior is covered by module tests in `src/coordinator.rs`,
`src/committee_selection.rs`, `src/data_selection.rs`, and `src/commitment.rs`.
