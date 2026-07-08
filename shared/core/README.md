# aether-core

Core types and utilities shared across the workspace.

## What Lives Here

- `BatchId` and token-size helpers used by training data and coordinator logic.
- `NodeIdentity` for node identifiers.
- `FixedVec` and `FixedString` for bounded serialization-friendly containers.
- `Bloom`, `MerkleTree`, `MerkleRoot`, and `OwnedProof` for compact membership/proof data.
- `Shuffle` and interval helpers for deterministic data ordering.
- `RunningAverage` and learning-rate schedule definitions.

## Features

- `rand`: enables RNG-oriented helpers.

## Commands

```sh
cargo test -p aether-core
cargo test -p aether-core --features rand
```
