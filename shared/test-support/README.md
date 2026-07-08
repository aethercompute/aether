# aether-test-support

Shared test helpers for first-party crates.

This crate is for development and tests only. It is not part of the runtime
architecture.

## Helpers

- `seeded_rng`: deterministic RNG for reproducible tests.
- `postcard_roundtrip`: encode/decode helper.
- `assert_postcard_roundtrip`: postcard round-trip assertion.
- `assert_serde_json_roundtrip`: serde JSON round-trip assertion.

## Commands

```sh
cargo test -p aether-test-support
```
