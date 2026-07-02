default:
    just --list

# ─────────────────────────────────────────────────────────────────────────────
# Crates that build & test on a CPU-only machine (no libtorch, no CUDA).
# Everything else (modeling/eval/network/client/centralized-*/python) pulls in
# `tch` unconditionally and needs the torch toolchain — see the *-torch recipes.
# ─────────────────────────────────────────────────────────────────────────────
torch-free-crates := "psyche-core psyche-coordinator psyche-event-sourcing psyche-metrics psyche-tui psyche-watcher psyche-data-provider psyche-centralized-shared psyche-centralized-volunteer psyche-centralized-local-testnet"

# Build the centralized training server used by the root Dockerfile.
build-server:
    cargo build --release -p psyche-centralized-server

# Run the centralized control dashboard from the root Dockerfile locally.
dashboard:
    python3 scripts/training-control-dashboard.py

# Spin up a local centralized testnet.
local-testnet *args='':
    cargo run -p psyche-centralized-local-testnet -- start --events-dir ./events/local-testnet {{ args }}

# Run centralized integration tests.
integration-test test_name="":
    if [ "{{ test_name }}" = "" ]; then \
        cargo test --release -p psyche-centralized-testing --test integration_tests; \
    else \
        cargo test --release -p psyche-centralized-testing --test integration_tests -- --nocapture "{{ test_name }}"; \
    fi

# ── formatting ───────────────────────────────────────────────────────────────
fmt:
    cargo fmt --all

fmt-check:
    cargo fmt --all --check

# ── lint ─────────────────────────────────────────────────────────────────────
# Lint the crates that don't need torch. Member crates inherit `warnings = deny`
# from [workspace.lints], so this fails on any warning in our own code.
clippy:
    cargo clippy {{ torch-free-crates }} --all-targets

# Lint the whole workspace. Requires the torch toolchain
# (LIBTORCH_USE_PYTORCH=1 + LD_LIBRARY_PATH pointing at torch's lib dir — see
# the root Dockerfile for the canonical setup).
clippy-torch:
    cargo clippy --workspace --all-targets

# ── tests ────────────────────────────────────────────────────────────────────
# Fast lane: torch-free crates. Runs in seconds and is the primary PR gate.
test-fast:
    cargo test {{ torch-free-crates }}

# Alias for test-fast.
test: test-fast

# Full lane: the whole workspace. Requires the torch toolchain (see clippy-torch).
test-torch:
    cargo test --workspace

# ── supply chain ─────────────────────────────────────────────────────────────
# Blocking gate: licenses, bans, sources. (cargo-deny: `cargo binstall -y cargo-deny`)
deny:
    cargo deny --workspace check bans licenses sources

# Informational: known advisories in the dependency graph. Non-blocking until the
# dependency graph is upgraded to clear the transitive hickory/rustls/pyo3/quick-xml
# advisories — then this becomes a blocking gate.
deny-advisories:
    cargo deny --workspace check advisories

# ── meta ─────────────────────────────────────────────────────────────────────
# Everything the fast CI lane runs, in one command.
ci-local: fmt-check clippy test-fast deny
    @echo "fast lane green"
