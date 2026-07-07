default:
    just --list

# All tests are treated as one suite. Rust commands that link torch run through
# scripts/with-libtorch-env.sh, which uses AETHER_PYTHON (default: python3.12)
# to locate PyTorch and configure LIBTORCH_USE_PYTORCH, LD_LIBRARY_PATH, and
# PYO3_PYTHON the same way CI does.

# Build the centralized training server used by the root Dockerfile.
build-server:
    bash scripts/with-libtorch-env.sh cargo build --release -p aether-centralized-server

# Run the centralized control dashboard from the root Dockerfile locally.
dashboard:
    python3 scripts/training-control-dashboard.py

# Spin up a local centralized testnet.
local-testnet *args='':
    bash scripts/with-libtorch-env.sh cargo run -p aether-centralized-local-testnet -- start --events-dir ./events/local-testnet {{ args }}

# Run centralized integration tests.
integration-test test_name="":
    if [ "{{ test_name }}" = "" ]; then \
        bash scripts/with-libtorch-env.sh cargo test --release -p aether-centralized-testing --test integration_tests; \
    else \
        bash scripts/with-libtorch-env.sh cargo test --release -p aether-centralized-testing --test integration_tests -- --nocapture "{{ test_name }}"; \
    fi

# ── formatting ───────────────────────────────────────────────────────────────
fmt:
    cargo fmt --all

fmt-check:
    cargo fmt --all --check

# ── lint ─────────────────────────────────────────────────────────────────────
# Member crates inherit `warnings = deny` from [workspace.lints], so this fails
# on any warning in our own code.
clippy:
    bash scripts/with-libtorch-env.sh cargo clippy --workspace --all-targets -- -D warnings

# Compatibility alias; all linting is now workspace-wide.
clippy-torch: clippy

# ── tests ────────────────────────────────────────────────────────────────────
# Run the full test suite.
test:
    bash scripts/with-libtorch-env.sh cargo test --workspace
    cd python && uv run --frozen --extra tests pytest

# Compatibility aliases; all tests are now one suite.
test-fast: test

test-torch: test

# ── supply chain ─────────────────────────────────────────────────────────────
# Full gate: advisories + bans + licenses + sources. (cargo-deny: `cargo binstall -y cargo-deny`)
# Transitive advisories needing major-version bumps are acknowledged in deny.toml.
deny:
    cargo deny --workspace check

# ── meta ─────────────────────────────────────────────────────────────────────
# Everything CI runs locally, in one command.
ci-local:
    bash scripts/ci-local.sh

# Same checks as ci-local, but run sequentially for easier debugging.
ci-local-sequential: fmt-check clippy test deny
    @echo "ci suite green"
