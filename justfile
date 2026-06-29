default:
    just --list

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
