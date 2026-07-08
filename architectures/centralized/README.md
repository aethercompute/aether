# Centralized Architecture

The centralized architecture runs one coordinator server and many training
clients. The server owns coordinator state and admissions. Clients connect to the
server for coordinator updates, then use shared crates for training, P2P model
sharing, data access, metrics, and event logs.

## Package Map

- `shared`: serde protocol between centralized server and clients.
- `server`: `aether-centralized-server`, the coordinator/data/dashboard server.
- `client`: `aether-centralized-client`, the training participant binary.
- `volunteer`: `aether-volunteer`, a lightweight onboarding launcher for volunteers.
- `local-testnet`: `aether-centralized-local-testnet`, a tmux-based local multi-client runner.
- `testing`: integration-test harness for in-process server/client scenarios.

## Message Flow

1. A client connects to the server TCP address.
2. The client sends `Join { run_id }`.
3. The server accepts the client when the run ID and optional allowlist match.
4. The server sends `Coordinator` state snapshots.
5. The client initializes local training state and sends `ReadyForEpoch`.
6. During a run, clients send witness, health-check, and checkpoint messages.
7. The server advances coordinator state and broadcasts updated snapshots.

## Server

Validate a coordinator config:

```sh
cargo run -p aether-centralized-server -- \
  validate-config \
  --state config/aether0-500m/state_distro.toml \
  --data-config config/aether0-500m/data.toml
```

Run a server:

```sh
bash scripts/with-libtorch-env.sh cargo run -p aether-centralized-server -- \
  run \
  --state config/aether0-500m/state_distro.toml \
  --data-config config/aether0-500m/data.toml \
  --server-port 39405 \
  --web-port 8081
```

Run an experiment with multiple state files:

```sh
bash scripts/with-libtorch-env.sh cargo run -p aether-centralized-server -- \
  run \
  --experiment config/experiment-run.toml \
  --data-config config/aether0-500m/data.toml \
  --server-port 39405 \
  --web-port 8081
```

Important server flags:

- `--server-port`: coordinator TCP port.
- `--web-port`: dashboard port.
- `--tui`: enable or disable the TUI.
- `--data-config`: data server config when the model points at server-hosted data.
- `--admission-allowlist`: file containing one 32-byte hex public key per line.
- `--save-state-dir`: directory for epoch-end TOML state snapshots.
- `--events-dir`: directory for coordinator event logs.
- `--withdraw-on-disconnect`: remove disconnected clients from active coordinator state.

## Client

Show the public identity for a secret key file:

```sh
cargo run -p aether-centralized-client -- \
  show-identity \
  --identity-secret-key-path .aethercompute/identity.key
```

Run a training client:

```sh
bash scripts/with-libtorch-env.sh cargo run -p aether-centralized-client -- \
  train \
  --server-addr 127.0.0.1:39405 \
  --run-id ds-v3-dense-100m-ufw \
  --identity-secret-key-path .aethercompute/identity.key
```

Common client options come from `shared/client/src/cli.rs` and include identity,
P2P, training parallelism, logging, metrics, events, checkpoint uploads, evals,
and W&B configuration.

## Volunteer Launcher

Build and run the lightweight volunteer UI:

```sh
cargo run -p aether-centralized-volunteer --bin aether-volunteer
```

The launcher detects a usable device, ensures an identity key, builds the heavy
centralized client if needed, and then execs `aether-centralized-client train`
with curated defaults.

## Local Testnet

Start a tmux-based local testnet:

```sh
just local-testnet --num-clients 2 --config-path config/aether0-500m
```

The testnet command builds server/client packages, validates the config, opens a
tmux session named `aether`, starts a server pane, and starts one pane per client.

## Integration Tests

```sh
just integration-test
just integration-test client_connection
```

The `testing` package provides handles for in-process server/client scenarios,
including client admission, state transitions, shutdown/replacement, checkpoint
progression, and witness health scoring.

## Ports And Environment

- `39405`: sample coordinator TCP port.
- `39406`: sample training data server port.
- `8081`: sample web dashboard port.
- `METRICS_LOCAL_PORT`: optional per-client local metrics TCP endpoint.
- `RAW_IDENTITY_SECRET_KEY`: raw 64-hex secret key alternative to key files.
- `HF_TOKEN`: required for private Hugging Face downloads/uploads.
- `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_RUN`, `WANDB_ENTITY`, `WANDB_GROUP`: W&B integration.
- `OLTP_AUTH_HEADER`, `OLTP_METRICS_URL`, `OLTP_TRACING_URL`, `OLTP_LOGS_URL`: telemetry sinks.
