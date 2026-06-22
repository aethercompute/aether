#!/usr/bin/env bash
set -e

# DisTrO distributed training test with real gradient compression
# No tmux needed - runs server and clients as background processes

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
export LD_LIBRARY_PATH="/home/alkin/.local/lib/python3.14/site-packages/torch/lib:$LD_LIBRARY_PATH"
export RUST_LOG="warn,psyche=info,psyche_network=warn,psyche_metrics=warn"

CONFIG_PATH="${1:-config/distro-test}"
NUM_CLIENTS="${2:-2}"
SERVER_PORT="${3:-20000}"
RUN_ID="${4:-distro-test}"

echo "╔═══════════════════════════════════════════════════╗"
echo "║      Psyche DisTrO Distributed Training Test      ║"
echo "╚═══════════════════════════════════════════════════╝"
echo ""
echo "Config:       $CONFIG_PATH"
echo "Clients:      $NUM_CLIENTS"
echo "Server port:  $SERVER_PORT"
echo "Run ID:       $RUN_ID"
echo ""

# Pre-build
echo "Pre-building binaries..."
cargo build -p psyche-centralized-server -p psyche-centralized-client --quiet 2>/dev/null
echo "Build done."
echo ""

# Validate config
echo "Validating config..."
cargo run -p psyche-centralized-server validate-config \
    --state "$CONFIG_PATH/state.toml" \
    2>&1 | head -5
echo ""

# Start server in background
echo "Starting server..."
cargo run -p psyche-centralized-server run \
    --state "$CONFIG_PATH/state.toml" \
    --server-port "$SERVER_PORT" \
    --tui false \
    > /tmp/psyche-server.log 2>&1 &
SERVER_PID=$!
echo "  Server PID: $SERVER_PID"

# Wait for server to be ready
for i in $(seq 1 30); do
    if timeout 1 bash -c "echo > /dev/tcp/localhost/$SERVER_PORT" 2>/dev/null; then
        echo "  Server is ready (attempt $i)"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "  ERROR: Server died! Check /tmp/psyche-server.log"
        exit 1
    fi
    sleep 1
done

echo ""

# Start clients
CLIENT_PIDS=()
for i in $(seq 1 "$NUM_CLIENTS"); do
    RAW_KEY=$(printf '%064x' "$i")
    CLIENT_LOG="/tmp/psyche-client-$i.log"
    echo "Starting client $i..."
    RUST_LOG="warn,psyche=info" \
    RAW_IDENTITY_SECRET_KEY="$RAW_KEY" \
    cargo run -p psyche-centralized-client train \
        --run-id "$RUN_ID" \
        --server-addr "localhost:$SERVER_PORT" \
        --logs console \
        --device auto \
        > "$CLIENT_LOG" 2>&1 &
    CLIENT_PIDS+=($!)
    echo "  Client $i PID: ${CLIENT_PIDS[$i]}"
    sleep 2
done

echo ""
echo "=========================================="
echo " All processes started!"
echo " Server PID:  $SERVER_PID"
for i in "${!CLIENT_PIDS[@]}"; do
    echo " Client $((i+1)) PID: ${CLIENT_PIDS[$i]}"
done
echo ""
echo " Logs:"
echo "   Server: tail -f /tmp/psyche-server.log"
for i in $(seq 1 "$NUM_CLIENTS"); do
    echo "   Client $i: tail -f /tmp/psyche-client-$i.log"
done
echo "=========================================="
echo ""

# Monitor loop
cleanup() {
    echo ""
    echo "Cleaning up..."
    kill "$SERVER_PID" 2>/dev/null || true
    for pid in "${CLIENT_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait "$SERVER_PID" 2>/dev/null || true
    for pid in "${CLIENT_PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    echo "Done."
    exit 0
}
trap cleanup INT TERM

# Wait for all processes
wait
