#!/usr/bin/env bash
set -e

# DisTrO/Psyche local distributed training test
# Runs a server + N clients without tmux

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
export LD_LIBRARY_PATH="/home/alkin/.local/lib/python3.14/site-packages/torch/lib:$LD_LIBRARY_PATH"
export RUST_LOG="warn,psyche=debug"

CONFIG_PATH="${1:-config/test}"
NUM_CLIENTS="${2:-2}"
SERVER_PORT="${3:-20000}"
RUN_ID="${4:-$(python3 -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["run_id"])' "$CONFIG_PATH/state.toml")}"
LOGDIR="logs/$RUN_ID"

echo "=== Psyche DisTrO Local Test ==="
echo "Config: $CONFIG_PATH"
echo "Run ID: $RUN_ID"
echo "Clients: $NUM_CLIENTS"
echo "Server port: $SERVER_PORT"
echo "Log dir: $LOGDIR"
echo ""

mkdir -p "$LOGDIR"

# Start server in background
echo "Starting server..."
cargo run -p psyche-centralized-server run \
    --state "$CONFIG_PATH/state.toml" \
    --server-port "$SERVER_PORT" \
    --tui false &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to start
sleep 3

# Start clients
CLIENT_PIDS=()
for i in $(seq 1 "$NUM_CLIENTS"); do
    RAW_KEY=$(printf '%064x' "$i")
    echo "Starting client $i (key: $RAW_KEY)..."
    RUST_LOG="$RUST_LOG" \
    RAW_IDENTITY_SECRET_KEY="$RAW_KEY" \
    cargo run -p psyche-centralized-client train \
        --run-id "$RUN_ID" \
        --server-addr "localhost:$SERVER_PORT" \
        --logs console \
        --device auto \
        --checkpoint-dir checkpoints \
        --iroh-relay disabled \
        --iroh-discovery local \
        --write-log "$LOGDIR/client-$i.log" &
    CLIENT_PIDS+=($!)
    sleep 2
done

# Start loss plot watcher (background, regenerates every 10 steps)
echo "Starting loss plot watcher (updates every 10 steps)..."
scripts/watch-loss.py "$LOGDIR/client-1.log" -o "$LOGDIR/loss-curve.png" -n 10 &
WATCHER_PID=$!
echo "Watcher PID: $WATCHER_PID"
echo ""

echo "=== All processes started ==="
echo "Server PID: $SERVER_PID"
for i in "${!CLIENT_PIDS[@]}"; do
    echo "Client $((i+1)) PID: ${CLIENT_PIDS[$i]}"
done
echo "Watcher PID: $WATCHER_PID"
echo ""
echo "Live plot: $LOGDIR/loss-curve.png (refreshes every ~10 steps)"
echo "Waiting for completion (Ctrl+C to stop)..."
echo ""

# Trap Ctrl+C to clean up
cleanup() {
    echo ""
    echo "Cleaning up..."
    kill $WATCHER_PID 2>/dev/null || true
    kill $SERVER_PID 2>/dev/null || true
    for pid in "${CLIENT_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait $WATCHER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    for pid in "${CLIENT_PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    # Final plot
    scripts/plot-loss.py "$LOGDIR/client-1.log" -o "$LOGDIR/loss-curve.png" 2>/dev/null || true
    echo "Done."
    exit 0
}
trap cleanup INT TERM

# Wait for all processes
wait

# Final plot on normal exit
scripts/plot-loss.py "$LOGDIR/client-1.log" -o "$LOGDIR/loss-curve.png" 2>/dev/null || true
echo ""
echo "=== Run complete ==="
echo "Logs: $LOGDIR/"
echo "Live plot: $LOGDIR/loss-curve.png"
echo "Checkpoints: checkpoints/"
echo ""
echo "To re-plot: scripts/plot-loss.py $LOGDIR/client-1.log"
