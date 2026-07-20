#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

log_dir="target/ci-local/logs"
mkdir -p "$log_dir"
rm -f "$log_dir"/*.log

declare -A pids
declare -A logs
declare -A statuses

run_job() {
  local name="$1"
  shift

  local log="$log_dir/${name}.log"
  logs["$name"]="$log"

  printf '[ci-local] starting %s\n' "$name"
  (
    set -euo pipefail
    export CARGO_TERM_COLOR=always
    export RUSTFLAGS="${RUSTFLAGS:--C debuginfo=0}"
    "$@"
  ) >"$log" 2>&1 &
  pids["$name"]="$!"
}

run_job fmt cargo fmt --all --check
run_job deny cargo deny --workspace check
run_job training-oracle env CARGO_TARGET_DIR=target/ci-local/training-oracle bash scripts/with-libtorch-env.sh cargo test -p aether-modeling --test training_oracle --test llama_oracle -- --nocapture
run_job clippy env CARGO_TARGET_DIR=target/ci-local/clippy bash scripts/with-libtorch-env.sh cargo clippy --workspace --all-targets -- -D warnings
run_job test bash scripts/with-libtorch-env.sh bash -c 'CARGO_TARGET_DIR=target/ci-local/test cargo test --workspace && cd python && uv run --frozen --extra tests pytest -m "not (gpu or distributed or vllm or slow)" --junitxml=test-results.xml && uv run --frozen python ../scripts/junit_summary.py test-results.xml --label "Python CPU required" --expected-tests 95 --forbid-skips'

failed=0
for name in fmt deny training-oracle clippy test; do
  if wait "${pids[$name]}"; then
    statuses["$name"]="passed"
    printf '[ci-local] passed %s\n' "$name"
  else
    statuses["$name"]="failed"
    failed=1
    printf '[ci-local] failed %s; log: %s\n' "$name" "${logs[$name]}"
  fi
done

if [[ "$failed" -ne 0 ]]; then
  printf '\n[ci-local] failures:\n'
  for name in fmt deny training-oracle clippy test; do
    if [[ "${statuses[$name]}" == "failed" ]]; then
      printf '\n===== %s (%s) =====\n' "$name" "${logs[$name]}"
      sed -n '1,240p' "${logs[$name]}"
      line_count="$(wc -l <"${logs[$name]}")"
      if [[ "$line_count" -gt 240 ]]; then
        printf '\n[ci-local] log truncated; full log: %s\n' "${logs[$name]}"
      fi
    fi
  done
  exit 1
fi

printf '[ci-local] suite green\n'
