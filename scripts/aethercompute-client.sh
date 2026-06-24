#!/usr/bin/env bash
set -euo pipefail

DEFAULT_RUN_ID="ds-v3-dense-250m-ufw"
DEFAULT_SERVER_HOST="train.aethercompute.org"
DEFAULT_SERVER_PORT="39405"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STATE_DIR="$REPO_ROOT/.aethercompute"
IDENTITY_KEY="$STATE_DIR/identity.key"
LOG_DIR="$STATE_DIR/logs"
EVENTS_DIR="$STATE_DIR/events"

bold="\033[1m"
cyan="\033[36m"
green="\033[32m"
yellow="\033[33m"
red="\033[31m"
reset="\033[0m"

print_banner() {
  printf "${cyan}${bold}"
  cat << 'BANNER_EOF'

  AETHERCOMPUTEAETHERCOMPUTEAETHERCOMPUTEAETHERCOMPUTEAETHERCOMPUTEAETHERC
  E:cccclllllllloooooooooodddddddddddodddddddddddddddddddddddddddddddooooO
  TcclllllllllllloooooooooddddddoooooooodddddddddddddddddddddddddddddooooM
  HllllllllllllllloooooolclllllllcllccllllllllllooooddddooddddddddddoooooP
  Ellllllllllllllllllccc:::clllllllc;:looooolccllllllllcloooddddddddoooooU
  Rccclllllllllcc:::ccccc:;cllllccc;':llloolc:clolllc:;:lllloooooodooooooT
  Ccccccccccccc:;;;:cccc:;'';:::;,'..;::::;,,;:ccc:,',:clllllllloolloooooE
  Occccccccc::;;:;,'';;;;'....'.... .''''...',;,'...';ccllllllllcccclllllA
  Mccccc:::;;;;;;;,........          ..     ...  ..,;::::::::::::ccllllllE
  Pc::::;,,,,,,''....            ..               ........'''',:cclllccccT
  U::::;,'''.................   .,,...........','...     ...',;::c:::::ccH
  T::;;,,''....',;:;........     .. .........,colc:,...    .......'',,;;;E
  E::;,,,'',,;:cllol:.......         ...'''',codddolc:,'...      ......',R
  A:;,'',,;;:cclooool:,......... ......''',;lddxdddoolc:,'..........',;;:C
  E,,',,,,;;;::ccllloll:,...............',:odddddoolcc:;,'.'''..',;;::cccO
  T''',,,,,,,,,,;;::::cc:;,'....'....'',:clooolllc:;,''''''',;;;;;:clllllM
  H,;;:::cc::;;;;;,,'.',,,,;;;;;,,;;;;;::::;:::;;,'...'',;::cclllllllllllP
  E:::::cccccccc:;,''''...,::;,...,;,...,,'...'',,,',,;:cccllllloooooooooU
  R::::::::::ccc:;,;:;'';;:lc;,..',;:,'..;:;,,;,,:ccccccloolcccllllloooooT
  C::ccc:::::::::::c:;;::::llc:,';:cccc:,:llccllc:clllcccllllllllllllooooE
  Occcccccccc::::::::::cc::ccc:;;::ccccc::clllcclc::cccccccllllllllooooooA
  Mllllccccccccccccc::ccccccccc::::ccccc:::cccccccc::ccccccllllloooooooooE
  Plllllllcccccccccccccccccccccc:cc:::::::::::::::cccccccllllooooooooooodT
  UoollllllllllllllcccccclllcccccccccccccccccccccccllllllloooooooodddddddH
  TEAETHERCOMPUTEAETHERCOMPUTEAETHERCOMPUTEAETHERCOMPUTEAETHERCOMPUTEAETHE
BANNER_EOF
  printf "${reset}\nVolunteer client launcher\n\n"
}

prompt_default() {
  local prompt="$1"
  local default="$2"
  local value
  read -r -p "$prompt [$default]: " value
  printf "%s" "${value:-$default}"
}

detect_torch_lib_dir() {
  python3 - <<'PY' 2>/dev/null || true
import pathlib
try:
    import torch
except Exception:
    raise SystemExit(0)
print(pathlib.Path(torch.__file__).resolve().parent / "lib")
PY
}

ensure_identity_key() {
  mkdir -p "$STATE_DIR" "$LOG_DIR" "$EVENTS_DIR"

  if [[ -f "$IDENTITY_KEY" ]]; then
    return
  fi

  printf "${yellow}No identity key found. Creating one at ${IDENTITY_KEY}.${reset}\n"
  if command -v openssl >/dev/null 2>&1; then
    openssl rand 32 > "$IDENTITY_KEY"
  else
    dd if=/dev/urandom of="$IDENTITY_KEY" bs=32 count=1 status=none
  fi
  chmod 600 "$IDENTITY_KEY"
}

select_device() {
  local detected_gpus=0

  >&2 printf "${bold}Select device${reset}\n"
  >&2 printf "  1) auto - let Psyche choose\n"
  >&2 printf "  2) cpu\n"

  if command -v nvidia-smi >/dev/null 2>&1; then
    detected_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
  fi

  if [[ "$detected_gpus" =~ ^[0-9]+$ ]] && (( detected_gpus > 0 )); then
    >&2 printf "  3) cuda - all visible NVIDIA GPUs\n"
    local i
    for ((i = 0; i < detected_gpus; i++)); do
      local name
      name="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$i" 2>/dev/null || true)"
      >&2 printf "  %d) cuda:%d%s\n" "$((i + 4))" "$i" "${name:+ - $name}"
    done
    >&2 printf "  c) custom, e.g. cuda:0,1 or mps\n"
  else
    >&2 printf "  c) custom, e.g. cuda, cuda:0, mps\n"
  fi

  local choice
  read -r -p "Device [1]: " choice
  choice="${choice:-1}"

  case "$choice" in
    1) printf "auto" ;;
    2) printf "cpu" ;;
    3)
      if [[ "$detected_gpus" =~ ^[0-9]$ ]] && (( detected_gpus > 0 )); then
        printf "cuda"
      else
        printf "auto"
      fi
      ;;
    c|C)
      local custom
      read -r -p "Custom device: " custom
      printf "%s" "$custom"
      ;;
    *)
      if [[ "$detected_gpus" =~ ^[0-9]+$ ]] && (( detected_gpus > 0 )) && [[ "$choice" =~ ^[0-9]+$ ]]; then
        local gpu_index=$((choice - 4))
        if (( gpu_index >= 0 && gpu_index < detected_gpus )); then
          printf "cuda:$gpu_index"
          return
        fi
      fi
      printf "auto"
      ;;
  esac
}

setup_libtorch_env() {
  export LIBTORCH_USE_PYTORCH="${LIBTORCH_USE_PYTORCH:-1}"
  export LIBTORCH_BYPASS_VERSION_CHECK="${LIBTORCH_BYPASS_VERSION_CHECK:-1}"

  local torch_lib_dir
  torch_lib_dir="$(detect_torch_lib_dir)"
  if [[ -n "$torch_lib_dir" && -d "$torch_lib_dir" ]]; then
    export LD_LIBRARY_PATH="$torch_lib_dir:${LD_LIBRARY_PATH:-}"
  else
    printf "${yellow}Warning: could not find Python torch/lib automatically.${reset}\n"
    printf "Install PyTorch or set LD_LIBRARY_PATH before launching if torch-sys cannot build.\n\n"
  fi
}

main() {
  cd "$REPO_ROOT"
  print_banner

  if ! command -v cargo >/dev/null 2>&1; then
    printf "${red}cargo is required but was not found.${reset}\n"
    exit 1
  fi

  ensure_identity_key
  setup_libtorch_env

  local run_id server_host server_port server_addr device micro_batch_size
  run_id="$(prompt_default "Run ID" "$DEFAULT_RUN_ID")"
  server_host="$(prompt_default "Training server host" "$DEFAULT_SERVER_HOST")"
  server_port="$(prompt_default "Training server port" "$DEFAULT_SERVER_PORT")"
  device="$(select_device)"
  micro_batch_size="$(prompt_default "Micro batch size" "1")"
  server_addr="$server_host:$server_port"

  printf "\n${green}${bold}Ready to train${reset}\n"
  printf "  Run ID:        %s\n" "$run_id"
  printf "  Server:        %s\n" "$server_addr"
  printf "  Device:        %s\n" "$device"
  printf "  Micro batch:   %s\n" "$micro_batch_size"
  printf "  Identity key:  %s\n" "$IDENTITY_KEY"
  printf "\n"

  read -r -p "Start client now? [Y/n]: " confirm
  case "${confirm:-Y}" in
    y|Y|yes|YES) ;;
    *) printf "Cancelled.\n"; exit 0 ;;
  esac

  exec cargo run --release -p psyche-centralized-client -- train \
    --run-id "$run_id" \
    --server-addr "$server_addr" \
    --device "$device" \
    --micro-batch-size "$micro_batch_size" \
    --identity-secret-key-path "$IDENTITY_KEY" \
    --events-dir "$EVENTS_DIR" \
    --write-log "$LOG_DIR/client.log"
}

main "$@"
