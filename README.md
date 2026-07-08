```
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

aethercompute.org
```

# aether

Aether is a distributed training system for language models. The workspace is
mostly Rust, with Python bindings and helper modules for model implementations,
sidecars, and optional vLLM inference.

This README is the front door. Detailed docs live beside the code they describe:

- [`shared/`](shared/README.md): reusable Rust crates for coordination, training, networking, data, metrics, TUI, event logs, and tests.
- [`architectures/centralized/`](architectures/centralized/README.md): the centralized server/client/volunteer architecture and local testnet.
- [`python/`](python/README.md): PyO3 extension, Python model backends, sidecar protocol, and vLLM bridge.
- [`config/`](config/README.md): sample coordinator, data, model, and experiment configs.
- [`scripts/`](scripts/README.md): launchers, local CI, data preparation, dashboard, and Hugging Face utilities.

## Workspace

First-party Rust packages are workspace members from `shared/*`,
`architectures/centralized/*`, and `python/`. The vendored `ts-rs` crates under
`vendor/` are path dependencies, not workspace members.

Useful entry points:

- `just --list` shows common developer commands.
- `just build-server` builds the centralized server through the libtorch setup wrapper.
- `just local-testnet` starts a local centralized tmux testnet.
- `just ci-local` runs formatting, linting, tests, oracle tests, cargo-deny, and Python tests.

## Requirements

- Rust toolchain from [`rust-toolchain.toml`](rust-toolchain.toml).
- Python 3.12 for Python-side tools and tests.
- PyTorch/libtorch available to builds that touch `tch` or PyO3 Torch bindings.
- `just`, `uv`, and `cargo-deny` are useful for local development workflows.

For Rust commands that link against PyTorch, prefer the wrapper:

```sh
bash scripts/with-libtorch-env.sh cargo test --workspace
```

## Quick Commands

Run the full local gate:

```sh
just ci-local
```

`just ci-local` runs formatting checks, cargo-deny, training oracle tests,
clippy, Rust workspace tests, and Python pytest. Use the targeted commands below
when you only want one part of the suite:

```sh
just fmt-check
just clippy
just test
just training-oracle
just deny
```

Run the centralized server against the sample training config:

```sh
bash scripts/with-libtorch-env.sh cargo run -p aether-centralized-server -- \
  run \
  --state config/aether0-500m/state_distro.toml \
  --data-config config/aether0-500m/data.toml \
  --server-port 39405 \
  --web-port 8081
```

Run a client against that server:

```sh
bash scripts/with-libtorch-env.sh cargo run -p aether-centralized-client -- \
  train \
  --server-addr 127.0.0.1:39405 \
  --run-id ds-v3-dense-100m-ufw
```
