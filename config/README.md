# Config

Sample configuration for centralized Aether training runs.

## Files

- `training-run.toml`: control dashboard/server config. Points at the default state file, server ports, dataset output path, and data-preparation parameters.
- `experiment-run.toml`: experiment config containing multiple coordinator state files.
- `aether0-500m/model-config.json`: DeepSeek V3-style model config used by scripts and sample runs.
- `aether0-500m/state_distro.toml`: coordinator state for a Distro optimizer run.
- `aether0-500m/state_muon.toml`: coordinator state for a Muon optimizer run.
- `aether0-500m/data.toml`: data-provider TCP server config for local preprocessed data.
- `llama3.2-1b-pirate-sft/`: sample Distro supervised fine-tuning config for `meta-llama/Llama-3.2-1B-Instruct` on `KafeisM/pirate-speak-dataset`.

## Validate Config

```sh
cargo run -p aether-centralized-server -- \
  validate-config \
  --state config/aether0-500m/state_distro.toml \
  --data-config config/aether0-500m/data.toml
```

## Run Config

```sh
bash scripts/with-libtorch-env.sh cargo run -p aether-centralized-server -- \
  run \
  --state config/aether0-500m/state_distro.toml \
  --data-config config/aether0-500m/data.toml \
  --server-port 39405 \
  --web-port 8081
```

## Experiment Config

```sh
bash scripts/with-libtorch-env.sh cargo run -p aether-centralized-server -- \
  run \
  --experiment config/experiment-run.toml \
  --data-config config/aether0-500m/data.toml \
  --server-port 39405 \
  --web-port 8081
```

## Data Path Assumptions

The sample data config points at `../../data/corpus-512-bin` from the config
file location. Prepare that directory with `scripts/prepare-ultra-fineweb-local.py`
or update `aether0-500m/data.toml` to point at an existing pre-tokenized dataset.

## SFT Example

Prepare masked-label SFT data:

```sh
python3 scripts/prepare-sft-local.py \
  --dataset KafeisM/pirate-speak-dataset \
  --split train \
  --prompt-field english \
  --response-field pirate \
  --tokenizer meta-llama/Llama-3.2-1B-Instruct \
  --output-dir data/pirate-speak-llama3.2-1b-sft-1024 \
  --sequence-length 1024 \
  --mode chat
```

Or use the admin panel with the SFT run config:

```sh
TRAINING_RUN_CONFIG=config/llama3.2-1b-pirate-sft/training-run.toml \
  python3 scripts/training-control-dashboard.py
```

Then click `Prepare dataset`. The panel will run `scripts/prepare-sft-local.py`
with `english` as the user turn and `pirate` as the assistant turn.

Validate the sample SFT config:

```sh
cargo run -p aether-centralized-server -- \
  validate-config \
  --state config/llama3.2-1b-pirate-sft/state.toml \
  --data-config config/llama3.2-1b-pirate-sft/data.toml
```

Run it with Python model support enabled:

```sh
bash scripts/with-libtorch-env.sh cargo run -p aether-centralized-server --features python -- \
  run \
  --state config/llama3.2-1b-pirate-sft/state.toml \
  --data-config config/llama3.2-1b-pirate-sft/data.toml \
  --server-port 39405 \
  --web-port 8081
```
