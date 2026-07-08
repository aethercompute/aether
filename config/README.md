# Config

Sample configuration for centralized Aether training runs. The default
dashboard config is now the Llama 3.2 1B pirate-speak SFT run.

## Files

- `training-run.toml`: control dashboard config. Points at the default state file, data-server config, server ports, dataset output path, and data-preparation parameters.
- `experiment-run.toml`: experiment config containing multiple coordinator state files.
- `aether0-500m/model-config.json`: DeepSeek V3-style model config used by scripts and sample runs.
- `aether0-500m/state_distro.toml`: coordinator state for a Distro optimizer run.
- `aether0-500m/state_muon.toml`: coordinator state for a Muon optimizer run.
- `aether0-500m/data.toml`: data-provider TCP server config for local binary pretraining data.
- `llama3.2-1b-pirate-sft/state.toml`: Distro SFT coordinator state for `meta-llama/Llama-3.2-1B-Instruct`.
- `llama3.2-1b-pirate-sft/data.toml`: data-provider TCP server config for the pirate-speak SFT Parquet data.

## Validate Config

```sh
cargo run -p aether-centralized-server -- \
  validate-config \
  --state config/llama3.2-1b-pirate-sft/state.toml \
  --data-config config/llama3.2-1b-pirate-sft/data.toml
```

## Run Config

```sh
bash scripts/with-libtorch-env.sh cargo run -p aether-centralized-server -- \
  run \
  --state config/llama3.2-1b-pirate-sft/state.toml \
  --data-config config/llama3.2-1b-pirate-sft/data.toml \
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

## Admin Panel

Run the dashboard with the default SFT config:

```sh
python3 scripts/training-control-dashboard.py
```

Then click `Prepare dataset`. The panel runs `scripts/prepare-sft-local.py`
with `english` as the user turn and `pirate` as the assistant turn.

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
