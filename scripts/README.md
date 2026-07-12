# Scripts

Operational and development scripts for Aether.

## Developer Scripts

- `with-libtorch-env.sh`: configures Python-installed Torch/libtorch paths for Cargo commands.
- `ci-local.sh`: runs the local CI suite in parallel.

Examples:

```sh
bash scripts/with-libtorch-env.sh cargo test --workspace
bash scripts/ci-local.sh
```

## Volunteer Launchers

- `aethercompute-client.sh`: one-command installer/launcher for volunteer clients.
- `aethercompute-seed.sh`: seed-mode wrapper that requires `HF_TOKEN`; `HUB_REPO` defaults to `aethercompute/llama3.2-1b-think`.

Supported `aethercompute-client.sh` modes include default launch, `seed`,
`update`, `doctor`, and `uninstall`.

## Training Operations

- `training-control-dashboard.py`: basic-auth web dashboard for editing config, preparing data, pushing initial models, validating config, and starting/stopping training.
- `prepare-ultra-fineweb-local.py`: streams Hugging Face datasets, tokenizes text, and writes binary shards plus metadata.
- `prepare-sft-local.py`: streams prompt/response datasets, applies chat templates, and writes masked-label SFT Parquet data.
- `push-new-model-hf.py`: initializes a random model from config and pushes it to Hugging Face Hub or saves locally.
- `merge-lora.py`: merges an adapter checkpoint into a standalone Hugging Face model.
- `run-inference.py`: simple Hugging Face Transformers checkpoint inference helper.

Run the dashboard locally:

```sh
python3 scripts/training-control-dashboard.py
```

The dashboard listens on `CONTROL_HOST=127.0.0.1` and `CONTROL_PORT=8080` by
default. Local-only runs generate and print a password if `CONTROL_PASSWORD` is
unset. Set a non-empty `CONTROL_PASSWORD` explicitly before setting
`CONTROL_HOST` to a non-loopback address. `CONTROL_USERNAME` defaults to
`admin`; authenticated forms include CSRF protection. Executable script fields
in the editable config are restricted to the dataset preparation and model
push scripts in this repository. Basic authentication does not encrypt traffic;
put the dashboard behind a TLS-terminating reverse proxy for any remote access.

The Docker image binds the dashboard to `0.0.0.0:8080`. If
`CONTROL_PASSWORD` is unset, its entrypoint generates a strong password,
persists it at `/app/.aether-control/control-password`, and prints the
credentials to `docker logs`. Mount `/app/.aether-control` persistently to keep
the generated password across container replacement. An explicit
`CONTROL_PASSWORD` takes precedence. Publish port `8080` or route it through a
TLS reverse proxy to access the panel outside the container.

Prepare a local dataset shard set:

```sh
python3 scripts/prepare-ultra-fineweb-local.py \
  --source 'dataset=openbmb/Ultra-FineWeb,split=en,text_field=content,weight=1.0' \
  --output-dir data/corpus-512-bin \
  --tokenizer deepseek-ai/DeepSeek-V3 \
  --sequence-length 512
```

Prepare Llama 3.2 1B Instruct SFT data from the pirate-speak dataset:

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

The SFT script stores prompt and padding labels as `-100`, so training loss is
computed only on assistant response tokens.

Initialize and push a model:

```sh
python3 scripts/push-new-model-hf.py \
  --config config/aether0-500m/model-config.json \
  --repo user/model \
  --tokenizer deepseek-ai/DeepSeek-V3 \
  --dtype bfloat16
```

Supported model initialization dtypes are `bfloat16`, `float16`, `float32`,
and `float64`.

## LoRA Training

LoRA is configured in the coordinator state and currently supports `HfAuto`, one
local device, and the AdamW or DisTrO optimizers. It targets all linear layers.
Existing states without `training_method` continue to use full training.

```toml
[model.LLM]
architecture = "HfAuto"
data_type = "Finetuning"
max_seq_len = 2048
cold_start_warmup_steps = 0

[model.LLM.checkpoint.Hub]
repo_id = "org/base-model"
revision = "immutable-commit-sha"

[model.LLM.training_method.Lora]
rank = 16
alpha = 32.0
dropout = 0.05
init_seed = 1337
adapter_checkpoint = "Fresh"
```

The base checkpoint remains immutable. Aether checkpoints and shares only the
adapter as `adapter_model.safetensors` plus `adapter_config.json`. To resume from
an uploaded adapter, replace `adapter_checkpoint = "Fresh"` with, for example:

```toml
[model.LLM.training_method.Lora.adapter_checkpoint.Hub]
repo_id = "org/aether-adapter"
revision = "immutable-commit-sha"
```

The seed client's upload destination is configured separately in the training
run config. Do not put the adapter repo in `[model.LLM.checkpoint.Hub]`; that
field must remain the immutable base model.

```toml
[checkpoint]
dir = "checkpoints/{run_id}"
hub_repo = "org/aether-adapter"
delete_old_steps = true
keep_steps = 3
epoch_interval = 1
```

Local DP/FSDP, tensor parallelism, native models, TorchTitan, and Muon are
rejected for LoRA until their dedicated implementations are available.

### Merging an Adapter

Merging is deliberately separate from live training because it requires a
second full-model materialization. Run it on a machine with enough memory:

```sh
python3 scripts/merge-lora.py \
  --base-model org/base-model \
  --base-revision immutable-commit-sha \
  --adapter ./checkpoints/run-step1000 \
  --output ./checkpoints/run-step1000-merged
```

## Common Environment

- `AETHER_PYTHON`: Python executable used by `with-libtorch-env.sh`, default `python3.12`.
- `TORCH_VERSION`: Torch version installed by helper scripts when needed.
- `LIBTORCH_USE_PYTORCH`: tells `tch`/`torch-sys` to use the Python Torch installation.
- `CONTROL_HOST`, `CONTROL_PORT`: dashboard bind host and port; defaults to `127.0.0.1:8080`.
- `CONTROL_USERNAME`, `CONTROL_PASSWORD`: dashboard Basic-auth credentials. A non-empty explicit password is required for non-loopback binds.
- `CONTROL_PASSWORD_FILE`: Docker entrypoint password file, default `/app/.aether-control/control-password`.
- `SERVER_PORT`, `WEB_PORT`: centralized server ports.
- `TRAINING_RUN_CONFIG`: dashboard config path.
- `HF_TOKEN`, `HUB_REPO`: Hugging Face credentials and destination repo. Seed mode defaults `HUB_REPO` to `aethercompute/llama3.2-1b-think`.
