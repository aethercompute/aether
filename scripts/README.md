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
- `aethercompute-seed.sh`: seed-mode wrapper that requires `HF_TOKEN` and `HUB_REPO`.

Supported `aethercompute-client.sh` modes include default launch, `seed`,
`update`, `doctor`, and `uninstall`.

## Training Operations

- `training-control-dashboard.py`: basic-auth web dashboard for editing config, preparing data, pushing initial models, validating config, and starting/stopping training.
- `prepare-ultra-fineweb-local.py`: streams Hugging Face datasets, tokenizes text, and writes binary shards plus metadata.
- `push-new-model-hf.py`: initializes a random model from config and pushes it to Hugging Face Hub or saves locally.
- `run-inference.py`: simple Hugging Face Transformers checkpoint inference helper.

Run the dashboard locally:

```sh
python3 scripts/training-control-dashboard.py
```

Prepare a local dataset shard set:

```sh
python3 scripts/prepare-ultra-fineweb-local.py \
  --source 'dataset=openbmb/Ultra-FineWeb,split=en,text_field=content,weight=1.0' \
  --output-dir data/corpus-512-bin \
  --tokenizer deepseek-ai/DeepSeek-V3 \
  --sequence-length 512
```

Initialize and push a model:

```sh
python3 scripts/push-new-model-hf.py \
  --config config/aether0-500m/model-config.json \
  --repo user/model \
  --tokenizer deepseek-ai/DeepSeek-V3
```

## Common Environment

- `AETHER_PYTHON`: Python executable used by `with-libtorch-env.sh`, default `python3.12`.
- `TORCH_VERSION`: Torch version installed by helper scripts when needed.
- `LIBTORCH_USE_PYTORCH`: tells `tch`/`torch-sys` to use the Python Torch installation.
- `CONTROL_PORT`, `SERVER_PORT`, `WEB_PORT`: dashboard and centralized server ports.
- `TRAINING_RUN_CONFIG`: dashboard config path.
- `HF_TOKEN`, `HUB_REPO`: Hugging Face credentials and destination repo.
