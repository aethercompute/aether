# aether-modeling

Torch/tch model runtime for inference, training, optimization, and model loading.

## Responsibilities

- Loads model configs, tokenizers, and checkpoints from local or pretrained sources.
- Provides causal language model abstractions and implementations.
- Runs sampling, training, optimizer steps, and gradient compression.
- Supports data parallel and tensor parallel execution paths where enabled.
- Can delegate selected model work to Python-backed implementations.

## Important Types

- `CausalLM`, `CausalLanguageModel`, `LanguageModelConfig`: model interfaces.
- `auto_model_for_causal_lm_from_pretrained`, `auto_tokenizer`, `AutoConfig`: auto-loading helpers.
- `PretrainedSource`: source descriptor for repo files or state dicts.
- `Trainer`, `LocalTrainer`, `Batch`: training runtime.
- `Optimizer`, `MuonOptimizer`, `Distro`, `DistroResult`: optimizer/update paths.
- `LogitsProcessor`, `Sampling`: text-generation helpers.
- `DataParallel`, `Devices`: parallel execution helpers.

## Examples

```sh
cargo run -p aether-modeling --example inference -- --model <HF_MODEL> "hello"
cargo run -p aether-modeling --example train -- --model <HF_MODEL> --data-path <DIR>
```

## Commands

```sh
bash scripts/with-libtorch-env.sh cargo test -p aether-modeling
just training-oracle
```

Optional features:

```sh
bash scripts/with-libtorch-env.sh cargo test -p aether-modeling --features parallelism
bash scripts/with-libtorch-env.sh cargo test -p aether-modeling --features python
```
