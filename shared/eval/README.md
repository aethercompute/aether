# aether-eval

Evaluation harness for language-model benchmark tasks.

## Responsibilities

- Loads benchmark datasets and normalizes task names.
- Prepares prompts, choices, and expected answers.
- Runs prepared tasks against a causal language model.
- Reports task-level metrics.

## Task Coverage

Implemented task families include ARC, BoolQ, CEval, HellaSwag, MMLU, MMLU-CF,
MMLU-Pro, OpenBookQA, and PIQA.

## Important Types

- `Task`, `PreparedTask`, `TaskType`: task definitions and prepared runs.
- `EvalTaskOptions`: limits and options for evaluation.
- `ALL_TASK_NAMES`: supported task-name list.
- `tasktype_from_name`: task-name parser.
- `load_dataset`: dataset loading helper.

## Commands

```sh
cargo test -p aether-eval
cargo run -p aether-eval --example evaluate -- --model <HF_MODEL> --tasks mmlu,piqa --limit 10
```

Optional features:

```sh
cargo test -p aether-eval --features parallelism
cargo test -p aether-eval --features python
```
