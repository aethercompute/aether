#!/usr/bin/env python3
"""Prepare supervised fine-tuning data for Aether.

The script writes Parquet files consumable by ``PreprocessedDataProvider`` with
fixed-length ``inputs`` and ``labels`` columns. Labels use PyTorch's common
``ignore_index`` convention: prompt and padding positions are ``-100`` and
assistant response positions are real token IDs.
"""
import argparse
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


IGNORE_INDEX = -100


def token_ids(value: Any) -> list[int]:
    """Normalize tokenizer outputs across Transformers versions.

    Some tokenizers return a raw ``list[int]`` from ``apply_chat_template``;
    newer/fast tokenizer paths can return a BatchEncoding-like mapping with an
    ``input_ids`` field.
    """
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if not isinstance(value, list) or not all(isinstance(x, int) for x in value):
        raise TypeError(f"expected token id list, got {type(value).__name__}: {value!r}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare masked prompt/response SFT Parquet data for Aether."
    )
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset name")
    parser.add_argument("--split", default="train", help="Dataset split to stream")
    parser.add_argument("--subset", default=None, help="Optional HF dataset config name")
    parser.add_argument("--tokenizer", required=True, help="HF tokenizer name or path")
    parser.add_argument("--output-dir", required=True, help="Output dataset directory")
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--num-sequences", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--buffer-docs", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--mode",
        choices=("chat", "prompt-response"),
        default="chat",
        help="chat uses tokenizer.apply_chat_template; prompt-response concatenates raw fields.",
    )
    parser.add_argument(
        "--prompt-field",
        default="english",
        help="User/prompt text column. For pirate-speak this is `english`.",
    )
    parser.add_argument(
        "--response-field",
        default="pirate",
        help="Assistant/response text column. For pirate-speak this is `pirate`.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system message prepended in chat mode.",
    )
    return parser.parse_args()


def nonempty_text(sample: dict[str, Any], field: str) -> str | None:
    value = sample.get(field)
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def common_prefix_len(left: list[int], right: list[int]) -> int:
    i = 0
    for a, b in zip(left, right):
        if a != b:
            break
        i += 1
    return i


def chat_tokens(
    tokenizer,
    prompt: str,
    response: str,
    system_prompt: str | None,
) -> tuple[list[int], list[int]]:
    prompt_messages: list[dict[str, str]] = []
    if system_prompt:
        prompt_messages.append({"role": "system", "content": system_prompt})
    prompt_messages.append({"role": "user", "content": prompt})

    full_messages = [*prompt_messages, {"role": "assistant", "content": response}]
    prompt_ids = token_ids(
        tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    full_ids = token_ids(
        tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
        )
    )
    return prompt_ids, full_ids


def prompt_response_tokens(tokenizer, prompt: str, response: str) -> tuple[list[int], list[int]]:
    prompt_text = f"{prompt}\n"
    full_text = f"{prompt_text}{response}"
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
    full_ids = tokenizer.encode(full_text, add_special_tokens=True)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None and (not full_ids or full_ids[-1] != eos_token_id):
        full_ids.append(eos_token_id)
    return prompt_ids, full_ids


def build_example(
    tokenizer,
    prompt: str,
    response: str,
    args: argparse.Namespace,
) -> tuple[list[int], list[int]] | None:
    if args.mode == "chat":
        prompt_ids, input_ids = chat_tokens(
            tokenizer, prompt, response, args.system_prompt
        )
    else:
        prompt_ids, input_ids = prompt_response_tokens(tokenizer, prompt, response)

    prefix_len = common_prefix_len(prompt_ids, input_ids)
    labels = [IGNORE_INDEX] * prefix_len + input_ids[prefix_len:]

    input_ids = input_ids[: args.sequence_length]
    labels = labels[: args.sequence_length]

    if len(input_ids) < args.sequence_length:
        pad = args.sequence_length - len(input_ids)
        input_ids.extend([tokenizer.pad_token_id] * pad)
        labels.extend([IGNORE_INDEX] * pad)

    if all(label == IGNORE_INDEX for label in labels):
        return None
    return input_ids, labels


def iter_samples(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "path": args.dataset,
        "split": args.split,
        "streaming": True,
    }
    if args.subset:
        kwargs["name"] = args.subset
    dataset = load_dataset(**kwargs)
    if args.buffer_docs > 0:
        dataset = dataset.shuffle(seed=args.seed, buffer_size=args.buffer_docs)
    return dataset


def write_shard(output_dir: Path, split: str, shard_index: int, rows: list[dict[str, list[int]]]) -> Path:
    path = output_dir / f"{split}-{shard_index:05d}.parquet"
    schema = pa.schema(
        [
            pa.field("inputs", pa.list_(pa.int32())),
            pa.field("labels", pa.list_(pa.int32())),
        ]
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path)
    return path


def main() -> None:
    args = parse_args()
    if args.sequence_length <= 1:
        raise ValueError("--sequence-length must be greater than 1")
    if args.shard_size <= 0:
        raise ValueError("--shard-size must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError(f"Tokenizer {args.tokenizer} has no pad or eos token")
        tokenizer.pad_token = tokenizer.eos_token

    if args.mode == "chat" and tokenizer.chat_template is None:
        raise RuntimeError(
            f"Tokenizer {args.tokenizer} does not define a chat template; use --mode prompt-response"
        )

    target = args.num_sequences
    progress = tqdm(total=target, desc="building SFT parquet")
    rows: list[dict[str, list[int]]] = []
    shard_index = 0
    total = 0
    skipped = 0
    missing_text = 0
    no_supervised_tokens = 0
    written_files: list[str] = []

    for sample in iter_samples(args):
        prompt = nonempty_text(sample, args.prompt_field)
        response = nonempty_text(sample, args.response_field)
        if prompt is None or response is None:
            skipped += 1
            missing_text += 1
            continue

        example = build_example(tokenizer, prompt, response, args)
        if example is None:
            skipped += 1
            no_supervised_tokens += 1
            continue
        input_ids, labels = example
        rows.append({"inputs": input_ids, "labels": labels})
        total += 1
        progress.update(1)

        if len(rows) >= args.shard_size:
            path = write_shard(output_dir, args.split, shard_index, rows)
            written_files.append(str(path.relative_to(output_dir)))
            print(f"wrote {len(rows)} examples to {path}")
            rows = []
            shard_index += 1

        if target is not None and total >= target:
            break

    progress.close()

    if rows:
        path = write_shard(output_dir, args.split, shard_index, rows)
        written_files.append(str(path.relative_to(output_dir)))
        print(f"wrote {len(rows)} examples to {path}")

    if total == 0:
        raise RuntimeError(
            "No SFT examples were produced "
            f"(missing prompt/response text: {missing_text}, "
            f"no supervised tokens after tokenization/truncation: {no_supervised_tokens}). "
            f"Check --prompt-field={args.prompt_field!r}, "
            f"--response-field={args.response_field!r}, --mode={args.mode!r}, "
            "and --sequence-length."
        )

    metadata = {
        "format": "aether-preprocessed-sft-parquet",
        "dataset": args.dataset,
        "subset": args.subset,
        "split": args.split,
        "tokenizer": args.tokenizer,
        "sequence_length": args.sequence_length,
        "num_sequences": total,
        "skipped_sequences": skipped,
        "missing_text_sequences": missing_text,
        "no_supervised_token_sequences": no_supervised_tokens,
        "mode": args.mode,
        "prompt_field": args.prompt_field,
        "response_field": args.response_field,
        "system_prompt": args.system_prompt,
        "label_ignore_index": IGNORE_INDEX,
        "files": written_files,
    }
    (output_dir / "subset_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"finished {total} SFT examples in {output_dir} ({skipped} skipped)")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
