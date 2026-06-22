#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


SCHEMA = pa.schema(
    [
        pa.field("inputs", pa.large_list(pa.int32())),
        pa.field("labels", pa.large_list(pa.int32())),
        pa.field("position_ids", pa.large_list(pa.int32())),
    ]
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stream a bounded Ultra-FineWeb subset and write Psyche preprocessed parquet."
    )
    parser.add_argument("--dataset", default="openbmb/Ultra-FineWeb")
    parser.add_argument("--split", default="en", help="Ultra-FineWeb exposes `en` and `zh` splits")
    parser.add_argument("--subset", default=None, help="Optional HF dataset config name")
    parser.add_argument("--text-field", default="content")
    parser.add_argument("--tokenizer", default="deepseek-ai/DeepSeek-V3")
    parser.add_argument("--output-dir", default="data/ultra-fineweb-deepseek-512")
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--num-sequences", type=int, default=8192)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--buffer-docs",
        type=int,
        default=0,
        help="Shuffle with this streaming buffer size. 0 keeps source order.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to tokenizer loading.",
    )
    return parser.parse_args()


def write_shard(output_dir: Path, shard_index: int, rows: list[dict]):
    table = pa.Table.from_pydict(
        {
            "inputs": [row["inputs"] for row in rows],
            "labels": [row["labels"] for row in rows],
            "position_ids": [row["position_ids"] for row in rows],
        },
        schema=SCHEMA,
    )
    path = output_dir / f"train-{shard_index:05d}.parquet"
    pq.write_table(table, path)
    print(f"wrote {len(rows)} sequences to {path}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code
    )
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise RuntimeError(f"Tokenizer {args.tokenizer} does not define eos_token_id")

    dataset_kwargs = {
        "path": args.dataset,
        "split": args.split,
        "streaming": True,
    }
    if args.subset:
        dataset_kwargs["name"] = args.subset

    dataset = load_dataset(**dataset_kwargs)
    if args.buffer_docs > 0:
        dataset = dataset.shuffle(seed=args.seed, buffer_size=args.buffer_docs)

    rng = random.Random(args.seed)
    token_buffer: list[int] = []
    rows: list[dict] = []
    total_sequences = 0
    shard_index = 0
    progress = tqdm(total=args.num_sequences, desc="building sequences")

    for sample in dataset:
        text = sample.get(args.text_field)
        if not isinstance(text, str) or not text.strip():
            continue

        token_buffer.extend(tokenizer.encode(text, add_special_tokens=False))
        token_buffer.append(eos_token_id)

        while len(token_buffer) >= args.sequence_length and total_sequences < args.num_sequences:
            start = 0
            if args.buffer_docs > 0 and len(token_buffer) > args.sequence_length:
                max_start = min(len(token_buffer) - args.sequence_length, args.sequence_length)
                start = rng.randint(0, max_start)
            sequence = token_buffer[start : start + args.sequence_length]
            del token_buffer[: start + args.sequence_length]

            rows.append(
                {
                    "inputs": sequence,
                    "labels": sequence,
                    "position_ids": list(range(args.sequence_length)),
                }
            )
            total_sequences += 1
            progress.update(1)

            if len(rows) >= args.shard_size:
                write_shard(output_dir, shard_index, rows)
                rows.clear()
                shard_index += 1

        if total_sequences >= args.num_sequences:
            break

    progress.close()
    if rows:
        write_shard(output_dir, shard_index, rows)

    if total_sequences == 0:
        raise RuntimeError("No training sequences were produced")

    metadata = {
        "dataset": args.dataset,
        "subset": args.subset,
        "split": args.split,
        "text_field": args.text_field,
        "tokenizer": args.tokenizer,
        "sequence_length": args.sequence_length,
        "num_sequences": total_sequences,
        "seed": args.seed,
    }
    (output_dir / "subset_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"finished {total_sequences} sequences in {output_dir}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
