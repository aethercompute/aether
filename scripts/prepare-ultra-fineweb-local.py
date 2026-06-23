#!/usr/bin/env python3
import argparse
import json
import os
import random
import struct
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stream Ultra-FineWeb and write Psyche LocalDataProvider binary shards."
    )
    parser.add_argument("--dataset", default="openbmb/Ultra-FineWeb")
    parser.add_argument("--split", default="en", help="Ultra-FineWeb exposes `en` and `zh` splits")
    parser.add_argument("--subset", default=None, help="Optional HF dataset config name")
    parser.add_argument("--text-field", default="content")
    parser.add_argument("--tokenizer", default="deepseek-ai/DeepSeek-V3")
    parser.add_argument("--output-dir", default="data/ultra-fineweb-deepseek-512-bin")
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--num-sequences", type=int, default=8192)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--token-bytes",
        type=int,
        choices=(2, 4),
        default=4,
        help="DeepSeek-V3 vocab requires 4-byte tokens.",
    )
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


def open_shard(output_dir: Path, shard_index: int):
    path = output_dir / f"train-{shard_index:05d}.bin"
    return path, path.open("wb")


def write_sequence(file, sequence: list[int], token_bytes: int):
    if token_bytes == 2:
        max_token = max(sequence, default=0)
        if max_token > 0xFFFF:
            raise ValueError(
                f"token id {max_token} does not fit in 2 bytes; rerun with --token-bytes 4"
            )
        file.write(struct.pack(f"<{len(sequence)}H", *sequence))
    else:
        file.write(struct.pack(f"<{len(sequence)}I", *sequence))


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
    total_sequences = 0
    shard_index = 0
    shard_sequences = 0
    shard_path, shard_file = open_shard(output_dir, shard_index)
    progress = tqdm(total=args.num_sequences, desc="building local binary sequences")

    try:
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

                write_sequence(shard_file, sequence, args.token_bytes)
                total_sequences += 1
                shard_sequences += 1
                progress.update(1)

                if shard_sequences >= args.shard_size:
                    shard_file.close()
                    print(f"wrote {shard_sequences} sequences to {shard_path}")
                    shard_index += 1
                    shard_sequences = 0
                    if total_sequences < args.num_sequences:
                        shard_path, shard_file = open_shard(output_dir, shard_index)

            if total_sequences >= args.num_sequences:
                break
    finally:
        progress.close()
        if not shard_file.closed:
            shard_file.close()

    if total_sequences == 0:
        raise RuntimeError("No training sequences were produced")

    if shard_sequences > 0:
        print(f"wrote {shard_sequences} sequences to {shard_path}")
    elif shard_path.exists() and shard_path.stat().st_size == 0:
        shard_path.unlink()

    metadata = {
        "dataset": args.dataset,
        "subset": args.subset,
        "split": args.split,
        "text_field": args.text_field,
        "tokenizer": args.tokenizer,
        "sequence_length": args.sequence_length,
        "num_sequences": total_sequences,
        "token_bytes": args.token_bytes,
        "format": "psyche-local-bin",
        "seed": args.seed,
    }
    (output_dir / "subset_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"finished {total_sequences} sequences in {output_dir}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
