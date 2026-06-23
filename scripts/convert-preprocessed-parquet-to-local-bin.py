#!/usr/bin/env python3
import argparse
import json
import os
import struct
import sys
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Psyche preprocessed parquet shards to LocalDataProvider binary shards."
    )
    parser.add_argument("--input-dir", default="data/ultra-fineweb-deepseek-512")
    parser.add_argument("--output-dir", default="data/ultra-fineweb-deepseek-512-bin")
    parser.add_argument("--column", default="inputs")
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--shard-size", type=int, default=8192)
    parser.add_argument(
        "--token-bytes",
        type=int,
        choices=(2, 4),
        default=4,
        help="DeepSeek-V3 vocab requires 4-byte tokens.",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
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
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(input_dir.glob("*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"No parquet files found in {input_dir}")

    total_sequences = 0
    shard_index = 0
    shard_sequences = 0
    shard_path, shard_file = open_shard(output_dir, shard_index)
    progress = tqdm(desc="converting sequences")

    try:
        for parquet_file in parquet_files:
            parquet = pq.ParquetFile(parquet_file)
            for batch in parquet.iter_batches(
                batch_size=args.batch_size, columns=[args.column]
            ):
                column = batch.column(0)
                for row_index in range(len(column)):
                    sequence = column[row_index].as_py()
                    if len(sequence) != args.sequence_length:
                        raise ValueError(
                            f"{parquet_file} row {row_index} has {len(sequence)} tokens; "
                            f"expected {args.sequence_length}"
                        )

                    write_sequence(shard_file, sequence, args.token_bytes)
                    total_sequences += 1
                    shard_sequences += 1
                    progress.update(1)

                    if shard_sequences >= args.shard_size:
                        shard_file.close()
                        print(f"wrote {shard_sequences} sequences to {shard_path}")
                        shard_index += 1
                        shard_sequences = 0
                        shard_path, shard_file = open_shard(output_dir, shard_index)
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
        "source_dir": str(input_dir),
        "source_format": "psyche-preprocessed-parquet",
        "column": args.column,
        "sequence_length": args.sequence_length,
        "num_sequences": total_sequences,
        "token_bytes": args.token_bytes,
        "format": "psyche-local-bin",
    }

    source_metadata = input_dir / "subset_metadata.json"
    if source_metadata.exists():
        metadata["source_metadata"] = json.loads(source_metadata.read_text())

    (output_dir / "subset_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"finished {total_sequences} sequences in {output_dir}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
