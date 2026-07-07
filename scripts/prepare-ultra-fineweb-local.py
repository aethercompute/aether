#!/usr/bin/env python3
"""Stream one or more HuggingFace datasets, tokenize them, and write Aether
``LocalDataProvider`` binary shards.

A "source" is a single HuggingFace dataset load with its own ``split``,
``subset`` (HF config name), ``text_field`` and ``weight``. Multiple sources
are mixed at the document level via a largest-deficit (a.k.a. ratio)
scheduler, so the emitted token stream interleaves documents from every
source in proportion to its weight while keeping a bounded memory footprint.

Backward compatibility
-----------------------
If no ``--source`` / ``--sources-json`` is supplied, the legacy
``--dataset`` / ``--split`` / ``--subset`` / ``--text-field`` flags are used
as a single source of weight 1.0, reproducing the original behaviour exactly.
"""
import argparse
import json
import os
import random
import struct
import sys
from pathlib import Path
from typing import Iterator

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


# --------------------------------------------------------------------------- #
# Source parsing
# --------------------------------------------------------------------------- #

# Keys we accept in a ``--source`` spec (long form, aliases lowercased).
_SOURCE_KEYS = {"dataset", "name", "split", "subset", "text_field", "weight"}


def parse_source(spec: str) -> dict:
    """Parse a ``key=value,key=value`` source spec into a normalized dict.

    Recognized keys: ``dataset`` (required), ``split``, ``subset``,
    ``text_field`` (default ``"text"``), ``weight`` (float, default ``1.0``).
    Both ``text_field`` and ``text-field`` are accepted; same for any other
    key with a hyphen (normalized to underscore).
    """
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"empty source spec: {spec!r}")

    raw: dict[str, str] = {}
    for part in parts:
        if "=" not in part:
            raise ValueError(
                f"invalid source fragment {part!r} (expected key=value); full spec: {spec!r}"
            )
        key, value = part.split("=", 1)
        key = key.strip().lower().replace("-", "_")
        value = value.strip()
        if key not in _SOURCE_KEYS:
            raise ValueError(
                f"unknown source key {key!r} in spec {spec!r}; "
                f"valid keys: {sorted(_SOURCE_KEYS)}"
            )
        raw[key] = value

    if "dataset" not in raw and "name" not in raw:
        raise ValueError(f"source spec is missing `dataset`: {spec!r}")
    if "dataset" in raw and "name" in raw:
        raise ValueError(f"source spec has both `dataset` and `name`: {spec!r}")

    dataset = raw.get("dataset") or raw.pop("name", None)
    weight = float(raw.get("weight", "1.0"))
    if weight <= 0 or not (weight == weight):  # NaN guard
        raise ValueError(f"source {dataset!r} has non-positive/NaN weight {weight!r}")

    return {
        "dataset": dataset,
        "split": raw.get("split"),
        "subset": raw.get("subset") or None,
        "text_field": raw.get("text_field", "text"),
        "weight": weight,
    }


def load_sources_json(arg: str) -> list[dict]:
    """Resolve ``--sources-json`` into a list of source dicts.

    ``arg`` may be a path to a JSON file or an inline JSON array.
    """
    path = Path(arg)
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        text = arg
    data = json.loads(text)
    if isinstance(data, dict):
        # Allow {"sources": [...]} or a single source object.
        if "sources" in data:
            data = data["sources"]
        else:
            data = [data]
    if not isinstance(data, list):
        raise ValueError("--sources-json must decode to a list of source objects")
    normalized = []
    for entry in data:
        if not isinstance(entry, dict):
            raise ValueError(f"--sources-json entry is not an object: {entry!r}")
        normalized.append(parse_source(",".join(f"{k}={v}" for k, v in entry.items())))
    return normalized


def build_sources(args: argparse.Namespace) -> list[dict]:
    """Resolve the effective source list from CLI args.

    Precedence: ``--sources-json`` > ``--source`` (repeatable) > legacy
    single-dataset flags.
    """
    if args.sources_json:
        sources = load_sources_json(args.sources_json)
    elif args.source:
        sources = [parse_source(s) for s in args.source]
    else:
        # Backward-compatible single-source fallback.
        sources = [
            {
                "dataset": args.dataset,
                "split": args.split,
                "subset": args.subset,
                "text_field": args.text_field,
                "weight": 1.0,
            }
        ]

    if not sources:
        raise ValueError("no dataset sources configured")

    # Normalize `None` text_field to the default, and fill missing splits.
    for src in sources:
        if not src.get("text_field"):
            src["text_field"] = "text"
    return sources


# --------------------------------------------------------------------------- #
# Tokenization & mixing
# --------------------------------------------------------------------------- #


def iter_source_tokens(
    source: dict, tokenizer, eos_token_id: int, seed: int, buffer_docs: int
) -> Iterator[list[int]]:
    """Yield ``[tokens...] + [eos]`` for each non-empty document in a source."""
    kwargs: dict = {
        "path": source["dataset"],
        "split": source.get("split") or "train",
        "streaming": True,
    }
    if source.get("subset"):
        kwargs["name"] = source["subset"]

    dataset = load_dataset(**kwargs)
    if buffer_docs > 0:
        dataset = dataset.shuffle(seed=seed, buffer_size=buffer_docs)

    text_field = source["text_field"]
    for sample in dataset:
        text = sample.get(text_field)
        if not isinstance(text, str) or not text.strip():
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False)
        tokens.append(eos_token_id)
        yield tokens


def pick_source(contributed: list[float], weights: list[float], exhausted: set[int]) -> int:
    """Largest-deficit scheduler: pick the index with the smallest
    ``contributed[i] / weights[i]`` ratio (i.e. the most under-served source).

    Ties break towards the lower index for determinism.
    """
    best_i = -1
    best_ratio = float("inf")
    for i in range(len(weights)):
        if i in exhausted:
            continue
        ratio = contributed[i] / weights[i]
        if ratio < best_ratio:
            best_ratio = ratio
            best_i = i
    return best_i


# --------------------------------------------------------------------------- #
# Shard I/O
# --------------------------------------------------------------------------- #


def open_shard(output_dir: Path, shard_index: int):
    path = output_dir / f"train-{shard_index:05d}.bin"
    return path, path.open("wb")


def write_sequence(file, sequence: list[int], token_bytes: int) -> None:
    if token_bytes == 2:
        max_token = max(sequence, default=0)
        if max_token > 0xFFFF:
            raise ValueError(
                f"token id {max_token} does not fit in 2 bytes; rerun with --token-bytes 4"
            )
        file.write(struct.pack(f"<{len(sequence)}H", *sequence))
    else:
        file.write(struct.pack(f"<{len(sequence)}I", *sequence))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream one or more HuggingFace datasets, tokenize them, and write "
            "Aether LocalDataProvider binary shards. Supports weighted mixing "
            "across multiple datasets (each with its own split/subset/text column)."
        )
    )

    mixing = parser.add_argument_group("mixing (multi-dataset)")
    mixing.add_argument(
        "--source",
        action="append",
        default=None,
        metavar="KEY=VALUE,...",
        help=(
            "A dataset source spec, repeatable. Recognized keys: "
            "dataset (required), split, subset, text_field, weight. "
            "Example: --source 'dataset=openbmb/Ultra-FineWeb,split=en,"
            "text_field=content,weight=0.6'"
        ),
    )
    mixing.add_argument(
        "--sources-json",
        default=None,
        metavar="PATH|JSON",
        help=(
            "Path to a JSON file (or inline JSON array) of source objects. "
            "Takes precedence over --source."
        ),
    )

    legacy = parser.add_argument_group("legacy single-dataset (used only if no --source)")
    legacy.add_argument("--dataset", default="openbmb/Ultra-FineWeb")
    legacy.add_argument("--split", default="en", help="HF split name (e.g. train, en, test)")
    legacy.add_argument("--subset", default=None, help="Optional HF dataset config name")
    legacy.add_argument(
        "--text-field",
        dest="text_field",
        default="content",
        help="Column holding the document text",
    )

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
        help=(
            "Per-source streaming shuffle buffer size. 0 keeps source order; "
            ">0 applies datasets.shuffle(buffer_size=N) to each source."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to tokenizer loading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = build_sources(args)
    total_weight = sum(s["weight"] for s in sources)
    print(
        f"mixing {len(sources)} source(s) (total weight {total_weight}):",
        file=sys.stderr,
    )
    for i, src in enumerate(sources):
        print(
            f"  [{i}] dataset={src['dataset']} split={src.get('split') or 'train'} "
            f"subset={src.get('subset') or '-'} text_field={src['text_field']} "
            f"weight={src['weight']} ({src['weight'] / total_weight:.1%})",
            file=sys.stderr,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code
    )
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise RuntimeError(f"Tokenizer {args.tokenizer} does not define eos_token_id")

    # One independent token-document generator per source. Each gets a
    # deterministic per-source seed derived from the global seed so that
    # replaying with the same --seed reproduces the mix exactly.
    generators: list[Iterator[list[int]]] = [
        iter_source_tokens(src, tokenizer, eos_token_id, args.seed + i, args.buffer_docs)
        for i, src in enumerate(sources)
    ]
    weights = [src["weight"] for src in sources]
    contributed = [0.0] * len(sources)
    exhausted: set[int] = set()

    rng = random.Random(args.seed)
    token_buffer: list[int] = []
    total_sequences = 0
    shard_index = 0
    shard_sequences = 0
    shard_path, shard_file = open_shard(output_dir, shard_index)
    progress = tqdm(total=args.num_sequences, desc="building local binary sequences")

    try:
        while total_sequences < args.num_sequences and len(exhausted) < len(sources):
            i = pick_source(contributed, weights, exhausted)
            if i < 0:  # all exhausted
                break
            try:
                doc_tokens = next(generators[i])
            except StopIteration:
                exhausted.add(i)
                remaining = len(sources) - len(exhausted)
                print(
                    f"source [{i}] ({sources[i]['dataset']}) exhausted after "
                    f"{int(contributed[i])} tokens; {remaining} source(s) remain",
                    file=sys.stderr,
                )
                continue

            contributed[i] += len(doc_tokens)
            token_buffer.extend(doc_tokens)

            while (
                len(token_buffer) >= args.sequence_length
                and total_sequences < args.num_sequences
            ):
                start = 0
                if args.buffer_docs > 0 and len(token_buffer) > args.sequence_length:
                    max_start = min(
                        len(token_buffer) - args.sequence_length, args.sequence_length
                    )
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

    if total_sequences < args.num_sequences:
        print(
            f"warning: produced {total_sequences}/{args.num_sequences} sequences "
            f"before all sources were exhausted",
            file=sys.stderr,
        )

    # Per-source final stats + provenance metadata.
    source_stats = []
    for src, tokens in zip(sources, contributed):
        source_stats.append(
            {
                "dataset": src["dataset"],
                "split": src.get("split") or "train",
                "subset": src.get("subset"),
                "text_field": src["text_field"],
                "weight": src["weight"],
                "share": (src["weight"] / total_weight) if total_weight else 0.0,
                "tokens_contributed": int(tokens),
            }
        )

    metadata = {
        # Mixed provenance: record the full source list.
        "sources": source_stats,
        "mixed": len(sources) > 1,
        # Legacy top-level fields (first source) for older readers.
        "dataset": sources[0]["dataset"],
        "subset": sources[0].get("subset"),
        "split": sources[0].get("split") or "train",
        "text_field": sources[0]["text_field"],
        # Shared training params.
        "tokenizer": args.tokenizer,
        "sequence_length": args.sequence_length,
        "num_sequences": total_sequences,
        "token_bytes": args.token_bytes,
        "format": "aether-local-bin",
        "seed": args.seed,
    }
    (output_dir / "subset_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"finished {total_sequences} sequences in {output_dir}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
