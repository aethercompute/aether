#!/usr/bin/env python3
"""Merge an Aether LoRA adapter into a standalone Hugging Face checkpoint."""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, help="Base model repo or directory")
    parser.add_argument("--adapter", required=True, help="Aether adapter repo or directory")
    parser.add_argument("--output", required=True, type=Path, help="Merged output directory")
    parser.add_argument("--base-revision", help="Immutable base model revision")
    parser.add_argument("--device", default="cpu", help="Merge device, such as cpu or cuda:0")
    parser.add_argument("--max-shard-size", default="5GB")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        revision=args.base_revision,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    ).to(args.device)
    model = PeftModel.from_pretrained(model, args.adapter)
    merged = model.merge_and_unload(safe_merge=True)

    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(
        args.output,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        revision=args.base_revision,
    )
    tokenizer.save_pretrained(args.output)

    if args.device.startswith("cuda"):
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
