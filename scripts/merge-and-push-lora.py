#!/usr/bin/env python3
"""Merge a LoRA adapter and upload the standalone Hugging Face model."""

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, help="Adapter repo or checkpoint directory")
    parser.add_argument("--repo", required=True, help="Destination Hugging Face model repo")
    parser.add_argument(
        "--base-model", default="meta-llama/Llama-3.2-3B-Instruct", help="Base model repo"
    )
    parser.add_argument("--output", type=Path, required=True, help="Merged model directory")
    parser.add_argument("--device", default="cuda:0", help="Merge device")
    parser.add_argument("--private", action="store_true", help="Create a private destination repo")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        parser.error("set HF_TOKEN or HUGGING_FACE_HUB_TOKEN")

    from huggingface_hub import HfApi
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype="auto", low_cpu_mem_usage=True, token=token
    ).to(args.device)
    merged = PeftModel.from_pretrained(model, args.adapter).merge_and_unload(safe_merge=True)
    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output, safe_serialization=True, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(args.base_model, token=token).save_pretrained(args.output)

    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo,
        repo_type="model",
        folder_path=str(args.output),
        commit_message=f"Merge LoRA adapter from {args.adapter}",
    )


if __name__ == "__main__":
    main()
