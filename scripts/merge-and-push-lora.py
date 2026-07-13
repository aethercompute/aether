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
        "--base-model",
        help="Base model repo (defaults to the model recorded in the adapter config)",
    )
    parser.add_argument("--base-revision", help="Immutable base model revision")
    parser.add_argument("--output", type=Path, required=True, help="Merged model directory")
    parser.add_argument("--device", default="cuda:0", help="Merge device")
    parser.add_argument("--private", action="store_true", help="Create a private destination repo")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        parser.error("set HF_TOKEN or HUGGING_FACE_HUB_TOKEN")

    from huggingface_hub import HfApi
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_config = PeftConfig.from_pretrained(args.adapter, token=token)
    base_model = args.base_model or adapter_config.base_model_name_or_path
    if not base_model:
        parser.error("adapter config has no base model; pass --base-model")
    base_revision = args.base_revision or adapter_config.revision

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        revision=base_revision,
        dtype="auto",
        low_cpu_mem_usage=True,
        token=token,
    ).to(args.device)
    merged = PeftModel.from_pretrained(
        model, args.adapter, config=adapter_config, token=token
    ).merge_and_unload(safe_merge=True)
    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output, safe_serialization=True, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(
        base_model, revision=base_revision, token=token
    ).save_pretrained(args.output)

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
