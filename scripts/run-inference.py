#!/usr/bin/env python3
"""Run inference with a trained Psyche checkpoint using HF Transformers."""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description='Run inference with trained Psyche model')
    parser.add_argument('checkpoint', default='checkpoints/ds-v3-dense-160m-ufw-step1999',
                        help='Path to checkpoint directory')
    parser.add_argument('--prompt', default='The future of AI is',
                        help='Input text prompt')
    parser.add_argument('--max-new-tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.7)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model from {args.checkpoint} on {device}...")

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    inputs = tokenizer(args.prompt, return_tensors='pt').to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=True,
    )
    print()
    print(tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == '__main__':
    main()
