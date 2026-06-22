#!/usr/bin/env python3
"""Extract loss values from Psyche client logs and plot the training curve."""
import argparse
import re
import matplotlib.pyplot as plt

ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')
CLIENT_LOSS_RE = re.compile(
    r'integration_test_log_marker\s*=\s*loss\s.*?step\s*=\s*(\d+)\s.*?loss\s*=\s*([\d.eE+\-]+)'
)


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub('', text)


def extract_losses(log_file: str, client_id: str | None = None):
    steps, losses = [], []
    with open(log_file) as f:
        for line in f:
            clean = strip_ansi(line)
            m = CLIENT_LOSS_RE.search(clean)
            if m:
                step = int(m.group(1))
                loss = float(m.group(2))
                if client_id is None or f'client_id={client_id}' in clean:
                    steps.append(step)
                    losses.append(loss)
    return steps, losses


def main():
    parser = argparse.ArgumentParser(
        description='Plot training loss curve from Psyche client log.'
    )
    parser.add_argument('log_file', help='Path to client log file')
    parser.add_argument('-c', '--client', help='Filter by client ID prefix')
    parser.add_argument('-o', '--output', default='loss-curve.png',
                        help='Output image path')
    args = parser.parse_args()

    steps, losses = extract_losses(args.log_file, args.client)
    if not steps:
        print("No client_loss entries found in log.")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, linewidth=0.8)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(f'Training Loss ({len(steps)} steps, final {losses[-1]:.4f})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved {args.output} ({len(steps)} points, loss {losses[-1]:.4f} final)")


if __name__ == '__main__':
    main()
