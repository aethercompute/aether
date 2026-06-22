#!/usr/bin/env python3
"""Watch a Psyche client log and regenerate a loss plot every N steps."""
import argparse
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt

ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')
CLIENT_LOSS_RE = re.compile(
    r'integration_test_log_marker\s*=\s*loss\s.*?step\s*=\s*(\d+)\s.*?loss\s*=\s*([\d.eE+\-]+)'
)


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub('', text)


def read_losses(log_file: str) -> tuple[list[int], list[float], int]:
    steps, losses = [], []
    max_step = 0
    with open(log_file) as f:
        for line in f:
            clean = strip_ansi(line)
            m = CLIENT_LOSS_RE.search(clean)
            if m:
                step = int(m.group(1))
                loss = float(m.group(2))
                steps.append(step)
                losses.append(loss)
                if step > max_step:
                    max_step = step
    return steps, losses, max_step


def save_plot(steps, losses, output: str):
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, linewidth=0.8)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(f'Training Loss ({len(steps)} points, step {steps[-1]}, loss {losses[-1]:.4f})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"[{time.strftime('%H:%M:%S')}] Plot saved: {output} (step {steps[-1]}, loss {losses[-1]:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description='Watch a Psyche client log and regenerate loss plot every N steps.'
    )
    parser.add_argument('log_file', help='Path to client log file')
    parser.add_argument('-o', '--output', default='loss-curve.png',
                        help='Output image path')
    parser.add_argument('-n', '--interval', type=int, default=10,
                        help='Regenerate plot every N steps (default: 10)')
    parser.add_argument('--poll', type=float, default=5.0,
                        help='Poll interval in seconds (default: 5)')
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Waiting for {args.log_file} to appear...")
        while not log_path.exists():
            time.sleep(1)

    last_plotted_step = -1

    print(f"Watching {args.log_file} — plotting every {args.interval} steps to {args.output}")
    try:
        while True:
            steps, losses, max_step = read_losses(args.log_file)
            if not steps:
                time.sleep(args.poll)
                continue

            # Find the latest completed interval boundary
            latest_interval = (max_step // args.interval) * args.interval
            if latest_interval > last_plotted_step and latest_interval >= args.interval:
                # Filter up to this boundary
                plot_steps = [s for s, l in zip(steps, losses) if s <= latest_interval]
                plot_losses = [l for s, l in zip(steps, losses) if s <= latest_interval]
                if plot_steps:
                    save_plot(plot_steps, plot_losses, args.output)
                    last_plotted_step = latest_interval

            # Also plot the final step if it exists and we haven't
            if max_step > last_plotted_step and max_step >= args.interval:
                plot_steps = [s for s, l in zip(steps, losses) if s <= max_step]
                plot_losses = [l for s, l in zip(steps, losses) if s <= max_step]
                if plot_steps:
                    save_plot(plot_steps, plot_losses, args.output)
                    last_plotted_step = max_step

            time.sleep(args.poll)
    except KeyboardInterrupt:
        # Final plot on exit
        steps, losses, _ = read_losses(args.log_file)
        if steps:
            save_plot(steps, losses, args.output)
        print("Stopped.")


if __name__ == '__main__':
    main()
