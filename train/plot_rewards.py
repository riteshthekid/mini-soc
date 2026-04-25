"""
Mini SOC — Reward Curve Plotter
================================
Reads the training log (JSON lines) and produces a publication-quality
reward curve chart for the blog post and submission.

Usage:
    python -m train.plot_rewards                        # default paths
    python -m train.plot_rewards --log training.jsonl --output curve.png

    # From Python:
    from train.plot_rewards import plot_reward_curve
    plot_reward_curve("training_log.jsonl", "reward_curve.png")
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def plot_reward_curve(
    log_file: str = "./outputs/mini-soc-grpo/training_log.jsonl",
    output_path: str = "./outputs/reward_curve.png",
    random_baseline: float = 0.09,
    show_plot: bool = False,
) -> str:
    """
    Read a JSONL training log and plot the reward curve.

    Args:
        log_file: Path to training_log.jsonl (one JSON object per line).
        output_path: Where to save the PNG chart.
        random_baseline: Mean reward of a random agent (dashed line).
        show_plot: Whether to call plt.show() (for notebooks).

    Returns:
        Absolute path to the saved chart image.
    """
    # Lazy import so module is importable without matplotlib
    import matplotlib
    if not show_plot:
        matplotlib.use("Agg")  # Non-interactive backend for headless servers
    import matplotlib.pyplot as plt

    # -----------------------------------------------------------------------
    # Parse log file
    # -----------------------------------------------------------------------
    steps, losses, rewards = _parse_training_log(log_file)

    if not steps:
        print(f"Warning: No training entries found in {log_file}")
        # Create a placeholder chart
        steps = [0]
        rewards = [0.0]
        losses = [0.0]

    # -----------------------------------------------------------------------
    # Create figure with dual y-axis
    # -----------------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Style
    plt.style.use("seaborn-v0_8-darkgrid")
    fig.patch.set_facecolor("#0d1117")
    ax1.set_facecolor("#161b22")

    # Primary axis: Reward
    color_reward = "#58a6ff"
    ax1.set_xlabel("Training Steps", fontsize=13, color="#c9d1d9", fontweight="bold")
    ax1.set_ylabel("Mean Episode Reward", fontsize=13, color=color_reward, fontweight="bold")

    if rewards:
        # Smooth the curve with a rolling average
        smoothed = _rolling_average(rewards, window=max(len(rewards) // 20, 3))
        ax1.plot(
            steps[:len(smoothed)], smoothed,
            color=color_reward, linewidth=2.5, label="Mean Reward (smoothed)",
            alpha=0.95, zorder=3,
        )
        # Raw data as faint scatter
        ax1.scatter(
            steps, rewards,
            color=color_reward, alpha=0.15, s=8, zorder=2,
        )

    # Random baseline
    ax1.axhline(
        y=random_baseline, color="#f85149", linestyle="--",
        linewidth=1.8, label=f"Random Agent Baseline ({random_baseline})",
        alpha=0.8, zorder=1,
    )

    ax1.tick_params(axis="y", labelcolor=color_reward)
    ax1.tick_params(axis="x", colors="#8b949e")

    # Secondary axis: Loss
    if losses and any(l > 0 for l in losses):
        ax2 = ax1.twinx()
        color_loss = "#f0883e"
        ax2.set_ylabel("Training Loss", fontsize=13, color=color_loss, fontweight="bold")
        smoothed_loss = _rolling_average(losses, window=max(len(losses) // 20, 3))
        ax2.plot(
            steps[:len(smoothed_loss)], smoothed_loss,
            color=color_loss, linewidth=1.5, linestyle="-.",
            label="Loss", alpha=0.7,
        )
        ax2.tick_params(axis="y", labelcolor=color_loss)

    # Title and legend
    ax1.set_title(
        "Mini SOC — GRPO Training Reward Curve",
        fontsize=16, color="#f0f6fc", fontweight="bold", pad=20,
    )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    if losses and any(l > 0 for l in losses):
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines1 += lines2
        labels1 += labels2

    legend = ax1.legend(
        lines1, labels1,
        loc="upper left", fontsize=11,
        facecolor="#21262d", edgecolor="#30363d",
        labelcolor="#c9d1d9",
    )

    # Grid styling
    ax1.grid(True, alpha=0.15, color="#30363d")

    # Spine styling
    for spine in ax1.spines.values():
        spine.set_color("#30363d")

    plt.tight_layout()

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Reward curve saved to: {os.path.abspath(output_path)}")

    if show_plot:
        plt.show()

    plt.close(fig)
    return os.path.abspath(output_path)


def plot_comparison(
    random_scores: Dict[str, float],
    trained_scores: Dict[str, float],
    output_path: str = "./outputs/comparison_chart.png",
    show_plot: bool = False,
) -> str:
    """
    Create a before/after comparison bar chart.

    Args:
        random_scores: {"alert_triage": 0.12, "incident_investigation": 0.05, ...}
        trained_scores: {"alert_triage": 0.85, "incident_investigation": 0.72, ...}
        output_path: Where to save the PNG.
        show_plot: Show interactive window.

    Returns:
        Absolute path to the saved chart.
    """
    import matplotlib
    if not show_plot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    tasks = list(random_scores.keys())
    task_labels = [t.replace("_", " ").title() for t in tasks]

    x = np.arange(len(tasks))
    width = 0.35

    random_vals = [random_scores.get(t, 0.0) for t in tasks]
    trained_vals = [trained_scores.get(t, 0.0) for t in tasks]

    bars1 = ax.bar(x - width / 2, random_vals, width, label="Before (Random)",
                   color="#f85149", alpha=0.8, edgecolor="#da3633")
    bars2 = ax.bar(x + width / 2, trained_vals, width, label="After (GRPO)",
                   color="#58a6ff", alpha=0.8, edgecolor="#388bfd")

    # Delta labels
    for i, (r, t) in enumerate(zip(random_vals, trained_vals)):
        delta = t - r
        if delta > 0:
            ax.annotate(
                f"+{delta:.2f}",
                xy=(x[i] + width / 2, t + 0.02),
                ha="center", fontsize=11, fontweight="bold",
                color="#3fb950",  # green
            )

    ax.set_ylabel("Final Score", fontsize=13, color="#c9d1d9", fontweight="bold")
    ax.set_title(
        "Mini SOC — Before vs After GRPO Training",
        fontsize=16, color="#f0f6fc", fontweight="bold", pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, fontsize=12, color="#c9d1d9")
    ax.set_ylim(0, 1.1)
    ax.tick_params(axis="y", colors="#8b949e")

    legend = ax.legend(
        fontsize=11, facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9",
    )

    ax.grid(True, axis="y", alpha=0.15, color="#30363d")
    for spine in ax.spines.values():
        spine.set_color("#30363d")

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Comparison chart saved to: {os.path.abspath(output_path)}")

    if show_plot:
        plt.show()

    plt.close(fig)
    return os.path.abspath(output_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_training_log(log_file: str) -> Tuple[List[int], List[float], List[float]]:
    """Parse JSONL training log into parallel lists."""
    steps = []
    losses = []
    rewards = []

    if not Path(log_file).exists():
        return steps, losses, rewards

    with open(log_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            step = entry.get("step", entry.get("global_step", None))
            if step is None:
                continue

            steps.append(int(step))

            # TRL logs reward as 'reward' or 'mean_reward'
            reward = entry.get("reward", entry.get("mean_reward",
                     entry.get("reward/mean", None)))
            rewards.append(float(reward) if reward is not None else 0.0)

            loss = entry.get("loss", entry.get("train_loss", 0.0))
            losses.append(float(loss) if loss is not None else 0.0)

    return steps, losses, rewards


def _rolling_average(values: List[float], window: int = 5) -> List[float]:
    """Compute a rolling average for smoothing."""
    if window <= 1 or len(values) <= window:
        return values

    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start:i + 1]
        smoothed.append(sum(chunk) / len(chunk))
    return smoothed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot Mini SOC training reward curve")
    parser.add_argument("--log", type=str, default="./outputs/mini-soc-grpo/training_log.jsonl")
    parser.add_argument("--output", type=str, default="./outputs/reward_curve.png")
    parser.add_argument("--baseline", type=float, default=0.09)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    plot_reward_curve(
        log_file=args.log,
        output_path=args.output,
        random_baseline=args.baseline,
        show_plot=args.show,
    )
