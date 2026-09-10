#!/usr/bin/env python3
"""Companion chart for paper9's Table 5 (tab:tier_b): macro-F1 across the primary
2x4 Tier B grid, one panel per Tier A channel (S0, S1), Tier B condition on the
x-axis, one bar per algorithm (Q-Learning, SARSA).

Hardcoded to the primary-grid numbers already in experiments9.tex (Table 5) rather
than re-reading run directories, since this is a companion visualization of that
exact table, not an independent computation -- keep the two in sync by hand if the
table numbers change.
"""
import matplotlib.pyplot as plt
import numpy as np

CONDITIONS = ["baseline", "S2", "S3", "S4"]

DATA = {
    "S0": {
        "Q-Learning": [0.635, 0.763, 0.605, 0.706],
        "SARSA":      [0.648, 0.764, 0.663, 0.790],
    },
    "S1": {
        "Q-Learning": [0.600, 0.770, 0.595, 0.733],
        "SARSA":      [0.661, 0.817, 0.621, 0.748],
    },
}

ALGO_COLORS = {"Q-Learning": "#78b7d8", "SARSA": "#d98ca5"}


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    x = np.arange(len(CONDITIONS), dtype=float)
    bar_width = 0.35

    for ax, tier_a in zip(axes, ["S0", "S1"]):
        for i, (algo, color) in enumerate(ALGO_COLORS.items()):
            offset = (i - 0.5) * bar_width
            vals = DATA[tier_a][algo]
            bars = ax.bar(x + offset, vals, width=bar_width, color=color,
                           edgecolor="black", linewidth=0.6, label=algo)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2.0, v + 0.015, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(CONDITIONS)
        ax.set_title(f"Tier A: {tier_a}", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 0.95)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_xlabel("Tier B condition")

    axes[0].set_ylabel("Macro-F1")
    axes[0].legend(loc="upper left", fontsize=9, frameon=False)
    fig.suptitle("Tier B ablation summary (Table 5): macro-F1 by condition and algorithm", fontsize=13)

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = "docs/papers/9-draft of future work in MARL_Coordination__Strategies/images/tier_b_summary.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
