#!/usr/bin/env python3
"""Final multi-seed figures for paper9's Tier B section, built on top of the
cell -> run registry in test_tools/paper9_multiseed.py (seeds 10/14/77 pooled,
see memory project-paper9-multiseed-pooling). Replaces the earlier
single-seed/mixed-seed versions of tier_b_summary.png and tier_b_combinations.png.

  python3 test_tools/plot_paper9_tier_b_final.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper9_multiseed import REGISTRY, cell_summary, QL, SARSA

OUT = "docs/papers/9-draft of future work in MARL_Coordination__Strategies/images"
QL_COLOR = "#78b7d8"
SARSA_COLOR = "#d98ca5"


def stats(cell_id, agent):
    return cell_summary(REGISTRY[cell_id], agent)


def bar_group(ax, x, labels, cells, title):
    """One panel: grouped QL/SARSA bars with sd error bars + per-seed dots,
    for the sequence of cells given (each rendered at one x position)."""
    w = 0.35
    rng = np.random.RandomState(0)
    for i, cid in enumerate(cells):
        for off, agent, color in [(-w / 2, QL, QL_COLOR), (w / 2, SARSA, SARSA_COLOR)]:
            s = stats(cid, agent)
            xi = x[i] + off
            ax.bar(xi, s["macro_mean"], w, yerr=s["macro_sd"], capsize=3,
                   color=color, edgecolor="black", linewidth=0.6,
                   label=("Q-Learning" if agent == QL else "SARSA") if i == 0 else None)
            pts = [v["macro"] for v in s["per_seed"].values()]
            jitter = (rng.rand(len(pts)) - 0.5) * w * 0.6
            ax.scatter(xi + jitter, pts, s=14, color="#1f3b52" if agent == QL else "#6b2b45",
                       zorder=5, edgecolor="white", linewidth=0.3)
            ax.text(xi, s["macro_mean"] + s["macro_sd"] + 0.015, f"{s['macro_mean']:.2f}",
                    ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def make_tier_b_summary():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, tier_a, cells in zip(axes, ["S0", "S1"],
                                 [["s0", "s0_s2", "s0_s3", "s0_s4"],
                                  ["s1", "s1_s2", "s1_s3", "s1_s4"]]):
        x = np.arange(4, dtype=float)
        bar_group(ax, x, ["baseline", "S2", "S3", "S4"], cells, f"Tier A: {tier_a}")
        ax.set_xlabel("Tier B condition")
    axes[0].set_ylabel("Macro-F1")
    axes[0].legend(loc="upper left", fontsize=9, frameon=False)
    fig.suptitle("Tier B ablation, multi-seed (n=3, seeds 10/14/77): "
                  "macro-F1 by condition and algorithm", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = f"{OUT}/tier_b_summary.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def make_secondary_probes():
    panels = [
        ("S0+S2+S4 (on S0)", ["s0", "s0_s2", "s0_s4", "s0_s2_s4"],
         ["baseline", "S2", "S4", "S2+S4"]),
        ("S1+S2+S4 (on S1)", ["s1", "s1_s2", "s1_s4", "s1_s2_s4"],
         ["baseline", "S2", "S4", "S2+S4"]),
        ("S1+S2+S3 (on S1)", ["s1", "s1_s2", "s1_s3", "s1_s2_s3"],
         ["baseline", "S2", "S3", "S2+S3"]),
        ("S0+S3+S4 (on S0)", ["s0", "s0_s4", "s0_s3", "s0_s3_s4"],
         ["baseline", "S4", "S3", "S3+S4"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), sharey=True)
    for ax, (title, cells, labels) in zip(axes.flat, panels):
        x = np.arange(4, dtype=float)
        bar_group(ax, x, labels, cells, title)
    axes[0, 0].legend(loc="upper left", fontsize=9, frameon=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Macro-F1")
    fig.suptitle("Tier B two-way combinations, multi-seed (n=3, seeds 10/14/77):\n"
                  "each combo vs. its baseline and the two contributing single mechanisms",
                  fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = f"{OUT}/tier_b_secondary_probes.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def make_tau_sweep():
    fig, ax = plt.subplots(figsize=(6, 4.4))
    x = np.arange(2, dtype=float)
    bar_group(ax, x, ["τ=0.8", "τ=0.9"], ["s1", "s1_tau09"],
              "UAQ threshold, multi-seed (S1, no Tier B)")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.set_ylabel("Macro-F1")
    fig.tight_layout()
    path = f"{OUT}/tau_multiseed.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    make_tier_b_summary()
    make_secondary_probes()
    make_tau_sweep()
