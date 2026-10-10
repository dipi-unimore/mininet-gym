#!/usr/bin/env python3
"""Multi-seed rewrite of the five *_metrics_comparison.png figures (experiments9.tex:
fig:tier_a_metrics, fig:tier_b_s2_metrics, fig:tier_b_s3_metrics, fig:tier_b_s4_metrics,
fig:tier_b_s2s4_metrics).

These figures show training-set macro precision/recall/F1 (pooled over the whole
100-episode training run, exploring policy) for each Tier A condition, Q-Learning and
SARSA. They previously used a single hardcoded seed (77) even though the corresponding
result TABLES were already updated to 3 seeds (10/14/77, see paper9_multiseed.py's
REGISTRY) -- the tables and the confusion-matrix/role-confusion figures were rewritten
for the multi-seed campaign, but nobody had rewritten these because no generator script
existed for them (they were committed as static images, commit e658c3a).

For each condition/algorithm this script pools the *whole* training run's confusion
matrix per seed (train_macro_metrics.train_confusion_per_episode(...).sum(axis=0)),
computes macro precision/recall/F1 from that pooled matrix
(train_macro_metrics.macro_prf_from_confusion), then reports the mean and sample s.d.
across the 3 seeds -- the same per-seed-then-aggregate convention already used for
every other multi-seed number in this paper (tab:s1_uaq, tab:tier_b, ...). Error bars
show the sample s.d.; there are only 3 seeds, so no bar is drawn when the corresponding
cell in REGISTRY has fewer.

Usage (repo root):
    python3 test_tools/plot_metrics_comparison_multiseed.py
    python3 test_tools/plot_metrics_comparison_multiseed.py --which tier_a tier_b_s2
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper9_multiseed import BASE, QL, SARSA, REGISTRY
from train_macro_metrics import train_confusion_per_episode, macro_prf_from_confusion

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = REPO_ROOT / "docs/papers/9-draft of future work in MARL_Coordination__Strategies/images"

ALGORITHMS = [("Q-learning", QL), ("Sarsa", SARSA)]

# figure key -> (title suffix, [(cell, display_label), (cell, display_label)])
FIGURES = {
    "tier_a": ("tier a", [("s0", "S0"), ("s1", "S1-UAQ")]),
    "tier_b_s2": ("tier b -- S2", [("s0_s2", "S0+S2"), ("s1_s2", "S1+S2")]),
    "tier_b_s3": ("tier b -- S3", [("s0_s3", "S0+S3"), ("s1_s3", "S1+S3")]),
    "tier_b_s4": ("tier b -- S4", [("s0_s4", "S0+S4"), ("s1_s4", "S1+S4")]),
    "tier_b_s2s4": ("tier b -- S2+S4", [("s0_s2_s4", "S0+S2+S4"), ("s1_s2_s4", "S1+S2+S4")]),
}

# left-condition / right-condition palettes (light -> dark, matching the original figures)
LEFT_COLORS = {"P": "#d3ecd9", "R": "#f2e6b3", "F1": "#f7d9df"}
RIGHT_COLORS = {"P": "#5aab84", "R": "#d69f2e", "F1": "#c15c78"}


def macro_prf(run: str, agent_dir: str):
    run_full = run if run.endswith("_1_5_5") else f"{run}_1_5_5"
    data_json = REPO_ROOT / BASE / run_full / agent_dir / "data.json"
    cms = train_confusion_per_episode(data_json)
    return macro_prf_from_confusion(cms.sum(axis=0))


def cell_stats(cell: str, agent_dir: str):
    """mean, sd (ddof=1, 0 if n<2) of (P, R, F1) across the cell's seeds."""
    seeds = REGISTRY[cell].seeds
    vals = np.array([macro_prf(run, agent_dir) for run in seeds.values()])
    mean = vals.mean(axis=0)
    sd = vals.std(axis=0, ddof=1) if len(vals) > 1 else np.zeros(3)
    return mean, sd, len(vals)


def plot_one(fig_key: str, title_suffix: str, conditions: list, out_file: str):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharey=True)
    metric_labels = ["Macro-P", "Macro-R", "Macro-F1"]
    n_seeds_seen = set()

    for ax, (algo_label, agent_dir) in zip(axes, ALGORITHMS):
        x = 0.0
        xticks, xticklabels = [], []
        group_centers = []
        for cond_idx, (cell, cond_label) in enumerate(conditions):
            mean, sd, n = cell_stats(cell, agent_dir)
            n_seeds_seen.add(n)
            colors = LEFT_COLORS if cond_idx == 0 else RIGHT_COLORS
            bar_xs = [x, x + 1, x + 2]
            heights = mean
            for bx, h, e, mlabel in zip(bar_xs, heights, sd, metric_labels):
                ax.bar(bx, h, width=0.9, color=colors[mlabel.replace("Macro-", "")],
                       edgecolor="black", linewidth=0.6, yerr=e, capsize=4, zorder=3)
                ax.text(bx, h + e + 0.015, f"{h * 100:.1f}", ha="center", va="bottom", fontsize=9)
            group_centers.append((x + 1, cond_label))
            x += 4  # 3 bars + 1 gap between conditions

        for cx, label in group_centers:
            ax.text(cx, 1.06, label, ha="center", va="bottom", fontsize=13,
                    fontweight="bold", color="#555")
        ax.set_xticks([])
        ax.set_ylim(0, 1.12)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_yticklabels([f"{int(v * 100)}%" for v in np.arange(0, 1.01, 0.2)])
        ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
        ax.set_xlabel(algo_label, fontsize=13, fontweight="bold", labelpad=10)

    axes[0].set_ylabel("Macro score (%) -- training set (mean +/- s.d., 3 seeds)")
    n_seeds = sorted(n_seeds_seen)
    fig.suptitle(f"Metrics comparison -- 100 training episodes -- {title_suffix} "
                 f"(n={'/'.join(str(n) for n in n_seeds)} seeds)", fontsize=13)

    left_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=LEFT_COLORS[m.replace("Macro-", "")],
                                   edgecolor="black", linewidth=0.6) for m in metric_labels]
    right_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=RIGHT_COLORS[m.replace("Macro-", "")],
                                    edgecolor="black", linewidth=0.6) for m in metric_labels]
    leg1 = fig.legend(left_handles, metric_labels, title=conditions[0][1], loc="lower left",
                       bbox_to_anchor=(0.06, -0.02), ncol=3, frameon=False, fontsize=10)
    fig.legend(right_handles, metric_labels, title=conditions[1][1], loc="lower right",
               bbox_to_anchor=(0.98, -0.02), ncol=3, frameon=False, fontsize=10)
    fig.add_artist(leg1)

    plt.tight_layout(rect=(0, 0.06, 1, 0.93))
    out_path = IMAGES_DIR / out_file
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", choices=list(FIGURES), default=list(FIGURES))
    args = ap.parse_args()
    for key in args.which:
        title_suffix, conditions = FIGURES[key]
        plot_one(key, title_suffix, conditions, f"{key}_metrics_comparison.png")


if __name__ == "__main__":
    main()
