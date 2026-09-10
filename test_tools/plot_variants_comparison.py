#!/usr/bin/env python3
"""
Prototype: one figure comparing the three configuration variants
(Reward Scaling / + Deltas / + Link Blocking) in a single chart, to replace the
three separate per-variant metric figures (paper Fig. 4-6).

Reads the same training-summary metrics as compare_episodes_metrics.py
(``train_metrics`` in each ``<algo>/data.json``), from the three per-variant
configs:  compare_config_reward_scaling.yaml / _deltas.yaml / _all_flags.yaml.

Usage:
    python3 test_tools/plot_variants_comparison.py \
        --metric f1_score --episodes 300 \
        --output_dir _training/_debug/variants
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from compare_episodes_metrics import (
    compute_metrics_for_experiment as _train_metrics,
    get_episodes,
)
from generate_ranking_table import (
    compute_metrics_for_experiment as _test_metrics,
)

VARIANT_CONFIGS = [
    ("Reward Scaling", "test_tools/compare_config_reward_scaling.yaml"),
    ("+ Deltas", "test_tools/compare_config_deltas.yaml"),
    ("+ Link Blocking", "test_tools/compare_config_all_flags.yaml"),
]
ALGO_ORDER = ["Q-Learning", "SARSA", "DQN", "PPO", "A2C"]
METRIC_LABEL = {
    "accuracy": "Accuracy", "precision": "Precision",
    "recall": "Recall", "f1_score": "F1",
}
# one hue per variant, light shade = 100 ep, dark = 300 ep
VARIANT_COLORS = {
    "Reward Scaling": ("#cfe3f5", "#3d7cb8"),
    "+ Deltas": ("#d3efd9", "#4ba36a"),
    "+ Link Blocking": ("#f8dcd0", "#d9713f"),
}

FS_TICK = 18
FS_AXIS = 17
FS_TITLE = 19
FS_VALUE = 14
FS_LEGEND = 14


def _norm_algo(name):
    m = {"Q-learning": "Q-Learning", "Sarsa": "SARSA"}
    base = name.split("_", 1)[0]
    return m.get(base, base)


def collect(metric, episodes, source):
    """-> {variant: {episode_count: {algo: value}}}

    source="train": training-set weighted metrics (train_metrics in data.json).
    source="test":  held-out test-set macro metrics (from TEST*/test.json) --
                    this is the same data as the paper's ranking table.
    """
    compute = _test_metrics if source == "test" else _train_metrics
    out = {}
    for variant, cfg_path in VARIANT_CONFIGS:
        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh)
        out[variant] = {}
        for exp in cfg["experiments"]:
            ep = get_episodes(exp["path"])
            if episodes not in (None, "all") and ep != episodes:
                continue
            m = compute(exp["path"])
            out[variant][ep] = {
                _norm_algo(a): vals.get(metric, 0.0) for a, vals in m.items()
            }
    return out


def plot(data, metric, episodes_shown, output_dir, source):
    os.makedirs(output_dir, exist_ok=True)
    variants = [v for v, _ in VARIANT_CONFIGS]
    algos = [a for a in ALGO_ORDER
             if any(a in data[v].get(ep, {}) for v in variants for ep in data[v])]
    eps = sorted({ep for v in variants for ep in data[v]})
    two_eps = len(eps) > 1

    n_algo, n_var, n_ep = len(algos), len(variants), len(eps)
    group_w = 0.86                         # fraction of the unit slot filled by one algo's bars
    bar_w = group_w / (n_var * n_ep)
    fig, ax = plt.subplots(figsize=(2.3 * n_algo + 1.6, 5.4))

    x = np.arange(n_algo)
    max_val = 0.0
    for vi, variant in enumerate(variants):
        light, dark = VARIANT_COLORS[variant]
        for ei, ep in enumerate(eps):
            slot = vi * n_ep + ei
            offset = (slot - (n_var * n_ep - 1) / 2) * bar_w
            vals = [data[variant].get(ep, {}).get(a, 0.0) * 100 for a in algos]
            max_val = max(max_val, *vals)
            color = light if (two_eps and ep == min(eps)) else dark
            bars = ax.bar(x + offset, vals, bar_w, color=color,
                          edgecolor="black", linewidth=0.5)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}",
                        ha="center", va="bottom", fontsize=FS_VALUE, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(algos, fontsize=FS_TICK, fontweight="bold")
    ax.tick_params(axis="y", labelsize=FS_TICK - 3)
    src_txt = "test set, macro avg" if source == "test" else "training set, weighted avg"
    ax.set_ylabel(f"{METRIC_LABEL.get(metric, metric)} (%)  ({src_txt})", fontsize=FS_AXIS - 3)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_ylim(0, max(102, max_val + 14))   # headroom for the rotated value labels
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ep_txt = " vs ".join(str(e) for e in eps) + " ep." if two_eps else f"{eps[0]} ep."
    ax.set_title(f"{METRIC_LABEL.get(metric, metric)} by configuration variant  ({ep_txt})",
                 fontsize=FS_TITLE)

    # legend: variant colours anchored to the left; if two episode counts, a
    # light/dark shade key anchored to the right, well separated from the colours
    colour_handles = [Patch(facecolor=VARIANT_COLORS[v][1], edgecolor="black", label=v)
                      for v in variants]
    legs = []
    if not two_eps:
        legs.append(ax.legend(handles=colour_handles, loc="upper center",
                              bbox_to_anchor=(0.5, -0.11), ncol=n_var,
                              frameon=False, fontsize=FS_LEGEND))
    else:
        legs.append(ax.legend(handles=colour_handles, loc="upper left",
                              bbox_to_anchor=(-0.02, -0.11), ncol=n_var,
                              frameon=False, fontsize=FS_LEGEND))
        ax.add_artist(legs[0])
        shade_handles = [
            Patch(facecolor="#cccccc", edgecolor="black", label=f"{min(eps)} ep."),
            Patch(facecolor="#555555", edgecolor="black", label=f"{max(eps)} ep."),
        ]
        legs.append(ax.legend(handles=shade_handles, loc="upper right",
                              bbox_to_anchor=(1.02, -0.11), ncol=2,
                              frameon=False, fontsize=FS_LEGEND,
                              title="light / dark:", title_fontsize=FS_LEGEND - 2))

    out = os.path.join(output_dir, f"variants_{source}_{metric}_{episodes_shown}.png")
    plt.savefig(out, dpi=180, bbox_inches="tight", bbox_extra_artists=legs, pad_inches=0.1)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="f1_score",
                    choices=["accuracy", "precision", "recall", "f1_score"])
    ap.add_argument("--source", default="train", choices=["train", "test"],
                    help="train = training-set weighted metrics; test = held-out macro metrics")
    ap.add_argument("--episodes", default="300", help="300 | 100 | all")
    ap.add_argument("--output_dir", default="_training/_debug/variants")
    args = ap.parse_args()
    episodes = args.episodes if args.episodes == "all" else int(args.episodes)
    data = collect(args.metric, episodes, args.source)
    plot(data, args.metric, args.episodes, args.output_dir, args.source)


if __name__ == "__main__":
    main()
