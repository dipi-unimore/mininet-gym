#!/usr/bin/env python3
"""Paper9 Section 5.5: zero-shot topology-generalization figures.

Each N=20 (10 host + 10 IoT) run in this comparison is a *warm-start,
per-host-only* load of a policy trained at N=10 (5 host + 5 IoT): every
per-host agent (Q-Learning, SARSA) is loaded from its matching N=10 run,
the coordinator is NOT loaded (its policy is conditioned on global
alert/sync traffic that is not topology-invariant), and only 3 further
training episodes are run before the frozen policy is evaluated over the
same 300-episode held-out test protocol used throughout the rest of the
paper. This is therefore a topology-*generalization* test, not a native
N=20 training run (Section~5.5's originally planned native-N=20 stress
test remains unrun).

Reads TEST_<agent>/test.json directly from each run directory, using the
same n_hosts-aware ground-truth remap as generate_ranking_table.py so
numbers match the rest of the paper's tables exactly.

Two cells (S1+S3, S0+S2+S4) were originally collected with a load_dir
mismatch (SARSA/DQN loaded from the wrong N=10 source) and have since
been re-run in full with corrected configs (2026-08-24); the corrected
runs are used below and supersede the earlier partial ones.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_ranking_table import get_num_hosts, read_test_payload  # noqa: E402

import numpy as np
import matplotlib.pyplot as plt

BASE = Path("_training/marl_pz")
AGENT_DIR = {"Q-Learning": "Q-learning_quick", "SARSA": "Sarsa_balanced"}

# (strategy label, N=10 source run, N=20 generalization run)
PAIRS = [
    ("S0",        "20260812-110417_1_5_5", "20260823-113920_1_10_10"),
    ("S1",        "20260815-181031_1_5_5", "20260823-125419_1_10_10"),
    ("S0+S2",     "20260813-091949_1_5_5", "20260823-151359_1_10_10"),
    ("S1+S2",     "20260816-174823_1_5_5", "20260823-160637_1_10_10"),
    ("S0+S3",     "20260814-100046_1_5_5", "20260823-172714_1_10_10"),
    ("S1+S3",     "20260817-091109_1_5_5", "20260824-150808_1_10_10"),  # corrected re-run
    ("S0+S4",     "20260814-183144_1_5_5", "20260824-075828_1_10_10"),
    ("S1+S4",     "20260818-150129_1_5_5", "20260824-084654_1_10_10"),
    ("S0+S2+S4",  "20260815-104903_1_5_5", "20260824-160134_1_10_10"),  # corrected re-run
    ("S1+S2+S4",  "20260819-135609_1_5_5", "20260824-102011_1_10_10"),
]


def load_gt_pred(run_dir, algo):
    exp_path = BASE / run_dir
    n_hosts = get_num_hosts(exp_path)
    agent_dir = AGENT_DIR[algo]
    test_json = exp_path / f"TEST_{agent_dir}" / "test.json"
    return read_test_payload(test_json, agent_dir, n_hosts)


def macro_f1(gt, pred):
    f1s = []
    for c in sorted(set(gt.tolist()) | set(pred.tolist())):
        tp = int(((pred == c) & (gt == c)).sum())
        fp = int(((pred == c) & (gt != c)).sum())
        fn = int(((pred != c) & (gt == c)).sum())
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0)
    return float(np.mean(f1s))


def recall_per_class(gt, pred, n_classes=3):
    recalls = []
    for c in range(n_classes):
        mask = gt == c
        recalls.append(float((pred[mask] == c).sum()) / mask.sum() if mask.sum() > 0 else np.nan)
    return recalls


def fig1_f1_by_strategy():
    labels = [p[0] for p in PAIRS]
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    bar_width = 0.32
    x = np.arange(len(labels), dtype=float)

    for ax, algo in zip(axes, ["Q-Learning", "SARSA"]):
        f10, f20 = [], []
        for label, src, tgt in PAIRS:
            gt10, pred10 = load_gt_pred(src, algo)
            gt20, pred20 = load_gt_pred(tgt, algo)
            f10.append(macro_f1(gt10, pred10))
            f20.append(macro_f1(gt20, pred20))
        f10, f20 = np.array(f10), np.array(f20)

        b1 = ax.bar(x - bar_width / 2, f10, width=bar_width,
                     color="#78b7d8", edgecolor="black", linewidth=0.6, label="N=10 (native)")
        b2 = ax.bar(x + bar_width / 2, f20, width=bar_width,
                     color="#d98ca5", edgecolor="black", linewidth=0.6, label="N=20 (zero-shot transfer)")
        for xi, v in zip(x - bar_width / 2, f10):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)
        for xi, v in zip(x + bar_width / 2, f20):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)

        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Macro-F1")
        ax.set_title(algo, fontsize=10, loc="left")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    axes[0].legend(loc="upper left", ncol=2, fontsize=9, frameon=False)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    fig.suptitle("Zero-shot topology generalization: N=10 (5+5) trained policy evaluated at N=20 (10+10),\n"
                 "per-host agents only (coordinator not loaded), same Tier A x Tier B grid as Table~3-4",
                 fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    out = "docs/papers/9-draft of future work in MARL_Coordination__Strategies/images/topology_generalization_f1.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def fig2_pooled_recall():
    class_names = ["normal", "victim", "attacker"]
    class_colors = ["#78b7d8", "#e6c56c", "#d98ca5"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for ax, algo in zip(axes, ["Q-Learning", "SARSA"]):
        gt10_all, pred10_all, gt20_all, pred20_all = [], [], [], []
        for label, src, tgt in PAIRS:
            g10, p10 = load_gt_pred(src, algo)
            g20, p20 = load_gt_pred(tgt, algo)
            gt10_all.append(g10); pred10_all.append(p10)
            gt20_all.append(g20); pred20_all.append(p20)
        gt10, pred10 = np.concatenate(gt10_all), np.concatenate(pred10_all)
        gt20, pred20 = np.concatenate(gt20_all), np.concatenate(pred20_all)

        r10 = recall_per_class(gt10, pred10)
        r20 = recall_per_class(gt20, pred20)
        n10 = [int((gt10 == c).sum()) for c in range(3)]
        n20 = [int((gt20 == c).sum()) for c in range(3)]

        x = np.arange(3, dtype=float)
        bar_width = 0.32
        b1 = ax.bar(x - bar_width / 2, r10, width=bar_width, color=class_colors,
                     edgecolor="black", linewidth=0.6)
        b2 = ax.bar(x + bar_width / 2, r20, width=bar_width, color=class_colors,
                     edgecolor="black", linewidth=0.6, hatch="///")
        for xi, v, n in zip(x - bar_width / 2, r10, n10):
            ax.text(xi, v + 0.02, f"{v:.2f}\n(n={n})", ha="center", va="bottom", fontsize=7)
        for xi, v, n in zip(x + bar_width / 2, r20, n20):
            ax.text(xi, v + 0.02, f"{v:.2f}\n(n={n})", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(class_names)
        ax.set_ylim(0, 1.15)
        ax.set_title(algo, fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    axes[0].set_ylabel("Recall")
    solid_proxy = plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", linewidth=0.6, label="N=10 (native)")
    hatch_proxy = plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", linewidth=0.6, hatch="///", label="N=20 (zero-shot transfer)")
    fig.legend(handles=[solid_proxy, hatch_proxy], loc="upper center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Per-class recall pooled across all 10 strategy cells per algorithm\n"
                 "(sample size n grows with N because it is per-host, not per-episode)", fontsize=9, y=1.0)
    plt.tight_layout(rect=(0, 0, 1, 0.86))
    out = "docs/papers/9-draft of future work in MARL_Coordination__Strategies/images/topology_generalization_recall.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    fig1_f1_by_strategy()
    fig2_pooled_recall()
