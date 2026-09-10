#!/usr/bin/env python3
"""Companion chart for paper9's Tier B two-way combination probe (Section
sec:exp_tier_b, Table tab:tier_b_combinations).

Two panels, one per tested pairing:
  * S2+S3 layered on the S1/UAQ channel  -> mechanistic prediction: partial redundancy
  * S3+S4 layered on the S0/naive channel -> mechanistic prediction: conflict of
    uncertain size

The combination bars are computed live (macro-F1, raw ground_truth/predicted) from
every Q-Learning run directory listed below -- multiple random seeds, cold- and
warm-start pooled. The single-mechanism / baseline bars are the single-seed values
already in Table tab:tier_b and are hardcoded here for reference; keep them in sync
with that table by hand if it changes.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

BASE = "_training/marl_pz"
AGENT = "Q-learning_quick"
OUT = "docs/papers/9-draft of future work in MARL_Coordination__Strategies/images/tier_b_combinations.png"

# --- combination run directories (Q-Learning, seeds 10 / 14 / 77) --------------
S2S3_RUNS = [  # S1 (uaq) + federated_sync + policy_exchange
    "20260824-114917", "20260903-122135", "20260903-160023",
    "20260904-100141", "20260904-120125", "20260904-174343",
]
S3S4_RUNS = [  # S0 (naive) + policy_exchange + experience_sharing
    "20260824-163643", "20260831-170423", "20260902-111527",
    "20260904-154306", "20260907-100421",
]

# --- single-seed reference cells, from Table tab:tier_b -----------------------
REF = {
    "S1 baseline": 0.600, "S1+S3": 0.595, "S1+S2": 0.770,
    "S0 baseline": 0.635, "S0+S3": 0.605, "S0+S4": 0.706,
}

BAR = "#78b7d8"
COMBO = "#3f6f94"


def macro_f1(run_dir):
    with open(f"{BASE}/{run_dir}_1_5_5/TEST_{AGENT}/test.json") as f:
        d = json.load(f)
    gt = np.array(d["ground_truth"], dtype=int)
    pred = np.array(d["predicted"][AGENT], dtype=int)
    return f1_score(gt, pred, average="macro", labels=[0, 1, 2], zero_division=0)


def main():
    v_s2s3 = np.array([macro_f1(r) for r in S2S3_RUNS])
    v_s3s4 = np.array([macro_f1(r) for r in S3S4_RUNS])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)

    panels = [
        dict(ax=axes[0], title="S2+S3 on the S1 (UAQ) channel",
             labels=["S1\nbaseline", "S1+S3", "S1+S2", "S1+S2+S3\n(multi-seed)"],
             singles=[REF["S1 baseline"], REF["S1+S3"], REF["S1+S2"]], combo=v_s2s3,
             note="prediction: partial redundancy\n(between S2-alone and S3-alone)"),
        dict(ax=axes[1], title="S3+S4 on the S0 (naive) channel",
             labels=["S0\nbaseline", "S0+S3", "S0+S4", "S0+S3+S4\n(multi-seed)"],
             singles=[REF["S0 baseline"], REF["S0+S3"], REF["S0+S4"]], combo=v_s3s4,
             note="prediction: conflict of uncertain size\n(combination below S4-alone)"),
    ]

    for p in panels:
        ax = p["ax"]
        x = np.arange(4)
        heights = list(p["singles"]) + [p["combo"].mean()]
        colors = [BAR, BAR, BAR, COMBO]
        errs = [0, 0, 0, p["combo"].std(ddof=1)]
        ax.bar(x, heights, width=0.6, color=colors, edgecolor="black",
               linewidth=0.6, yerr=errs, capsize=5)
        jitter = (np.random.RandomState(0).rand(len(p["combo"])) - 0.5) * 0.28
        ax.scatter(3 + jitter, p["combo"], color="#1f3b52", s=26, zorder=5,
                   edgecolor="white", linewidth=0.5)
        for xi, h, e in zip(x, heights, errs):
            pad = 0.05 if e else 0.02
            ax.text(xi, h + e + pad, f"{h:.3f}", ha="center", va="bottom", fontsize=9.5)
        ax.text(3, 0.04, f"n={len(p['combo'])}", ha="center", fontsize=8.5, color="white")
        ax.set_xticks(x)
        ax.set_xticklabels(p["labels"], fontsize=9)
        ax.set_title(p["title"], fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.text(0.02, 0.97, p["note"], transform=ax.transAxes, fontsize=8,
                va="top", style="italic", color="#555")

    axes[0].set_ylabel("Macro-F1  (Q-Learning)")
    fig.suptitle("Tier B two-way combinations, multi-seed: mechanistic predictions vs. data",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}")
    print(f"  S2+S3 (n={len(v_s2s3)}): mean {v_s2s3.mean():.3f} +/- {v_s2s3.std(ddof=1):.3f}")
    print(f"  S3+S4 (n={len(v_s3s4)}): mean {v_s3s4.mean():.3f} +/- {v_s3s4.std(ddof=1):.3f}")


if __name__ == "__main__":
    main()
