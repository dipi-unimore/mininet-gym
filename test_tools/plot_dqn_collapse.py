#!/usr/bin/env python3
"""Two-panel chart illustrating DQN's majority-class collapse across the paper9
Tier A x Tier B grid: per-class recall (top) and control-plane messages per step
(bottom), one group per (Tier A, Tier B) condition.

Reads TEST_<agent>/test.json (ground_truth/predicted) and <agent>/comm_stats.json
directly from each run directory -- not part of the generic compare_* pipeline
since it plots one specific algorithm (DQN) across many runs rather than several
algorithms within one run.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE = Path("_training/marl_pz")
AGENT_DIR = "DQN_quick"

# (x-axis label, run directory, agent dir override or None for AGENT_DIR)
RUNS = [
    ("S0", "20260812-110417_1_5_5", None),
    ("S0+S2", "20260813-091949_1_5_5", None),
    ("S0+S3", "20260814-100046_1_5_5", None),
    ("S0+S4", "20260814-183144_1_5_5", None),
    ("S0+S2+S4", "20260815-104903_1_5_5", None),
    ("S1", "20260815-181031_1_5_5", None),
    ("S1+S2", "20260816-174823_1_5_5", None),
    ("S1+S3", "20260817-091109_1_5_5", None),
    ("S1+S4", "20260818-150129_1_5_5", None),
    ("S1+S2+S4", "20260819-135609_1_5_5", None),
]

# Episode-scaling check: same S1+S2+S4 / S1+S4 conditions re-run at 5x the
# episode budget (500 vs. 100 training episodes), to test whether the
# collapse is an under-training artifact. Kept visually distinct (separate
# group, hatched bars) since they are not part of the fixed-episode grid above.
SCALING_RUNS = [
    ("S1+S2+S4\n(500ep)", "20260820-131454_1_5_5", "DQN_balanced"),
    ("S1+S4\n(500ep)", "20260821-124419_1_5_5", "DQN_balanced"),
]


def recall_per_class(gt, pred, n_classes=3):
    gt = np.array(gt, dtype=int)
    pred = np.array(pred, dtype=int)
    recalls = []
    for c in range(n_classes):
        mask = gt == c
        recalls.append(float((pred[mask] == c).sum()) / mask.sum() if mask.sum() > 0 else np.nan)
    return recalls


def msgs_per_step(comm_stats_path):
    with open(comm_stats_path) as f:
        c = json.load(f)
    rows = c.get("alert_rows", [])
    cols = c.get("alert_columns", [])
    if not rows or "total" not in cols:
        return None
    ti = cols.index("total")
    return sum(r[ti] for r in rows) / len(rows)


def main():
    all_runs = list(RUNS) + list(SCALING_RUNS)
    n_grid = len(RUNS)

    labels, recalls, msgs = [], [], []
    for label, run_dir, agent_dir_override in all_runs:
        agent_dir = agent_dir_override or AGENT_DIR
        run_path = BASE / run_dir
        test_json = run_path / f"TEST_{agent_dir}" / "test.json"
        comm_json = run_path / agent_dir / "comm_stats.json"
        with open(test_json) as f:
            t = json.load(f)
        gt = t["ground_truth"]
        pred = t["predicted"][agent_dir]
        labels.append(label)
        recalls.append(recall_per_class(gt, pred))
        msgs.append(msgs_per_step(comm_json))

    recalls = np.array(recalls)  # shape (n_grid + n_scaling, 3)
    x = np.arange(len(labels), dtype=float)
    # Push the scaling-check bars away from the grid so the extra separator
    # line reads as a distinct group rather than more grid cells.
    x[n_grid:] += 0.8
    bar_width = 0.26
    class_names = ["normal", "under_attack", "attacking"]
    class_colors = ["#78b7d8", "#e6c56c", "#d98ca5"]
    is_scaling = [False] * n_grid + [True] * len(SCALING_RUNS)

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(13.5, 7), sharex=True,
                                             gridspec_kw={"height_ratios": [2, 1]})

    for i, (cname, color) in enumerate(zip(class_names, class_colors)):
        offset = (i - 1) * bar_width
        vals = recalls[:, i]
        hatches = [None if not s else "///" for s in is_scaling]
        for xi, v, hatch in zip(x + offset, vals, hatches):
            bar = ax_top.bar(xi, np.nan_to_num(v), width=bar_width, color=color,
                              edgecolor="black", linewidth=0.6, hatch=hatch)[0]
            if not np.isnan(v):
                ax_top.text(bar.get_x() + bar.get_width() / 2.0, v + 0.02, f"{v:.2f}",
                            ha="center", va="bottom", fontsize=7, rotation=90)
    # Legend proxies (avoid one per bar)
    for cname, color in zip(class_names, class_colors):
        ax_top.bar(0, 0, color=color, edgecolor="black", linewidth=0.6, label=cname)

    ax_top.axvline(4.5, color="gray", linestyle=":", linewidth=1)
    ax_top.axvline((x[n_grid - 1] + x[n_grid]) / 2.0, color="black", linestyle="-", linewidth=1)
    ax_top.set_ylim(0, 1.15)
    ax_top.set_ylabel("Recall")
    ax_top.set_title("DQN majority-class collapse across the Tier A x Tier B grid, "
                      "and under 5x more training episodes")
    ax_top.legend(loc="upper right", ncol=3, fontsize=9, frameon=False)
    ax_top.grid(axis="y", linestyle="--", alpha=0.3)

    bar_colors = ["#9aa5b1" if "S4" not in lab else "#c0392b" for lab in labels]
    hatches_bottom = [None if not s else "///" for s in is_scaling]
    for xi, v, color, hatch in zip(x, msgs, bar_colors, hatches_bottom):
        bar = ax_bottom.bar(xi, v, width=0.55, color=color, edgecolor="black",
                             linewidth=0.6, hatch=hatch)[0]
        ax_bottom.text(bar.get_x() + bar.get_width() / 2.0, v + 0.5, f"{v:.1f}",
                        ha="center", va="bottom", fontsize=8)
    ax_bottom.axvline(4.5, color="gray", linestyle=":", linewidth=1)
    ax_bottom.axvline((x[n_grid - 1] + x[n_grid]) / 2.0, color="black", linestyle="-", linewidth=1)
    ax_bottom.set_ylabel("Msgs/step")
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(labels, rotation=30, ha="right")
    ax_bottom.grid(axis="y", linestyle="--", alpha=0.3)

    fig.text(0.11, 0.965, "S0 (naive_broadcast) conditions", fontsize=9, color="dimgray")
    fig.text(0.52, 0.965, "S1 (UAQ) conditions", fontsize=9, color="dimgray")
    fig.text(0.85, 0.965, "100 vs 500 episodes\n(hatched)", fontsize=9, color="dimgray")

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = "docs/papers/9-draft of future work in MARL_Coordination__Strategies/images/dqn_collapse.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
