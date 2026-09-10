#!/usr/bin/env python3
"""Companion figure for the "DQN under S1" paragraph in experiments9.tex:
DQN's per-episode training metric looks like a converging, reasonably good
policy, while every test-time metric collapses -- illustrating in one
picture why this paper insists on macro-averaged F1 (Section 6.1, point 1)
rather than the weighted-averaged figures the dashboard also stores.

Left panel:  DQN per-episode training F1 (train_metrics.f1_score), S0 vs
             S1 (tau=0.8) -- both climb to ~80% and look like healthy
             convergence.
Right panel: three bars per run -- the same in-training F1 metric at the
             end of training, the dashboard's weighted test accuracy, and
             this paper's macro-averaged test F1. The first two look fine
             (both dominated by the ~83% normal-class share); the third
             exposes that DQN predicts "normal" for essentially every one
             of the 3000 test steps (per-class recall
             [normal, victim, attacker] = [100%, 0%, 0%] in both runs).
             Deliberately NOT y-zoomed like the QL/SARSA tau-sweep figures
             -- the point here is the size of the drop, so compressing the
             axis would undersell it.

Only S0 and S1 (tau=0.8) are available for DQN (the 2026-08-22 tau=0.6/0.9
sweep runs only cover Q-Learning and SARSA), so there is no tau axis here.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = REPO_ROOT / "docs/papers/9-draft of future work in MARL_Coordination__Strategies/images"
AGENT_DIR = "DQN_quick"
SMOOTH_WINDOW = 7
N_CLASSES = 3

RUNS = {
    "S0": REPO_ROOT / "_training/marl_pz/20260812-110417_1_5_5",
    "S1 (tau=0.8)": REPO_ROOT / "_training/marl_pz/20260815-181031_1_5_5",
}

COLORS = {"S0": "#9a9a9a", "S1 (tau=0.8)": "#e76f51"}
METRIC_COLORS = {
    "Train F1\n(in-training, weighted)": "#78b7d8",
    "Test accuracy\n(weighted)": "#e6c56c",
    "Test macro-F1\n(this paper's metric)": "#d98ca5",
}
NORMAL_CLASS_SHARE = 0.83


def load_f1_series(run_dir: Path) -> np.ndarray:
    with open(run_dir / AGENT_DIR / "data.json") as f:
        data = json.load(f)
    return np.array(data["train_metrics"]["f1_score"], dtype=float)


def load_test(run_dir: Path):
    with open(run_dir / f"TEST_{AGENT_DIR}" / "test.json") as f:
        return json.load(f)


def macro_f1(gt, pred, n_classes=N_CLASSES):
    gt = np.array(gt, dtype=int)
    pred = np.array(pred, dtype=int)
    f1s = []
    for c in range(n_classes):
        tp = int(((pred == c) & (gt == c)).sum())
        fp = int(((pred == c) & (gt != c)).sum())
        fn = int(((pred != c) & (gt == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append((2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0)
    return float(np.mean(f1s))


def zoom_ylim(values, pad_frac=0.08, floor=None, ceil=None):
    lo, hi = float(min(values)), float(max(values))
    span = hi - lo if hi > lo else max(abs(hi), 1e-6)
    pad = span * pad_frac
    lo, hi = lo - pad, hi + pad
    if floor is not None:
        lo = max(lo, floor)
    if ceil is not None:
        hi = min(hi, ceil)
    return lo, hi


def smooth(series: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or series.size < window:
        return series
    kernel = np.ones(window) / window
    return np.convolve(series, kernel, mode="valid")


def plot_training_panel(ax, series_by_label):
    for label, raw in series_by_label.items():
        episodes_raw = np.arange(1, len(raw) + 1)
        ax.plot(episodes_raw, raw, color=COLORS[label], alpha=0.18, linewidth=1.0)
        sm = smooth(raw, SMOOTH_WINDOW)
        offset = (len(raw) - len(sm)) / 2.0
        episodes_sm = np.arange(len(sm)) + 1 + offset
        ax.plot(episodes_sm, sm, color=COLORS[label], linewidth=2.2, label=label)

    ax.set_title("DQN training F1: S0 vs S1 (tau=0.8)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("F1 score (train, per-episode)")
    all_raw = np.concatenate(list(series_by_label.values()))
    ax.set_ylim(*zoom_ylim(all_raw, pad_frac=0.06, floor=0.0, ceil=1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, frameon=False)


def plot_gap_panel(ax, metrics_by_run):
    metric_names = list(METRIC_COLORS.keys())
    run_labels = list(metrics_by_run.keys())
    n_metrics = len(metric_names)
    bar_width = 0.36
    x = np.arange(n_metrics, dtype=float)

    for i, run_label in enumerate(run_labels):
        offset = (i - (len(run_labels) - 1) / 2.0) * bar_width
        vals = [metrics_by_run[run_label][m] for m in metric_names]
        edgecolor = COLORS[run_label]
        bars = ax.bar(x + offset, vals, width=bar_width, color=[METRIC_COLORS[m] for m in metric_names],
                       edgecolor=edgecolor, linewidth=2.2, hatch=("///" if i == 1 else None))
        for xi, v in zip(x + offset, vals):
            ax.text(xi, v + 0.02, f"{v * 100:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.axhline(NORMAL_CLASS_SHARE, color="black", linestyle=":", linewidth=1.2, alpha=0.6)
    ax.text(0.55, NORMAL_CLASS_SHARE + 0.025, "normal-class share (~83%)", fontsize=8, alpha=0.7, ha="center")

    # Legend proxies: one per metric (color) + one per run (edge style)
    from matplotlib.patches import Patch
    metric_handles = [Patch(facecolor=METRIC_COLORS[m], edgecolor="none", label=m.split("\n")[0]) for m in metric_names]
    run_handles = [Patch(facecolor="white", edgecolor=COLORS[r], linewidth=2.2, hatch=("///" if i == 1 else None), label=r)
                   for i, r in enumerate(run_labels)]

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=8.5)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_title("Same runs, three metrics: the drop only macro-F1 exposes", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(handles=metric_handles + run_handles, loc="upper right", fontsize=8, frameon=True,
              facecolor="white", edgecolor="none", framealpha=0.9, ncol=1)


def main():
    train_series = {label: load_f1_series(path) for label, path in RUNS.items()}

    metrics_by_run = {}
    for label, path in RUNS.items():
        train_end = float(np.mean(train_series[label][-10:]))
        test = load_test(path)
        gt = test["ground_truth"]
        pred = test["predicted"][AGENT_DIR]
        metrics_by_run[label] = {
            "Train F1\n(in-training, weighted)": train_end,
            "Test accuracy\n(weighted)": test["metrics"]["accuracy"],
            "Test macro-F1\n(this paper's metric)": macro_f1(gt, pred),
        }

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    plot_training_panel(axes[0], train_series)
    plot_gap_panel(axes[1], metrics_by_run)

    fig.suptitle(
        "DQN: training looks like convergence, macro-F1 shows total majority-class collapse",
        fontsize=10,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    out_path = IMAGES_DIR / "dqn_train_test_gap.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
