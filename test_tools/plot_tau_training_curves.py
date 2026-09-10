#!/usr/bin/env python3
"""Companion figure for the tau sweep (experiments9.tex, Table
tab:tau_sweep): two panels side by side, one figure per algorithm.

Left panel:  S0 (naive_broadcast) vs S1 (UAQ) at tau in {0.6, 0.8, 0.9},
             training curves -- per-episode train F1, effect of the
             coordination mechanism and of the entropy threshold on
             training dynamics (S0 in grey, S1 variants in warm colors).
Right panel: S1 alone at tau in {0.6, 0.8, 0.9} -- test-set macro-F1
             (left axis, solid) and Msgs/step (right axis, dashed) at the
             end of training, i.e. the actual tab:tau_sweep numbers as an
             explicit Pareto view, plus an S0 macro-F1 reference line.

Left panel reads <agent_dir>/data.json -> train_metrics.f1_score (one
value per training episode). Right panel recomputes macro-F1 from
TEST_<agent_dir>/test.json's ground_truth/predicted (macro-averaged over
the 3 classes, matching compare_episodes_metrics.py's
precision_recall_f1_macro) and Msgs/step from <agent_dir>/comm_stats.json
(mean of the "total" column across alert_rows) directly from each run
directory, rather than hardcoding the table values -- for Q-Learning both
were checked to reproduce Table tab:tau_sweep exactly (macro-F1
0.600/0.656/0.655, Msgs/step 4.70/5.28/3.30 for tau=0.8/0.6/0.9).

Note: train_metrics.f1_score (left panel) is the per-episode metric
logged during *training*; it is not the same figure as the macro-F1 in
Table tab:tau_sweep (right panel), which is computed on the separate
3000-step *test* set after training. The two panels intentionally use
different metrics for this reason -- left panel shows training was
equally stable across conditions, right panel shows where the tau effect
actually appears.

DQN is deliberately not covered here: it majority-class-collapses under
both S0 and S1 regardless of tau (see the "DQN under S1" paragraph in
experiments9.tex) and is excluded from every other Tier A/B table and
figure in the paper for the same reason -- a training-curve/Pareto view
of a collapsed predictor would not show anything the rest of the paper
doesn't already state.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = REPO_ROOT / "docs/papers/9-draft of future work in MARL_Coordination__Strategies/images"
SMOOTH_WINDOW = 7

RUNS = {
    "S0": REPO_ROOT / "_training/marl_pz/20260812-110417_1_5_5",
    "S1 (tau=0.6)": REPO_ROOT / "_training/marl_pz/20260822-114801_1_5_5",
    "S1 (tau=0.8)": REPO_ROOT / "_training/marl_pz/20260815-181031_1_5_5",
    "S1 (tau=0.9)": REPO_ROOT / "_training/marl_pz/20260822-162154_1_5_5",
}

ALGORITHMS = {
    "Q-Learning": {"agent_dir": "Q-learning_quick", "out_file": "tau_training_curves_qlearning.png"},
    "SARSA": {"agent_dir": "Sarsa_balanced", "out_file": "tau_training_curves_sarsa.png"},
}

LEFT_PANEL = ["S0", "S1 (tau=0.6)", "S1 (tau=0.8)", "S1 (tau=0.9)"]
TAU_ORDER = ["S1 (tau=0.6)", "S1 (tau=0.8)", "S1 (tau=0.9)"]
TAU_VALUES = {"S1 (tau=0.6)": 0.6, "S1 (tau=0.8)": 0.8, "S1 (tau=0.9)": 0.9}

COLORS = {
    "S0": "#9a9a9a",
    "S1 (tau=0.6)": "#f4a261",
    "S1 (tau=0.8)": "#e76f51",
    "S1 (tau=0.9)": "#9d0208",
}

F1_COLOR = "#4472a8"
MSGS_COLOR = "#d98ca5"


def load_f1_series(run_dir: Path, agent_dir: str) -> np.ndarray:
    data_json = run_dir / agent_dir / "data.json"
    with open(data_json, "r") as f:
        data = json.load(f)
    return np.array(data["train_metrics"]["f1_score"], dtype=float)


def macro_f1_from_test(run_dir: Path, agent_dir: str, n_classes: int = 3) -> float:
    test_json = run_dir / f"TEST_{agent_dir}" / "test.json"
    with open(test_json, "r") as f:
        data = json.load(f)
    gt = np.array(data["ground_truth"], dtype=int)
    pred = np.array(data["predicted"][agent_dir], dtype=int)
    f1s = []
    for c in range(n_classes):
        tp = int(((pred == c) & (gt == c)).sum())
        fp = int(((pred == c) & (gt != c)).sum())
        fn = int(((pred != c) & (gt == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append((2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0)
    return float(np.mean(f1s))


def msgs_per_step(run_dir: Path, agent_dir: str) -> float:
    comm_stats_json = run_dir / agent_dir / "comm_stats.json"
    with open(comm_stats_json, "r") as f:
        data = json.load(f)
    rows = data["alert_rows"]
    total_idx = data["alert_columns"].index("total")
    return float(np.mean([row[total_idx] for row in rows]))


def zoom_ylim(values, pad_frac=0.08, floor=None, ceil=None):
    """Tightest y-limits that fit all `values` plus a proportional pad, so the
    axis is 'as zoomed as possible' instead of a fixed 0-1/0-100% range."""
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


def plot_training_panel(ax, series_by_label, title):
    for label in series_by_label:
        raw = series_by_label[label]
        episodes_raw = np.arange(1, len(raw) + 1)
        ax.plot(episodes_raw, raw, color=COLORS[label], alpha=0.18, linewidth=1.0)

        sm = smooth(raw, SMOOTH_WINDOW)
        offset = (len(raw) - len(sm)) / 2.0
        episodes_sm = np.arange(len(sm)) + 1 + offset
        ax.plot(episodes_sm, sm, color=COLORS[label], linewidth=2.2, label=label)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Training episode")
    all_raw = np.concatenate(list(series_by_label.values()))
    ax.set_ylim(*zoom_ylim(all_raw, pad_frac=0.06, floor=0.0, ceil=1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, frameon=False)


def plot_tau_pareto_panel(ax, run_paths, s0_path, agent_dir, title):
    taus = [TAU_VALUES[label] for label in TAU_ORDER]
    macro_f1 = [macro_f1_from_test(run_paths[label], agent_dir) for label in TAU_ORDER]
    msgs = [msgs_per_step(run_paths[label], agent_dir) for label in TAU_ORDER]
    s0_f1 = macro_f1_from_test(s0_path, agent_dir)

    l0 = ax.axhline(s0_f1, color=COLORS["S0"], linestyle=":", linewidth=1.5, zorder=1,
                     label=f"S0 macro-F1 ({s0_f1 * 100:.1f}%, reference)")

    l1, = ax.plot(taus, macro_f1, color=F1_COLOR, marker="o", linewidth=2.2, label="Macro-F1 (test, left axis)")
    for t, v in zip(taus, macro_f1):
        ax.annotate(f"{v * 100:.1f}%", (t, v), textcoords="offset points", xytext=(-11, 10), ha="right", fontsize=9, color=F1_COLOR,
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.75))
    ax.set_ylim(*zoom_ylim(macro_f1 + [s0_f1], pad_frac=0.35))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylabel("Macro-F1 (test set)", color=F1_COLOR)
    ax.tick_params(axis="y", labelcolor=F1_COLOR)

    ax2 = ax.twinx()
    l2, = ax2.plot(taus, msgs, color=MSGS_COLOR, marker="s", linestyle="--", linewidth=2.2, label="Msgs/step (right axis)")
    for t, v in zip(taus, msgs):
        ax2.annotate(f"{v:.2f}", (t, v), textcoords="offset points", xytext=(16, -22), ha="left", fontsize=9, color=MSGS_COLOR,
                     bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.75))
    ax2.set_ylim(*zoom_ylim(msgs, pad_frac=0.3))
    ax2.set_ylabel("Msgs/step (test set)", color=MSGS_COLOR)
    ax2.tick_params(axis="y", labelcolor=MSGS_COLOR)

    ax.set_xticks(taus)
    ax.set_xlim(0.47, 1.03)
    ax.set_xlabel("Entropy threshold (tau)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(handles=[l1, l2, l0], loc="lower center", fontsize=9, frameon=False)


def build_figure(algo_label: str, agent_dir: str, out_file: str):
    train_series = {label: load_f1_series(RUNS[label], agent_dir) for label in LEFT_PANEL}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

    plot_training_panel(axes[0], train_series, f"S0 vs S1 (tau=0.6/0.8/0.9) -- training")
    plot_tau_pareto_panel(
        axes[1],
        run_paths=RUNS,
        s0_path=RUNS["S0"],
        agent_dir=agent_dir,
        title="Entropy threshold: S1 alone -- test set",
    )

    axes[0].set_ylabel("F1 score (train, per-episode)")
    fig.suptitle(
        f"{algo_label}: S0 vs S1 training dynamics across tau (left, per-episode train F1) vs "
        "end-of-training test macro-F1 / Msgs/step by tau (right)",
        fontsize=10,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.93))
    out_path = IMAGES_DIR / out_file
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    for algo_label, cfg in ALGORITHMS.items():
        build_figure(algo_label, cfg["agent_dir"], cfg["out_file"])


if __name__ == "__main__":
    main()
