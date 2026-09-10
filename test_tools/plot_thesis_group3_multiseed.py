#!/usr/bin/env python3
"""Thesis Ch6: multi-seed Group-3 (20-node zero-shot transfer) analysis + figures.

Extends test_tools/plot_group3_role_confusion_multiseed.py (which only covers the
Reward-Scaling-Only variant, for the ICAART paper) to all three configuration
variants, and writes the three role-confusion figures into the thesis image dir:

  docs/papers/thesis/images/group3_reward_scaling_role_confusion_rates.png
  docs/papers/thesis/images/group3_deltas_role_confusion_rates.png
  docs/papers/thesis/images/group3_all_flags_role_confusion_rates.png

Each bar is the mean conditional error rate over seeds 77/10/14, the black
whisker is the min--max spread, and the dots are the individual seeds.

It also prints, per variant / seed / algorithm, the macro-F1 and the
victim/attacker recall, so the numbers in Table~\\ref{tab:group3_ranking} and the
G3 columns of Table~\\ref{tab:bias_summary} can be regenerated/verified.
"""
import sys
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_ranking_table import get_num_hosts  # noqa: E402
from generate_confusion_plots import (  # noqa: E402
    read_test_payload,
    compute_victim_as_attacker_stats,
    compute_attacker_as_victim_stats,
    algorithm_sort_key,
    normalize_algorithm_name,
)

# seed 77 dirs come from test_tools/compare_config_group3_resources.yaml;
# seeds 10/14 were run on 2026-09-07/08 for the thesis multi-seed pass.
VARIANTS = {
    "reward_scaling": {
        77: "_training/attacks_ho/20260818-122525_1_10_10",
        14: "_training/attacks_ho/20260908-114049_1_10_10",
        10: "_training/attacks_ho/20260908-122857_1_10_10",
        "title": "Reward Scaling Only",
    },
    "deltas": {
        77: "_training/attacks_ho/20260808-122406_1_10_10",
        14: "_training/attacks_ho/20260907-151359_1_10_10",
        10: "_training/attacks_ho/20260908-093609_1_10_10",
        "title": "Deltas",
    },
    "all_flags": {
        77: "_training/attacks_ho/20260819-123046_1_10_10",
        14: "_training/attacks_ho/20260907-120809_1_10_10",
        10: "_training/attacks_ho/20260908-103525_1_10_10",
        "title": "All Flags",
    },
}
OUT_DIR = Path("docs/papers/thesis/images")


def collect(seed_runs):
    """-> {algo: {'v2a': [per seed], 'a2v': [...], 'f1': [...], 'vrec': [...], 'arec': [...]}}"""
    per_algo = {}
    for seed, run in seed_runs.items():
        if not isinstance(seed, int):
            continue
        run_dir = Path(run)
        n_hosts = get_num_hosts(run_dir)
        for test_dir in sorted(glob.glob(str(run_dir / "TEST_*"))):
            test_dir = Path(test_dir)
            agent_hint = test_dir.name.replace("TEST_", "")
            algo = normalize_algorithm_name(agent_hint)
            df = read_test_payload(test_dir / "test.json", agent_hint, n_hosts)
            v2a, _, _ = compute_victim_as_attacker_stats(df)
            a2v, _, _ = compute_attacker_as_victim_stats(df)
            y_true = df["true_label"].to_numpy()
            y_pred = df["predicted_label"].to_numpy()
            f1 = f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)
            vrec = float(((y_true == 1) & (y_pred == 1)).sum()) / max(1, int((y_true == 1).sum()))
            arec = float(((y_true == 2) & (y_pred == 2)).sum()) / max(1, int((y_true == 2).sum()))
            d = per_algo.setdefault(algo, {k: [] for k in ("v2a", "a2v", "f1", "vrec", "arec")})
            d["v2a"].append(v2a); d["a2v"].append(a2v); d["f1"].append(f1)
            d["vrec"].append(vrec); d["arec"].append(arec)
    return per_algo


def plot_variant(variant_key, per_algo):
    algos = sorted(per_algo.keys(), key=algorithm_sort_key)
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    x = np.arange(len(algos), dtype=float)
    bw = 0.38

    def series(key, offset, color, label):
        means, lo_err, hi_err = [], [], []
        for a in algos:
            vals = np.array(per_algo[a][key], dtype=float)
            m = float(vals.mean())
            means.append(m); lo_err.append(m - float(vals.min())); hi_err.append(float(vals.max()) - m)
        ax.bar(x + offset, means, width=bw, color=color, edgecolor="black",
               linewidth=0.5, label=label, zorder=2)
        ax.errorbar(x + offset, means, yerr=[lo_err, hi_err], fmt="none",
                    ecolor="black", elinewidth=0.9, capsize=2.5, zorder=3)
        for i, a in enumerate(algos):
            vals = per_algo[a][key]
            ax.scatter([x[i] + offset] * len(vals), vals, s=9, color="black", zorder=4, alpha=0.75)
        return means

    m_v2a = series("v2a", -bw / 2.0, "coral", r"Victim$\to$Attacker")
    m_a2v = series("a2v", +bw / 2.0, "steelblue", r"Attacker$\to$Victim")
    ax.set_xticks(x); ax.set_xticklabels(algos, fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylabel("Conditional error rate", fontsize=9)
    ax.set_title(f"Role confusion, 20-node transfer ({VARIANTS[variant_key]['title']}), 3 seeds",
                 fontsize=9)
    top = max(a + b for a, b in zip(m_v2a, m_a2v))
    ax.set_ylim(0, min(1.0, top + 0.32))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_yticks(np.arange(0, 1.0001, 0.2))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=8,
              handlelength=1.2, columnspacing=1.0)
    out = OUT_DIR / f"group3_{variant_key}_role_confusion_rates.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    for vk, seed_runs in VARIANTS.items():
        per_algo = collect(seed_runs)
        plot_variant(vk, per_algo)
        print(f"\n=== {VARIANTS[vk]['title']} (seeds 77/14/10) ===")
        for a in sorted(per_algo.keys(), key=algorithm_sort_key):
            d = per_algo[a]
            f1 = np.array(d["f1"]); vr = np.array(d["vrec"]); ar = np.array(d["arec"])
            print(f"  {a:11s} F1 {f1.mean():.2f}+-{f1.std():.2f} {[round(v,2) for v in d['f1']]}"
                  f"  Vrec {vr.mean():.2f} {[round(v,2) for v in d['vrec']]}"
                  f"  Arec {ar.mean():.2f} {[round(v,2) for v in d['arec']]}")
        print()


if __name__ == "__main__":
    main()
