#!/usr/bin/env python3
"""Paper 8 (ICAART): Group-3 Reward-Scaling-Only role-confusion, multi-seed.

Regenerates images/group3_reward_scaling_role_confusion_rates.png as a 3-seed
version of the single-seed figure produced by generate_confusion_plots.py: for
each algorithm the bar is the mean conditional error rate over seeds 77/10/14,
the black whisker is the min--max spread, and the dots are the individual seeds.

Run dirs are the Group-3 (20-node) Reward-Scaling-Only transfer, one per seed:
  seed 77 -> _training/attacks_ho/20260818-122525_1_10_10  (the single-seed run
             used everywhere else in the paper)
  seed 14 -> _training/attacks_ho/20260908-114049_1_10_10
  seed 10 -> _training/attacks_ho/20260908-122857_1_10_10
"""
import sys
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_ranking_table import get_num_hosts  # noqa: E402
from generate_confusion_plots import (  # noqa: E402
    read_test_payload,
    compute_victim_as_attacker_stats,
    compute_attacker_as_victim_stats,
    algorithm_sort_key,
    normalize_algorithm_name,
)

RUNS = {
    77: "_training/attacks_ho/20260818-122525_1_10_10",
    14: "_training/attacks_ho/20260908-114049_1_10_10",
    10: "_training/attacks_ho/20260908-122857_1_10_10",
}
OUT = Path("docs/papers/8-draft of future work in Single_agent_Enhanced_Scenarios/images/group3_reward_scaling_role_confusion_rates.png")


def collect():
    """-> {algo: {'v2a': [rate per seed], 'a2v': [rate per seed]}}"""
    per_algo = {}
    for seed, run in RUNS.items():
        run_dir = Path(run)
        n_hosts = get_num_hosts(run_dir)
        for test_dir in sorted(glob.glob(str(run_dir / "TEST_*"))):
            test_dir = Path(test_dir)
            agent_hint = test_dir.name.replace("TEST_", "")
            algo = normalize_algorithm_name(agent_hint)
            df = read_test_payload(test_dir / "test.json", agent_hint, n_hosts)
            v2a, _, _ = compute_victim_as_attacker_stats(df)
            a2v, _, _ = compute_attacker_as_victim_stats(df)
            per_algo.setdefault(algo, {"v2a": [], "a2v": []})
            per_algo[algo]["v2a"].append(v2a)
            per_algo[algo]["a2v"].append(a2v)
    return per_algo


def main():
    per_algo = collect()
    algos = sorted(per_algo.keys(), key=algorithm_sort_key)

    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    x = np.arange(len(algos), dtype=float)
    bw = 0.38

    def series(key, offset, color, label):
        means, lo_err, hi_err = [], [], []
        for a in algos:
            vals = np.array(per_algo[a][key], dtype=float)
            m = float(vals.mean())
            means.append(m)
            lo_err.append(m - float(vals.min()))
            hi_err.append(float(vals.max()) - m)
        ax.bar(x + offset, means, width=bw, color=color, edgecolor="black",
               linewidth=0.5, label=label, zorder=2)
        ax.errorbar(x + offset, means, yerr=[lo_err, hi_err], fmt="none",
                    ecolor="black", elinewidth=0.9, capsize=2.5, zorder=3)
        for i, a in enumerate(algos):
            vals = per_algo[a][key]
            ax.scatter([x[i] + offset] * len(vals), vals, s=9, color="black",
                       zorder=4, alpha=0.75)
        return means

    m_v2a = series("v2a", -bw / 2.0, "coral", r"Victim$\to$Attacker")
    m_a2v = series("a2v", +bw / 2.0, "steelblue", r"Attacker$\to$Victim")

    ax.set_xticks(x)
    ax.set_xticklabels(algos, fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylabel("Conditional error rate", fontsize=9)
    ax.set_title("Role confusion, 20-node transfer (Reward Scaling Only), 3 seeds",
                 fontsize=9)
    top = max(m_v2a + m_a2v)
    ax.set_ylim(0, min(1.0, top + 0.32))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_yticks(np.arange(0, 1.0001, 0.2))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=8,
              handlelength=1.2, columnspacing=1.0)

    fig.savefig(OUT, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {OUT}")
    for a in algos:
        print(f"  {a:11s} V->A {np.mean(per_algo[a]['v2a']):.2f} "
              f"(seeds {[round(v,2) for v in per_algo[a]['v2a']]})  "
              f"A->V {np.mean(per_algo[a]['a2v']):.2f} "
              f"(seeds {[round(v,2) for v in per_algo[a]['a2v']]})")


if __name__ == "__main__":
    main()
