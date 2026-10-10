#!/usr/bin/env python3
"""Macro-F1 vs. msgs/step Pareto scatter for paper9 (todos.md A8).

One point per tested configuration -- S0, S1 at tau in {0.6, 0.8, 0.9}, and the
primary 2x4 Tier A x Tier B grid (S0/S1 x baseline/S2/S3/S4) -- for Q-Learning and
SARSA. Highlights the Pareto frontier: the configurations for which no other point
has both lower msgs/step and equal-or-higher macro-F1.

Hardcoded to the numbers already in experiments9.tex (Tables tab:s1_uaq,
tab:tau_sweep, tab:tier_b) rather than re-reading run directories, since this is a
companion visualization of those exact tables, not an independent computation --
keep in sync by hand if the table numbers change. Mean values only (no error bars):
a Pareto scatter is about the trade-off shape, not per-seed spread, which the
tables it accompanies already report.
"""
import matplotlib.pyplot as plt

# label -> (msgs/step, macro-F1)
DATA = {
    "Q-Learning": {
        "S0":            (5.31, 0.691),
        r"S1 ($\tau$=0.6)": (5.28, 0.656),
        r"S1 ($\tau$=0.8)": (4.57, 0.707),
        r"S1 ($\tau$=0.9)": (3.26, 0.706),
        "S0+S2":         (4.76, 0.748),
        "S0+S3":         (6.44, 0.597),
        "S0+S4":         (4.63, 0.737),
        "S1+S2":         (3.94, 0.722),
        "S1+S3":         (6.10, 0.636),
        "S1+S4":         (4.12, 0.687),
    },
    "SARSA": {
        "S0":            (10.38, 0.705),
        r"S1 ($\tau$=0.6)": (9.57, 0.656),
        r"S1 ($\tau$=0.8)": (8.64, 0.694),
        r"S1 ($\tau$=0.9)": (7.63, 0.741),
        "S0+S2":         (9.75, 0.769),
        "S0+S3":         (11.37, 0.635),
        "S0+S4":         (9.64, 0.763),
        "S1+S2":         (7.75, 0.792),
        "S1+S3":         (10.74, 0.638),
        "S1+S4":         (7.69, 0.723),
    },
}

ALGO_COLORS = {"Q-Learning": "#78b7d8", "SARSA": "#d98ca5"}


def pareto_frontier(points):
    """points: dict label -> (x, y). Maximize y, minimize x.
    Returns the set of labels not dominated by any other point."""
    labels = list(points.keys())
    frontier = set()
    for i, li in enumerate(labels):
        xi, yi = points[li]
        dominated = False
        for j, lj in enumerate(labels):
            if i == j:
                continue
            xj, yj = points[lj]
            if xj <= xi and yj >= yi and (xj < xi or yj > yi):
                dominated = True
                break
        if not dominated:
            frontier.add(li)
    return frontier


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    for ax, algo in zip(axes, ["Q-Learning", "SARSA"]):
        points = DATA[algo]
        frontier = pareto_frontier(points)
        color = ALGO_COLORS[algo]

        for label, (x, y) in points.items():
            on_frontier = label in frontier
            ax.scatter(x, y,
                       s=90 if on_frontier else 55,
                       facecolor=color if on_frontier else "white",
                       edgecolor=color, linewidth=1.6, zorder=3)
            dx = 0.08 if algo == "Q-Learning" else 0.15
            ax.annotate(label, (x, y), xytext=(x + dx, y),
                       fontsize=8, va="center",
                       fontweight="bold" if on_frontier else "normal")

        frontier_pts = sorted((points[l] for l in frontier), key=lambda p: p[0])
        fx = [p[0] for p in frontier_pts]
        fy = [p[1] for p in frontier_pts]
        ax.plot(fx, fy, "--", color=color, alpha=0.6, linewidth=1.2, zorder=2)

        ax.set_xlabel("Msgs/step (lower is better)")
        ax.set_title(algo, fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.margins(x=0.25)

    axes[0].set_ylabel("Macro-F1 (higher is better)")
    fig.suptitle(
        "Macro-F1 vs. communication overhead: filled markers + dashed line = Pareto frontier",
        fontsize=12,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.93))
    out_path = ("docs/papers/9-draft of future work in "
                "MARL_Coordination__Strategies/images/pareto_f1_msgs.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
