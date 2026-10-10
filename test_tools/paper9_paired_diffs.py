#!/usr/bin/env python3
"""Per-seed differences between paired configurations for paper9 (Appendix app:paired).

For each primary comparison (a treatment cell vs its baseline cell), prints the
macro-F1 difference for each of the three seeds (10, 14, 77), their mean, and how many
of the three seeds have the same sign as the mean. With n=3 this is a descriptive
consistency check, not a significance test. Caveat: the seed-77 runs predate the
2026-09-04 RNG-seed fix (see paper9_multiseed.py), so seed 77 is a matched label, not
a guaranteed identical scenario draw.

Usage (repo root): python3 test_tools/paper9_paired_diffs.py [--latex]
"""
import sys
import numpy as np
sys.path.insert(0, "test_tools")
from paper9_multiseed import REGISTRY, cell_summary, QL, SARSA

PAIRS = [
    ("S1 vs S0 (UAQ, $\\tau{=}0.8$)", "s1", "s0"),
    ("S1 $\\tau{=}0.9$ vs $\\tau{=}0.8$", "s1_tau09", "s1"),
    ("S0+S2 vs S0", "s0_s2", "s0"), ("S0+S3 vs S0", "s0_s3", "s0"), ("S0+S4 vs S0", "s0_s4", "s0"),
    ("S1+S2 vs S1", "s1_s2", "s1"), ("S1+S3 vs S1", "s1_s3", "s1"), ("S1+S4 vs S1", "s1_s4", "s1"),
]
SEEDS = [10, 14, 77]


def diffs(t, b, agent):
    ct, cb = cell_summary(REGISTRY[t], agent), cell_summary(REGISTRY[b], agent)
    return [ct["per_seed"][s]["macro"] - cb["per_seed"][s]["macro"] for s in SEEDS]


def main():
    latex = "--latex" in sys.argv
    rows = []
    for label, t, b in PAIRS:
        cells = []
        for agent in (QL, SARSA):
            d = diffs(t, b, agent)
            m = float(np.mean(d))
            same = sum(1 for x in d if np.sign(x) == np.sign(m) and x != 0)
            cells.append((d, m, same))
        rows.append((label, cells))
    if latex:
        for label, cells in rows:
            parts = []
            for d, m, same in cells:
                parts.append(" / ".join(f"{x:+.3f}" for x in d) + f" & {m:+.3f} & {same}/3")
            print(f"{label} & " + " & ".join(parts) + " \\\\")
    else:
        for label, cells in rows:
            print(label, [(["%+.3f" % x for x in d], "%+.3f" % m, same) for d, m, same in cells])


if __name__ == "__main__":
    main()
