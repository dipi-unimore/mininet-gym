#!/usr/bin/env python3
"""Paper 9 multi-seed results pipeline.

Single source of truth for the cell -> run-directory matrix of the §B multi-seed
campaign, plus reusable computation (macro-F1 recomputed from raw
ground_truth/predicted, F1 on the normal+attacker sub-problem, per-class recall,
msgs/step, victim ground-truth count) and output (a LaTeX summary table and a
mean +/- sd figure with per-seed dots), for any subset of cells that has runs.

Cells that do not yet have every seed are reported anyway, marked preliminary.

Seeds 77, 10, 14 are pooled as one equivalent 3-seed set for QL and SARSA (user
decision 2026-09-10, see memory project-paper9-multiseed-pooling), even though the
seed-77 runs predate the Aug-19 eval fixes and the Sep-4 RNG-seed fix. DQN comes
from the seed-77 runs only and is reported as a bounded failure analysis, kept out
of the seed-validated table and figure.

Usage
-----
  python3 test_tools/paper9_multiseed.py --cells s0 s1
  python3 test_tools/paper9_multiseed.py --cells s0 s1 --out-dir docs/papers/9-.../
  python3 test_tools/paper9_multiseed.py --list

Extend REGISTRY as campaign runs land; keep RUN_MATRIX.md / ch08_experiment_log.md
in sync.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

BASE = "_training/marl_pz"
QL = "Q-learning_quick"
SARSA = "Sarsa_balanced"
DQN = "DQN_quick"
CLASSES = [0, 1, 2]            # normal, victim, attacker
CLASS_NAMES = ["normal", "victim", "attacker"]

# The seed-77 grid runs (2026-08-10 -> 2026-09-07) predate three code changes the
# seed-10/14 runs have: b77c9e6 "fix bug seed" (2026-09-04, np.random/torch now
# seeded), fe30f03 "no reset link every episodes" (2026-08-19, continuous-trace
# eval), c9c854f "incoming attack state one step before" (2026-08-19). USER
# DECISION 2026-09-10: pool all three seeds as equivalent for QL & SARSA; do not
# down-weight seed 77. Kept here only as a provenance note in verbose output.
# See memory project-paper9-multiseed-pooling.
SEED77_PREFIX_RUNS = {
    "20260812-110417_1_5_5", "20260813-091949_1_5_5", "20260814-100046_1_5_5",
    "20260814-183144_1_5_5", "20260815-104903_1_5_5", "20260815-181031_1_5_5",
    "20260816-174823_1_5_5", "20260817-091109_1_5_5", "20260818-150129_1_5_5",
    "20260819-135609_1_5_5", "20260822-162154_1_5_5",
}


@dataclass
class Cell:
    label: str            # paper label, e.g. "S1+S2"
    tier_a: str           # "S0" | "S1"
    tier_b: str           # "-", "S2", "S3", "S4", "S2+S3", ...
    seeds: dict = field(default_factory=dict)   # seed -> run dir (no _1_5_5 needed)


# --- cell -> run matrix -------------------------------------------------------
# Only list what exists on disk. Seed 77 baselines/combos are the pre-fix paper
# runs, pooled as an equivalent third seed (user decision, see memory
# project-paper9-multiseed-pooling) rather than re-run.
REGISTRY: dict[str, Cell] = {
    # --- Tier 1 spine (todos.md §B) ---
    "s0": Cell("S0", "S0", "-", {
        77: "20260812-110417_1_5_5",
        10: "20260908-140651_1_5_5",
        14: "20260909-084902_1_5_5",
    }),
    "s1": Cell("S1", "S1", "-", {
        77: "20260815-181031_1_5_5",
        10: "20260909-132530_1_5_5",
        14: "20260909-192933_1_5_5",
    }),
    "s0_s2": Cell("S0+S2", "S0", "S2", {
        77: "20260813-091949_1_5_5",
        10: "20260912-114224_1_5_5",
        14: "20260912-154105_1_5_5",
    }),
    "s1_s2": Cell("S1+S2", "S1", "S2", {
        77: "20260816-174823_1_5_5",
        10: "20260913-113106_1_5_5",
        14: "20260913-152901_1_5_5",
    }),
    "s0_s3": Cell("S0+S3", "S0", "S3", {
        77: "20260814-100046_1_5_5",
        10: "20260912-193901_1_5_5",
        14: "20260912-233716_1_5_5",
    }),
    "s1_s3": Cell("S1+S3", "S1", "S3", {
        77: "20260817-091109_1_5_5",
        10: "20260913-192637_1_5_5",
        14: "20260913-232522_1_5_5",
    }),
    "s0_s4": Cell("S0+S4", "S0", "S4", {
        77: "20260814-183144_1_5_5",
        10: "20260913-033549_1_5_5",
        14: "20260913-073328_1_5_5",
    }),
    "s1_s4": Cell("S1+S4", "S1", "S4", {
        77: "20260818-150129_1_5_5",
        10: "20260914-032346_1_5_5",
        14: "20260914-072133_1_5_5",
    }),

    # --- Tier 2 (todos.md §B) ---
    "s0_s2_s4": Cell("S0+S2+S4", "S0", "S2+S4", {
        77: "20260815-104903_1_5_5",   # pre-fix, warm-start (original secondary probe)
        10: "20260910-085617_1_5_5",
        14: "20260910-160251_1_5_5",
    }),
    "s1_s2_s4": Cell("S1+S2+S4", "S1", "S2+S4", {
        77: "20260819-135609_1_5_5",   # pre-fix, warm-start (original secondary probe)
        10: "20260911-194921_1_5_5",
        14: "20260911-234714_1_5_5",
    }),
    "s1_tau09": Cell("S1 (τ=0.9)", "S1", "-", {
        77: "20260822-162154_1_5_5",
        10: "20260912-034456_1_5_5",
        14: "20260912-074320_1_5_5",
    }),
    # SARSA coverage of the two Round-4 combination probes (ch08_experiment_log.md).
    # Fresh matched cold-start QL+SARSA runs at seeds 10/14/77, kept separate from
    # Round 4's QL-only seed set (mixed cold/warm, more seeds) rather than merged
    # one-run-per-seed -- see ch08_experiment_log.md for the full QL-only picture.
    "s1_s2_s3": Cell("S1+S2+S3", "S1", "S2+S3", {
        77: "20260911-155013_1_5_5",
        10: "20260911-075341_1_5_5",
        14: "20260911-115156_1_5_5",
    }),
    "s0_s3_s4": Cell("S0+S3+S4", "S0", "S3+S4", {
        77: "20260911-035608_1_5_5",
        10: "20260910-200102_1_5_5",
        14: "20260910-235837_1_5_5",
    }),
}


def _test_json(run: str, agent: str) -> str:
    run = run if run.endswith("_1_5_5") else f"{run}_1_5_5"
    return f"{BASE}/{run}/TEST_{agent}/test.json"


def _comm_json(run: str, agent: str) -> str:
    run = run if run.endswith("_1_5_5") else f"{run}_1_5_5"
    return f"{BASE}/{run}/{agent}/comm_stats.json"


def run_metrics(run: str, agent: str) -> dict | None:
    tj = _test_json(run, agent)
    if not os.path.exists(tj):
        return None
    d = json.load(open(tj))
    if agent not in d.get("predicted", {}):
        return None
    gt = np.asarray(d["ground_truth"], dtype=int)
    pr = np.asarray(d["predicted"][agent], dtype=int)
    macro = f1_score(gt, pr, average="macro", labels=CLASSES, zero_division=0)
    macro_na = f1_score(gt, pr, average="macro", labels=[0, 2], zero_division=0)
    _, rec, _, sup = precision_recall_fscore_support(
        gt, pr, labels=CLASSES, zero_division=0)
    msg = None
    cj = _comm_json(run, agent)
    if os.path.exists(cj):
        c = json.load(open(cj))
        rows, cols = c.get("alert_rows", []), c.get("alert_columns", [])
        if rows and "total" in cols:
            ti = cols.index("total")
            msg = sum(r[ti] for r in rows) / len(rows)
    return dict(macro=macro, macro_na=macro_na, recall=list(rec),
               support=list(sup), msg=msg)


def cell_summary(cell: Cell, agent: str) -> dict:
    per_seed = {}
    for seed, run in sorted(cell.seeds.items()):
        m = run_metrics(run, agent)
        if m is not None:
            per_seed[seed] = dict(m, run=run,
                                  s77_prefix=run in SEED77_PREFIX_RUNS)
    macros = np.array([v["macro"] for v in per_seed.values()])
    macros_na = np.array([v["macro_na"] for v in per_seed.values()])
    msgs = np.array([v["msg"] for v in per_seed.values()
                     if v["msg"] is not None])
    n = len(per_seed)
    return dict(
        per_seed=per_seed, n=n,
        macro_mean=float(macros.mean()) if n else float("nan"),
        macro_sd=float(macros.std(ddof=1)) if n > 1 else 0.0,
        macro_na_mean=float(macros_na.mean()) if n else float("nan"),
        macro_na_sd=float(macros_na.std(ddof=1)) if n > 1 else 0.0,
        msg_mean=float(msgs.mean()) if len(msgs) else float("nan"),
        msg_sd=float(msgs.std(ddof=1)) if len(msgs) > 1 else 0.0,
    )


def print_report(cell_ids: list[str]) -> dict:
    out = {}
    for cid in cell_ids:
        cell = REGISTRY[cid]
        print(f"\n=== {cell.label}  (Tier A {cell.tier_a}, Tier B {cell.tier_b}) ===")
        out[cid] = {}
        # DQN is seed-77-only by design (bounded failure analysis, todos.md A9);
        # shown for completeness, kept out of the seed-validated table/figure.
        for agent, tag in [(QL, "Q-Learning"), (SARSA, "SARSA"), (DQN, "DQN")]:
            s = cell_summary(cell, agent)
            out[cid][agent] = s
            if s["n"] == 0:
                if agent != DQN:
                    print(f"  {tag:11}: no runs")
                continue
            note = "  (seed-77 only, failure analysis)" if agent == DQN else ""
            print(f"  {tag:11}: macro-F1 {s['macro_mean']:.3f} "
                  f"+/- {s['macro_sd']:.3f}  (n={s['n']})   "
                  f"F1(norm+att) {s['macro_na_mean']:.3f} +/- {s['macro_na_sd']:.3f}   "
                  f"msgs/step {s['msg_mean']:.2f}{note}")
            for seed, v in s["per_seed"].items():
                flag = "  (s77 pre-fix run)" if v["s77_prefix"] else ""
                print(f"      seed {seed:<3} macroF1={v['macro']:.3f}  "
                      f"F1(n+a)={v['macro_na']:.3f}  "
                      f"recall(n/v/a)={v['recall'][0]:.2f}/{v['recall'][1]:.2f}/"
                      f"{v['recall'][2]:.2f}  gtVic={v['support'][1]}  "
                      f"msg/s={v['msg']:.2f}{flag}")
            if agent != DQN and s["n"] < 3:
                print(f"      -> PRELIMINARY: {s['n']}/3 seeds")
    return out


def emit_table(cell_ids: list[str], path: str) -> None:
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Multi-seed Tier 1 spine, macro-F1 recomputed from raw "
        r"predictions (seeds 10/14/77; $\pm$ sample s.d.). Cells with fewer "
        r"than three seeds are marked preliminary.}",
        r"\label{tab:multiseed_tier1}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
        r"\textbf{Tier A} & \textbf{Tier B} & \textbf{QL F1} & "
        r"\textbf{SARSA F1} & \textbf{QL msg/s} & \textbf{SARSA msg/s} \\",
        r"\midrule",
    ]

    def fmt(s):
        if s["n"] == 0:
            return "--"
        body = f"{s['macro_mean']:.3f}"
        if s["n"] > 1:
            body += rf" $\pm$ {s['macro_sd']:.3f}"
        if s["n"] < 3:
            body += rf"\textsuperscript{{\dag}}"
        return body

    def fmt_msg(s):
        if not np.isfinite(s["msg_mean"]):
            return "--"
        return f"{s['msg_mean']:.2f}"

    last_a = None
    for cid in cell_ids:
        cell = REGISTRY[cid]
        ql = cell_summary(cell, QL)
        sa = cell_summary(cell, SARSA)
        a = cell.tier_a if cell.tier_a != last_a else ""
        last_a = cell.tier_a
        tb = "--- (baseline)" if cell.tier_b == "-" else cell.tier_b
        lines.append(f"        {a} & {tb} & {fmt(ql)} & {fmt(sa)} & "
                     f"{fmt_msg(ql)} & {fmt_msg(sa)} \\\\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\\[2pt]\footnotesize\textsuperscript{\dag}\,preliminary "
        r"($<3$ seeds available).",
        r"\end{table}", "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nwrote {path}")


def emit_figure(cell_ids: list[str], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, ql_m, ql_sd, sa_m, sa_sd = [], [], [], [], []
    ql_pts, sa_pts = [], []
    for cid in cell_ids:
        cell = REGISTRY[cid]
        ql = cell_summary(cell, QL)
        sa = cell_summary(cell, SARSA)
        labels.append(cell.label)
        ql_m.append(ql["macro_mean"]); ql_sd.append(ql["macro_sd"])
        sa_m.append(sa["macro_mean"]); sa_sd.append(sa["macro_sd"])
        ql_pts.append([v["macro"] for v in ql["per_seed"].values()])
        sa_pts.append([v["macro"] for v in sa["per_seed"].values()])

    x = np.arange(len(labels), dtype=float)
    w = 0.38
    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 3, 4.4))
    ax.bar(x - w / 2, ql_m, w, yerr=ql_sd, capsize=4, color="#78b7d8",
           edgecolor="black", linewidth=0.6, label="Q-Learning")
    ax.bar(x + w / 2, sa_m, w, yerr=sa_sd, capsize=4, color="#d98ca5",
           edgecolor="black", linewidth=0.6, label="SARSA")
    rng = np.random.RandomState(0)
    for xi, pts in zip(x - w / 2, ql_pts):
        ax.scatter(xi + (rng.rand(len(pts)) - 0.5) * w * 0.7, pts, s=18,
                   color="#1f3b52", zorder=5, edgecolor="white", linewidth=0.4)
    for xi, pts in zip(x + w / 2, sa_pts):
        ax.scatter(xi + (rng.rand(len(pts)) - 0.5) * w * 0.7, pts, s=18,
                   color="#6b2b45", zorder=5, edgecolor="white", linewidth=0.4)
    for xi, m, sd in zip(x - w / 2, ql_m, ql_sd):
        ax.text(xi, m + sd + 0.02, f"{m:.2f}", ha="center", fontsize=8)
    for xi, m, sd in zip(x + w / 2, sa_m, sa_sd):
        ax.text(xi, m + sd + 0.02, f"{m:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Macro-F1"); ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Multi-seed macro-F1 (mean \u00b1 s.d.; dots = individual seeds)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=None,
                    help="cell ids from the registry (default: all)")
    ap.add_argument("--out-dir", default=None,
                    help="if set, write tier_1_multiseed.{tex,png} there "
                         "(expects images/ and tables/ subdirs)")
    ap.add_argument("--list", action="store_true", help="list registry and exit")
    args = ap.parse_args()

    if args.list:
        for cid, c in REGISTRY.items():
            seeds = ", ".join(str(s) for s in sorted(c.seeds))
            print(f"  {cid:8} {c.label:8} seeds: {seeds}")
        return

    cell_ids = args.cells or list(REGISTRY)
    unknown = [c for c in cell_ids if c not in REGISTRY]
    if unknown:
        ap.error(f"unknown cells: {unknown}")

    report = print_report(cell_ids)

    if args.out_dir:
        od = args.out_dir.rstrip("/")
        emit_table(cell_ids, f"{od}/tables/tier_1_multiseed.tex")
        emit_figure(cell_ids, f"{od}/images/tier_1_multiseed.png")
    else:
        os.makedirs("/tmp/p9ms", exist_ok=True)
        emit_table(cell_ids, "/tmp/p9ms/tier_1_multiseed.tex")
        emit_figure(cell_ids, "/tmp/p9ms/tier_1_multiseed.png")

    # headline deltas vs the single-seed table currently in the paper
    print("\n--- vs. single-seed paper numbers ---")
    paper = {"s0": (0.635, 0.648), "s1": (0.600, 0.661)}
    for cid, (pq, ps) in paper.items():
        if cid not in report:
            continue
        q = report[cid][QL]["macro_mean"]
        s = report[cid][SARSA]["macro_mean"]
        print(f"  {REGISTRY[cid].label}: QL {pq:.3f} -> {q:.3f} "
              f"({q - pq:+.3f}) | SARSA {ps:.3f} -> {s:.3f} ({s - ps:+.3f})")


if __name__ == "__main__":
    main()
