#!/usr/bin/env python3
"""Tables for the Paper 9 re-run (after the marl_pz fixes, todos.md A24).

Replaces paper9_multiseed.py (hand-kept REGISTRY of the invalid 3-seed campaign) and
security_metrics_table_multiseed.py for the new runs. Run folders come from
paper9_rerun_registry.discover(), so re-running this script after more batches finish
updates every number without editing anything.

For every cell with at least one finished seed, and for Q-Learning, SARSA and DQN:
  * test macro-F1 (mean +/- sample s.d. over seeds) and per-class recall;
  * paired per-seed difference against the independent-learner baseline (`none`), with
    the number of seeds where the difference is positive;
  * alerts during training (comm_stats.json: mean alerts per host per training episode,
    the unit of the old "msgs/step" column, see A24) and at test time
    (test_comm_stats.msgs_per_round: alerts per test round, summed over hosts);
  * the operational IPS metrics of security_metrics_table.collect(), computed per seed
    and then aggregated (never pooled across seeds).

Writes LaTeX snippets and a JSON dump to --output-dir; prints a plain-text report.

Usage (repo root):
    python3 test_tools/paper9_rerun_tables.py --since 20260923 --output-dir /tmp/p9_rerun
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paper9_rerun_registry import BASE, discover  # noqa: E402
from paper9_multiseed import run_metrics  # noqa: E402
from security_metrics_table import COLS, collect  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

# run-directory agent name -> (paper label, name used by security_metrics_table.collect)
AGENTS = {
    "Q-learning_quick": ("Q-Learning", "Q-Learning"),
    "Sarsa_balanced": ("SARSA", "SARSA"),
    "DQN_quick": ("DQN", "DQN"),
}
BASELINE = "none"
# cell id -> paper label, in table order
CELLS = {
    "none": "Independent",
    "s0": "S0",
    "s1": "S1 (UAQ)",
    "s1_tau06": r"S1 ($\tau{=}0.6$)",
    "s1_tau09": r"S1 ($\tau{=}0.9$)",
    "none_s2": "Ind.+S2",
    "none_s3": "Ind.+S3",
    "none_s4": "Ind.+S4",
    "s0_s2": "S0+S2",
    "s0_s3": "S0+S3",
    "s0_s4": "S0+S4",
    "s1_s2": "S1+S2",
    "s1_s3": "S1+S3",
    "s1_s4": "S1+S4",
    "s0_s2_s4": "S0+S2+S4",
    "s1_s2_s4": "S1+S2+S4",
    "s1_s2_s3": "S1+S2+S3",
    "s0_s3_s4": "S0+S3+S4",
}


def _mean_sd(vals):
    a = np.asarray([v for v in vals if v is not None and not np.isnan(v)], dtype=float)
    if not a.size:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0


def _test_msgs(run: str, agent: str):
    tj = f"{BASE}/{run}/TEST_{agent}/test.json"
    if not os.path.isfile(tj):
        return None
    st = json.load(open(tj)).get("test_comm_stats") or {}
    return st.get("msgs_per_round")


def _alert_split(run: str, agent: str) -> dict:
    """UAQ split of the alerts: confident (read by the coordinator) and uncertain
    (sent but ignored), during training (per host per episode, comm_stats.json) and
    at test time (totals over the test rounds, test_comm_stats). NaN when absent."""
    out = {k: float("nan") for k in ("train_conf", "train_unc", "test_total", "test_conf", "test_unc")}
    cs = f"{BASE}/{run}/{agent}/comm_stats.json"
    if os.path.isfile(cs):
        c = json.load(open(cs))
        cols, rows = c.get("alert_columns", []), c.get("alert_rows", [])
        for name, key in (("confident", "train_conf"), ("uncertain", "train_unc")):
            if rows and name in cols:
                i = cols.index(name)
                out[key] = sum(r[i] for r in rows) / len(rows)
    tj = f"{BASE}/{run}/TEST_{agent}/test.json"
    if os.path.isfile(tj):
        st = json.load(open(tj)).get("test_comm_stats") or {}
        for name, key in (("alerts_total", "test_total"), ("confident", "test_conf"), ("uncertain", "test_unc")):
            if st.get(name) is not None:
                out[key] = float(st[name])
    return out


def summarize(reg: dict) -> dict:
    """{cell: {agent_label: summary}} for every cell/agent with at least one seed."""
    base_runs = reg.get(BASELINE, {})
    out = {}
    for cell in CELLS:
        runs = reg.get(cell, {})
        if not runs:
            continue
        out[cell] = {}
        for agent, (label, ips_name) in AGENTS.items():
            per_seed = {}
            for seed, run in sorted(runs.items()):
                m = run_metrics(run, agent)
                if m is None:
                    continue
                m["test_msgs"] = _test_msgs(run, agent)
                m.update(_alert_split(run, agent))
                d = json.load(open(f"{BASE}/{run}/TEST_{agent}/test.json"))
                m["f1_class"] = f1_score(d["ground_truth"], d["predicted"][agent], labels=[0, 1, 2],
                                         average=None, zero_division=0).tolist()
                m["ips"] = collect(f"{BASE}/{run}", excluded=set()).get(ips_name)
                per_seed[seed] = m
            if not per_seed:
                continue
            diffs = {}
            if cell != BASELINE:
                for seed, m in per_seed.items():
                    b = run_metrics(base_runs[seed], agent) if seed in base_runs else None
                    if b is not None:
                        diffs[seed] = m["macro"] - b["macro"]
            rec = np.array([m["recall"] for m in per_seed.values()], dtype=float)
            ips = {}
            for key, _ in COLS:
                ips[key] = _mean_sd([m["ips"][key] for m in per_seed.values() if m["ips"]])
            diffs_na = {}
            if cell != BASELINE:
                for seed, m in per_seed.items():
                    b = run_metrics(base_runs[seed], agent) if seed in base_runs else None
                    if b is not None:
                        diffs_na[seed] = m["macro_na"] - b["macro_na"]
            out[cell][label] = dict(
                seeds=sorted(per_seed),
                macro=_mean_sd([m["macro"] for m in per_seed.values()]),
                macro_na=_mean_sd([m["macro_na"] for m in per_seed.values()]),
                diff_na=_mean_sd(list(diffs_na.values())) if diffs_na else None,
                diff_na_negative=sum(d < 0 for d in diffs_na.values()),
                macro_per_seed={s: m["macro"] for s, m in per_seed.items()},
                recall=rec.mean(axis=0).tolist(),
                f1_class=np.array([m["f1_class"] for m in per_seed.values()]).mean(axis=0).tolist(),
                train_alerts=_mean_sd([m["msg"] for m in per_seed.values()]),
                **{k: _mean_sd([m[k] for m in per_seed.values()])
                   for k in ("train_conf", "train_unc", "test_total", "test_conf", "test_unc")},
                test_msgs=_mean_sd([m["test_msgs"] for m in per_seed.values()]),
                diff=_mean_sd(list(diffs.values())) if diffs else None,
                diff_per_seed=diffs,
                diff_positive=sum(d > 0 for d in diffs.values()),
                ips=ips,
            )
    return out


def _f(ms, nd=3):
    m, s = ms
    if np.isnan(m):
        return "--"
    return f"{m:.{nd}f}" if s == 0 or np.isnan(s) else rf"${m:.{nd}f}\pm{s:.{nd}f}$"


def print_report(summary: dict) -> None:
    for cell, by_agent in summary.items():
        print(f"\n{CELLS[cell]}")
        for label, s in by_agent.items():
            line = (f"  {label:10s} n={len(s['seeds'])} F1 {s['macro'][0]:.3f}+/-{s['macro'][1]:.3f}"
                    f"  rec n/v/a {s['recall'][0]:.2f}/{s['recall'][1]:.2f}/{s['recall'][2]:.2f}"
                    f"  alerts train {s['train_alerts'][0]:.2f} test/round {s['test_msgs'][0]:.2f}")
            if s["diff"]:
                line += (f"  vs ind. {s['diff'][0]:+.3f} ({s['diff_positive']}/"
                         f"{len(s['diff_per_seed'])} seeds positive)")
            print(line)
            print("             per seed: " + " ".join(
                f"{k}:{v:.3f}" for k, v in s["macro_per_seed"].items()))


def latex_main(summary: dict) -> str:
    agents = ["Q-Learning", "SARSA"]
    lines = [
        "% generated by test_tools/paper9_rerun_tables.py -- do not edit by hand",
        r"\begin{tabular}{@{}l" + "cc" * len(agents) + "c@{}}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Q-Learning} & \multicolumn{2}{c}{SARSA} & \\",
        r"Cell & Macro-F1 & $\Delta$ vs Ind. & Macro-F1 & $\Delta$ vs Ind. & Seeds \\",
        r"\midrule",
    ]
    for cell, by_agent in summary.items():
        cells = []
        n = 0
        for a in agents:
            s = by_agent.get(a)
            if s is None:
                cells += ["--", "--"]
                continue
            n = max(n, len(s["seeds"]))
            cells.append(_f(s["macro"]))
            cells.append("--" if not s["diff"] else
                         rf"${s['diff'][0]:+.3f}$ ({s['diff_positive']}/{len(s['diff_per_seed'])})")
        lines.append(f"{CELLS[cell]} & " + " & ".join(cells) + f" & {n} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def latex_baselines(summary: dict, cells=("none", "s0")) -> str:
    """Independent learners vs S0: macro-F1 (with and without VICTIM), recalls, alerts.

    Delta columns: mean difference from independent learners and the number of seeds
    in which the cell is lower.
    """
    lines = [
        "% generated by test_tools/paper9_rerun_tables.py -- do not edit by hand",
        r"\begin{tabular}{@{}llcccccc@{}}",
        r"\toprule",
        r"Cell & Alg. & Macro-F1 & $\Delta$ (lower) & F1 \textsc{n}+\textsc{a} & $\Delta$ (lower) "
        r"& F1 \textsc{v} / \textsc{a} & Alerts tr./test \\",
        r"\midrule",
    ]
    for cell in cells:
        for a in ("Q-Learning", "SARSA"):
            s = summary.get(cell, {}).get(a)
            if s is None:
                continue
            d = dn = "--"
            if s["diff"]:
                n = len(s["diff_per_seed"])
                d = rf"${s['diff'][0]:+.3f}$ ({n - s['diff_positive']}/{n})"
                dn = rf"${s['diff_na'][0]:+.3f}$ ({s['diff_na_negative']}/{n})"
            tr = "--" if np.isnan(s["train_alerts"][0]) else f"{s['train_alerts'][0]:.2f}"
            lines.append(
                f"{CELLS[cell]} & {a} & {_f(s['macro'])} & {d} & {_f(s['macro_na'])} & {dn} & "
                f"{s['f1_class'][1]:.2f} / {s['f1_class'][2]:.2f} & {tr} / {s['test_msgs'][0]:.2f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def latex_tier_a(summary: dict, cells=("s0", "s1")) -> str:
    """S0 vs UAQ: macro-F1 with its differences from S0 and from independent learners,
    and the alerts sent (UAQ split into confident / uncertain)."""
    lines = [
        "% generated by test_tools/paper9_rerun_tables.py -- do not edit by hand",
        r"\begin{tabular}{@{}llccccc@{}}",
        r"\toprule",
        r"Cell & Alg. & Macro-F1 & $\Delta$ vs S0 (higher) & $\Delta$ vs Ind.\ (lower) "
        r"& Alerts train: total (conf./unc.) & Alerts test: total (conf./unc.) \\",
        r"\midrule",
    ]
    for a in ("Q-Learning", "SARSA"):
        s0 = summary.get("s0", {}).get(a)
        for cell in cells:
            s = summary.get(cell, {}).get(a)
            if s is None:
                continue
            d_s0 = "--"
            if cell != "s0" and s0 is not None:
                diffs = [s["macro_per_seed"][k] - s0["macro_per_seed"][k]
                         for k in s["macro_per_seed"] if k in s0["macro_per_seed"]]
                if diffs:
                    d_s0 = rf"${np.mean(diffs):+.3f}$ ({sum(d > 0 for d in diffs)}/{len(diffs)})"
            d_ind = "--"
            if s["diff"]:
                n = len(s["diff_per_seed"])
                d_ind = rf"${s['diff'][0]:+.3f}$ ({n - s['diff_positive']}/{n})"
            tr = f"{s['train_alerts'][0]:.2f}"
            te = f"{s['test_total'][0]:.0f}"
            if cell.startswith("s1") and not np.isnan(s["train_conf"][0]):
                tr += f" ({s['train_conf'][0]:.2f} / {s['train_unc'][0]:.2f})"
                te += f" ({s['test_conf'][0]:.0f} / {s['test_unc'][0]:.0f})"
            lines.append(f"{CELLS[cell]} & {a} & {_f(s['macro'])} & {d_s0} & {d_ind} & {tr} & {te} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def latex_paired(summary: dict, seeds=(10, 14, 77, 2015, 2019)) -> str:
    """Appendix table: per-seed macro-F1 difference of every cell from independent learners.

    'same sign' counts the seeds whose difference has the sign of the mean (a consistency
    check, not a significance test).
    """
    head = "/".join(str(s) for s in seeds)
    lines = [
        "% generated by test_tools/paper9_rerun_tables.py -- do not edit by hand",
        r"\begin{tabular}{@{}lcccccc@{}}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Q-Learning} & \multicolumn{3}{c}{SARSA} \\",
        rf"Cell vs Ind. & seeds {head} & mean & same sign & seeds {head} & mean & same sign \\",
        r"\midrule",
    ]
    for cell, by_agent in summary.items():
        if cell == BASELINE:
            continue
        cols = []
        for a in ("Q-Learning", "SARSA"):
            s = by_agent.get(a)
            if s is None or not s["diff_per_seed"]:
                cols += ["--", "--", "--"]
                continue
            per = " / ".join(f"{s['diff_per_seed'][k]:+.3f}" if k in s["diff_per_seed"] else "--"
                             for k in seeds)
            mean = s["diff"][0]
            same = sum(np.sign(d) == np.sign(mean) for d in s["diff_per_seed"].values())
            cols += [per, f"{mean:+.3f}", f"{same}/{len(s['diff_per_seed'])}"]
        lines.append(f"{CELLS[cell]} & " + " & ".join(cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def latex_ips(summary: dict) -> str:
    lines = [
        "% generated by test_tools/paper9_rerun_tables.py -- do not edit by hand",
        r"\begin{tabular}{@{}ll" + "c" * len(COLS) + r"@{}}",
        r"\toprule",
        "Cell & Alg. & " + " & ".join(c[1] for c in COLS) + r" \\",
        r"\midrule",
    ]
    for cell, by_agent in summary.items():
        for a in ("Q-Learning", "SARSA"):
            s = by_agent.get(a)
            if s is None:
                continue
            vals = " & ".join(_f(s["ips"][k], nd=1 if k == "median_ttm" else 3) for k, _ in COLS)
            lines.append(f"{CELLS[cell]} & {a} & {vals} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="20260923", help="YYYYMMDD: first day of the re-run")
    ap.add_argument("--output-dir", default="/tmp/p9_rerun")
    args = ap.parse_args()

    summary = summarize(discover(args.since))
    print_report(summary)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rerun_macro_f1.tex").write_text(latex_main(summary))
    (out / "rerun_ips.tex").write_text(latex_ips(summary))
    (out / "rerun_baselines.tex").write_text(latex_baselines(summary))
    (out / "rerun_paired.tex").write_text(latex_paired(summary))
    (out / "rerun_tier_a.tex").write_text(latex_tier_a(summary))
    (out / "rerun_summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    print(f"\nwrote {out}/rerun_macro_f1.tex, rerun_ips.tex, rerun_summary.json")


if __name__ == "__main__":
    main()
