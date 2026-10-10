#!/usr/bin/env python3
"""Sanity checks on finished runs of the Paper 9 re-run (after the marl_pz fixes).

For every complete N=10 run found since --since (skips the no-communication cells):
  1. the saved host Q-table must have visited states with the message feature in a bin
     above 0 (before the fix it was always bin 0: the hosts could not see the broadcast);
  2. TEST_<agent>/test.json must contain test_comm_stats (test-time message counts), and
     hosts must have seen a coordinator alert in at least one test round.
Exit code 1 if any run fails a check.

Usage (repo root): python3 test_tools/check_rerun_runs.py --since 20260922 [--run <run_dir>]
"""
import argparse
import json
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paper9_rerun_registry import BASE, AGENTS, discover, is_complete  # noqa: E402


def check_run(run: str) -> list:
    problems = []
    cfg = yaml.safe_load(open(f"{BASE}/{run}/config.yaml"))
    if cfg["env_params"]["communication"]["strategy"] == "none":
        return problems
    for ag in AGENTS:
        q = np.array(json.load(open(f"{BASE}/{run}/{ag}/{ag}.json"))["q_table"])
        visited = np.abs(q).sum(axis=-1) > 0
        ax = q.ndim - 2
        per_bin = [int(np.take(visited, b, axis=ax).sum()) for b in range(q.shape[ax])]
        if sum(per_bin[1:]) == 0:
            problems.append(f"{ag}: message axis never leaves bin 0 {per_bin}")
        test = json.load(open(f"{BASE}/{run}/TEST_{ag}/test.json"))
        st = test.get("test_comm_stats")
        if not st:
            problems.append(f"{ag}: no test_comm_stats in test.json")
        elif st.get("rounds_hosts_saw_broadcast", 0) == 0:
            problems.append(f"{ag}: hosts never saw a coordinator alert during the test")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since")
    ap.add_argument("--run", help="check one run directory name instead of discovering")
    args = ap.parse_args()
    runs = [args.run] if args.run else sorted(
        r for by_seed in discover(args.since).values() for r in by_seed.values())
    bad = 0
    for run in runs:
        if not is_complete(run):
            print(f"SKIP {run}: not complete")
            continue
        problems = check_run(run)
        print(("FAIL " if problems else "ok   ") + run + ("  " + "; ".join(problems) if problems else ""))
        bad += bool(problems)
    print(f"\n{len(runs)} runs checked, {bad} failing")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
