#!/usr/bin/env python3
"""Find the completed N=10 runs of the Paper 9 re-run and map (cell, seed) -> run directory.

The re-run has no hand-kept table like paper9_multiseed.REGISTRY: run folders are found by
reading each run's own config.yaml, so nothing has to be typed after a batch finishes.

A run counts as complete for a cell/seed when it was started on or after --since, has the
N=10 topology suffix (_1_5_5), was trained for at least TRAIN_EPISODES episodes (so smoke
runs are ignored), and both Q-Learning and SARSA have a saved model and a
TEST_<agent>/test.json. If several runs match, the latest wins.

Usage (repo root):
    python3 test_tools/paper9_rerun_registry.py --since 20260922
    python3 test_tools/paper9_rerun_registry.py --since 20260922 --write   # config/marl_pz/rerun/registry.json
"""
import argparse
import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from gen_paper9_rerun_configs import all_cells, DEFAULT_SEEDS, TRAIN_EPISODES  # noqa: E402

BASE = os.path.join(ROOT, "_training", "marl_pz")
AGENTS = ("Q-learning_quick", "Sarsa_balanced")
TOPOLOGY_SUFFIX = "_1_5_5"


def _identity(cfg: dict):
    comm = cfg["env_params"]["communication"]
    tau = (comm.get("uaq") or {}).get("tau", 0.8)
    return (comm["strategy"], tuple(sorted(comm.get("strategies") or [])), round(float(tau), 3))


def cell_identities() -> dict:
    out = {}
    for cell, (strategy, strategies, patch) in all_cells().items():
        tau = (patch.get("uaq") or {}).get("tau", 0.8)
        out[(strategy, tuple(sorted(strategies)), round(float(tau), 3))] = cell
    return out


def is_complete(run: str) -> bool:
    for ag in AGENTS:
        if not os.path.isfile(f"{BASE}/{run}/{ag}/{ag}.json"):
            return False
        if not os.path.isfile(f"{BASE}/{run}/TEST_{ag}/test.json"):
            return False
    return True


def discover(since: str, min_episodes: int = TRAIN_EPISODES) -> dict:
    ids = cell_identities()
    found = {}
    for run in sorted(os.listdir(BASE)):
        if not run.endswith(TOPOLOGY_SUFFIX) or run[:8] < since:
            continue
        cfg_path = f"{BASE}/{run}/config.yaml"
        if not os.path.isfile(cfg_path) or not is_complete(run):
            continue
        cfg = yaml.safe_load(open(cfg_path))
        # smoke runs (--smoke: 3 training episodes) share a cell identity with real runs
        if int(cfg["env_params"].get("episodes", 0)) < min_episodes:
            continue
        cell = ids.get(_identity(cfg))
        if cell is None:
            continue
        found.setdefault(cell, {})[int(cfg["random_seed"])] = run  # sorted -> latest wins
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", required=True, help="YYYYMMDD: ignore runs started before this day")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    reg = discover(args.since)
    missing = []
    for cell in all_cells():
        have = sorted(reg.get(cell, {}))
        print(f"{cell:10s} seeds done: {have}")
        missing += [(cell, s) for s in args.seeds if s not in have]
    print(f"\n{sum(len(v) for v in reg.values())} complete runs, {len(missing)} (cell, seed) missing")
    if args.write:
        out = os.path.join(ROOT, "config", "marl_pz", "rerun", "registry.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(reg, open(out, "w"), indent=1, sort_keys=True)
        print("wrote", out)


if __name__ == "__main__":
    main()
