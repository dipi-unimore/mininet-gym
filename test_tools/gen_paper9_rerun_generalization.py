#!/usr/bin/env python3
"""Generate the zero-shot topology-transfer batches of the Paper 9 re-run.

Evaluation only: each config loads the N=10 team trained in the re-run (found by
paper9_rerun_registry.py) and tests it for 300 episodes on a larger topology. There is no
training pass at all (skip_learn: true), so nothing about the transfer has to be explained
as "a few episodes of settling". Run it after the training batches are done.

The host model is the federated average of the N=10 hosts, broadcast to every host on load
(that is how the framework saves a team). The coordinator is not loaded by default
(--coordinator fresh): its feature ranges scale with N, so an N=10 table would be misaligned
at N=20/40, and a fresh, untrained coordinator broadcasts "no attack" (all-zero Q-table).
Use --coordinator loaded to load it anyway and compare.

Writes config/marl_pz/rerun/generalization_n<N>[_coordloaded]/<cell>__seed<NN>.yaml.

Usage (repo root):
    python3 test_tools/gen_paper9_rerun_generalization.py --since 20260922
    python3 test_tools/gen_paper9_rerun_generalization.py --since 20260922 --n 20 --coordinator loaded
Then: sudo python3 utility/run_batch.py config/marl_pz/rerun/generalization_n20
"""
import argparse
import copy
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.path.join(HERE, "paper9_generalization_base.yaml")
OUT_ROOT = os.path.join(ROOT, "config", "marl_pz", "rerun")

sys.path.insert(0, HERE)
from gen_paper9_rerun_configs import all_cells, DEFAULT_SEEDS  # noqa: E402
from paper9_rerun_registry import discover  # noqa: E402

PRIMARY_CELLS = ["s0", "s1", "s0_s2", "s1_s2", "s0_s3", "s1_s3", "s0_s4", "s1_s4",
                 "s0_s2_s4", "s1_s2_s4"]
HOURS_PER_RUN = 0.3  # evaluation only: 300 test episodes for Q-Learning and SARSA


def build(base: dict, seed: int, strategy: str, strategies: list, half: int, run: str,
          load_coordinator: bool) -> dict:
    cfg = copy.deepcopy(base)
    cfg["random_seed"] = seed
    comm = cfg["env_params"]["communication"]
    comm["strategy"] = strategy
    comm["strategies"] = list(strategies)
    cfg["env_params"]["net_params"]["num_hosts"] = half
    cfg["env_params"]["net_params"]["num_iots"] = half
    cfg["env_params"]["episodes"] = 1
    for agent in cfg["agents"]:
        if agent["name"] in ("Q-learning_quick", "Sarsa_balanced"):
            agent["load"] = True
            agent["skip_learn"] = True
            agent["load_coordinator"] = load_coordinator
            agent["load_dir"] = f"marl_pz/{run}/{agent['name']}/{agent['name']}.json"
        else:
            agent["enabled"] = False
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", required=True, help="YYYYMMDD start of the re-run (see paper9_rerun_registry.py)")
    ap.add_argument("--n", type=int, nargs="+", default=[20, 40])
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--cells", nargs="+", default=PRIMARY_CELLS)
    ap.add_argument("--coordinator", choices=["fresh", "loaded"], default="fresh")
    ap.add_argument("--smoke-source", metavar="RUN_DIR",
                    help="write one tiny N=20 evaluation-only config (10 test episodes) that "
                         "loads this run (the output of the 00_smoke training run)")
    args = ap.parse_args()

    if args.smoke_source:
        with open(BASE) as fh:
            base = yaml.safe_load(fh)
        cfg = build(base, 10, "uaq", ["federated_sync"], 10, args.smoke_source, False)
        cfg["env_params"]["test_episodes"] = 10
        out_dir = os.path.join(OUT_ROOT, "00_smoke_transfer")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "s1_s2__smoke_n20.yaml")
        with open(path, "w") as fh:
            yaml.dump(cfg, fh, sort_keys=False, default_flow_style=False)
        print("wrote", path)
        return

    cells = all_cells()
    unknown = [c for c in args.cells if c not in cells]
    if unknown:
        ap.error(f"unknown cells: {unknown}")
    reg = discover(args.since)
    with open(BASE) as fh:
        base = yaml.safe_load(fh)

    total, skipped = 0, []
    suffix = "" if args.coordinator == "fresh" else "_coordloaded"
    for n in args.n:
        if n % 2:
            ap.error(f"--n {n} must be even")
        out_dir = os.path.join(OUT_ROOT, f"generalization_n{n}{suffix}")
        os.makedirs(out_dir, exist_ok=True)
        for cell in args.cells:
            strategy, strategies, _ = cells[cell]
            for seed in args.seeds:
                run = reg.get(cell, {}).get(seed)
                if run is None:
                    skipped.append((n, cell, seed))
                    continue
                cfg = build(base, seed, strategy, strategies, n // 2, run,
                            args.coordinator == "loaded")
                with open(os.path.join(out_dir, f"{cell}__seed{seed}.yaml"), "w") as fh:
                    fh.write(f"# transfer N=10 -> N={n} | {cell} | seed {seed} | source run {run}\n")
                    fh.write("# evaluation only (skip_learn), coordinator "
                             f"{args.coordinator}; generated by gen_paper9_rerun_generalization.py\n")
                    yaml.dump(cfg, fh, sort_keys=False, default_flow_style=False)
                total += 1
    print(f"wrote {total} configs under {OUT_ROOT} (~{total * HOURS_PER_RUN:.0f} h of machine time)")
    if skipped:
        print(f"{len(skipped)} (N, cell, seed) skipped: source training run not complete yet, e.g. {skipped[:3]}")


if __name__ == "__main__":
    main()
