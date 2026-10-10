#!/usr/bin/env python3
"""Generate the training batches for the Paper 9 re-run (after the marl_pz fixes).

Why a re-run: the message feature was invisible to tabular hosts (bin bug), host tables
were averaged in place before every test, and test rounds never carried messages. All
three are fixed in the code, so every earlier result must be re-measured. See
docs/papers/9-.../todos.md A24.

Writes config/marl_pz/rerun/<batch>/<cell>__seed<NN>.yaml, one folder per batch, in
priority order so the machine can be stopped after any batch:

  01_core                 S0, S1, none (independent, no communication), and each of S2/S3/S4
                          on the S0 and S1 channels                       9 cells
  02_extra                S2+S4 (both channels), tau=0.9, S1+S2+S3, S0+S3+S4   5 cells
  03_independent_tierb    S2/S3/S4 with no coordinator at all                3 cells (optional)

Each run trains Q-Learning, SARSA and (by default, see DQN below) DQN on the same
scenario. Base protocol: test_tools/paper9_multiseed_base.yaml, except for two
changes applied by every config this script writes: episodes 100 -> 200 (see
TRAIN_EPISODES below) and DQN on by default.

Usage:
    python3 test_tools/gen_paper9_rerun_configs.py
    python3 test_tools/gen_paper9_rerun_configs.py --seeds 10 14 77 2015 2019
    python3 test_tools/gen_paper9_rerun_configs.py --dqn-seeds 77 2015   # DQN for only these seeds
    python3 test_tools/gen_paper9_rerun_configs.py --dqn-seeds          # DQN off everywhere
Then, one batch at a time (root needed for Mininet):
    sudo python3 utility/run_batch.py config/marl_pz/rerun/01_core
"""
import argparse
import copy
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.path.join(HERE, "paper9_multiseed_base.yaml")
OUT_ROOT = os.path.join(ROOT, "config", "marl_pz", "rerun")

DEFAULT_SEEDS = [10, 14, 77, 2015, 2019]
TRAIN_EPISODES = 200  # base yaml has 100; see the module docstring and todos.md A24/A16(c)
# Observed in the previous campaign (Sep 2026): ~3h/run at 100 train + 300 test episodes,
# QL+SARSA only. TRAIN_EPISODES doubled and DQN is now on by default (slower per episode
# than the tabular agents), so this number is a placeholder -- recalibrate from the
# 00_smoke run's actual wall-clock time before trusting the day estimates below.
HOURS_PER_RUN = 5.0

BATCHES = {
    "01_core": {
        "s0":    ("naive_broadcast", [], {}),
        "s1":    ("uaq", [], {}),
        "none":  ("none", [], {}),
        "s0_s2": ("naive_broadcast", ["federated_sync"], {}),
        "s1_s2": ("uaq", ["federated_sync"], {}),
        "s0_s3": ("naive_broadcast", ["policy_exchange"], {}),
        "s1_s3": ("uaq", ["policy_exchange"], {}),
        "s0_s4": ("naive_broadcast", ["experience_sharing"], {}),
        "s1_s4": ("uaq", ["experience_sharing"], {}),
    },
    "02_extra": {
        "s0_s2_s4": ("naive_broadcast", ["federated_sync", "experience_sharing"], {}),
        "s1_s2_s4": ("uaq", ["federated_sync", "experience_sharing"], {}),
        "s1_tau06": ("uaq", [], {"uaq": {"tau": 0.6}}),
        "s1_tau09": ("uaq", [], {"uaq": {"tau": 0.9}}),
        "s1_s2_s3": ("uaq", ["federated_sync", "policy_exchange"], {}),
        "s0_s3_s4": ("naive_broadcast", ["policy_exchange", "experience_sharing"], {}),
    },
    "03_independent_tierb": {
        "none_s2": ("none", ["federated_sync"], {}),
        "none_s3": ("none", ["policy_exchange"], {}),
        "none_s4": ("none", ["experience_sharing"], {}),
    },
}

LABEL = {
    "none": "independent agents, no communication, no coordinator",
    "s1_tau06": "S1 UAQ tau=0.6",
    "s1_tau09": "S1 UAQ tau=0.9",
}


def all_cells() -> dict:
    """cell -> (tier_a, tier_b, patch) across every batch (used by the registry/generalization scripts)."""
    out = {}
    for cells in BATCHES.values():
        out.update(cells)
    return out


def build(base: dict, seed: int, strategy: str, strategies: list, patch: dict,
          dqn: bool, episodes: int = TRAIN_EPISODES) -> dict:
    cfg = copy.deepcopy(base)
    cfg["random_seed"] = seed
    cfg["env_params"]["episodes"] = episodes
    comm = cfg["env_params"]["communication"]
    comm["strategy"] = strategy
    comm["strategies"] = list(strategies)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(comm.get(k), dict):
            comm[k].update(v)
        else:
            comm[k] = v
    for agent in cfg["agents"]:
        if agent["algorithm"] == "DQN":
            agent["enabled"] = dqn
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--dqn-seeds", type=int, nargs="*", default=None,
                    help="seeds for which DQN is also trained (default: all --seeds; "
                         "pass with no values to disable DQN everywhere)")
    ap.add_argument("--episodes", type=int, default=TRAIN_EPISODES,
                    help=f"training episodes per run (default: {TRAIN_EPISODES})")
    ap.add_argument("--smoke", action="store_true",
                    help="only write one tiny S1+S2 config (3 training and 10 test episodes) "
                         "to config/marl_pz/rerun/00_smoke, to check the fixes before the batches")
    args = ap.parse_args()
    dqn_seeds = args.seeds if args.dqn_seeds is None else args.dqn_seeds

    with open(BASE) as fh:
        base = yaml.safe_load(fh)

    if args.smoke:
        out_dir = os.path.join(OUT_ROOT, "00_smoke")
        os.makedirs(out_dir, exist_ok=True)
        cfg = build(base, 10, "uaq", ["federated_sync"], {}, False)
        cfg["env_params"]["episodes"] = 3
        cfg["env_params"]["max_steps"] = 20
        cfg["env_params"]["test_episodes"] = 10
        with open(os.path.join(out_dir, "s1_s2__smoke.yaml"), "w") as fh:
            fh.write("# smoke test: S1+S2, 3 training episodes x 20 steps, 10 test episodes\n")
            yaml.dump(cfg, fh, sort_keys=False, default_flow_style=False)
        print("wrote", os.path.join(out_dir, "s1_s2__smoke.yaml"))
        return

    lines = ["# Paper 9 re-run: training batches", "",
             f"Seeds: {args.seeds}. Training episodes: {args.episodes}. "
             f"DQN also trained for seeds: {dqn_seeds or 'none'}.",
             f"Estimated machine time at ~{HOURS_PER_RUN:.0f} h per run "
             "(placeholder, see HOURS_PER_RUN in this script -- recalibrate from 00_smoke).", ""]
    total = 0
    for batch, cells in BATCHES.items():
        out_dir = os.path.join(OUT_ROOT, batch)
        os.makedirs(out_dir, exist_ok=True)
        runs = len(cells) * len(args.seeds)
        total += runs
        lines.append(f"## {batch}: {len(cells)} cells x {len(args.seeds)} seeds = {runs} runs "
                     f"(~{runs * HOURS_PER_RUN / 24:.1f} days)")
        for cell, (strategy, strategies, patch) in cells.items():
            for seed in args.seeds:
                cfg = build(base, seed, strategy, strategies, patch, seed in dqn_seeds,
                            episodes=args.episodes)
                with open(os.path.join(out_dir, f"{cell}__seed{seed}.yaml"), "w") as fh:
                    fh.write(f"# {LABEL.get(cell, cell)} | seed {seed} | re-run after the marl_pz fixes\n")
                    fh.write("# generated by test_tools/gen_paper9_rerun_configs.py\n")
                    yaml.dump(cfg, fh, sort_keys=False, default_flow_style=False)
            lines.append(f"- `{cell}`: Tier A `{strategy}`, Tier B `{strategies or '[]'}`")
        lines.append("")
    lines.append(f"**Total: {total} training runs (~{total * HOURS_PER_RUN / 24:.1f} days).**")
    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(os.path.join(OUT_ROOT, "RUN_MATRIX.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
