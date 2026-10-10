#!/usr/bin/env python3
"""
Generate the ready-to-run config files for the Paper 9 topology-generalization
batch: zero-shot transfer of N=10-trained per-host policies (from the multi-seed
campaign's own REGISTRY, see paper9_multiseed.py) into a larger topology (N=20
and/or N=40), matching experiments9.tex's existing N=20 protocol (a dummy 3-episode/
60-step "load-then-test" pass to let per-host state discretization settle -- this
is a workaround, not additional training, and is never discussed as such in the
paper -- then the usual 300-episode frozen-policy test; coordinator not loaded).

Q-Learning and SARSA transfer at every seed. DQN transfers too, but only where a
seed actually trained one: seed 77's N=10 runs have a DQN model (checked against
the run directory, not assumed), seeds 10/14 do not (the multi-seed campaign is
QL+SARSA-only by design) -- DQN stays disabled for those.

The existing N=20 seed-77 cells (2026-08-23/24, 10 runs) are NOT regenerated --
this script only fills in what's missing: seeds 10/14 at N=20, and all three
seeds at N=40 (fully new, unrun and unvalidated at that scale).

Writes config/marl_pz/generalization_n<N>/<cell>__seed<NN>.yaml.

Usage:
    python3 test_tools/gen_paper9_generalization_configs.py                # N=20 (seeds 10,14) + N=40 (seeds 10,14,77)
    python3 test_tools/gen_paper9_generalization_configs.py --n 20 --seeds 10 14
    python3 test_tools/gen_paper9_generalization_configs.py --n 40 --seeds 77 --cells s0 s1
"""
import argparse
import copy
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.path.join(HERE, "paper9_generalization_base.yaml")
OUT_ROOT = os.path.join(ROOT, "config", "marl_pz")

sys.path.insert(0, HERE)
from paper9_multiseed import REGISTRY  # noqa: E402

# Same 10-cell grid as the existing N=20 runs (primary 2x4 grid + S2+S4 secondary
# probe on both channels) -- cell key -> (Tier A strategy, Tier B strategies).
CELLS = {
    "s0":        ("naive_broadcast", []),
    "s1":        ("uaq",             []),
    "s0_s2":     ("naive_broadcast", ["federated_sync"]),
    "s1_s2":     ("uaq",             ["federated_sync"]),
    "s0_s3":     ("naive_broadcast", ["policy_exchange"]),
    "s1_s3":     ("uaq",             ["policy_exchange"]),
    "s0_s4":     ("naive_broadcast", ["experience_sharing"]),
    "s1_s4":     ("uaq",             ["experience_sharing"]),
    "s0_s2_s4":  ("naive_broadcast", ["federated_sync", "experience_sharing"]),
    "s1_s2_s4":  ("uaq",             ["federated_sync", "experience_sharing"]),
}

# Cells that already have a completed N=20 seed-77 run (2026-08-23/24) -- skip
# regenerating seed 77 at N=20 for these; still generate it fresh at N=40.
EXISTING_N20_SEED77 = set(CELLS)  # all 10 cells already have their seed-77 N=20 run


def source_run(cell: str, seed: int) -> str:
    """The N=10 run dir (no leading _training/) whose Q-Learning/SARSA policy gets
    loaded for this cell/seed -- reuses the same seed->run mapping as the main
    multi-seed analysis (paper9_multiseed.REGISTRY), so "the N=10 policy we
    transfer" and "the N=10 policy the paper's Table tab:tier_b numbers come
    from" are guaranteed to be the same run."""
    if cell not in REGISTRY:
        raise KeyError(f"cell '{cell}' not in paper9_multiseed.REGISTRY")
    seeds = REGISTRY[cell].seeds
    if seed not in seeds:
        raise KeyError(f"cell '{cell}' has no seed {seed} in paper9_multiseed.REGISTRY")
    return seeds[seed]


def build(base: dict, seed: int, strategy: str, strategies: list, n_hosts_half: int,
          run_dir: str, has_dqn_source: bool) -> dict:
    cfg = copy.deepcopy(base)
    cfg["random_seed"] = seed
    comm = cfg["env_params"]["communication"]
    comm["strategy"] = strategy
    comm["strategies"] = list(strategies)
    cfg["env_params"]["net_params"]["num_hosts"] = n_hosts_half
    cfg["env_params"]["net_params"]["num_iots"] = n_hosts_half
    for agent in cfg["agents"]:
        if agent["name"] == "Q-learning_quick":
            agent["load_dir"] = f"marl_pz/{run_dir}/Q-learning_quick/Q-learning_quick.json"
        elif agent["name"] == "Sarsa_balanced":
            agent["load_dir"] = f"marl_pz/{run_dir}/Sarsa_balanced/Sarsa_balanced.json"
        elif agent["name"] == "DQN_quick":
            # Only seed 77's N=10 runs trained DQN (seeds 10/14 are QL+SARSA-only,
            # per the multi-seed campaign's "no DQN" decision) -- transfer it where
            # a source model actually exists, keep it disabled otherwise.
            if has_dqn_source:
                agent["enabled"] = True
                agent["load_dir"] = f"marl_pz/{run_dir}/DQN_quick/DQN_quick.zip"
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[20, 40],
                    help="total host counts to generate for (N/2 workstations + N/2 IoT each)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="default: 10,14 for N=20 (seed 77 already run); 10,14,77 for any other N")
    ap.add_argument("--cells", nargs="+", default=list(CELLS),
                    help="default: all 10 cells (primary 2x4 grid + S2+S4 both channels)")
    args = ap.parse_args()

    unknown = [c for c in args.cells if c not in CELLS]
    if unknown:
        ap.error(f"unknown cells: {unknown} (choices: {list(CELLS)})")

    with open(BASE) as fh:
        base = yaml.safe_load(fh)

    total = 0
    matrix_lines = ["# Paper 9 topology-generalization campaign — run matrix", "",
                     "Zero-shot transfer: N=10-trained per-host policies (from the seeds "
                     "already used for `docs/papers/9-.../experiments9.tex` Tables "
                     "tab:tier_b/tab:tier_b_combinations) loaded into a larger topology, "
                     "then a dummy 3-episode/60-step load-then-test pass (workaround, not "
                     "real training -- never described as such in the paper) and the usual "
                     "300-episode frozen-policy test. Q-Learning + SARSA transfer at every "
                     "seed; DQN transfers too wherever seed 77's N=10 run actually trained "
                     "one (checked per cell, not assumed), stays disabled for seeds 10/14 "
                     "(no source model there -- out of scope for those by design). "
                     "Coordinator not loaded (see experiments9.tex sec:exp_generalization).", ""]

    for n in args.n:
        if n % 2:
            ap.error(f"--n {n} must be even (split N/2 workstations + N/2 IoT)")
        half = n // 2
        seeds = args.seeds
        if seeds is None:
            seeds = [10, 14] if n == 20 else [10, 14, 77]

        out_dir = os.path.join(OUT_ROOT, f"generalization_n{n}")
        os.makedirs(out_dir, exist_ok=True)
        n_runs = 0
        matrix_lines.append(f"## N={n}  ({half} workstations + {half} IoT)")
        matrix_lines.append("")
        for cell in args.cells:
            strategy, strategies = CELLS[cell]
            for seed in seeds:
                if n == 20 and seed == 77 and cell in EXISTING_N20_SEED77:
                    continue  # already run 2026-08-23/24, don't regenerate
                try:
                    run_dir = source_run(cell, seed)
                except KeyError as exc:
                    print(f"skip {cell} seed {seed}: {exc}")
                    continue
                dqn_zip = os.path.join(ROOT, "_training", "marl_pz", run_dir,
                                        "DQN_quick", "DQN_quick.zip")
                has_dqn_source = os.path.exists(dqn_zip)
                cfg = build(base, seed, strategy, strategies, half, run_dir, has_dqn_source)
                fname = f"{cell}__seed{seed}.yaml"
                path = os.path.join(out_dir, fname)
                dqn_note = "DQN transferred" if has_dqn_source else "DQN disabled, no source"
                with open(path, "w") as fh:
                    fh.write(f"# N={n} transfer | {cell} ({strategy}, {strategies or '[]'}) "
                              f"| seed {seed} | source _training/marl_pz/{run_dir} | {dqn_note}\n")
                    fh.write("# generated by test_tools/gen_paper9_generalization_configs.py "
                              "-- edit test_tools/paper9_generalization_base.yaml or the script\n")
                    yaml.dump(cfg, fh, sort_keys=False, default_flow_style=False)
                total += 1
                n_runs += 1
            gen_seeds = [s for s in seeds if not (n == 20 and s == 77 and cell in EXISTING_N20_SEED77)]
            dqn_seeds = [s for s in gen_seeds if s == 77]
            dqn_suffix = f", DQN also transferred for seed(s) {dqn_seeds}" if dqn_seeds else ""
            matrix_lines.append(f"- `{cell}` -> Tier A `{strategy}`, Tier B `{strategies or '[]'}`, "
                                 f"seeds `{gen_seeds}`{dqn_suffix}")
        matrix_lines.append(f"\n{n_runs} configs written to `config/marl_pz/generalization_n{n}/`.\n")

    matrix_lines.append(f"**Total new configs: {total}** (each evaluates Q-Learning + SARSA; "
                         "DQN too on seed-77 cells, where a source model exists). Launch a "
                         "batch with e.g.:\n"
                         "```\nsudo python3 utility/run_batch.py config/marl_pz/generalization_n20\n"
                         "sudo python3 utility/run_batch.py config/marl_pz/generalization_n40\n```")
    with open(os.path.join(OUT_ROOT, "GENERALIZATION_RUN_MATRIX.md"), "w") as fh:
        fh.write("\n".join(matrix_lines) + "\n")

    print("\n".join(matrix_lines))
    print(f"\nWrote {total} config files under {OUT_ROOT}/generalization_n*/")


if __name__ == "__main__":
    main()
