#!/usr/bin/env python3
"""Benchmark the per-micro-step decision cost of each trained agent, in isolation
from Mininet/network emulation, to empirically validate the O(1) decision-time
claim of the Sequential Micro-Step Decomposition architecture.

For Q-Learning/SARSA: times the Q-table lookup (np.argmax over q_table[state_index]),
using the real trained q_table loaded from a Group 2 (300-episode, reward_scaling
variant) experiment.

For DQN/PPO/A2C: times model.predict(obs, deterministic=True) on the real trained
stable-baselines3 model from the same experiment, using synthetic normalized
observations of the correct shape (matching what the environment wrapper produces).

This isolates the RL agent's own decision cost (the thing the O(1) claim is about)
from network-emulation overhead, which is a separate, unrelated cost.
"""
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "_training/attacks_ho/20260802-235108_1_5_5"  # reward_scaling, G2 (300ep, 10 nodes)
N_REPEATS = 5000
N_WARMUP = 200
SEED = 0


def bench_q_table(algo_name, folder):
    path = EXP / folder / f"{folder}.json"
    with open(path) as f:
        data = json.load(f)
    q_table = np.array(data["q_table"])
    state_shape = q_table.shape[:-1]

    rng = np.random.default_rng(SEED)

    def one_call():
        idx = tuple(int(rng.integers(0, s)) for s in state_shape)
        return int(np.argmax(q_table[idx]))

    for _ in range(N_WARMUP):
        one_call()

    t0 = time.perf_counter()
    for _ in range(N_REPEATS):
        one_call()
    t1 = time.perf_counter()

    per_call_us = (t1 - t0) / N_REPEATS * 1e6
    print(f"{algo_name:12s} q_table shape={q_table.shape}  {per_call_us:8.2f} us/decision")
    return per_call_us


def bench_sb3(algo_name, folder, cls):
    path = EXP / folder / f"{folder}.zip"
    model = cls.load(str(path))
    obs_shape = model.observation_space.shape

    rng = np.random.default_rng(SEED)

    def one_call():
        obs = rng.uniform(0.0, 1.0, size=obs_shape).astype(np.float32)
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    for _ in range(N_WARMUP):
        one_call()

    t0 = time.perf_counter()
    for _ in range(N_REPEATS):
        one_call()
    t1 = time.perf_counter()

    per_call_us = (t1 - t0) / N_REPEATS * 1e6
    print(f"{algo_name:12s} obs_shape={obs_shape}      {per_call_us:8.2f} us/decision")
    return per_call_us


def main():
    from stable_baselines3 import A2C, DQN, PPO

    print(f"Loading real trained models from: {EXP}")
    print(f"Timing {N_REPEATS} predict() calls per algorithm (after {N_WARMUP} warmup calls)\n")

    per_decision_us = {}
    per_decision_us["Q-Learning"] = bench_q_table("Q-Learning", "Q-learning_quick")
    per_decision_us["SARSA"] = bench_q_table("SARSA", "Sarsa_balanced")
    per_decision_us["DQN"] = bench_sb3("DQN", "DQN_quick", DQN)
    per_decision_us["PPO"] = bench_sb3("PPO", "PPO_quick", PPO)
    per_decision_us["A2C"] = bench_sb3("A2C", "A2C_quick", A2C)

    # N=10 and N=20 match the actual tested topologies (Groups 1-2 and Group 3
    # respectively); no experiment used N=5, so it is intentionally not included here.
    print("\n--- Per-network-step decision time (per-decision x N), microseconds ---")
    print(f"{'Algorithm':12s} {'N=10':>10s} {'N=20':>10s}")
    for algo, per_call in per_decision_us.items():
        row = [f"{per_call * n:10.1f}" for n in (10, 20)]
        print(f"{algo:12s} " + " ".join(row))

    out = {
        "per_decision_us": per_decision_us,
        "per_network_step_us": {
            algo: {str(n): per_call * n for n in (10, 20)}
            for algo, per_call in per_decision_us.items()
        },
        "n_repeats": N_REPEATS,
        "source_experiment": str(EXP.relative_to(REPO)),
    }
    out_path = REPO / "_training/_debug/generate/decision_overhead_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
