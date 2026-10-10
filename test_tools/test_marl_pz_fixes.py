#!/usr/bin/env python3
"""Regression checks for the marl_pz fixes that do not need Mininet.

1. The host message feature must be distinguishable (0 vs 1) at every topology size.
2. Saving the team model must not modify the live host agents.

Not covered here (needs a live run): messages carried across test rounds in
_evaluate_marl_agent, and the test_comm_stats block written into test.json. Check those
with one smoke run (see config/marl_pz/rerun/README.md).

Usage (repo root): python3 test_tools/test_marl_pz_fixes.py
"""
import sys
import numpy as np

sys.path.insert(0, ".")
from reinforcement_learning.scenarios.marl_pz.marl_pz_env import HOST_MSG_MAX, _discretize_obs
from reinforcement_learning.scenarios.marl_pz.marl_pz_main import _save_team_model_without_mutation


def check_message_feature_is_visible():
    n_traffic = 8
    low = np.zeros(n_traffic + 1, dtype=np.float32)
    counter_idx, var_idx = {0, 2, 4, 6}, {1, 3, 5, 7}
    states = {}
    for n_hosts in (10, 20, 40):
        high = np.append(np.full(n_traffic, 100.0), HOST_MSG_MAX).astype(np.float32)
        s = []
        for msg in (0.0, 1.0):
            obs = np.append(np.full(n_traffic, 5.0), msg).astype(np.float32)
            s.append(_discretize_obs(obs, low, high, 4, counter_idx, var_idx)[-1])
        assert s[0] != s[1], f"message 0 and 1 share a bin at N={n_hosts}: {s}"
        states[n_hosts] = tuple(s)
    assert len(set(states.values())) == 1, f"message bins depend on N: {states}"
    print("ok  message feature: 0 and 1 use different bins, same for N=10/20/40:", states[10])


class _FakeHost:
    def __init__(self, table):
        self.q_table = table
        self.saved = None

    def save(self, name):
        self.saved = self.q_table.copy()


def check_save_does_not_mutate():
    rng = np.random.default_rng(0)
    hosts = [(f"h{i}", _FakeHost(rng.normal(size=(4, 3)))) for i in range(5)]
    before = [m.q_table.copy() for _, m in hosts]
    expected_mean = np.mean(before, axis=0)
    _save_team_model_without_mutation(hosts, [], "unused")
    assert np.allclose(hosts[0][1].saved, expected_mean), "saved table is not the host average"
    for (_, m), b in zip(hosts, before):
        assert np.array_equal(m.q_table, b), "a live host table was modified by saving"
    assert len({m.q_table.tobytes() for _, m in hosts}) == len(hosts), "hosts became identical"
    print("ok  saving writes the average and leaves the 5 live host tables distinct and unchanged")


if __name__ == "__main__":
    check_message_feature_is_visible()
    check_save_does_not_mutate()
    print("all checks passed")
