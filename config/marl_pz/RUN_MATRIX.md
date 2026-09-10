# Paper 9 multi-seed campaign — run matrix

Seeds: [10, 14, 77]

## tier1  (8 cells x 3 seeds = 24 runs)

- `s0` — S0 (naive broadcast, no Tier B) -> Tier A `naive_broadcast`, Tier B `[]`
- `s1` — S1 (UAQ tau=0.8, no Tier B) -> Tier A `uaq`, Tier B `[]`
- `s0_s2` — S0 + S2 federated_sync -> Tier A `naive_broadcast`, Tier B `['federated_sync']`
- `s1_s2` — S1 + S2 federated_sync -> Tier A `uaq`, Tier B `['federated_sync']`
- `s0_s4` — S0 + S4 experience_sharing -> Tier A `naive_broadcast`, Tier B `['experience_sharing']`
- `s1_s4` — S1 + S4 experience_sharing -> Tier A `uaq`, Tier B `['experience_sharing']`
- `s0_s3` — S0 + S3 policy_exchange -> Tier A `naive_broadcast`, Tier B `['policy_exchange']`
- `s1_s3` — S1 + S3 policy_exchange -> Tier A `uaq`, Tier B `['policy_exchange']`

## tier2  (5 cells x 3 seeds = 15 runs)

- `s0_s2_s4` — S0 + S2 + S4 -> Tier A `naive_broadcast`, Tier B `['federated_sync', 'experience_sharing']`
- `s1_s2_s4` — S1 + S2 + S4 -> Tier A `uaq`, Tier B `['federated_sync', 'experience_sharing']`
- `s1_tau09` — S1 UAQ, tau=0.9, no Tier B -> Tier A `uaq`, Tier B `[]`
- `s1_s2_s3` — S1 + S2 + S3 (SARSA coverage; QL already multi-seed) -> Tier A `uaq`, Tier B `['federated_sync', 'policy_exchange']`
- `s0_s3_s4` — S0 + S3 + S4 (SARSA coverage; QL already multi-seed) -> Tier A `naive_broadcast`, Tier B `['policy_exchange', 'experience_sharing']`

**Total: 39 runs** (each run trains Q-Learning + SARSA on the same scenario).
