# Paper 9 — multi-seed experiment configs

Ready-to-run configs for the multi-seed campaign in
`docs/papers/9-.../todos.md` §B.

`tier1/` and `tier2/` are **generated** — `git`-ignored by the repo's `config/**/*.yaml`
rule, like the rest of `config/`. The tracked source of truth is
`test_tools/paper9_multiseed_base.yaml` (the fixed protocol) +
`test_tools/gen_paper9_multiseed_configs.py` (the cell/seed matrix). Regenerate anytime:

```bash
python3 test_tools/gen_paper9_multiseed_configs.py
```

This `README.md`, `stage.sh`, and `RUN_MATRIX.md` are tracked.

## Answers to the scoping questions

**How many runs?**
- **Tier 1 (spine): 8 cells × 3 seeds = 24 runs.** This is the minimum that turns the
  paper's core claims into seed-validated ones.
- **Tier 2 (strongly recommended): 5 cells × 3 seeds = 15 runs.** Decide after Tier 1.
- **Total if both: 39 runs.** Each run trains Q-Learning **and** SARSA on the same
  scenario, so this is 39 *runs*, not 78.

**DQN? — No.** DQN majority-class-collapses at S0 already, independent of any
coordination mechanism (macro-F1 ≈ 0.30, recall 1/0/0). Multi-seed will not change that
story, and the paper reframes DQN as a bounded "failure analysis" (todos.md A9) using the
runs that already exist. DQN is disabled in every config here. Re-enable it only if the
separate DQN fix lands and you want a fresh full-grid pass.

**SARSA? — Yes, worth it.** The single best cell in the paper is **S1+S2 SARSA = 0.817**,
and SARSA behaves *differently* from Q-Learning on Tier A (S1: QL −6% F1, SARSA +2%). Only
seeding QL would leave the headline number without an error bar and invite "why only one
of your two tabular algorithms?". SARSA runs are tabular — same cost as QL — and both
train in the same run, so it is nearly free to include.

**Seeds: 10, 14, 77** (matches the existing S2+S3 / S3+S4 multi-seed probes → consistent).
To go to 5 seeds:
`python3 test_tools/gen_paper9_multiseed_configs.py --seeds 10 14 21 42 77`

## The cells

### Tier 1 — `tier1/`  (blocking for any submission)

| file prefix | cell | Tier A | Tier B | why it's on the list |
|---|---|---|---|---|
| `s0` | S0 baseline | naive_broadcast | — | anchor for every comparison |
| `s1` | UAQ | uaq (τ=0.8) | — | Tier A headline; QL's −6% is within noise |
| `s0_s2` / `s1_s2` | Federated Sync | both channels | federated_sync | strongest single mechanism (+0.13 F1); S1+S2 = best cell |
| `s0_s4` / `s1_s4` | Experience Sharing | both channels | experience_sharing | second reliable mechanism |
| `s0_s3` / `s1_s3` | Policy Exchange | both channels | policy_exchange | the "S3 trades F1 for recall / hurts" claim (−0.03, needs seeds) |

### Tier 2 — `tier2/`  (strongly recommended; decide after Tier 1)

| file prefix | cell | note |
|---|---|---|
| `s0_s2_s4` / `s1_s2_s4` | S2+S4 | resolve "substitutes vs. synergy" — currently single-seed |
| `s1_tau09` | UAQ τ=0.9, no Tier B | the "lower overhead *and* equal/better F1" lead; needed for the Pareto figure (A8) |
| `s1_s2_s3` / `s0_s3_s4` | combos, **SARSA** | QL already multi-seed (Round 4, n=5–6); these add SARSA coverage |

Not here (Tier 3 / journal-only): native N=20 training, zero-shot N=20 multi-seed,
`sync_interval` sweep, PPO/A2C, DQN fix. See todos.md §B Tier 3.

## Running a config

The app only reads `config/default.yaml`, so each run means swapping that file:

```bash
# from repo root
./config/marl_pz/stage.sh config/marl_pz/tier1/s0__seed10.yaml   # backs up default.yaml, copies the chosen one in
python3 main.py                                                  # then launch the run from the web UI as usual
```

`stage.sh` keeps the previous `config/default.yaml` at
`config/default.yaml.bak` so you can restore your interactive config afterwards.

For unattended batches, set `enable_web_interface: false` in
`test_tools/paper9_multiseed_base.yaml` and regenerate
— `main.py` then runs `start_experiment` directly on load. (The Telegram
"experiment-finished" notification in todos.md §B "Infra dependency" is the real unlock
for running the 24–39 runs without babysitting.)

## Tracking what's already been run

```bash
python3 tools/manage_configs.py            # tag every config with its run status
python3 tools/manage_configs.py --dry-run  # preview only
```

`tools/manage_configs.py` is gym-type-generic: it tags configs under any
`config/<gym_type>/` directory the results panel understands, skipping the ones
that don't exist. It walks `_training/<gym_type>/` and renames each config here to
record its state:

| suffix | meaning |
|---|---|
| `__seedNN_<ddMMyyyyHHmm>.yaml` | a **complete** run exists (stamp = most recent one) |
| `__seedNN_incomplete.yaml` | a matching run folder exists but no run is complete yet |
| `__seedNN.yaml` (no suffix) | not run yet |

"Complete" is decided by the same function the `/results` panel uses
(`build_results_dir_list`), so it never drifts from the panel. A config is matched
to a run by scenario identity — gym type, network shape, `random_seed`, episode /
step / bin counts, and (marl_pz) Tier A `strategy` + Tier B `strategies` + UAQ
`tau` — **plus** the requirement that every learning algorithm the config enables
was trained in the run. So a Q-Learning **and** SARSA config is *not* matched by a
QL-only run (e.g. the Round-4 `s1_s2_s3` / `s0_s3_s4` combos still need their SARSA
pass). Attack thresholds / timing and agent hyper-parameters are ignored — they
drift with protocol tweaks; diff `config.yaml` by hand before reusing an old run.

The script is idempotent — re-run it after every batch. It never stacks markers:
it strips the one it finds and recomputes, so a file only moves when the facts
move (newer complete run → stamp bumped; `_incomplete` run finishes → becomes a
stamp; last complete run deleted → falls back to `_incomplete`). It also
de-duplicates the untagged copies `gen_paper9_multiseed_configs.py` recreates.
`stage.sh` accepts the tagged paths unchanged.

## After the runs

Follow the standing rules in `[[project-paper9-marl-experiments]]`:
1. `diff` each new `config.yaml` against a reference before treating runs as comparable.
2. Recompute macro-F1 from raw `ground_truth`/`predicted` in `TEST_*/test.json`, **not**
   the dashboard's weighted figures.
3. Update `ch08_experiment_log.md` first, then the paper tables/figures
   (`test_tools/plot_tier_b_combinations.py` and the `compare_config_tier_*` pipeline
   already exist and are reusable).

### Reusing the existing seed-77 cells

The paper's current single-seed cells are all seed 77 and mostly match this protocol
(Round 3 table, 2026-08-12 onward). You *can* reuse them instead of re-running the
`*__seed77.yaml` files — but **diff the config first** (there have been param
additions/renames, e.g. `unblock_min_hold_rounds`). Regenerating seed 77 too gives one
uniform set under the current code, which is cleaner for the paper.
