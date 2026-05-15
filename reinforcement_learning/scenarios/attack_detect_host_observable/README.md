# Scenario: attack_ho — Attack Detection with Per-Host Observable

## Overview

`attack_ho` (**Attack Detection, Host Observable**) is a single-agent reinforcement learning scenario where the agent monitors a star-topology Mininet network and must classify the traffic state of each host independently: whether it is idle/normal, a victim of an incoming attack, or an active attacker.

The key design principle is that the agent sees the network **one host at a time** (via `PerHostScanWrapper`), making the observation and action spaces constant regardless of the total number of hosts. This keeps agent complexity O(1) with respect to network size while still enabling per-host decisions.

---

## Network Topology

- **Switch topology:** single OVS switch (star)
- **Hosts:** configurable (`num_hosts` in net params), typically 3–10
- **SDN controller:** OpenDaylight (ODL) — manages flow rules for traffic blocking
- **Traffic generation:** `iperf`-based synthetic workloads launched by the scenario generator

---

## Working Mode

The scenario runs in four sequential phases:

### 1. Scenario Generation
A JSON sequence of traffic tasks is generated and saved to disk (`scenario.json`). It contains:
- **Training sequence:** `episodes × steps` tasks (default: 30 ep × 120 steps)
- **Evaluation sequence:** 180 single-step episodes at fixed attack rate (30%)

Each task specifies, for each host: task type (`normal`, `short_attack`, `long_attack`), source, destination, and duration.

### 2. Training
For each enabled agent (sequentially):
1. The training sequence is loaded and replayed step-by-step.
2. At each step, actual `iperf` traffic is launched in Mininet, OVS counter statistics are read, and the agent observes the result.
3. The agent produces a per-host classification decision via the `PerHostScanWrapper` micro-step mechanism.
4. Rewards are applied and the agent updates its policy.

Attack likelihood during training: **90%** (`likely_train: 0.9`), so the agent is exposed to many attack examples to counteract class imbalance.

### 3. Evaluation
Each agent is evaluated on the 180-episode test sequence (attack rate: 30%). Metrics collected per agent:
- Accuracy, Precision, Recall, F1 (macro and per-class)
- Confusion matrix
- Mitigation ratio (fraction of victim episodes where all attackers were blocked)

### 4. Summary
Comparative table and plots are saved to the experiment directory.

---

## Observation Space

**Space type:** `Box(shape=(8,), dtype=float32)`

The observation is always a fixed-size **8-feature vector for a single host**, regardless of how many hosts are in the network. The `PerHostScanWrapper` slices the full N×8 network state and delivers one host slice at a time.

| Index | Feature | Bounds |
|-------|---------|--------|
| 0 | RX packets (received) | `[0, threshold_packets × num_hosts]` |
| 1 | ΔRX packets (% change) | `[-threshold_var_packets, +threshold_var_packets]` |
| 2 | RX bytes (received) | `[0, threshold_bytes × num_hosts]` |
| 3 | ΔRX bytes (% change) | `[-threshold_var_bytes, +threshold_var_bytes]` |
| 4 | TX packets (transmitted) | `[0, threshold_packets × num_hosts]` |
| 5 | ΔTX packets (% change) | `[-threshold_var_packets, +threshold_var_packets]` |
| 6 | TX bytes (transmitted) | `[0, threshold_bytes × num_hosts]` |
| 7 | ΔTX bytes (% change) | `[-threshold_var_bytes, +threshold_var_bytes]` |

**Default thresholds** (`scenario_env_param.yaml`):

```yaml
thresholds:
  packets: 22000
  var_packets: 50       # ±50% change
  bytes: 423000000
  var_bytes: 30         # ±30% change
```

Observations are returned **raw** (un-normalised). Each agent type handles them differently:
- **SB3 agents (DQN, PPO, A2C):** receive raw values directly; the neural network learns the scale.
- **Tabular agents (Q-Learning, SARSA):** call `env.get_discretized_state(obs)` to convert to discrete bins via logarithmic binning for counters and linear binning for deltas.

---

## Action Space

**Space type:** `Discrete(3)`

| Action | ID | Meaning |
|--------|----|---------|
| `NORMAL_TRAFFIC` | 0 | Host traffic is normal; unblock its link if it was previously blocked |
| `ATTACK_IN` | 1 | Host is a victim (under attack); do not restrict its incoming traffic |
| `ATTACK_OUT` | 2 | Host is an attacker; block its outgoing traffic via SDN drop rule |

Actions are applied **per host** through the wrapper. The base environment still exposes a joint action space of size `3^N` (base-3 encoding), but the wrapper always presents `Discrete(3)` to the agent and accumulates decisions across the N micro-steps of a round.

---

## State Representation

The internal **global state** is an N×8 flat array (N = number of hosts), updated once per round by reading OVS `dump-ports` counters.

Each host is assigned a **status ID**:

| ID | Status | Description |
|----|--------|-------------|
| 0 | `normal` | No active attack |
| 1 | `under_attack` | Receiving attack traffic |
| 2 | `attacking` | Sending attack traffic |
| 3 | `incoming_blocked_attack` | Victim, but all its attackers are currently blocked (mitigated) |
| 4 | `out_attack_blocked` | Attacker whose outgoing link has been dropped |

The wrapper maps `incoming_blocked_attack` (ID=3) back to `under_attack` (ID=1) for reward and correctness purposes, since from the agent's perspective the host is still a victim — the effective status is what drives the reward signal.

---

## Reward Function

Rewards are computed **per host per micro-step** inside `PerHostScanWrapper`. All values are scaled by an **attack reward scale** that compensates for class imbalance:

```
attack_reward_scale = clamp((num_hosts − 1) / 2, 1.0, 8.0)
```

Examples: 3 hosts → ×1.0 | 6 hosts → ×2.5 | 10 hosts → ×4.5

| Condition | Base reward | Notes |
|-----------|-------------|-------|
| Correct normal classification | +0.5 | Scaled (majority class, lower weight) |
| Correct victim detection (`ATTACK_IN`) | +3.0 | Scaled |
| Correct attacker detection (`ATTACK_OUT`) | +3.0 | Scaled |
| False positive (normal classified as attack) | −1.5 | Scaled |
| False negative (attack classified as normal) | −1.5 | Scaled |
| Wrong attack direction (IN↔OUT swapped) | −2.0 | Scaled — penalised more than FP to prevent hedging |
| Unnecessary link block (`LINK_OFF`) | −0.1 | Unscaled |

The asymmetric penalty for wrong direction (`-2.0` vs `-1.5`) discourages the agent from "guessing" attack presence without correctly distinguishing victim from attacker.

---

## PerHostScanWrapper — Key Design Choices

The wrapper is the critical component that makes the per-host observable pattern work:

### Micro-step decomposition
A single logical round (one call to `base_env.update_state()`) is split into N sequential micro-steps:
1. `reset()` → reads full N×8 snapshot → returns slice for host 0.
2. `step(action)` for host i → computes reward for host i → executes SDN action → returns slice for host i+1.
3. After host N−1 → calls `base_env.update_state()` → new snapshot is acquired.

All N micro-steps in a round see the **same frozen network snapshot**, guaranteeing consistent ground truth across the round.

### Ground truth capture
`host_status_id` is read from `global_state` **once at the very beginning of `step()`**, before any action is executed. This prevents reward leakage caused by state mutations during action execution.

### Link unblock policy (anti-oscillation)
To prevent rapid block/unblock cycles, a blocked host's link is only restored when:
1. At least `unblock_min_hold_rounds` (default: 2) full rounds have elapsed since blocking.
2. The agent has predicted `NORMAL_TRAFFIC` for that host in at least `unblock_required_normal_streak` (default: 2) consecutive rounds.

### Observation mode
The wrapper always returns **raw** observations. Returning normalised values was found to cause two bugs:
- DQN received `[0,1]` values that collapsed Q-value learning.
- Tabular agents passed `[0,1]` values to `get_discretized_state()`, which expects raw units, causing all values to fall into the first bin.

---

## Configuration Parameters

Default values in `scenario_env_param.yaml`:

```yaml
attacks:
  likely: 0.45                       # General attack probability
  likely_train: 0.9                  # Attack rate during training (high, to expose rare class)
  likely_eval: 0.3                   # Attack rate during evaluation
  short_attack_duration: 5           # Short attack lasts 5 steps
  long_attack_duration: 25           # Long attack lasts 25 steps
  no_attack_timeout: 3               # Steps between consecutive attacks on the same host
  unblock_min_hold_rounds: 2         # Minimum rounds a link stays blocked
  unblock_required_normal_streak: 2  # Consecutive NORMAL decisions needed to unblock
  apply_drop_rules: false            # Whether SDN drop rules are actually pushed to OVS
  thresholds:
    packets: 22000
    var_packets: 50
    bytes: 423000000
    var_bytes: 30
```

---

## File Structure

| File | Role |
|------|------|
| [attack_detect_ho.py](attack_detect_ho.py) | Main entry point: generates scenario, trains agents sequentially, evaluates, prints summary |
| [network_env_attack_detect_per_host_observable.py](network_env_attack_detect_per_host_observable.py) | Gymnasium environment backed by live Mininet; reads OVS counters; replays scenario tasks |
| [per_host_scan_wrapper.py](per_host_scan_wrapper.py) | Decomposes N-host round into N micro-steps; constant `Box(8,)` / `Discrete(3)` spaces; per-host rewards |
| [constants.py](constants.py) | Action IDs, reward values, host/agent status mappings |
| [instant_state.py](instant_state.py) | Per-host feature container; updates `global_state` from OVS counter reads |
| [scenario_env_param.yaml](scenario_env_param.yaml) | Default scenario parameters (attack rates, thresholds, unblock policy) |
