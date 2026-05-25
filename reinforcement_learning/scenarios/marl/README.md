# Scenario: marl — Multi-Agent Attack Detection

> **Status: Work in Progress**
> This scenario is partially implemented. Core infrastructure is in place, but several features are incomplete or disabled. See the [Known Limitations](#known-limitations) section.

## Overview

`marl` implements a **hierarchical multi-agent** attack detection system. Unlike `attack_ho` where a single agent scans hosts sequentially, here each host runs its own independent agent, plus a **coordinator agent** that monitors the network globally. Agents communicate through a shared state object, allowing the coordinator to broadcast alerts to host agents.

The goal is distributed attack detection: host agents classify their own local traffic, while the coordinator provides a network-wide binary verdict (normal vs. attack).

---

## Architecture

```
Coordinator Agent  ←──── global network state (packets, bytes, Δ)
       │
       │  alert message (action broadcast)
       ▼
Host Agent h1 | Host Agent h2 | ... | Host Agent hN
   ↕ local state (TX + RX per host + message count)
      OVS Switch  (SDN, OpenDaylight)
```

- **Coordinator:** one agent, observes aggregated network totals, decides NORMAL vs. ATTACK (2 actions)
- **Host agents:** one per host, observe per-host TX+RX statistics plus incoming message count, decide NORMAL / ATTACK\_IN / ATTACK\_OUT (3 actions)
- **Communication:** actions are written into a shared `InstantState` object; host agents read the count of non-zero messages as an extra observation dimension

---

## Observation Spaces

### Coordinator — `Box(shape=(5,), dtype=float32)`

| Index | Feature | Bounds |
|-------|---------|--------|
| 0 | Total RX packets | `[0, threshold_packets × num_hosts]` |
| 1 | ΔRX packets (% change) | `[-threshold_var_packets, +threshold_var_packets]` |
| 2 | Total RX bytes | `[0, threshold_bytes × num_hosts]` |
| 3 | ΔRX bytes (% change) | `[-threshold_var_bytes, +threshold_var_bytes]` |
| 4 | Incoming attack message count | `[0, num_hosts]` |

### Host Agent — `Box(shape=(9,), dtype=float32)`

| Index | Feature | Bounds |
|-------|---------|--------|
| 0 | TX packets | `[0, threshold_packets × num_hosts]` |
| 1 | ΔTX packets (% change) | `[-threshold_var_packets, +threshold_var_packets]` |
| 2 | TX bytes | `[0, threshold_bytes × num_hosts]` |
| 3 | ΔTX bytes (% change) | `[-threshold_var_bytes, +threshold_var_bytes]` |
| 4 | RX packets | `[0, threshold_packets × num_hosts]` |
| 5 | ΔRX packets (% change) | `[-threshold_var_packets, +threshold_var_packets]` |
| 6 | RX bytes | `[0, threshold_bytes × num_hosts]` |
| 7 | ΔRX bytes (% change) | `[-threshold_var_bytes, +threshold_var_bytes]` |
| 8 | Incoming message count | `[0, num_hosts]` |

Index 8 is a social signal: it counts how many other agents (coordinator and peers) sent a non-zero (i.e. attack) action in the previous step.

Observations are **normalised** before being passed to SB3 agents via `get_normalized_state()`. Tabular agents call `get_discretized_state()` directly on raw values.

---

## Action Spaces

### Coordinator — `Discrete(2)`

| Action | ID | Meaning |
|--------|----|---------|
| `NORMAL_TRAFFIC` | 0 | No network-wide attack detected |
| `ATTACK` | 1 | Broadcast attack alert to all host agents |

### Host Agent — `Discrete(3)`

| Action | ID | Meaning |
|--------|----|---------|
| `NORMAL_TRAFFIC` | 0 | Host traffic is normal; unblock link if blocked |
| `INCOMING_ATTACK` | 1 | Host is a victim; do not restrict incoming traffic |
| `OUTGOING_ATTACK` | 2 | Host is an attacker; apply SDN drop rule |

---

## Reward Functions

### Host Agent

| Condition | Reward |
|-----------|--------|
| Correct normal classification | +1.0 |
| Correct incoming attack detection | +3.0 |
| Correct outgoing attack detection | +3.0 |
| Wrong attack direction (IN↔OUT swapped) | −1.0 |
| False positive (normal classified as attack) | −2.0 |
| False negative (attack classified as normal) | −3.0 |
| Link blocked (OUTGOING\_ATTACK action applied) | −0.1 |

### Coordinator

| Condition | Reward |
|-----------|--------|
| Correct attack alert | +2.0 |
| Correct normal (no alert) | +1.0 |
| False alarm (alert on normal) | −1.0 |
| Missed alert (normal on attack) | −2.0 |

### Team Bonus (currently disabled)

A team bonus (`TEAM_SUCCESSFUL = 0`) was designed to reward all agents when a sufficient fraction predict correctly in the same step. It is disabled by setting the constant to 0. When enabled, a full-team correct prediction would add `+5.0` and a 50%+ correct round would add `+2.5`.

---

## State Representation

The **global state** is a shared `InstantState` object that holds:
- Per-host 8-feature arrays (same structure as `attack_ho`, see that README)
- Coordinator state: `[packets, Δpackets, bytes, Δbytes]`
- Status dicts for the coordinator and each host
- Message store: `messages[agent_name][host_name] = action_id`

**Coordinator status IDs:**

| ID | Status |
|----|--------|
| −1 | idle |
| 0 | normal |
| 1 | attack |

**Host status IDs:**

| ID | Status |
|----|--------|
| −1 | idle |
| 0 | normal |
| 1 | under\_attack |
| 2 | attacking |
| 3 | attacking/under\_attack (not yet used) |

The coordinator status is derived automatically: if any host is under attack or attacking, the coordinator ground truth is set to `attack`.

---

## Agent Interaction & Training

Training is **parallel by thread**: each agent (coordinator + N host agents) trains in its own thread, all sharing the same live environment state.

```
Thread: coordinator agent ──┐
Thread: host agent h1       ├──► shared NetworkEnvMarlAttackDetect
Thread: host agent h2       │    (global_state updated by background thread)
...                         ┘
```

Each sub-environment (`CoordinatorEnv`, `HostAgentEnv`) wraps the shared `global_state` and exposes its own `observation_space`, `action_space`, `step()`, `reset()`, and `calculate_reward()`. The container class `NetworkEnvMarlAttackDetect` has stub implementations of `step()` / `execute_action()` / `calculate_reward()` — these are intentionally empty (dispatching is done in sub-envs).

A background thread updates `global_state` every `wait_after_read` seconds by reading OVS counters from the live Mininet network.

---

## Configuration Parameters

Default values in `scenario_env_param.yaml`:

```yaml
attacks:
  likely: 0.45
  likely_train: 0.9
  likely_eval: 0.3
  short_attack_duration: 5
  long_attack_duration: 25
  no_attack_timeout: 3
  unblock_min_hold_rounds: 2
  unblock_required_normal_streak: 2
  apply_drop_rules: false
  thresholds:
    packets: 22000
    var_packets: 50
    bytes: 423000000
    var_bytes: 30
```

Parameters mirror `attack_ho`. See that README for threshold semantics.

---

## Known Limitations

The following items are known to be incomplete or disabled:

- **Team reward disabled** — `TEAM_SUCCESSFUL = 0` in `constants.py`. The cooperative bonus logic exists in both `coordinator_env.py` and `host_agent_env.py` but produces no signal.
- **Link unblock logic incomplete** — the reward branch for a blocked host is a commented-out TODO in `host_agent_env.py`. Currently `LINK_OFF` is returned unconditionally when the link is down, regardless of the action taken.
- **Message encoding not validated** — the message appended to the observation is the raw action integer (0/1/2). Whether this encoding is meaningful for the receiving agent has not been verified (TODO in `host_agent_env.py`).
- **Team-level metrics not aggregated** — evaluation computes per-agent, per-host metrics but the cross-agent team accuracy/F1 is not implemented (TODO in `marl_attack_detect.py`).
- **Confusion matrix printing** — the code that converts multi-hot ground truth to scalar labels for `sklearn` confusion matrix is commented out.
- **Dataset mode unverified** — the state update path for `GYM_TYPE[MARL_ATTACKS]` mode has a commented-out `update_state()` call; dataset replay in MARL mode may not update state correctly.

---

## File Structure

| File | Role |
|------|------|
| [marl_attack_detect.py](marl_attack_detect.py) | Orchestration: parallel training threads, evaluation loop, metrics, plotting |
| [network_env_marl_attack_detect.py](network_env_marl_attack_detect.py) | Container env: initialises sub-envs, manages shared global state, background update thread |
| [coordinator_env.py](coordinator_env.py) | Coordinator sub-environment: 5-dim obs, 2-action space, coordinator reward |
| [host_agent_env.py](host_agent_env.py) | Host sub-environment: 9-dim obs, 3-action space, per-host reward, link control |
| [communication_bus.py](communication_bus.py) | Message queue abstraction (direct send + broadcast); currently superseded by shared `global_state` messages dict |
| [constants.py](constants.py) | Action IDs, reward values, status ID mappings for both coordinator and hosts |
| [instant_state.py](instant_state.py) | Shared state container: per-host arrays, coordinator state, message store |
| [scenario_env_param.yaml](scenario_env_param.yaml) | Default attack/threshold parameters |
