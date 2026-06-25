# scenario_generator.py
"""
Synthetic scenario generator for the ATTACKS_HO sequential mode.

Generates a deterministic sequence of host task assignments (who does what,
towards whom) WITHOUT executing any real network traffic.  The sequence is
saved to scenario.json in the experiment directory and later replayed by
the environment to drive real traffic generation — one step at a time,
one agent at a time — so all agents are trained and evaluated on the exact
same scenario.

Scenario file format
--------------------
{
    "training": [
        {
            "step": 0,          -- global step index across all episodes
            "episode": 0,       -- episode index (0-based)
            "episode_step": 0,  -- step index within the episode (0-based)
            "h1": {"task_type": "normal",       "traffic_type": "tcp",
                   "destination": "h3"},
            "h2": {"task_type": "short_attack", "traffic_type": "attack",
                   "destination": "h5", "attack_subtype": "udp_flood"},
            ...
        },
        ...  (episodes * max_steps entries)
    ],
    "evaluation": [
        ...  (test_episodes entries, 1 step per episode)
        -- evaluation is generated with attack_likely = EVAL_ATTACK_RATE
        -- (default 0.3) to target roughly 30% attack selection at decision time.
    ],
    "statistics": {
        "training": {
            "episodes":    30,
            "max_steps":   80,
            "total_steps": 2400,
            "global": {
                "normal":       1900,
                "short_attack": 350,
                "long_attack":  150,
                "tcp":          600,
                "udp":          400,
                "ping":         300,
                "none":         600,
                "attack":       500
            },
            "hosts": {
                "h1": {
                    "normal":       {"tcp": 45, "udp": 20, "ping": 15, "none": 10},
                    "short_attack": 8,
                    "long_attack":  2
                },
                ...
            }
        },
        "evaluation": { ... same structure ... }
    }
}
"""

import json
import os
import random
import time
from collections import defaultdict
from typing import List, Dict

from utility.constants import LONG_ATTACK, NORMAL, SHORT_ATTACK
from utility.my_log import information


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Available attack subtypes — mirrors adversarial_agent.py
_ATTACK_SUBTYPES = ["udp_flood", "syn_flood", "icmp_flood"]

# Default target fraction of steps that should be attacks for ATTACKS_HO.
# These are FRACTIONS (not per-decision probabilities); _generate_sequence
# converts them to per-decision probabilities via the Markov steady-state
# inverse so the actual fraction of attack steps matches these values.
DEFAULT_ATTACK_LIKELY_TRAIN = 0.5   # 50 % attack steps: balanced training
DEFAULT_ATTACK_LIKELY_EVAL = 0.3    # 30 % attack episodes during evaluation
DEFAULT_NO_ATTACK_TIMEOUT = 15

_DEFAULT_DURATION_STEPS = {
    NORMAL: 3,
    SHORT_ATTACK: 5,
    LONG_ATTACK: 30,
}

# Average task durations in steps (1 step ≈ 1 second)
_DURATION_STEPS = dict(_DEFAULT_DURATION_STEPS)
_NO_ATTACK_TIMEOUT = DEFAULT_NO_ATTACK_TIMEOUT


def _normalize_probability(value, default=0.0):
    """
    Normalize probability-like inputs to [0, 1].

    Supports both fractions (0.3) and percentages (30).
    """
    try:
        p = float(value)
    except (TypeError, ValueError):
        p = float(default)

    if p > 1.0:
        p = p / 100.0
    return max(0.0, min(1.0, p))




# ──────────────────────────────────────────────────────────────────────────────
# Internal host state tracker

def _set_attack_timing_params(short_duration=None, long_duration=None, 
                               no_attack_timeout=None) -> None:
    """
    Dynamically configure attack timing parameters for a scenario run.
    
    Allows users to override default values from config YAML.
    If a parameter is None, the global default is kept.
    
    Args:
        short_duration: Duration (steps) for SHORT_ATTACK (default 5, range 2-10)
        long_duration: Duration (steps) for LONG_ATTACK (default 30, range 10-60)
        no_attack_timeout: Timeout (steps) between attacks (default 15, range 5-30)
    """
    global _DURATION_STEPS, _NO_ATTACK_TIMEOUT

    # Reset to deterministic defaults on every call to avoid cross-run leakage.
    _DURATION_STEPS = dict(_DEFAULT_DURATION_STEPS)
    _NO_ATTACK_TIMEOUT = DEFAULT_NO_ATTACK_TIMEOUT
    
    if short_duration is not None:
        short_duration = max(2, min(int(short_duration), 10))  # Clamp to [2, 10]
        _DURATION_STEPS[SHORT_ATTACK] = short_duration
        information(f"[Scenario] SHORT_ATTACK duration set to {short_duration} steps\n")
    
    if long_duration is not None:
        long_duration = max(10, min(int(long_duration), 60))  # Clamp to [10, 60]
        _DURATION_STEPS[LONG_ATTACK] = long_duration
        information(f"[Scenario] LONG_ATTACK duration set to {long_duration} steps\n")
    
    if no_attack_timeout is not None:
        no_attack_timeout = max(5, min(int(no_attack_timeout), 30))  # Clamp to [5, 30]
        _NO_ATTACK_TIMEOUT = no_attack_timeout
        information(f"[Scenario] No-attack timeout set to {no_attack_timeout} steps\n")
# ──────────────────────────────────────────────────────────────────────────────

class _HostState:
    """
    Tracks the synthetic task state for a single host across steps.
    Mirrors the host_tasks dict used by the live adversarial agent.
    """
    __slots__ = ('task_type', 'traffic_type', 'destination',
                 'attack_subtype', 'remaining_steps')

    def __init__(self, traffic_types: list):
        self.task_type       = NORMAL
        self.traffic_type    = random.choice(traffic_types)
        self.destination     = None
        self.attack_subtype  = None
        self.remaining_steps = 0

    def to_dict(self) -> dict:
        d = {
            "task_type":    self.task_type,
            "traffic_type": self.traffic_type,
            "destination":  self.destination,
        }
        if self.attack_subtype:
            d["attack_subtype"] = self.attack_subtype
        return d


# ──────────────────────────────────────────────────────────────────────────────
# Task type selection — uses synthetic clock instead of time.time()
# ──────────────────────────────────────────────────────────────────────────────

def _choose_task_type(attack_likely: float,
                      last_short_ts: float,
                      last_long_ts:  float,
                      now:           float,
                      max_attack_percentage: float = 0.5) -> tuple:
    """
    Synthetic version of adversarial_agent.choose_task_type().

    Uses a synthetic clock (now = global_step counter) instead of time.time()
    so that cooldown comparisons work correctly during instant batch generation.
    Without this fix, all cooldown timestamps are set to wall-clock future values
    that are never reached during the loop, blocking all attacks after the first.

    Returns (task_type, new_last_short_ts, new_last_long_ts).
    """
    attack_likely = max(0.0, min(attack_likely, max_attack_percentage))
    rand      = random.random()
    pct_short = 0.66

    # Cooldown active — force normal traffic
    if now < last_long_ts or now < last_short_ts:
        return NORMAL, last_short_ts, last_long_ts

    if rand < attack_likely * pct_short:
        new_short = now + _DURATION_STEPS[SHORT_ATTACK] + _NO_ATTACK_TIMEOUT
        return SHORT_ATTACK, new_short, last_long_ts

    if rand < attack_likely:
        new_long = now + _DURATION_STEPS[LONG_ATTACK] + _NO_ATTACK_TIMEOUT
        return LONG_ATTACK, last_short_ts, new_long

    return NORMAL, last_short_ts, last_long_ts


# ──────────────────────────────────────────────────────────────────────────────
# Core sequence generator
# ──────────────────────────────────────────────────────────────────────────────

def _generate_sequence(hosts,
                        traffic_types:      list,
                        attack_likely_init: float,
                        n_episodes:         int,
                        max_steps:          int,
                        max_attack_percentage: float = 0.5) -> List[Dict]:
    """
    Generate n_episodes * max_steps synthetic task assignments.

    At most ONE host attacks at a time, targeting ONE victim, for the full
    duration of the attack (_DURATION_STEPS: 5 steps for SHORT_ATTACK,
    30 for LONG_ATTACK by default).  After the attack ends a cooldown of
    _NO_ATTACK_TIMEOUT steps is enforced before a new attack can start.

    _choose_task_type() uses a synthetic clock (global_step) for cooldown
    comparisons so all cooldowns are correctly honoured even though the
    entire sequence is generated in a tight loop without wall-clock time.

    Each entry in the returned list is a step dict with keys:
      "step", "episode", "episode_step", and one key per host name.
    """
    host_names = [h.name for h in hosts]
    pct_short  = 0.66

    # Interpret attack_likely_init as the TARGET FRACTION of steps that should
    # be attacks rather than a raw per-decision probability.  When multi-step
    # attacks are possible (max_steps > 1), the effective attack fraction in
    # steady-state is p*E/(1+p*(E-1)) where p is the per-decision probability
    # and E is the mean attack duration.  We invert that formula so that the
    # user-facing parameter directly controls the fraction of attack steps:
    #   p = fraction / (E - fraction*(E-1))
    # For max_steps==1 every episode is a fresh decision so fraction==p.
    target_fraction = max(0.0, min(float(attack_likely_init), max_attack_percentage))
    if max_steps > 1:
        mean_duration = (pct_short * _DURATION_STEPS[SHORT_ATTACK] +
                         (1.0 - pct_short) * _DURATION_STEPS[LONG_ATTACK])
        if mean_duration > 1.0 and target_fraction < 1.0:
            denom = mean_duration - target_fraction * (mean_duration - 1.0)
            attack_likely = min(1.0, target_fraction / denom) if denom > 0.0 else 1.0
        else:
            attack_likely = target_fraction
        attack_likely = max(0.0, attack_likely)
    else:
        attack_likely = target_fraction

    steps = []
    global_step = 0

    # Global attack state — at most one attacker and one victim at a time.
    # No cooldown between attacks: attack_likely directly controls the
    # probability of starting a new attack at each non-attack decision step,
    # so the parameter is honoured as configured.
    g_attacker  = None   # name of the host currently attacking
    g_victim    = None   # name of the host being attacked
    g_task_type = None   # SHORT_ATTACK or LONG_ATTACK
    g_subtype   = None   # udp_flood / syn_flood / icmp_flood
    g_remaining = 0      # steps still left in the current attack

    for episode in range(n_episodes):
        # Stop any in-progress attack at each episode boundary so multi-step
        # attacks do not bleed across episodes.
        g_attacker  = None
        g_victim    = None
        g_task_type = None
        g_subtype   = None
        g_remaining = 0

        for episode_step in range(max_steps):
            step_entry = {
                "step":         global_step,
                "episode":      episode,
                "episode_step": episode_step,
            }

            if g_remaining > 0:
                # Ongoing attack — continue for one more step.
                g_remaining -= 1
            else:
                # No active attack — start a new one with probability attack_likely.
                g_attacker  = None
                g_victim    = None
                g_task_type = None
                g_subtype   = None

                if random.random() < attack_likely:
                    g_attacker = random.choice(host_names)
                    others     = [n for n in host_names if n != g_attacker]
                    g_victim   = random.choice(others) if others else g_attacker
                    g_task_type = (SHORT_ATTACK if random.random() < pct_short
                                   else LONG_ATTACK)
                    g_subtype   = random.choice(_ATTACK_SUBTYPES)
                    # -1 because the current step already counts as step 1.
                    g_remaining = _DURATION_STEPS[g_task_type] - 1

            # Build per-host entries for this step.
            for name in host_names:
                others = [n for n in host_names if n != name]
                if name == g_attacker:
                    step_entry[name] = {
                        "task_type":      g_task_type,
                        "traffic_type":   "attack",
                        "destination":    g_victim,
                        "attack_subtype": g_subtype,
                    }
                else:
                    step_entry[name] = {
                        "task_type":    NORMAL,
                        "traffic_type": random.choice(traffic_types),
                        "destination":  random.choice(others) if others else None,
                    }

            steps.append(step_entry)
            global_step += 1

    return steps


# ──────────────────────────────────────────────────────────────────────────────
# Statistics computation
# ──────────────────────────────────────────────────────────────────────────────

def _compute_statistics(sequence: List[Dict],
                         host_names: List[str],
                         n_episodes: int,
                         max_steps:  int) -> dict:
    """
    Compute per-host and global traffic statistics for a sequence.

        Returns a dict with:
            episodes, max_steps, total_steps,
            global {
                    normal: {total, none, ping, udp, tcp},
                    attack: {total, short, long}
            },
            hosts   {host_name: {normal: {tcp:n, udp:n, ...},
                                                     short_attack: n, long_attack: n}}
    """
    total_steps = len(sequence)

    # Per-host counters
    host_stats = {
        name: {
            "normal":       defaultdict(int),  # traffic_type → count
            "short_attack": 0,
            "long_attack":  0,
        }
        for name in host_names
    }

    # Global counters
    global_counts = defaultdict(int)

    for step in sequence:
        for name in host_names:
            info = step.get(name, {})
            task    = info.get("task_type",    NORMAL)
            traffic = info.get("traffic_type", "none")

            if task == NORMAL:
                host_stats[name]["normal"][traffic] += 1
                global_counts["normal"]  += 1
                global_counts[traffic]   += 1
            elif task == SHORT_ATTACK:
                host_stats[name]["short_attack"] += 1
                global_counts["short_attack"]    += 1
                global_counts["attack"]          += 1
            elif task == LONG_ATTACK:
                host_stats[name]["long_attack"]  += 1
                global_counts["long_attack"]     += 1
                global_counts["attack"]          += 1

    # Convert defaultdicts to plain dicts for JSON serialisation
    hosts_out = {}
    for name in host_names:
        hosts_out[name] = {
            "normal":       dict(host_stats[name]["normal"]),
            "short_attack": host_stats[name]["short_attack"],
            "long_attack":  host_stats[name]["long_attack"],
        }

    normal_counts = {
        "none": global_counts.get("none", 0),
        "ping": global_counts.get("ping", 0),
        "udp":  global_counts.get("udp", 0),
        "tcp":  global_counts.get("tcp", 0),
    }
    normal_counts["total"] = sum(normal_counts.values())

    attack_counts = {
        "short": global_counts.get("short_attack", 0),
        "long":  global_counts.get("long_attack", 0),
    }
    attack_counts["total"] = attack_counts["short"] + attack_counts["long"]

    return {
        "episodes":    n_episodes,
        "max_steps":   max_steps,
        "total_steps": total_steps,
        "global": {
            "normal": normal_counts,
            "attack": attack_counts,
        },
        "hosts":       hosts_out,
    }


def _build_attack_step_series(sequence: List[Dict], host_names: List[str]) -> List[int]:
    """Return a 0/1 series indicating whether at least one host is attacking at each step."""
    series = []
    for step in sequence:
        has_attack = 0
        for name in host_names:
            info = step.get(name, {}) if isinstance(step, dict) else {}
            if str(info.get("task_type", NORMAL)) != NORMAL:
                has_attack = 1
                break
        series.append(has_attack)
    return series


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def generate_and_save_scenario(base_env,
                                config,
                                scenario_path: str,
                                train_attack_likely: float = DEFAULT_ATTACK_LIKELY_TRAIN,
                                eval_attack_likely: float = DEFAULT_ATTACK_LIKELY_EVAL) -> dict:
    """
    Generate training + evaluation sequences, compute statistics, and save
    everything to scenario_path as a single JSON file.

    Training and evaluation use fixed attack_likely values:
    - training:   0.9 (or config.env_params.attacks.likely_train if specified)
    - evaluation: 0.3 (or config.env_params.attacks.likely_eval if specified)
    
    These values can be configured per experiment in the YAML config file
    under env_params.attacks.likely_train and env_params.attacks.likely_eval
    to control attack frequency during training vs evaluation phases.

    Args:
        base_env:            NetworkEnvAttackDetectPerHostObservable instance.
        config:              Experiment config (may contain likely_train/likely_eval overrides).
        scenario_path:       Full path for scenario.json.
        train_attack_likely: Override attack likelihood for training (default from DEFAULT_ATTACK_LIKELY_TRAIN).
        eval_attack_likely:  Override attack likelihood for evaluation (default from DEFAULT_ATTACK_LIKELY_EVAL).

    Returns:
        The scenario dict.
    """
    hosts              = base_env.hosts
    host_names         = [h.name for h in hosts]
    traffic_types      = base_env.net.traffic_types
    attack_likely_init = base_env.init_attack_likely
    episodes           = config.env_params.episodes
    max_steps          = config.env_params.max_steps
    test_episodes      = config.env_params.test_episodes
    num_hosts          = len(hosts)

    # Load optional attack timing and percentage parameters from config (for ATTACKS_HO distribution)
    # If not specified, uses global _DURATION_STEPS and _NO_ATTACK_TIMEOUT constants
    attack_config = getattr(config.env_params, 'attacks', {})
    if isinstance(attack_config, dict):
        short_duration = attack_config.get('short_attack_duration', None)
        long_duration  = attack_config.get('long_attack_duration', None)
        no_attack_tmout = attack_config.get('no_attack_timeout', None)
        max_attack_percentage = attack_config.get('max_attack_percentage', 0.99)
    else:
        short_duration = getattr(attack_config, 'short_attack_duration', None)
        long_duration  = getattr(attack_config, 'long_attack_duration', None)
        no_attack_tmout = getattr(attack_config, 'no_attack_timeout', None)
        max_attack_percentage = getattr(attack_config, 'max_attack_percentage', 0.99)
    
    # Set global timing parameters for this run
    _set_attack_timing_params(
        short_duration=short_duration,
        long_duration=long_duration,
        no_attack_timeout=no_attack_tmout
    )

    n_train = episodes * max_steps
    n_eval = test_episodes

    # Use explicit scenario-level attack rates, clamped by max_attack_percentage.
    max_attack_percentage = _normalize_probability(max_attack_percentage, default=0.99)
    attack_likely_train = min(max_attack_percentage, _normalize_probability(train_attack_likely, default=DEFAULT_ATTACK_LIKELY_TRAIN))
    attack_likely_eval = min(max_attack_percentage, _normalize_probability(eval_attack_likely, default=DEFAULT_ATTACK_LIKELY_EVAL))

    information(
        f"[Scenario] Generating training sequence: "
        f"{episodes} episodes × {max_steps} steps = {n_train} steps "
        f"(attack_likely_config={attack_likely_init} → used "
        f"{attack_likely_train:.4f}; hosts={num_hosts})\n"
    )
    training_seq = _generate_sequence(
        hosts, traffic_types, attack_likely_train,
        episodes, max_steps, max_attack_percentage
    )

    information(
        f"[Scenario] Generating evaluation sequence: "
        f"{test_episodes} episodes = {n_eval} steps "
        f"(attack_likely_config={attack_likely_init} → used "
        f"{attack_likely_eval:.4f}; hosts={num_hosts})\n"
    )
    evaluation_seq = _generate_sequence(
        hosts, traffic_types, attack_likely_eval,
        test_episodes, 1, max_attack_percentage
    )

    # Compute statistics
    train_stats = _compute_statistics(
        training_seq, host_names, episodes, max_steps
    )
    eval_stats  = _compute_statistics(
        evaluation_seq, host_names, test_episodes, 1
    )
    # Record the actual attack_likely values used for reproducibility
    train_stats["attack_likely_used"] = attack_likely_train
    eval_stats["attack_likely_used"]  = attack_likely_eval
    train_stats["attack_likely_config"] = attack_likely_init
    eval_stats["attack_likely_config"]  = attack_likely_init
    # Step-level attack series needed by the UI chart in the saved scenario popup
    train_stats["attack_step_series"] = _build_attack_step_series(training_seq, host_names)
    eval_stats["attack_step_series"]  = _build_attack_step_series(evaluation_seq, host_names)

    scenario = {
        "training":   training_seq,
        "evaluation": evaluation_seq,
        "statistics": {
            "training":   train_stats,
            "evaluation": eval_stats,
        },
    }

    with open(scenario_path, 'w') as f:
        json.dump(scenario, f, indent=2)

    # Print a brief summary to console
    _print_statistics_summary(train_stats, eval_stats)

    information(f"[Scenario] Saved to {scenario_path}\n")
    return scenario


def preview_scenario_statistics_from_config(config_dict: dict,
                                            train_attack_likely: float = DEFAULT_ATTACK_LIKELY_TRAIN,
                                            eval_attack_likely: float = DEFAULT_ATTACK_LIKELY_EVAL) -> dict:
    """
    Generate ATTACKS_HO scenario statistics in memory from a config dict.

    This function does not save any scenario.json file. It is intended for
    quick UI previews so users can evaluate scenario balance before training.

    Returns:
        {"training": {...}, "evaluation": {...}} statistics dict.
    """
    config_dict = config_dict or {}
    env_params = config_dict.get("env_params", {}) if isinstance(config_dict.get("env_params", {}), dict) else {}
    attacks = env_params.get("attacks", {}) if isinstance(env_params.get("attacks", {}), dict) else {}
    net_params = env_params.get("net_params", {}) if isinstance(env_params.get("net_params", {}), dict) else {}

    episodes = int(env_params.get("episodes", 0) or 0)
    max_steps = int(env_params.get("max_steps", 0) or 0)
    test_episodes = int(env_params.get("test_episodes", 0) or 0)
    num_hosts = int(net_params.get("num_hosts", 0) or 0)
    num_iots = int((net_params.get("num_iots", None) if isinstance(net_params, dict) else None) or net_params.get("num_iot", 0) or 0)
    traffic_types = net_params.get("traffic_types", ["none", "ping", "udp", "tcp"])
    if not isinstance(traffic_types, list) or len(traffic_types) == 0:
        traffic_types = ["none", "ping", "udp", "tcp"]

    if episodes <= 0 or max_steps <= 0 or test_episodes <= 0 or num_hosts <= 0:
        raise ValueError(
            "Invalid config for scenario preview: episodes, max_steps, test_episodes, and num_hosts must be > 0"
        )

    class _PreviewHost:
        __slots__ = ("name",)

        def __init__(self, name: str):
            self.name = name

    hosts = [_PreviewHost(f"h{i + 1}") for i in range(num_hosts)]
    hosts += [_PreviewHost(f"iot{i + 1}") for i in range(num_iots)] if num_iots > 0 else []
    host_names = [h.name for h in hosts]
    attack_likely_config = float(attacks.get("likely", 0.0) or 0.0)

    short_duration = attacks.get("short_attack_duration", None)
    long_duration = attacks.get("long_attack_duration", None)
    no_attack_tmout = attacks.get("no_attack_timeout", None)
    max_attack_percentage = _normalize_probability(attacks.get("max_attack_percentage", 0.99), default=0.99)

    _set_attack_timing_params(
        short_duration=short_duration,
        long_duration=long_duration,
        no_attack_timeout=no_attack_tmout,
    )

    attack_likely_train = min(max_attack_percentage, _normalize_probability(train_attack_likely, default=DEFAULT_ATTACK_LIKELY_TRAIN))
    attack_likely_eval = min(max_attack_percentage, _normalize_probability(eval_attack_likely, default=DEFAULT_ATTACK_LIKELY_EVAL))

    training_seq = _generate_sequence(
        hosts,
        traffic_types,
        attack_likely_train,
        episodes,
        max_steps,
        max_attack_percentage,
    )
    evaluation_seq = _generate_sequence(
        hosts,
        traffic_types,
        attack_likely_eval,
        test_episodes,
        1,
        max_attack_percentage,
    )

    train_stats = _compute_statistics(training_seq, host_names, episodes, max_steps)
    eval_stats = _compute_statistics(evaluation_seq, host_names, test_episodes, 1)

    train_stats["attack_step_series"] = _build_attack_step_series(training_seq, host_names)
    eval_stats["attack_step_series"] = _build_attack_step_series(evaluation_seq, host_names)

    train_stats["attack_likely_used"] = attack_likely_train
    eval_stats["attack_likely_used"] = attack_likely_eval
    train_stats["attack_likely_config"] = attack_likely_config
    eval_stats["attack_likely_config"] = attack_likely_config

    return {
        "training": train_stats,
        "evaluation": eval_stats,
    }


def load_scenario(scenario_path: str) -> dict:
    """
    Load a scenario from scenario.json.
    Raises FileNotFoundError with a clear message if the file is missing.
    """
    if not os.path.isfile(scenario_path):
        raise FileNotFoundError(
            f"[Scenario] scenario.json not found at '{scenario_path}'.\n"
            f"To use attacks_ho_from_dataset you must provide a scenario.json "
            f"generated by a previous attacks_ho run.\n"
            f"Set gym_type: attacks_ho to generate a new one."
        )

    with open(scenario_path, 'r') as f:
        scenario = json.load(f)

    n_train = len(scenario.get("training",   []))
    n_eval  = len(scenario.get("evaluation", []))
    information(
        f"[Scenario] Loaded from {scenario_path} "
        f"({n_train} training + {n_eval} evaluation steps)\n"
    )

    # Print statistics if available
    if "statistics" in scenario:
        _print_statistics_summary(
            scenario["statistics"].get("training",   {}),
            scenario["statistics"].get("evaluation", {}),
        )

    return scenario


# ──────────────────────────────────────────────────────────────────────────────
# Console summary
# ──────────────────────────────────────────────────────────────────────────────

def _print_statistics_summary(train_stats: dict, eval_stats: dict):
    """Print a compact statistics summary to the console."""
    line = "─" * 60

    def _section(label: str, stats: dict):
        if not stats:
            return
        g = stats.get("global", {})
        # Support both formats:
        # - new: {normal:{total,none,ping,udp,tcp}, attack:{total,short,long}}
        # - legacy: flat keys {normal, short_attack, long_attack, tcp, udp, ping, none, attack}
        if isinstance(g.get("normal"), dict):
            g_normal = g.get("normal", {})
            g_attack = g.get("attack", {})
            normal_total = g_normal.get("total", 0)
            normal_tcp = g_normal.get("tcp", 0)
            normal_udp = g_normal.get("udp", 0)
            normal_ping = g_normal.get("ping", 0)
            normal_none = g_normal.get("none", 0)
            short_attack = g_attack.get("short", 0)
            long_attack = g_attack.get("long", 0)
            attack_total = g_attack.get("total", short_attack + long_attack)
        else:
            normal_total = g.get("normal", 0)
            normal_tcp = g.get("tcp", 0)
            normal_udp = g.get("udp", 0)
            normal_ping = g.get("ping", 0)
            normal_none = g.get("none", 0)
            short_attack = g.get("short_attack", 0)
            long_attack = g.get("long_attack", 0)
            attack_total = g.get("attack", short_attack + long_attack)

        total_host_steps = (stats.get('total_steps', 1)
                            * len(stats.get('hosts', {'_': None})))
        attack_pct = 100 * attack_total / max(1, total_host_steps)
        al_used   = stats.get("attack_likely_used",   "?")
        al_config = stats.get("attack_likely_config", "?")
        information(
            f"\n  {label}\n"
            f"    Episodes: {stats.get('episodes', '?')}  "
            f"Steps/ep: {stats.get('max_steps', '?')}  "
            f"Total: {stats.get('total_steps', '?')}\n"
            f"    attack_likely config={al_config}  used={al_used:.4f}\n"
            f"    Normal:       {normal_total:>6}  "
            f"(tcp={normal_tcp} udp={normal_udp} "
            f"ping={normal_ping} none={normal_none})\n"
            f"    Short attack: {short_attack:>6}\n"
            f"    Long attack:  {long_attack:>6}\n"
            f"    Attack total: {attack_total:>6}  "
            f"({attack_pct:.1f}% of host-steps)\n"
        )
        # Per-host summary
        for host_name, hstats in stats.get("hosts", {}).items():
            normal_total  = sum(hstats.get("normal", {}).values())
            attack_total  = (hstats.get("short_attack", 0)
                             + hstats.get("long_attack",  0))
            normal_detail = ", ".join(
                f"{k}={v}"
                for k, v in sorted(hstats.get("normal", {}).items())
                if v > 0
            )
            information(
                f"    {host_name:>4}  normal={normal_total:>5} ({normal_detail})  "
                f"attack={attack_total:>4} "
                f"(short={hstats.get('short_attack',0)} "
                f"long={hstats.get('long_attack',0)})\n"
            )

    information(f"\n{line}\n  SCENARIO STATISTICS\n{line}\n")
    _section("TRAINING",   train_stats)
    _section("EVALUATION", eval_stats)
    information(f"{line}\n")