"""
MarlPzEnv — PettingZoo-style multi-agent environment for MARL attack detection.

Architecture
------------
* Inherits from NetworkEnv (not from PettingZoo ParallelEnv) to avoid the
  gymnasium.Env / ParallelEnv MRO conflict.
* Implements the PettingZoo Parallel interface via duck typing:
    reset()  → dict[agent_id, obs],  dict[agent_id, info]
    step(actions_dict) → obs, rewards, terminations, truncations, infos
* SingleAgentView (defined at module bottom) wraps MarlPzEnv to expose a
  standard gymnasium.Env interface per agent_id for tabular and SB3 agents.

State layout
------------
* Host agents observe their own traffic counters (+% variations if enabled)
  plus one "message" feature counting ATTACK alerts received from other agents.
* Coordinator observes aggregate counters over all hosts plus a message count.
* All observations are normalized to [0, 1].
* Sequential scenario replay via _apply_scenario_step() (ported from HO).
"""

import copy
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np
from colorama import Fore
from gymnasium import spaces
import gymnasium as gym

from reinforcement_learning.network_env import (
    NetworkEnv,
    get_custom_bin_index,
    get_linear_bin_index,
    get_log_bin_index,
    get_normalized_state,
)
from utility.constants import (
    GYM_TYPE,
    LONG_ATTACK,
    MARL_PZ,
    MARL_PZ_FROM_DATASET,
    NORMAL,
    SHORT_ATTACK,
    HostStatus,
    SystemLevels,
    SystemModes,
    SystemStatus,
    TrafficTypes,
)
from utility.my_log import debug, error, information, notify_client
from utility.network_configurator import (
    block_flow_drop,
    comunicate_in_attack_detected,
    comunicate_normal_traffic_detected,
    comunicate_out_attack_detected,
    format_bytes,
    unblock_flow_delete,
)

from .constants import (
    AGENT_ACTIONS,
    COORDINATOR,
    COORDINATOR_ACTIONS,
    COORDINATOR_REWARDS,
    COORDINATOR_STATUS_ID_MAPPING,
    HOST_STATUS_ID_MAPPING,
    NORMALIZED,
    RAW,
    REWARDS,
    CoordinatorActions,
)
from .instant_state import MarlPzInstantState


# ────────────────────────────────────────────────────────────────────────────
# Discretization helpers
# ────────────────────────────────────────────────────────────────────────────

def _discretize_obs(obs: np.ndarray, low: np.ndarray, high: np.ndarray,
                    n_bins: int,
                    counter_indices: set, variation_indices: set) -> tuple:
    """Map a raw observation vector to a discrete state tuple."""
    discrete = []
    for i, val in enumerate(obs):
        if i in variation_indices:
            idx = get_linear_bin_index(val, low[i], high[i], n_bins - 1) + 1
        elif i in counter_indices:
            if val <= 0:
                idx = 0
            else:
                h_safe = max(float(high[i]), 1.0)
                idx = get_log_bin_index(val, 1.0, h_safe, n_bins - 1) + 1
        else:
            idx = get_linear_bin_index(val, low[i], high[i], n_bins)
        discrete.append(int(idx))
    return tuple(discrete)


# ────────────────────────────────────────────────────────────────────────────
# MarlPzEnv
# ────────────────────────────────────────────────────────────────────────────

class MarlPzEnv(NetworkEnv):
    """
    PettingZoo-style parallel MARL environment for network attack detection.

    Each host has its own agent observing local traffic.
    A coordinator agent aggregates signals from all hosts.
    All agents act simultaneously each step.
    """

    metadata = {"name": "marl_pz_v1", "render_modes": ["human"]}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, params, server_user, existing_net=None):
        params.actions_number = AGENT_ACTIONS.NUMBER
        super().__init__(params, server_user, existing_net)
        self.params = params

        self.hosts = self.net.hosts
        self.num_hosts = len(self.hosts)

        # Traffic generation thread pool (one worker per host + 2 slack)
        _pool = self.num_hosts + getattr(params.net_params, 'num_iots', 0) + 2
        self._traffic_executor = ThreadPoolExecutor(
            max_workers=_pool, thread_name_prefix="marl_pz_traffic"
        )
        self._host_futures: Dict[str, object] = {}

        # Observation config
        self._include_pct_var = bool(
            getattr(params.attacks, 'include_percentage_variations', False)
        )
        self._apply_drop_rules = bool(
            getattr(params.attacks, 'apply_drop_rules', False)
        )
        self._use_coordinator = bool(
            getattr(params.attacks, 'use_coordinator', True)
        )
        self._unblock_min_hold = int(
            getattr(params.attacks, 'unblock_min_hold_rounds', 2)
        )
        self._unblock_normal_streak = int(
            getattr(params.attacks, 'unblock_required_normal_streak', 2)
        )

        # PettingZoo agent lists (coordinator included only if enabled)
        self.possible_agents: List[str] = (
            [h.name for h in self.net.hosts] + ([COORDINATOR] if self._use_coordinator else [])
        )
        self.agents: List[str] = list(self.possible_agents)

        # Per-host link/blocking state
        self.status_links = [True for _ in self.hosts]
        self._block_started_step: Dict[str, int] = {h.name: 0 for h in self.hosts}
        self._normal_streak_blocked: Dict[str, int] = {h.name: 0 for h in self.hosts}

        # Thresholds (used for obs bounds and discretization)
        self.threshold_packets = params.attacks.thresholds.packets
        self.threshold_var_packets = params.attacks.thresholds.var_packets
        self.threshold_bytes = params.attacks.thresholds.bytes
        self.threshold_var_bytes = params.attacks.thresholds.var_bytes

        # Build per-agent observation bounds and spaces
        self._build_spaces()

        # n_bins for tabular agents
        self.n_bins = params.n_bins

        # Shared state
        coordinator_state = np.zeros(4, dtype=np.float32)
        messages = self._init_messages()
        self.global_prev_state = self.global_state = MarlPzInstantState(
            self.hosts, coordinator_state, messages
        )

        # Episode counters (reset every episode)
        self._step_count: int = 0
        self._correct_count: int = 0
        self._total_count: int = 0

        # Per-step correct counts per agent (for info dict)
        self._agent_corrects: Dict[str, int] = {a: 0 for a in self.possible_agents}

        # Display flag
        self.show_complete_network_status = False
        self.statuses = []

        # One-time initialization flag (read from net/scenario once before training)
        self._env_initialized = False

        if self.gym_type == GYM_TYPE[MARL_PZ]:
            self.attack_likely = self.init_attack_likely = params.attacks.likely
            self.update_state_thread_instance = threading.Thread(
                target=self.update_state_thread
            )
            self.update_state_thread_instance.start()

    # ------------------------------------------------------------------
    # Spaces
    # ------------------------------------------------------------------

    def _build_spaces(self):
        # Host observation: counters [+ pct variations] + 1 message feature
        if self._include_pct_var:
            n_host_raw = 8
            host_low_raw = np.array([
                0, -self.threshold_var_packets,
                0, -self.threshold_var_bytes,
                0, -self.threshold_var_packets,
                0, -self.threshold_var_bytes,
            ], dtype=np.float32)
            host_high_raw = np.array([
                self.threshold_packets, self.threshold_var_packets,
                self.threshold_bytes,   self.threshold_var_bytes,
                self.threshold_packets, self.threshold_var_packets,
                self.threshold_bytes,   self.threshold_var_bytes,
            ], dtype=np.float32)
            self._host_counter_idx = {0, 2, 4, 6}
            self._host_var_idx = {1, 3, 5, 7}
        else:
            n_host_raw = 4
            host_low_raw = np.array([0, 0, 0, 0], dtype=np.float32)
            host_high_raw = np.array([
                self.threshold_packets, self.threshold_bytes,
                self.threshold_packets, self.threshold_bytes,
            ], dtype=np.float32)
            self._host_counter_idx = {0, 1, 2, 3}
            self._host_var_idx: set = set()

        # Message feature appended only when coordinator is enabled
        if self._use_coordinator:
            self._host_low_raw = np.append(host_low_raw, 0.0).astype(np.float32)
            self._host_high_raw = np.append(host_high_raw, float(self.num_hosts)).astype(np.float32)
            self._host_msg_idx = n_host_raw   # index of the message feature
        else:
            self._host_low_raw = host_low_raw.astype(np.float32)
            self._host_high_raw = host_high_raw.astype(np.float32)
            self._host_msg_idx = None         # no message feature

        # Coordinator: [total_pkts, pkts_pct, total_bytes, bytes_pct, msg_count]
        self._coord_low_raw = np.array([
            0, -self.threshold_var_packets,
            0, -self.threshold_var_bytes,
            0,
        ], dtype=np.float32)
        self._coord_high_raw = np.array([
            self.threshold_packets * self.num_hosts, self.threshold_var_packets,
            self.threshold_bytes   * self.num_hosts, self.threshold_var_bytes,
            float(self.num_hosts),
        ], dtype=np.float32)
        self._coord_counter_idx = {0, 2}
        self._coord_var_idx = {1, 3}

        n_host_feat = n_host_raw + (1 if self._use_coordinator else 0)
        n_coord_feat = 5

        # Per-agent spaces (all normalized to [0,1])
        self._observation_spaces: Dict[str, spaces.Box] = {}
        self._action_spaces: Dict[str, spaces.Discrete] = {}

        for agent_id in self.possible_agents:
            if agent_id == COORDINATOR:
                self._observation_spaces[agent_id] = spaces.Box(
                    low=np.zeros(n_coord_feat, dtype=np.float32),
                    high=np.ones(n_coord_feat, dtype=np.float32),
                    dtype=np.float32,
                )
                self._action_spaces[agent_id] = spaces.Discrete(COORDINATOR_ACTIONS.NUMBER)
            else:
                self._observation_spaces[agent_id] = spaces.Box(
                    low=np.zeros(n_host_feat, dtype=np.float32),
                    high=np.ones(n_host_feat, dtype=np.float32),
                    dtype=np.float32,
                )
                self._action_spaces[agent_id] = spaces.Discrete(AGENT_ACTIONS.NUMBER)

        # Gymnasium compat (first host space is the default)
        first_host = self.hosts[0].name if self.hosts else COORDINATOR
        self.observation_space = self._observation_spaces[first_host]
        self.action_space = self._action_spaces[first_host]

        # Normalize bounds shortcuts
        self.low  = self._host_low_raw
        self.high = self._host_high_raw
        self.low_to_normalize  = self.low
        self.high_to_normalize = self.high

    def _init_messages(self) -> dict:
        """Create empty message dicts: each agent receives messages from all others."""
        agent_list = list(self.possible_agents)
        msgs = {}
        for agent_id in agent_list:
            msgs[agent_id] = {s: 0 for s in agent_list if s != agent_id}
        return msgs

    # ------------------------------------------------------------------
    # PettingZoo interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """
        Reset the episode.

        Does NOT advance the scenario (no update_state call here).
        update_state() is called at the end of each step(), consuming
        exactly `episodes * max_steps` df entries for training.
        The first episode uses whatever state is in global_state (zeros on
        first run, or the last state of the previous episode otherwise).
        """
        self._step_count = 0
        self._correct_count = 0
        self._total_count = 0
        self.agents = list(self.possible_agents)
        self._agent_corrects = {a: 0 for a in self.possible_agents}

        # Reset anti-oscillation counters
        for h in self.hosts:
            self._block_started_step[h.name] = 0
            self._normal_streak_blocked[h.name] = 0

        # Re-init messages to zero
        self.global_state.messages = self._init_messages()

        # One-time network read if not yet initialized
        if not self._env_initialized:
            self._env_initialized = True
            if self.gym_type == GYM_TYPE[MARL_PZ]:
                # Live mode: read from network now (update_state_thread does the rest)
                if self.read_from_network():
                    self._evaluate_traffic()
            # For sequential/dataset mode, the first step will consume from df.

        obs = {a: self._get_obs_raw(a) for a in self.possible_agents}
        infos = {a: {} for a in self.possible_agents}
        return obs, infos

    def step(self, actions: Dict[str, int]):
        """
        PettingZoo-style parallel step.

        All agents act simultaneously. The env advances its state once per step.

        Returns
        -------
        obs, rewards, terminations, truncations, infos  — all dicts keyed by agent_id
        """
        while hasattr(self, 'pause_event') and self.pause_event.is_set():
            notify_client(
                level=SystemLevels.STATUS,
                status=SystemStatus.PAUSED,
                message="Paused marl_pz agents...",
                mode=SystemModes.TRAINING,
            )
            time.sleep(1)

        # Snapshot ground-truth BEFORE applying actions
        ground_truths = self._snapshot_ground_truths()

        # Calculate per-agent rewards
        rewards = {a: self._reward_for_agent(a, actions.get(a, 0), ground_truths[a])
                   for a in self.possible_agents}

        # Execute per-agent actions (drop rules, messages)
        for a in self.possible_agents:
            self._execute_agent_action(a, actions.get(a, 0), ground_truths[a])

        # Advance step counter
        self._step_count += 1
        for a in self.possible_agents:
            action = actions.get(a, 0)
            gt = ground_truths[a]
            if self._is_action_correct(a, action, gt):
                self._agent_corrects[a] = self._agent_corrects.get(a, 0) + 1
                self._correct_count += 1
            self._total_count += 1

        # Done / truncated
        pct = self._correct_count / max(1, self._total_count)
        done, truncated = self.check_if_done_or_truncated(self._step_count, pct)

        # Advance env state (reads next scenario step or network)
        if not done and not truncated:
            self.update_state()

        # Build next observations
        obs = {a: self._get_obs_raw(a) for a in self.possible_agents}
        terminations = {a: done for a in self.possible_agents}
        truncations  = {a: truncated for a in self.possible_agents}
        infos = {
            a: self._build_info(a, actions.get(a, 0), rewards[a], ground_truths[a],
                                done, truncated)
            for a in self.possible_agents
        }

        if done or truncated:
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------
    # Done / truncated
    # ------------------------------------------------------------------

    def check_if_done_or_truncated(self, current_step, pct_correct):
        if (current_step >= self.max_steps * self.steps_min_percentage
                and pct_correct > self.min_accuracy):
            return False, True
        if current_step >= self.max_steps:
            return True, False
        return False, False

    # ------------------------------------------------------------------
    # State update (sequential scenario replay + live mode)
    # ------------------------------------------------------------------

    def update_state(self):
        """Advance the environment state one step."""
        if self.gym_type in (GYM_TYPE[MARL_PZ],):
            if hasattr(self, 'df') and self.df:
                step_plan = self.df.pop(0)
                self._apply_scenario_step(step_plan)
                if self.read_from_network():
                    self._evaluate_traffic()
            else:
                if self.read_from_network():
                    self._evaluate_traffic()
        elif self.gym_type == GYM_TYPE[MARL_PZ_FROM_DATASET]:
            try:
                if not hasattr(self, 'df'):
                    self.state = self.initial_state
                elif self.df:
                    status = self.df.pop(0)
                    self.global_prev_state = copy.copy(self.global_state)
                    self.global_state.set_state(status)
                    self.state = self.global_state.state
                    self.host_tasks = {}
                    for host_name, host_status in status["hostStatusesStructured"].items():
                        self.host_tasks[host_name] = {
                            'traffic_type': host_status['trafficType'],
                            'task_type':    host_status['taskType'],
                            'destination':  (
                                None if host_status['trafficType'] == TrafficTypes.NONE
                                else host_status.get('destination')
                            ),
                            'end_time': None,
                        }
                    self._evaluate_traffic()
            except Exception as exc:
                debug(f"[marl_pz] dataset state read error: {exc}\n")

    def _apply_scenario_step(self, step_plan: dict):
        """
        Apply one scenario step to the Mininet network (ported from HO).

        Launches planned traffic tasks via ThreadPoolExecutor and waits 1s
        for OVS counters to reflect traffic.
        """
        from reinforcement_learning.agents.adversarial_agent import (
            _generate_normal_traffic_once,
            _launch_attack_once,
            _name_to_attack_type,
        )

        if self.host_tasks is None:
            self.host_tasks = {}

        for host in self.net.hosts:
            plan = step_plan.get(host.name)
            if plan is None:
                continue

            task_type        = plan.get("task_type",    NORMAL)
            traffic_type     = plan.get("traffic_type", "none")
            destination_name = plan.get("destination")
            destination      = next(
                (h for h in self.net.hosts if h.name == destination_name), None
            )

            self.host_tasks[host.name] = {
                "task_type":    task_type,
                "traffic_type": traffic_type,
                "destination":  destination_name,
                "end_time":     time.time() + 2.0,
            }

            if destination is None or traffic_type == "none":
                continue

            prev = self._host_futures.get(host.name)
            if prev is not None and not prev.done():
                continue

            if task_type == NORMAL:
                self._host_futures[host.name] = self._traffic_executor.submit(
                    _generate_normal_traffic_once,
                    host, destination, traffic_type, 1.0,
                )
            elif task_type in (SHORT_ATTACK, LONG_ATTACK):
                attack_type = _name_to_attack_type(
                    plan.get("attack_subtype", "udp_flood")
                )
                self._host_futures[host.name] = self._traffic_executor.submit(
                    _launch_attack_once,
                    host, destination, attack_type, 1.0,
                )

        time.sleep(1.0)

    def _evaluate_traffic(self):
        """Update per-host statuses from network, remap blocked attacks, notify client."""
        statuses, _ = self.update_hosts_status()

        if self._apply_drop_rules:
            for i, host in enumerate(self.hosts):
                if not self.status_links[i] and statuses.get(host.name) == HostStatus.ATTACKING:
                    statuses[host.name] = HostStatus.OUT_ATTACK_BLOCKED

        # Derive coordinator status (only when coordinator is enabled)
        host_names = {h.name for h in self.hosts}
        if self._use_coordinator:
            is_attack = any(
                statuses.get(n) in (HostStatus.ATTACKING, HostStatus.UNDER_ATTACK,
                                    HostStatus.WAR, HostStatus.OUT_ATTACK_BLOCKED)
                for n in host_names
            )
            statuses[COORDINATOR] = 'attack' if is_attack else 'normal'

            total_pkts = sum(
                float(self.global_state.host_states.get(h.name, [0])[0])
                for h in self.hosts
            )
            avg_pkts_pct = np.mean([
                float(self.global_state.host_states.get(h.name, [0, 0])[1])
                for h in self.hosts
            ])
            total_bytes = sum(
                float(self.global_state.host_states.get(h.name, [0, 0, 0])[2])
                for h in self.hosts
            )
            avg_bytes_pct = np.mean([
                float(self.global_state.host_states.get(h.name, [0, 0, 0, 0])[3])
                for h in self.hosts
            ])
            self.global_state.coordinator_state = np.array(
                [total_pkts, avg_pkts_pct, total_bytes, avg_bytes_pct], dtype=np.float32
            )

        self.global_state.update_statuses(statuses)

        # Sync link status
        if not hasattr(self.global_state, 'links_status'):
            self.global_state.links_status = {}
        for i, host in enumerate(self.hosts):
            self.global_state.links_status[host.name] = 1 if self.status_links[i] else 0

        traffic_data = self.global_state.get_network_traffic_status()
        notify_client(level=SystemLevels.DATA, traffic_data=traffic_data)

        if (self.gym_type == GYM_TYPE[MARL_PZ]
                and hasattr(self, 'host_tasks') and self.host_tasks):
            for host_name, task in list(self.host_tasks.items()):
                if host_name in traffic_data.get("hostStatusesStructured", {}):
                    traffic_data["hostStatusesStructured"][host_name].update({
                        'trafficType': task['traffic_type'],
                        'taskType':    task['task_type'],
                        'destination': task['destination'],
                    })
            self.statuses.append(traffic_data)
            self._show_network_status()

    def _show_network_status(self):
        status = self.global_state.status
        ids = status.get("id", [])
        total_pkts = int(status.get("packets", 0))
        pkt_var = int(status.get("packets_percentage_change", 0))
        total_bytes = int(status.get("bytes", 0))
        byte_var = int(status.get("bytes_percentage_change", 0))

        if isinstance(ids, (list, np.ndarray)) and len(ids) > 0:
            if any(s == 2 for s in ids):
                color, label = Fore.RED,    "attack"
            elif any(s == 1 for s in ids):
                color, label = Fore.YELLOW, "under_attack"
            else:
                color, label = Fore.GREEN,  "normal"
        else:
            color, label = Fore.GREEN, "normal"

        information(
            color + f"{label} "
            + Fore.BLUE + "Packet "
            + Fore.WHITE + f"{total_pkts} {pkt_var}%"
            + Fore.BLUE + " - "
            + Fore.WHITE + f"{format_bytes(total_bytes)}B {byte_var}%\n"
            + Fore.WHITE
        )

    def initialize_storage(self):
        pass

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_obs_raw(self, agent_id: str) -> np.ndarray:
        """Return the raw (non-normalized) observation for one agent."""
        if agent_id == COORDINATOR:
            cs = self.global_state.coordinator_state
            msg_count = float(sum(
                1 for v in self.global_state.get_messages(COORDINATOR).values()
                if v != 0
            ))
            raw = np.array([cs[0], cs[1], cs[2], cs[3], msg_count], dtype=np.float32)
            return np.clip(raw, self._coord_low_raw, self._coord_high_raw)
        else:
            host_state = np.array(
                self.global_state.host_states.get(agent_id,
                                                  np.zeros(8, dtype=np.float32)),
                dtype=np.float32,
            )
            if self._include_pct_var:
                raw_traffic = host_state[:8]
            else:
                raw_traffic = host_state[[0, 2, 4, 6]]

            if self._use_coordinator:
                msg_count = float(sum(
                    1 for v in self.global_state.get_messages(agent_id).values()
                    if v != 0
                ))
                raw = np.append(raw_traffic, msg_count).astype(np.float32)
            else:
                raw = raw_traffic.astype(np.float32)
            return np.clip(raw, self._host_low_raw, self._host_high_raw)

    def _get_obs_normalized(self, agent_id: str) -> np.ndarray:
        """Return normalized [0,1] observation for one agent."""
        raw = self._get_obs_raw(agent_id)
        if agent_id == COORDINATOR:
            return get_normalized_state(raw, self._coord_low_raw, self._coord_high_raw)
        return get_normalized_state(raw, self._host_low_raw, self._host_high_raw)

    # ------------------------------------------------------------------
    # Discretization (for tabular agents via SingleAgentView)
    # ------------------------------------------------------------------

    def get_discretized_state_for_agent(self, agent_id: str,
                                         obs: np.ndarray) -> tuple:
        """
        Discretize a RAW observation for the given agent.

        `obs` must be the raw (non-normalized) array as returned by _get_obs_raw().
        """
        n = self.n_bins
        if agent_id == COORDINATOR:
            return _discretize_obs(obs, self._coord_low_raw, self._coord_high_raw,
                                   n, self._coord_counter_idx, self._coord_var_idx)
        # Host agent: traffic counters use log binning; message feature (last, only
        # when coordinator is enabled) uses linear binning — log-scale bins would be
        # non-monotone when num_hosts < 10.
        n_traffic = (len(obs) - 1) if self._use_coordinator else len(obs)
        counter_idx = {i for i in self._host_counter_idx if i < n_traffic}
        var_idx = {i for i in self._host_var_idx if i < n_traffic}
        # message feature excluded from counter_idx → linear bin (no-op when no coordinator)
        return _discretize_obs(obs, self._host_low_raw, self._host_high_raw,
                                n, counter_idx, var_idx)

    # Gymnasium compat: called by base_agent on the SingleAgentView
    def get_discretized_state(self, state) -> tuple:
        """Discretize using the first host's space (gym compat)."""
        if self.hosts:
            return self.get_discretized_state_for_agent(self.hosts[0].name, state)
        return tuple()

    # ------------------------------------------------------------------
    # Ground truth helpers
    # ------------------------------------------------------------------

    def _snapshot_ground_truths(self) -> Dict[str, dict]:
        """Snapshot the current ground-truth status for every agent."""
        truths = {}
        for host in self.hosts:
            hs = self.global_state.host_statuses.get(host.name, {})
            if isinstance(hs, dict):
                truths[host.name] = hs
            else:
                truths[host.name] = {'id': 0, 'status': 'normal'}
        # Coordinator ground truth
        coord = self.global_state.coordinator_status
        if isinstance(coord, dict):
            truths[COORDINATOR] = coord
        else:
            truths[COORDINATOR] = {'id': 0, 'status': 'normal'}
        return truths

    def _is_action_correct(self, agent_id: str, action: int,
                            ground_truth: dict) -> bool:
        """True if the agent's action matches the ground truth."""
        gt_id = ground_truth.get('id', 0)
        if agent_id == COORDINATOR:
            # Coordinator: action 0=normal, 1=attack; gt 0=normal, 1=attack
            gt_binary = 0 if gt_id <= 0 else 1
            return action == gt_binary
        else:
            # Host agent: 0=normal, 1=attack_in (under_attack), 2=attack_out (attacking/blocked)
            if gt_id == HOST_STATUS_ID_MAPPING['normal'] or gt_id < 0:
                expected = AGENT_ACTIONS.NORMAL_TRAFFIC
            elif gt_id == HOST_STATUS_ID_MAPPING['under_attack']:
                expected = AGENT_ACTIONS.ATTACK_IN
            elif gt_id in (HOST_STATUS_ID_MAPPING['attacking'],
                           HOST_STATUS_ID_MAPPING['out_attack_blocked']):
                expected = AGENT_ACTIONS.ATTACK_OUT
            elif gt_id == HOST_STATUS_ID_MAPPING['incoming_blocked_attack']:
                expected = AGENT_ACTIONS.ATTACK_IN
            else:
                expected = AGENT_ACTIONS.NORMAL_TRAFFIC
            return action == expected

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _reward_for_agent(self, agent_id: str, action: int,
                           ground_truth: dict) -> float:
        """Per-agent reward based on action correctness."""
        gt_id = ground_truth.get('id', 0)

        if agent_id == COORDINATOR:
            gt_binary = 0 if gt_id <= 0 else 1
            if action == CoordinatorActions.NO_ATTACK and gt_binary == 0:
                return COORDINATOR_REWARDS.CORRECT_NO_ATTACK
            elif action == CoordinatorActions.ATTACK and gt_binary == 1:
                return COORDINATOR_REWARDS.CORRECT_ATTACK
            elif action == CoordinatorActions.NO_ATTACK and gt_binary == 1:
                return COORDINATOR_REWARDS.FALSE_NEGATIVE
            else:
                return COORDINATOR_REWARDS.FALSE_POSITIVE
        else:
            # Host agent
            host_index = next(
                (i for i, h in enumerate(self.hosts) if h.name == agent_id), 0
            )
            is_blocked = not self.status_links[host_index]
            reward = 0.0

            if gt_id == HOST_STATUS_ID_MAPPING['normal'] or gt_id < 0:
                gt_label = HostStatus.NORMAL
            elif gt_id == HOST_STATUS_ID_MAPPING['under_attack']:
                gt_label = HostStatus.UNDER_ATTACK
            elif gt_id in (HOST_STATUS_ID_MAPPING['attacking'],
                           HOST_STATUS_ID_MAPPING['out_attack_blocked']):
                gt_label = HostStatus.ATTACKING
            elif gt_id == HOST_STATUS_ID_MAPPING['incoming_blocked_attack']:
                gt_label = HostStatus.UNDER_ATTACK
            else:
                gt_label = HostStatus.NORMAL

            if is_blocked and gt_label == HostStatus.NORMAL:
                reward += REWARDS.LINK_OFF

            if action == AGENT_ACTIONS.NORMAL_TRAFFIC:
                if gt_label == HostStatus.NORMAL:
                    reward += REWARDS.CORRECT_NORMAL_TRAFFIC
                else:
                    reward += REWARDS.FALSE_NEGATIVE
            elif action == AGENT_ACTIONS.ATTACK_IN:
                if gt_label == HostStatus.UNDER_ATTACK:
                    reward += REWARDS.CORRECT_UNDER_ATTACK_DETECTION
                elif gt_label == HostStatus.NORMAL:
                    reward += REWARDS.FALSE_POSITIVE
                elif gt_label == HostStatus.ATTACKING:
                    reward += REWARDS.WRONG_ATTACK_DIRECTION_DETECTED
            elif action == AGENT_ACTIONS.ATTACK_OUT:
                if gt_label == HostStatus.ATTACKING:
                    reward += REWARDS.CORRECT_ATTACK_DETECTION
                elif gt_label == HostStatus.NORMAL:
                    reward += REWARDS.FALSE_POSITIVE
                elif gt_label == HostStatus.UNDER_ATTACK:
                    reward += REWARDS.WRONG_ATTACK_DIRECTION_DETECTED

            return reward

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def _execute_agent_action(self, agent_id: str, action: int,
                               ground_truth: dict):
        """Execute action side-effects: drop rules and messaging."""
        if agent_id == COORDINATOR:
            # Broadcast attack alert to all hosts
            if action == CoordinatorActions.ATTACK:
                for h in self.hosts:
                    self.global_state.set_message(h.name, COORDINATOR, 1)
            else:
                for h in self.hosts:
                    self.global_state.set_message(h.name, COORDINATOR, 0)
            return

        # Host agent
        host_index = next(
            (i for i, h in enumerate(self.hosts) if h.name == agent_id), None
        )
        if host_index is None:
            return

        # Send message to coordinator (only when coordinator is enabled)
        if self._use_coordinator:
            self.global_state.set_message(
                COORDINATOR, agent_id,
                1 if action != AGENT_ACTIONS.NORMAL_TRAFFIC else 0
            )

        if not self._apply_drop_rules:
            return

        if action == AGENT_ACTIONS.NORMAL_TRAFFIC:
            # Unblock if link is down AND we've held long enough with clean streak
            if not self.status_links[host_index]:
                self._normal_streak_blocked[agent_id] = (
                    self._normal_streak_blocked.get(agent_id, 0) + 1
                )
                held = self._step_count - self._block_started_step.get(agent_id, 0)
                if (held >= self._unblock_min_hold
                        and self._normal_streak_blocked.get(agent_id, 0)
                        >= self._unblock_normal_streak):
                    if unblock_flow_delete(self.net, agent_id):
                        time.sleep(0.02)
                        self.status_links[host_index] = True
                        self._normal_streak_blocked[agent_id] = 0
            else:
                self._normal_streak_blocked[agent_id] = 0

        elif action == AGENT_ACTIONS.ATTACK_OUT:
            if self.status_links[host_index]:
                if block_flow_drop(self.net, agent_id):
                    time.sleep(0.02)
                    self.status_links[host_index] = False
                    self._block_started_step[agent_id] = self._step_count
                    self._normal_streak_blocked[agent_id] = 0

    # ------------------------------------------------------------------
    # Info dict
    # ------------------------------------------------------------------

    def _build_info(self, agent_id: str, action: int, reward: float,
                    ground_truth: dict, done: bool, truncated: bool) -> dict:
        """Build the infos dict for a single agent (compatible with base_agent.manage_step_data)."""
        gt_id = ground_truth.get('id', 0)
        gt_status = ground_truth.get('status', 'normal')
        is_correct = self._is_action_correct(agent_id, action, ground_truth)

        n_actions = (COORDINATOR_ACTIONS.NUMBER if agent_id == COORDINATOR
                     else AGENT_ACTIONS.NUMBER)
        gt_one_hot = np.zeros(n_actions, dtype=np.float32)
        pred_one_hot = np.zeros(n_actions, dtype=np.float32)

        if agent_id == COORDINATOR:
            gt_binary = 0 if gt_id <= 0 else 1
            gt_one_hot[min(gt_binary, n_actions - 1)] = 1.0
            pred_one_hot[min(action, n_actions - 1)] = 1.0
            status = dict(ground_truth)
        else:
            safe_gt = gt_id if 0 <= gt_id < n_actions else 0
            gt_one_hot[safe_gt] = 1.0
            pred_one_hot[min(action, n_actions - 1)] = 1.0
            status = dict(ground_truth)

        return {
            'action_correct':      gt_id,
            'text_action_correct': gt_status,
            'status':              status,
            'is_correct_action':   is_correct,
            'TimeLimit.truncated': truncated,
            'Ground_truth_step':   gt_one_hot,
            'Predicted_step':      pred_one_hot,
        }

    # ------------------------------------------------------------------
    # NetworkEnv abstract method stubs (used via SingleAgentView pathway)
    # ------------------------------------------------------------------

    def execute_action(self, action, show_action=False, name=None, reward=0):
        pass

    def calculate_reward(self, status, action) -> float:
        return 0.0

    # ------------------------------------------------------------------
    # Stop / cleanup
    # ------------------------------------------------------------------

    def stop(self):
        if hasattr(self, 'stop_update_event'):
            self.stop_update_event.set()
        if hasattr(self, '_traffic_executor'):
            self._traffic_executor.shutdown(wait=False, cancel_futures=True)
        super().stop()


# ────────────────────────────────────────────────────────────────────────────
# SingleAgentView — thin gymnasium.Env wrapper for one agent in MarlPzEnv
# ────────────────────────────────────────────────────────────────────────────

class SingleAgentView(gym.Env):
    """
    Expose MarlPzEnv as a single-agent gymnasium.Env for one agent_id.

    This allows standard tabular agents (QLearning, SARSA) and SB3 models
    (DQN, PPO, A2C) to train on the multi-agent env without modification.
    During training, all other agents submit action 0 (NORMAL_TRAFFIC).

    state_mode: RAW (for tabular, returns raw obs for manual discretization)
                NORMALIZED (default, returns [0,1] obs for SB3)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, marl_env: MarlPzEnv, agent_id: str,
                 state_mode: str = NORMALIZED):
        super().__init__()
        self.marl_env = marl_env
        self.agent_id = agent_id
        self.state_mode = state_mode
        self.is_coordinator = (agent_id == COORDINATOR)

        self.observation_space = marl_env._observation_spaces[agent_id]
        self.action_space = marl_env._action_spaces[agent_id]

        # Attributes expected by tabular agents
        self.n_bins = marl_env.n_bins
        self.gym_type = marl_env.gym_type
        self.max_steps = marl_env.max_steps

    # Forward shared env attributes to the parent env
    @property
    def global_state(self):
        return self.marl_env.global_state

    @property
    def pause_event(self):
        return getattr(self.marl_env, 'pause_event', None)

    @property
    def stop_event(self):
        return getattr(self.marl_env, 'stop_event', None)

    @property
    def hosts(self):
        return self.marl_env.hosts

    # ------------------------------------------------------------------
    # gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """
        Reset via MarlPzEnv and return this agent's observation.

        options keys forwarded from tabular agents:
          is_discretized_state (bool)
          is_real_state        (bool)
        """
        obs_dict, infos_dict = self.marl_env.reset(seed=seed, options=options)
        obs = obs_dict[self.agent_id]

        is_discretized = (options or {}).get('is_discretized_state', False)
        is_real = (options or {}).get('is_real_state', False)
        obs = self._process_obs(obs, is_discretized, is_real)

        return obs, infos_dict.get(self.agent_id, {})

    def step(self, action, options=None):
        """
        Single-agent step: submit this agent's action; all others act with 0.

        options keys used:
          is_discretized_state (bool)
          current_step         (int)
          correct_predictions  (int)
          show_action          (bool)
          name                 (str)
        """
        opts = options or {}
        actions = {a: 0 for a in self.marl_env.possible_agents}
        actions[self.agent_id] = int(action)

        obs_dict, rewards, terms, truncs, infos = self.marl_env.step(actions)

        obs = obs_dict[self.agent_id]
        is_discretized = opts.get('is_discretized_state', False)
        obs = self._process_obs(obs, is_discretized, False)

        return (obs,
                rewards[self.agent_id],
                terms.get(self.agent_id, False),
                truncs.get(self.agent_id, False),
                infos.get(self.agent_id, {}))

    def _process_obs(self, raw_obs: np.ndarray, is_discretized: bool,
                     is_real: bool):
        """Post-process raw obs according to state_mode and flags."""
        if is_real:
            return raw_obs
        if is_discretized:
            return self.get_discretized_state(raw_obs)
        if self.state_mode == RAW:
            return raw_obs
        # Default: normalize
        return self.marl_env._get_obs_normalized(self.agent_id)

    # ------------------------------------------------------------------
    # Discretization (forwarded to MarlPzEnv)
    # ------------------------------------------------------------------

    def get_discretized_state(self, obs: np.ndarray) -> tuple:
        return self.marl_env.get_discretized_state_for_agent(self.agent_id, obs)

    # ------------------------------------------------------------------
    # Misc gymnasium compat
    # ------------------------------------------------------------------

    def initialize_storage(self):
        return self.marl_env.initialize_storage()

    def clean_network_state(self):
        return self.marl_env.clean_network_state()

    def render(self):
        pass

    def close(self):
        pass
