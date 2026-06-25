import copy
import json as jsonlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
from colorama import Fore
from gymnasium import spaces

from reinforcement_learning.network_env import (
    NetworkEnv,
    get_agent_name,
    get_custom_bin_index,
    get_linear_bin_index,
    get_log_bin_index,
    get_normalized_state,
)
from utility.constants import *
from utility.my_log import debug, error, information, notify_client
from utility.network_configurator import (
    block_flow_drop,
    comunicate_in_attack_detected,
    comunicate_normal_traffic_detected,
    comunicate_out_attack_detected,
    format_bytes,
    unblock_flow_delete,
)
from utility.params import Params

from .constants import AGENT_ACTIONS, REWARDS, HOST_STATUS_ID_MAPPING
from .instant_state import InstantState


def discretize_attack_detect_ho_state(state, low, high, n_bins):
    """Discretize one per-host ATTACKS_HO state slice using env rules.

    Handles both the full 8-feature state (counters + percentage variations)
    and the compact 4-feature state (counters only: RX pkts, RX bytes,
    TX pkts, TX bytes) produced when include_percentage_variations=False.
    """
    if state is None:
        return tuple()

    if isinstance(state, tuple):
        state = state[0]

    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    n_bins = max(2, int(n_bins))

    n_features = len(state)
    if n_features == 4:
        # Compact mode: all features are counters (log-bin)
        counter_indices = {0, 1, 2, 3}
        variation_indices: set = set()
    else:
        # Full 8-feature mode
        counter_indices = {0, 2, 4, 6}
        variation_indices = {1, 3, 5, 7}

    discrete_state = []
    for i, val in enumerate(state):
        if i in variation_indices:
            bin_index = get_linear_bin_index(val, low[i], high[i], n_bins - 1) + 1
        elif i in counter_indices:
            if val <= 0:
                bin_index = 0
            else:
                high_safe = max(float(high[i]), 1.0)
                bin_index = get_log_bin_index(val, 1.0, high_safe, n_bins - 1) + 1
        else:
            bin_index = get_linear_bin_index(val, low[i], high[i], n_bins)
        discrete_state.append(int(bin_index))

    return tuple(discrete_state)


class NetworkEnvAttackDetectPerHostObservable(NetworkEnv):
    """Custom Environment that follows gym interface.
    This is a simple env where an agent must detect if there is an attack or normal traffic.

    The agent can choose between two actions: NORMAL_TRAFFIC, ATTACK.
    The environment provides a reward based on the correctness of the detection.
    The network topology is a simple star topology with one switch and multiple hosts.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, params, server_user, existing_net=None):
        params.actions_number = AGENT_ACTIONS.NUMBER
        super().__init__(params, server_user, existing_net)
        self.params = params
        self.show_complete_network_status = False
        self.statuses = []
        self._scenario_traffic_threads = []
        # Bounded pool: one worker per host + 2 slack, capping OS thread count.
        _pool_size = (getattr(params.net_params, 'num_hosts', 5)
                      + getattr(params.net_params, 'num_iots', 0) + 2)
        self._traffic_executor = ThreadPoolExecutor(max_workers=_pool_size, thread_name_prefix="traffic")
        self._host_futures: dict = {}   # last submitted Future per host name
        self.hosts = self.net.hosts  # Access hosts from the parent class's network
        self.status_links = [True for _ in self.hosts]  # Initially, all links are up
        # Network params
        self.threshold_packets = params.attacks.thresholds.packets
        self.threshold_var_packets = params.attacks.thresholds.var_packets
        self.threshold_bytes = params.attacks.thresholds.bytes
        self.threshold_var_bytes = params.attacks.thresholds.var_bytes
        # Observation includes per-host statistics
        self.num_hosts = len(self.net.hosts)

        # Whether to include percentage-variation features in the observation.
        # True  → shape (8,): [RX_pkts, ΔRX_pkts%, RX_bytes, ΔRX_bytes%,
        #                       TX_pkts, ΔTX_pkts%, TX_bytes, ΔTX_bytes%]
        # False → shape (4,): [RX_pkts, RX_bytes, TX_pkts, TX_bytes]
        self._include_pct_var = bool(
            getattr(params.attacks, 'include_percentage_variations', True)
        )

        if self._include_pct_var:
            per_host_features = 8
            self.low = np.array([
                0, -self.threshold_var_packets,
                0, -self.threshold_var_bytes,
                0, -self.threshold_var_packets,
                0, -self.threshold_var_bytes,
            ], dtype=np.float32)
            self.high = np.array([
                self.threshold_packets, self.threshold_var_packets,
                self.threshold_bytes,   self.threshold_var_bytes,
                self.threshold_packets, self.threshold_var_packets,
                self.threshold_bytes,   self.threshold_var_bytes,
            ], dtype=np.float32)
        else:
            per_host_features = 4
            self.low = np.array([0, 0, 0, 0], dtype=np.float32)
            self.high = np.array([
                self.threshold_packets, self.threshold_bytes,
                self.threshold_packets, self.threshold_bytes,
            ], dtype=np.float32)

        # Define action and observation space (single-host).
        # PerHostScanWrapper overrides these, but keeping them here
        # makes the base env self-consistent.
        self.observation_space = spaces.Box(
            low=self.low,
            high=self.high,
            shape=(per_host_features,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(AGENT_ACTIONS.NUMBER)

        # Define the number of discrete bins for each observation dimension
        self.n_bins = params.n_bins
        self.low_to_normalize  = self.low
        self.high_to_normalize = self.high

        self.global_prev_state = self.global_state = InstantState(self.hosts)

        self.last_short_attack_timestamp = time.time()
        self.last_long_attack_timestamp  = time.time()

        if self.gym_type == GYM_TYPE[ATTACKS_HO]:
            self.attack_likely = self.init_attack_likely = params.attacks.likely
            self.update_state_thread_instance = threading.Thread(
                target=self.update_state_thread
            )
            self.update_state_thread_instance.start()

        self.reset()

    # ------------------------------------------------------------------
    # Action encoding / decoding
    # ------------------------------------------------------------------

    def action_to_per_host(self, action_int) -> List[int]:
        """
        Convert single discrete action to per-host actions.

        Args:
            action_int: Integer from 0 to (3^num_hosts - 1)

        Returns:
            List of actions, one per host (each in range 0-2)

        Example with 3 hosts:
            action_int=0  -> [0, 0, 0]
            action_int=1  -> [1, 0, 0]
            action_int=26 -> [2, 2, 2]
        """
        per_host_actions = []
        remaining = action_int if isinstance(action_int, int) else action_int.item()

        for _ in range(self.num_hosts):
            per_host_actions.append(remaining % self.params.actions_number)
            remaining //= self.params.actions_number

        return per_host_actions

    def per_host_to_action(self, per_host_actions) -> int:
        """
        Convert per-host actions back to single discrete action.

        Args:
            per_host_actions: List of actions [a0, a1, a2, ...]

        Returns:
            Single integer action
        """
        action_int   = 0
        multiplier   = 1

        for host_action in per_host_actions:
            action_int += host_action * multiplier
            multiplier *= self.params.actions_number

        return action_int if isinstance(action_int, int) else action_int.item()

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    def update_state(self):
        """
        Update environment state based on gym_type.

        Three cases:

        ATTACKS_HO + self.df set  (sequential scenario replay)
            Pop the next step plan from self.df, apply the planned tasks to
            the real network via _apply_scenario_step(), then read OVS
            counters as usual.

        ATTACKS_HO + no self.df   (live mode)
            Read directly from OVS — traffic is generated by the
            adversarial_agent thread running in background.

        ATTACKS_HO_FROM_DATASET
            Pop the next pre-recorded full state from self.df and apply it
            directly (original behaviour, unchanged).
        """
        if self.gym_type == GYM_TYPE[ATTACKS_HO] and self.state is not None:

            if hasattr(self, 'df') and self.df: #is from scenario.json
                # ── Sequential scenario replay ─────────────────────────
                step_plan = self.df.pop(0)          # {host_name: task_info}
                self._apply_scenario_step(step_plan)
                # _apply_scenario_step already waits 1 s; read OVS now
                if self.read_from_network():
                    self.evaluate_traffic()
            else:
                # ── Live mode ──────────────────────────────────────────
                if self.read_from_network():
                    self.evaluate_traffic()

        elif self.gym_type == GYM_TYPE[ATTACKS_HO_FROM_DATASET]:
            # ── Full pre-recorded state replay (unchanged) ─────────────
            try:
                if not hasattr(self, 'df'):
                    self.state = self.initial_state
                elif len(self.df) > 0:
                    status = self.df.pop(0)
                    self.global_prev_state = copy.copy(self.global_state)
                    self.global_state.set_state(status)
                    self.state = self.global_state.state
                    self.host_tasks = {}
                    for host_name, host_status in \
                            status["hostStatusesStructured"].items():
                        self.host_tasks[host_name] = {
                            'traffic_type': host_status['trafficType'],
                            'task_type':    host_status['taskType'],
                            'destination':  (
                                None
                                if host_status['trafficType'] == TrafficTypes.NONE
                                else host_status.get('destination', None)
                            ),
                            'end_time': None,
                        }
                    self.evaluate_traffic()
                else:
                    debug("Missing dataset row: no status read\n")
            except Exception as e:
                debug(f"Reading status error: {e}\n")

    def _apply_scenario_step(self, step_plan: dict):
        """
        Apply one scenario step to the real Mininet network.

        continuous_traffic_generation is NOT running in sequential mode
        (excluded in main.py), so there are no background threads to
        conflict with.  For each host:
          1. Update self.host_tasks with the planned task so that
             update_hosts_status() computes the correct ground truth.
          2. Launch the planned traffic in a short-lived daemon thread
             (fire-and-forget, duration ~1 s).
        Then wait 1 second for OVS counters to reflect the traffic.

        Args:
            step_plan: {host_name: {"task_type": ..., "traffic_type": ...,
                                    "destination": ...,
                                    "attack_subtype": ...}}
        """
        # Lazy import to avoid circular imports at module load time
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
                (h for h in self.net.hosts if h.name == destination_name),
                None,
            )

            # Update ground truth — update_hosts_status() reads host_tasks
            self.host_tasks[host.name] = {
                "task_type":    task_type,
                "traffic_type": traffic_type,
                "destination":  destination_name,
                "end_time":     time.time() + 2.0,  # valid for this step
            }

            # Nothing to send if no destination or no traffic
            if destination is None or traffic_type == "none":
                continue

            # Skip if the previous task for this host is still running or queued.
            # For normal traffic (iperf 1-5 s) this avoids stacking multiple iperf
            # runs; the existing traffic keeps flowing so the OVS counter is correct.
            # For attacks (future completes in ~0.6 s) the check almost never fires,
            # so each step still submits a fresh 1 s hping3 burst.
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

        # Wait for traffic to propagate to OVS counters
        time.sleep(1.0)

    def stop(self):
        """Stop traffic threads before tearing down Mininet."""
        if hasattr(self, 'stop_event'):
            self.stop_event.set()

        if hasattr(self, '_traffic_executor'):
            self._traffic_executor.shutdown(wait=False, cancel_futures=True)

        super().stop()

    # ------------------------------------------------------------------
    # Traffic evaluation and display
    # ------------------------------------------------------------------

    def evaluate_traffic(self):
        statuses, _ = self.update_hosts_status()
        # Only remap ATTACKING→OUT_ATTACK_BLOCKED when drop rules are actually
        # being applied.  With apply_drop_rules=False the link is not really
        # blocked in the network, so the ground-truth label stays ATTACKING.
        apply_drop_rules = getattr(
            getattr(self.params, 'attacks', None), 'apply_drop_rules', True
        )
        if apply_drop_rules:
            for host_index, host in enumerate(self.hosts):
                if not self.status_links[host_index] and statuses.get(host.name) == HostStatus.ATTACKING:
                    statuses[host.name] = HostStatus.OUT_ATTACK_BLOCKED
        self.global_state.update_statuses(statuses)
        # Sync status_links to global_state so network_env can expose linkStatus to the dashboard
        if not hasattr(self.global_state, 'links_status'):
            self.global_state.links_status = {}
        for host_index, host in enumerate(self.hosts):
            self.global_state.links_status[host.name] = 1 if self.status_links[host_index] else 0
        traffic_data = self.global_state.get_network_traffic_status()
        notify_client(level=SystemLevels.DATA, traffic_data=traffic_data)
        if self.gym_type == GYM_TYPE[ATTACKS_HO] \
                and hasattr(self, 'host_tasks') \
                and self.host_tasks is not None:
            for host_name, host_task in list(self.host_tasks.items()):
                traffic_data["hostStatusesStructured"][host_name].update({
                    'trafficType': host_task['traffic_type'],
                    'taskType':    host_task['task_type'],
                    'destination': host_task['destination'],
                })
            self.statuses.append(traffic_data)
            self.show_network_status()

    def show_network_status(self):
        host_tasks = dict(self.host_tasks) if isinstance(self.host_tasks, dict) else {}
        status     = self.global_state.status
        ids        = status.get("id", [])
        total_pkts = int(status.get("packets", 0))
        pkt_var    = int(status.get("packets_percentage_change", 0))
        total_bytes = int(status.get("bytes", 0))
        byte_var   = int(status.get("bytes_percentage_change", 0))

        # Overall label: worst status across all hosts
        if any(s == 2 for s in ids):
            overall_color = Fore.RED
            overall_label = "attack"
        elif any(s == 1 for s in ids):
            overall_color = Fore.YELLOW
            overall_label = "under_attack"
        else:
            overall_color = Fore.GREEN
            overall_label = "normal"

        # Per-host ID string, each colored
        _id_colors = {0: Fore.GREEN, 1: Fore.YELLOW, 2: Fore.RED}
        id_str = " ".join(
            _id_colors.get(s, Fore.WHITE) + str(s) + Fore.WHITE for s in ids
        )

        information(
            overall_color + f"{overall_label} "
            + Fore.BLUE + f"Packet "
            + Fore.WHITE + f"{total_pkts} {pkt_var}%"
            + Fore.BLUE + " - "
            + Fore.WHITE + f"{format_bytes(total_bytes)}B {byte_var}%"
            + Fore.CYAN + f" - [{id_str}" + Fore.CYAN + "]\n"
            + Fore.WHITE
        )

        # Per-host detail
        level_fn = information if self.show_complete_network_status else debug
        for host in self.global_state.host_states.keys():
            host_state  = self.global_state.get_host_state(host)
            host_status = self.global_state.get_host_status(host)
            if host_status is None or host_state is None:
                continue

            raw = np.array(host_state, dtype=np.float32)  # always 8 features
            s   = host_status['status']
            # For discretize/normalize, filter to the configured obs size
            obs_raw = raw if self._include_pct_var else raw[[0, 2, 4, 6]]

            if s in (HostStatus.ATTACKING, HostStatus.OUT_ATTACK_BLOCKED, HostStatus.WAR):
                disc = self.get_discretized_state(obs_raw)
                norm = get_normalized_state(obs_raw, self.low, self.high)
                disc_str = " ".join(str(int(v)) for v in disc)
                norm_str = " ".join(f"{float(v):.3f}" for v in norm)
                information(
                    Fore.WHITE + f"  {host} "
                    + Fore.RED
                    + f"{s} "
                    + f"{host_tasks.get(host, {}).get('attack_subtype', '').upper()}"
                    + f" - RX Pkt {int(raw[0])} {int(raw[1])}%"
                    + f" - {format_bytes(int(raw[2]))}B {int(raw[3])}%"
                    + f" - TX Pkt {int(raw[4])} {int(raw[5])}%"
                    + f" - {format_bytes(int(raw[6]))}B {int(raw[7])}%"
                    + Fore.CYAN  + f" - disc:[{disc_str}]"
                    + Fore.MAGENTA + f" norm:[{norm_str}]\n"
                    + Fore.WHITE
                )
            elif s == HostStatus.UNDER_ATTACK:
                disc = self.get_discretized_state(obs_raw)
                norm = get_normalized_state(obs_raw, self.low, self.high)
                disc_str = " ".join(str(int(v)) for v in disc)
                norm_str = " ".join(f"{float(v):.3f}" for v in norm)
                information(
                    Fore.WHITE + f"  {host} "
                    + Fore.YELLOW
                    + f"{s}-"
                    + f"{host_tasks.get(host, {}).get('traffic_type', '').upper()}"
                    + f" - RX Pkt {int(raw[0])} {int(raw[1])}%"
                    + f" - {format_bytes(int(raw[2]))}B {int(raw[3])}%"
                    + f" - TX Pkt {int(raw[4])} {int(raw[5])}%"
                    + f" - {format_bytes(int(raw[6]))}B {int(raw[7])}%"
                    + Fore.CYAN  + f" - disc:[{disc_str}]"
                    + Fore.MAGENTA + f" norm:[{norm_str}]\n"
                    + Fore.WHITE
                )
            else:
                level_fn(
                    Fore.WHITE + f"  {host} "
                    + Fore.GREEN
                    + f"{s}-"
                    + f"{host_tasks.get(host, {}).get('traffic_type', '').upper()} "
                    + f"to {host_tasks.get(host, {}).get('destination')} "
                    + f" - RX Pkt {int(raw[0])} {int(raw[1])}%"
                    + f" - {format_bytes(int(raw[2]))}B {int(raw[3])}%"
                    + f" - TX Pkt {int(raw[4])} {int(raw[5])}%"
                    + f" - {format_bytes(int(raw[6]))}B {int(raw[7])}%\n"
                    + Fore.WHITE
                )

    def initialize_storage(self):
        pass

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action, options={
        "is_discretized_state": False, "is_real_state": False,
        "current_step": -1, "correct_predictions": 0,
        "show_action": False, "name": None,
    }):
        """
        Execute action and return next state, reward, done, info.

        Args:
            action:  Single discrete action integer.
            options: Dict with additional options (see keys above).

        Returns:
            observation, reward, done, truncated, info
        """
        while hasattr(self, "pause_event") and self.pause_event.is_set():
            notify_client(
                level=SystemLevels.STATUS,
                status=SystemStatus.PAUSED,
                message="Paused training agents...",
                mode=SystemModes.TRAINING,
            )
            time.sleep(1)
            continue

        # Calculate reward
        status              = self.global_state.status.copy()
        action_correct      = status["id"]
        text_action_correct = status["status"]
        reward              = self.calculate_reward(status, action)
        self.execute_action(action, show_action=options["show_action"],
                            reward=reward)

        ground_truth_step = self.per_host_to_action(
            [host_status_id for host_status_id in status["id"]]
        )
        predicted_step     = action.item()
        is_action_correct  = ground_truth_step == predicted_step

        if is_action_correct:
            options["correct_predictions"] += 1
        percentage_correct_predictions = (
            options["correct_predictions"] / options["current_step"]
            if options["current_step"] > 0 else 0
        )

        debug(Fore.CYAN + f"Environment reward {reward}" + Fore.WHITE)

        # Check if the episode is done
        done, truncated = self.check_if_done_or_truncated(
            options["current_step"], percentage_correct_predictions
        )
        # Update state
        if not done and not truncated:
            if self.gym_type == GYM_TYPE[ATTACKS]:
                time.sleep(1)
            else:
                self.update_state()

        next_state = self.get_current_state(
            is_discretized_state=options["is_discretized_state"]
        )
        return next_state, reward, done, truncated, {
            'action_correct':      action_correct,
            'text_action_correct': text_action_correct,
            'status':              status,
            'is_correct_action':   is_action_correct,
            'TimeLimit.truncated': truncated,
            'Ground_truth_step':   ground_truth_step,
            'Predicted_step':      predicted_step,
        }

    # ------------------------------------------------------------------
    # Done / truncated
    # ------------------------------------------------------------------

    def check_if_done_or_truncated(self, current_step,
                                    percentage_correct_predictions):
        """
        Check if the episode is done.
        1. Maximum steps reached.
        2. Accuracy above min_accuracy after steps_min_percentage of steps.
           (Disabled in sequential mode by setting min_accuracy = 2.0.)
        """
        if (current_step >= self.max_steps * self.steps_min_percentage
                and percentage_correct_predictions > self.min_accuracy):
            return False, True  # early truncation on high accuracy

        if current_step >= self.max_steps:
            return True, False

        return False, False

    # ------------------------------------------------------------------
    # Execute action
    # ------------------------------------------------------------------

    def execute_action(self, action: int, show_action=False,
                        name=None, reward=0):
        per_host_actions = self.action_to_per_host(action)
        attacking_hosts  = []
        victimized_hosts = []
        apply_drop_rules = getattr(self.params.attacks, 'apply_drop_rules', True)

        for host_index, host in enumerate(self.hosts):
            if per_host_actions[host_index] == AGENT_ACTIONS.NORMAL_TRAFFIC:
                if apply_drop_rules and not self.status_links[host_index] \
                        and unblock_flow_delete(self.net, host.name):
                    time.sleep(0.02)
                    self.status_links[host_index] = True
                continue
            elif per_host_actions[host_index] == AGENT_ACTIONS.ATTACK_IN:
                victimized_hosts.append(host)
            elif per_host_actions[host_index] == AGENT_ACTIONS.ATTACK_OUT:
                attacking_hosts.append(host)
                if apply_drop_rules and self.status_links[host_index] \
                        and block_flow_drop(self.net, host.name):
                    time.sleep(0.02)
                    self.status_links[host_index] = False

        if len(attacking_hosts) == 0 and len(victimized_hosts) == 0:
            msg = comunicate_normal_traffic_detected()
        else:
            msg = ""
            if len(attacking_hosts) > 0:
                msg = (f"{comunicate_out_attack_detected()} for hosts: "
                       f"{[h.name for h in attacking_hosts]}.")
            if len(victimized_hosts) > 0:
                msg = (f"{comunicate_in_attack_detected()} for hosts: "
                       f"{[h.name for h in victimized_hosts]}.")

        if show_action:
            agent_name = get_agent_name() if name is None else name
            information(f"{msg} R: {reward}\n", agent_name)

        return msg

    # ------------------------------------------------------------------
    # Host status helpers
    # ------------------------------------------------------------------

    def get_attacking_hosts(self):
        return [
            host_name
            for host_name, host_status in self.global_state.host_statuses.items()
            if host_status["status"] == HostStatus.ATTACKING
        ]

    def get_victims(self):
        return [
            host_name
            for host_name, host_status in self.global_state.host_statuses.items()
            if host_status["status"] == HostStatus.UNDER_ATTACK
        ]

    def get_attacking_hosts_index(self, status):
        _ids = (
            HOST_STATUS_ID_MAPPING[HostStatus.ATTACKING],
            HOST_STATUS_ID_MAPPING[HostStatus.OUT_ATTACK_BLOCKED],
        )
        return [
            index for index, value in enumerate(status["id"])
            if value in _ids
        ]

    def get_victims_index(self, status):
        _ids = (
            HOST_STATUS_ID_MAPPING[HostStatus.UNDER_ATTACK],
            HOST_STATUS_ID_MAPPING.get(HostStatus.INCOMING_BLOCKED_ATTACK, 3),
        )
        return [
            index for index, value in enumerate(status["id"])
            if value in _ids
        ]

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def calculate_reward(self, status, action) -> float:
        """Reward based on accurate identification AND appropriate action."""
        reward = 0.0
        attacking_hosts_index    = self.get_attacking_hosts_index(status)
        under_attack_hosts_index = self.get_victims_index(status)
        is_normal = (len(attacking_hosts_index) == 0
                     and len(under_attack_hosts_index) == 0)

        actions_per_host = self.action_to_per_host(action)
        for host_index, action_type in enumerate(actions_per_host):
            if is_normal:
                g_t = HostStatus.NORMAL
            elif host_index in attacking_hosts_index:
                g_t = HostStatus.ATTACKING
            elif host_index in under_attack_hosts_index:
                g_t = HostStatus.UNDER_ATTACK
            else:
                g_t = HostStatus.NORMAL

            if not self.status_links[host_index] and g_t == HostStatus.NORMAL:
                reward += REWARDS.LINK_OFF

            if action_type == AGENT_ACTIONS.NORMAL_TRAFFIC \
                    and g_t == HostStatus.NORMAL:
                reward += REWARDS.CORRECT_NORMAL_TRAFFIC
            elif action_type == AGENT_ACTIONS.NORMAL_TRAFFIC \
                    and g_t != HostStatus.NORMAL:
                reward += REWARDS.FALSE_NEGATIVE
            elif action_type != AGENT_ACTIONS.NORMAL_TRAFFIC \
                    and g_t == HostStatus.NORMAL:
                reward += REWARDS.FALSE_POSITIVE
            else:
                if action_type == AGENT_ACTIONS.ATTACK_IN \
                        and g_t == HostStatus.UNDER_ATTACK:
                    reward += REWARDS.CORRECT_ATTACK_DETECTION
                elif action_type == AGENT_ACTIONS.ATTACK_IN \
                        and g_t == HostStatus.ATTACKING:
                    reward += REWARDS.WRONG_ATTACK_DIRECTION_DETECTED
                elif action_type == AGENT_ACTIONS.ATTACK_OUT \
                        and g_t == HostStatus.ATTACKING:
                    reward += REWARDS.CORRECT_ATTACK_DETECTION
                elif action_type == AGENT_ACTIONS.ATTACK_OUT \
                        and g_t == HostStatus.UNDER_ATTACK:
                    reward += REWARDS.WRONG_ATTACK_DIRECTION_DETECTED

        return reward

    # ------------------------------------------------------------------
    # Discretized state (for Q-Learning / SARSA)
    # ------------------------------------------------------------------

    def get_discretized_state(self, state):
        if state is None:
            return np.array(np.zeros(len(self.low)), dtype=np.float32)

        n_bins = self.n_bins
        low    = self.low
        high   = self.high

        if isinstance(state, tuple):
            state = state[0]

        if self._include_pct_var:
            counter_indices   = {0, 2, 4, 6}
            variation_indices = {1, 3, 5, 7}
        else:
            # Compact mode: all 4 features are counters (log-bin)
            counter_indices   = {0, 1, 2, 3}
            variation_indices = set()

        discrete_state = []
        for i, val in enumerate(state):
            if i in variation_indices:
                bin_index = get_linear_bin_index(val, low[i], high[i],
                                                  n_bins - 1) + 1
            elif i in counter_indices:
                # Zero-aware heavy-tail discretization for counters.
                #   bin 0   -> exactly zero / non-positive
                #   bins 1+ -> logarithmic scale on positive values
                if val <= 0:
                    bin_index = 0
                else:
                    high_safe = max(float(high[i]), 1.0)
                    bin_index = get_log_bin_index(val, 1.0, high_safe, n_bins - 1) + 1
            else:
                bin_index = get_linear_bin_index(val, low[i], high[i], n_bins)
            discrete_state.append(bin_index)

        return tuple(discrete_state)

    # ------------------------------------------------------------------
    # get_current_state override
    # ------------------------------------------------------------------

    def get_current_state(self, is_discretized_state=False,
                           is_real_state=False):
        """
        Override of NetworkEnv.get_current_state.

        global_state.state is always a flat N*8 list.
        We return only the first host's slice.  When include_percentage_variations
        is False the 4 variation features are stripped, yielding shape (4,);
        otherwise shape (8,) as usual.
        The PerHostScanWrapper manages host iteration externally and reads
        slices directly via _raw_slice() / _normalized_slice().
        """
        full_state = self.global_state.state  # N*8 list
        self.real_state = self.state = full_state

        raw8 = np.array(full_state[:8], dtype=np.float32)
        slice_state = raw8 if self._include_pct_var else raw8[[0, 2, 4, 6]]

        if is_real_state:
            return slice_state
        if is_discretized_state:
            return self.get_discretized_state(slice_state)
        return get_normalized_state(slice_state, self.low, self.high)

    @property
    def _per_host_features(self):
        return 8 if self._include_pct_var else 4


# ──────────────────────────────────────────────────────────────────────────────
# Module-level helpers (called via lazy import in _apply_scenario_step)
# These are intentionally at module level, not class methods, to keep
# _apply_scenario_step free of direct adversarial_agent imports at class load.
# ──────────────────────────────────────────────────────────────────────────────
# NOTE: _generate_normal_traffic_once, _launch_attack_once, _name_to_attack_type
# live in reinforcement_learning/agents/adversarial_agent.py as you placed them.


if __name__ == '__main__':
    env_params = {
        'net_params': {
            'num_hosts': 3,
            'num_switches': 1,
            'num_iots': 0,
            'controller': {
                'ip': '192.168.1.226',
                'port': 6633,
                'usr': 'admin',
                'pwd': 'admin',
            },
        },
        'K_steps': 2,
        'steps_min_percentage': 0.9,
        'accuracy_min': 0.9,
    }
    env_params = jsonlib.loads(jsonlib.dumps(env_params), object_hook=Params)
    env = NetworkEnvAttackDetectPerHostObservable(env_params, 'server_user')
    observation, info = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        observation, reward, done, _, _ = env.step(action)
        if done:
            break
    env.close()
