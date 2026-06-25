"""
PerHostScanWrapper
==================
Wrapper for NetworkEnvAttackDetectPerHostObservable that decomposes a single
"logical step" into N sequential micro-steps, one per host.

Observation space : Box(shape=(8,))  — constant regardless of number of hosts
Action space      : Discrete(3)      — NORMAL_TRAFFIC / ATTACK_IN / ATTACK_OUT

Observation convention
----------------------
The wrapper always returns RAW (un-normalised) observations.  Each agent is
responsible for its own transformation:

  DQN / PPO / A2C (SB3)
      SB3 does NOT normalise internally by default.  The network receives the
      raw values directly.  If normalisation is desired, add a VecNormalize
      wrapper at the SB3 level, or normalise in the policy's pre-processing.
      In our setup we pass raw values and let the network learn to handle them.

  Q-Learning / SARSA (tabular)
      After each step(), the agent calls env.get_discretized_state(obs) to
      convert the raw slice into a Q-table index.  The discretization uses
      env.low and env.high which are in raw units — consistent with raw obs.

  Evaluation (attack_detect_ho.py)
      Uses raw obs directly with model.predict().  SB3 models are trained on
      raw obs so they predict correctly from raw obs.  Tabular models call
      get_discretized_state() in their predict() method.

Why raw and not normalised
--------------------------
Returning normalised observations caused two bugs:
  1. DQN received values already in [0,1]; SB3 observation_space bounds are
     also [0,1] (since we set low/high to normalised bounds) — no further
     normalisation occurs, but the Q-table learned incorrect value ranges.
  2. Tabular agents received [0,1] values and passed them to get_discretized_state()
     which expects raw per-host values (range 0..threshold_packets).
     All values collapsed into the first bin, making the Q-table useless.

Key invariant
-------------
host_status_id (ground truth) is read from global_state ONCE at the very
beginning of step(), before _execute_action_for_host() and before
update_state() can mutate global_state.

Special case: out_attack_blocked
--------------------------------
When an attacker host is blocked, the environment may expose the original
status as OUT_ATTACK_BLOCKED (id=4) for telemetry/history purposes.
For reward and action correctness, the wrapper still treats that case as
ATTACKING until the dedicated class remapping step is enabled.

infos (compatible with CustomCallback.manage_step_data)
---------------------------------------------------------
  action_correct        int  0/1/2   ground truth for the current host
  text_action_correct   str          ground truth as a string
  status                dict         per-host traffic data
  is_correct_action     bool
  Ground_truth_step     int  0/1/2   scalar, directly usable by sklearn
  Predicted_step        int  0/1/2   scalar, directly usable by sklearn
  host_idx              int
  host_name             str
  TimeLimit.truncated   bool

done / truncated
----------------
Always False for intermediate micro-steps.
Propagated from the base env only at the end of a full round (after host N-1).
"""

import time

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from reinforcement_learning.network_env import get_normalized_state
from utility.my_log import notify_client

from .network_env_attack_detect_per_host_observable import (
    NetworkEnvAttackDetectPerHostObservable,
)
from .constants import AGENT_ACTIONS, NORMALIZED, RAW, REWARDS
from utility.constants import HostStatus, SystemLevels, SystemModes, SystemStatus


# Map host_status_id (int) → HostStatus string constant
_ID_TO_HOST_STATUS = {
    0: HostStatus.NORMAL,
    1: HostStatus.UNDER_ATTACK,
    2: HostStatus.ATTACKING,
    3: HostStatus.INCOMING_BLOCKED_ATTACK,
}


def _compute_attack_reward_scale(num_hosts: int) -> float:
    """
    Compute reward multiplier for attack-class detections proportional to
    the class imbalance introduced by the number of hosts.

    With N hosts and 2 attack-related hosts per round, the imbalance ratio
    is (N-2)/2.  We use (N-1)/2 as a conservative estimate, clamped to [1,8].

        num_hosts=3  → scale=1.0
        num_hosts=6  → scale=2.5
        num_hosts=10 → scale=4.5
    """
    return float(max(1.0, min((num_hosts - 1) / 2.0, 8.0)))


class PerHostScanWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that exposes constant spaces Box(8,) / Discrete(3)
    regardless of the number of hosts in the network.

        Returns observations according to selected mode:
            - normalized (default)
            - raw
    """

    # Internal storage always uses 8 features per host (raw from OVS/global_state).
    # The output observation may be 4 or 8 depending on include_percentage_variations.
    _INTERNAL_FEATURES = 8
    _COUNTER_INDICES   = np.array([0, 2, 4, 6], dtype=np.intp)

    def __init__(self, env: NetworkEnvAttackDetectPerHostObservable):
        super().__init__(env)

        self._num_hosts         = env.num_hosts
        self._per_host_features = env._per_host_features  # 4 or 8
        self.deep_state_mode    = NORMALIZED

        self._set_observation_space()
        self.action_space = spaces.Discrete(AGENT_ACTIONS.NUMBER)

        # Internal pointer and counters
        self._current_host_idx:   int = 0
        # _full_obs always stores N*8 (all raw features from global_state)
        self._full_obs: np.ndarray    = np.zeros(
            self._num_hosts * self._INTERNAL_FEATURES, dtype=np.float32
        )
        self._current_micro_step: int = 0
        self._current_round_step: int = 0
        self._correct_predictions:int = 0
        self._episode_under_attack_count: int = 0
        self._episode_mitigated_under_attack_count: int = 0

        # Attack reward scale — proportional to class imbalance
        self._attack_reward_scale: float = _compute_attack_reward_scale(
            self._num_hosts
        )
        from utility.my_log import information
        information(
            f"PerHostScanWrapper: num_hosts={self._num_hosts}, "
            f"attack_reward_scale={self._attack_reward_scale:.2f}\n"
        )

        # Blocked hosts now keep receiving live dump-ports counters.
        # Kept only for backward compatibility / saved wrapper state.
        self._frozen_obs: dict = {}
        self._block_started_round: dict = {}
        self._normal_streak_while_blocked: dict = {}

        # Anti-oscillation policy for unblock decisions.
        attacks_cfg = getattr(self.env.params, "attacks", None)
        self._unblock_min_hold_rounds = int(getattr(attacks_cfg, "unblock_min_hold_rounds", 2) or 2)
        self._unblock_required_normal_streak = int(getattr(attacks_cfg, "unblock_required_normal_streak", 2) or 2)

    # ------------------------------------------------------------------
    # Transparent attribute forwarding to base env
    # ------------------------------------------------------------------

    _WRAPPER_OWN_ATTRS = frozenset({
        'env',
        '_num_hosts', '_per_host_features',
        'deep_state_mode',
        'observation_space', 'action_space',
        '_current_host_idx', '_full_obs',
        '_current_micro_step', '_current_round_step', '_correct_predictions',
        '_episode_under_attack_count', '_episode_mitigated_under_attack_count',
        '_attack_reward_scale',
        '_block_started_round', '_normal_streak_while_blocked',
        '_unblock_min_hold_rounds', '_unblock_required_normal_streak',
        '_frozen_obs',
        'metadata', 'render_mode', 'reward_range', 'spec',
    })

    def __getattr__(self, name: str):
        try:
            base = object.__getattribute__(self, 'env')
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        return getattr(base, name)

    def __setattr__(self, name: str, value) -> None:
        if name in PerHostScanWrapper._WRAPPER_OWN_ATTRS:
            object.__setattr__(self, name, value)
        else:
            try:
                base = object.__getattribute__(self, 'env')
                setattr(base, name, value)
            except AttributeError:
                object.__setattr__(self, name, value)

    # ------------------------------------------------------------------
    # Explicit forwarding for methods called by AgentManager / SB3
    # ------------------------------------------------------------------

    def initialize_storage(self):
        """Forward to base env — called by AgentManager.__init__."""
        return self.env.initialize_storage()

    # ------------------------------------------------------------------
    # Read-only properties for events
    # ------------------------------------------------------------------

    @property
    def stop_event(self):
        return self.env.stop_event

    @property
    def pause_event(self):
        return self.env.pause_event

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _refresh_full_obs(self) -> None:
        """
        Copy the full N*8 RAW state from global_state into _full_obs,
        keeping blocked hosts live so dropped outgoing traffic is still visible.
        """
        live = np.array(self.env.global_state.state, dtype=np.float32)
        self._frozen_obs.clear()
        self._full_obs = live

    def _freeze_host_now(self, host_idx: int) -> None:
        """
        Deprecated no-op.

        Blocked hosts should continue to expose live counters, including
        packets that hit the switch port and are then dropped by policy.
        """
        self._frozen_obs.pop(host_idx, None)

    def _is_incoming_attack_mitigated(self, victim_host_idx: int) -> bool:
        """
        True when victim is tagged UNDER_ATTACK but all known attackers
        targeting that victim are currently blocked.
        """
        host_tasks = getattr(self.env, "host_tasks", None)
        if not isinstance(host_tasks, dict) or not self.env.hosts:
            return False

        victim_name = self.env.hosts[victim_host_idx].name
        attacker_indexes = []
        for idx, host in enumerate(self.env.hosts):
            task = host_tasks.get(host.name, {}) if isinstance(host_tasks.get(host.name, {}), dict) else {}
            task_type = task.get("task_type")
            destination = task.get("destination")
            if destination == victim_name and task_type in ("short_attack", "long_attack"):
                attacker_indexes.append(idx)

        if not attacker_indexes:
            return False

        return all(not self.env.status_links[idx] for idx in attacker_indexes)

    def _effective_ground_truth(self, host_idx: int, host_status_id: int, host_status_text: str):
        """
        Convert UNDER_ATTACK to INCOMING_BLOCKED_ATTACK when incoming attack is mitigated.
        
        Mapping rules:
        - OUT_ATTACK_BLOCKED (status 2) → stays as ATTACKING (class 2)
        - UNDER_ATTACK with all attackers blocked → INCOMING_BLOCKED_ATTACK (status 3)
        - Everything else → unchanged
        """
        if host_status_text == HostStatus.OUT_ATTACK_BLOCKED:
            return 2, HostStatus.ATTACKING, False
        if host_status_text == HostStatus.UNDER_ATTACK and self._is_incoming_attack_mitigated(host_idx):
            return 3, HostStatus.INCOMING_BLOCKED_ATTACK, True
        return host_status_id, host_status_text, False

    def _raw_slice(self, host_idx: int) -> np.ndarray:
        """
        Return the RAW observation slice for the given host.
        _full_obs always stores N*8; when include_percentage_variations is False
        the 4 variation columns are stripped to return a (4,) counter-only slice.
        """
        start = host_idx * self._INTERNAL_FEATURES
        raw8  = self._full_obs[start: start + self._INTERNAL_FEATURES]
        if not self.env._include_pct_var:
            return raw8[self._COUNTER_INDICES].copy()
        return raw8.copy()

    def _normalized_slice(self, host_idx: int) -> np.ndarray:
        raw_slice = self._raw_slice(host_idx)
        normalized = get_normalized_state(
            raw_slice,
            self.env.low_to_normalize,
            self.env.high_to_normalize,
        )
        return np.array(normalized, dtype=np.float32)

    def _set_observation_space(self) -> None:
        if self.deep_state_mode == NORMALIZED:
            low = np.zeros(self._per_host_features, dtype=np.float32)
            high = np.ones(self._per_host_features, dtype=np.float32)
        else:
            low = np.array(self.env.low, dtype=np.float32)
            high = np.array(self.env.high, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=low,
            high=high,
            shape=(self._per_host_features,),
            dtype=np.float32,
        )

    def set_state_mode(self, mode: str) -> None:
        selected = str(mode).strip().lower()
        self.deep_state_mode = RAW if selected == RAW else NORMALIZED
        self._set_observation_space()

    def _get_output_slice(self, host_idx: int) -> np.ndarray:
        if self.deep_state_mode == NORMALIZED:
            return self._normalized_slice(host_idx)
        return self._raw_slice(host_idx)

    def _reward_for_host(self, host_idx: int, action: int,
                         host_status_id: int) -> float:
        """
        Compute the reward for a single host with attack class scaling.
        host_status_id is passed in explicitly (captured before any mutation).
        """
        g_t    = _ID_TO_HOST_STATUS.get(host_status_id, HostStatus.NORMAL)
        link_up = self.env.status_links[host_idx]
        reward  = 0.0
        scale   = self._attack_reward_scale

        if not link_up and g_t == HostStatus.NORMAL:
            reward += REWARDS.LINK_OFF

        if action == AGENT_ACTIONS.NORMAL_TRAFFIC:
            if g_t == HostStatus.NORMAL:
                # Not scaled: reward for normal stays constant regardless of host count
                # so that "always predict normal" does not become more attractive
                # with larger networks.
                reward += REWARDS.CORRECT_NORMAL_TRAFFIC
            else:
                reward += REWARDS.FALSE_NEGATIVE * scale

        elif action == AGENT_ACTIONS.ATTACK_IN:
            if g_t == HostStatus.NORMAL:
                reward += REWARDS.FALSE_POSITIVE * scale
            elif g_t == HostStatus.UNDER_ATTACK:
                reward += REWARDS.CORRECT_ATTACK_DETECTION * scale
            elif g_t == HostStatus.ATTACKING:
                # Also scaled so that wrong-direction stays costlier than FP
                # even when attack_reward_scale > 1 (more than 3 hosts).
                reward += REWARDS.WRONG_ATTACK_DIRECTION_DETECTED * scale

        elif action == AGENT_ACTIONS.ATTACK_OUT:
            if g_t == HostStatus.NORMAL:
                reward += REWARDS.FALSE_POSITIVE * scale
            elif g_t == HostStatus.ATTACKING:
                reward += REWARDS.CORRECT_ATTACK_DETECTION * scale
            elif g_t == HostStatus.UNDER_ATTACK:
                reward += REWARDS.WRONG_ATTACK_DIRECTION_DETECTED * scale

        return reward

    def _execute_action_for_host(self, host_idx: int, action: int,
                                  reward: float, show_action: bool,
                                  name) -> None:
        """
        Execute a block/unblock action on host_idx ONLY.

        We do NOT go through execute_action() on the base env because that
        method always iterates over all N hosts and may call unblock_flow_delete
        on hosts that happen to have their link down for unrelated reasons.

        Instead we replicate only the single-host SDN logic:
          NORMAL_TRAFFIC → if link is down, unblock it (agent says "normal now")
          ATTACK_IN      → if link is down, unblock it (victim should not be blocked)
          ATTACK_OUT     → if link is up,   block  it (attacker should be dropped)
        """
        from utility.network_configurator import block_flow_drop, unblock_flow_delete
        from utility.my_log import information as _info, debug as _debug
        from reinforcement_learning.network_env import get_agent_name

        apply_drop_rules = getattr(
            getattr(self.env.params, 'attacks', None), 'apply_drop_rules', True
        )

        host      = self.env.hosts[host_idx]
        link_up   = self.env.status_links[host_idx]

        if action == AGENT_ACTIONS.NORMAL_TRAFFIC:
            # Agent says normal — unblock only after hold + consistent normal streak.
            if not link_up:
                self._normal_streak_while_blocked[host_idx] = self._normal_streak_while_blocked.get(host_idx, 0) + 1
                held_rounds = self._current_round_step - self._block_started_round.get(host_idx, self._current_round_step)
                if (held_rounds >= self._unblock_min_hold_rounds
                        and self._normal_streak_while_blocked.get(host_idx, 0) >= self._unblock_required_normal_streak):
                    if apply_drop_rules and unblock_flow_delete(self.env.net, host.name):
                        import time as _t; _t.sleep(0.02)
                        self.env.status_links[host_idx] = True
                        self._frozen_obs.pop(host_idx, None)
                        self._normal_streak_while_blocked.pop(host_idx, None)
                        self._block_started_round.pop(host_idx, None)

        elif action == AGENT_ACTIONS.ATTACK_IN:
            # ATTACK_IN alone is not enough evidence to unblock.
            if not link_up:
                self._normal_streak_while_blocked[host_idx] = 0

        elif action == AGENT_ACTIONS.ATTACK_OUT:
            # Attacker — block its outgoing traffic (OVS drop rule only when enabled)
            if link_up and apply_drop_rules and block_flow_drop(self.env.net, host.name):
                import time as _t; _t.sleep(0.02)
                self.env.status_links[host_idx] = False
                self._block_started_round[host_idx] = self._current_round_step
                self._normal_streak_while_blocked[host_idx] = 0

        if show_action:
            agent_name = get_agent_name() if name is None else name
            _info(
                f"Host {host.name} action={action} "
                f"link={'UP' if self.env.status_links[host_idx] else 'DOWN'} "
                f"R:{reward:.2f}\n",
                agent_name,
            )

        # Freeze raw obs immediately if the link just went down
        if link_up and not self.env.status_links[host_idx]:
            self._freeze_host_now(host_idx)

    def _build_infos(self, host_idx: int, action: int,
                     host_status_id: int, host_status_text: str,
                     original_status_id: int, original_status_text: str,
                     incoming_attack_mitigated: bool,
                     reward: float, done: bool, truncated: bool) -> dict:
        """Build infos dict compatible with CustomCallback.manage_step_data."""
        host_name = self.env.hosts[host_idx].name
        host_data = self.env.global_state.host_statuses.get(host_name, {})

        status = {
            "id":     [host_status_id],
            "status": [host_status_text],
            "received_packets":
                host_data.get("received_packets", 0),
            "received_packets_percentage_change":
                host_data.get("received_packets_percentage_change", 0),
            "received_bytes":
                host_data.get("received_bytes", 0),
            "received_bytes_percentage_change":
                host_data.get("received_bytes_percentage_change", 0),
            "transmitted_packets":
                host_data.get("transmitted_packets", 0),
            "transmitted_packets_percentage_change":
                host_data.get("transmitted_packets_percentage_change", 0),
            "transmitted_bytes":
                host_data.get("transmitted_bytes", 0),
            "transmitted_bytes_percentage_change":
                host_data.get("transmitted_bytes_percentage_change", 0),
        }

        return {
            "action_correct":      host_status_id,
            "text_action_correct": host_status_text,
            "action_correct_original":      original_status_id,
            "text_action_correct_original": original_status_text,
            "incoming_attack_mitigated": incoming_attack_mitigated,
            "status":              status,
            "is_correct_action":   int(action) == host_status_id,
            "Ground_truth_step":   host_status_id,
            "Predicted_step":      int(action),
            "TimeLimit.truncated": truncated,
            "host_idx":            host_idx,
            "host_name":           host_name,
            "reward":              reward,
        }

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """
        Reset and return RAW observation for host 0.

        options is accepted for API compatibility (Q-Learning/SARSA pass
        is_discretized_state etc.) but ignored here — the wrapper always
        returns raw obs; each agent transforms as needed.
        """
        if options is None:
            options = {}

        # Remove any OVS drop rules left over from the previous episode.
        # If we skip this, _block_started_round.clear() below resets the
        # block timestamp so held_rounds always evaluates to 0, making the
        # unblock condition (held_rounds >= _unblock_min_hold_rounds) permanently
        # unsatisfiable and leaving the drop rule active for the entire next episode.
        apply_drop_rules = getattr(
            getattr(self.env.params, 'attacks', None), 'apply_drop_rules', True
        )
        if apply_drop_rules:
            from utility.network_configurator import unblock_flow_delete
            for host_idx in range(self._num_hosts):
                if not self.env.status_links[host_idx]:
                    host = self.env.hosts[host_idx]
                    unblock_flow_delete(self.env.net, host.name)
                    self.env.status_links[host_idx] = True

        _, info = self.env.reset(seed=seed, options=options)

        self._frozen_obs.clear()
        self._refresh_full_obs()
        self._current_host_idx    = 0
        self._current_micro_step  = 0
        self._current_round_step  = 0
        self._correct_predictions = 0
        self._episode_under_attack_count = 0
        self._episode_mitigated_under_attack_count = 0
        self._block_started_round.clear()
        self._normal_streak_while_blocked.clear()

        return self._get_output_slice(0), info

    def step(self, action: int, options: dict = None):
        """
        Micro-step for host self._current_host_idx.

        Semantic contract
        -----------------
        The full N*8 observation (self._full_obs) is acquired ONCE at the
        beginning of each round — either at reset() or after the previous
        round's update_state() — and stays FROZEN for all N micro-steps.

        Each micro-step:
          1. Reads the 8-feature slice for the current host from _full_obs
             (already captured, no new network read).
          2. Computes reward for (slice, action, ground_truth).
          3. Executes the SDN action (block/unblock) on that host ONLY.
          4. Returns the slice for the NEXT host (still from the same frozen
             _full_obs), plus reward and infos. No new obs from the network.

        Only after micro-step N-1 (last host in the round):
          - update_state() is called → reads new OVS counters → _full_obs updated.
          - done/truncated are evaluated.

        This guarantees that all N micro-steps within a round see the same
        network snapshot, which is correct because only one real network tick
        has elapsed between rounds.

        Returns RAW obs shape (8,). Each agent normalises or discretizes
        independently:
          - SB3 agents: use raw obs directly
          - Tabular agents: call get_discretized_state(obs) after step()

        options is accepted for backward compatibility with Q-Learning/SARSA.
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
        host_idx = self._current_host_idx

        # ── 1. Capture ground truth BEFORE any mutation ────────────────
        host_status_id   = self.env.global_state.status["id"][host_idx]
        host_status_text = self.env.global_state.status["status"][host_idx]
        effective_status_id, effective_status_text, incoming_attack_mitigated = self._effective_ground_truth(
            host_idx,
            host_status_id,
            host_status_text,
        )
        if host_status_text == HostStatus.UNDER_ATTACK:
            self._episode_under_attack_count += 1
            if incoming_attack_mitigated:
                self._episode_mitigated_under_attack_count += 1

        # ── 2. Reward with attack scaling ───────────────────────────────
        reward = self._reward_for_host(host_idx, action, effective_status_id)

        # ── 3. Execute action — may freeze obs if link goes down ────────
        self._execute_action_for_host(
            host_idx, int(action), reward,
            show_action=False, name=None
        )

        # ── 4. Update step counters ─────────────────────────────────────
        self._current_micro_step += 1
        if int(action) == effective_status_id:
            self._correct_predictions += 1
        percentage_correct = (
            self._correct_predictions / self._current_micro_step
            if self._current_micro_step > 0 else 0.0
        )

        # ── 5. Advance host pointer ─────────────────────────────────────
        done          = False
        truncated     = False
        next_host_idx = host_idx + 1

        if next_host_idx >= self._num_hosts:
            self._current_round_step += 1

            done, truncated = self.env.check_if_done_or_truncated(
                self._current_round_step, percentage_correct
            )
            # Only advance the scenario when the episode continues.
            # If done/truncated, the next reset() will call update_state() to
            # load the next step — calling it here too would waste one df entry
            # and cause every episode to consume two scenario steps instead of one.
            if not done and not truncated:
                self.env.update_state()
                self._refresh_full_obs()
            self._current_host_idx = 0
            next_host_idx          = 0
        else:
            self._current_host_idx = next_host_idx

        # ── 6. Build infos ──────────────────────────────────────────────
        infos = self._build_infos(
            host_idx, int(action),
            effective_status_id, effective_status_text,
            host_status_id, host_status_text,
            incoming_attack_mitigated,
            reward, done, truncated
        )
        infos["micro_step"] = self._current_micro_step
        infos["round_step"] = self._current_round_step
        infos["episode_under_attack_count"] = self._episode_under_attack_count
        infos["episode_mitigated_under_attack_count"] = self._episode_mitigated_under_attack_count

        if self._episode_under_attack_count > 0:
            mitigated_ratio = self._episode_mitigated_under_attack_count / self._episode_under_attack_count
        else:
            mitigated_ratio = 0.0
        infos["episode_mitigated_under_attack_ratio"] = mitigated_ratio

        if done or truncated:
            from utility.my_log import information
            information(
                f"[PerHostScanWrapper] episode mitigation stats: "
                f"under_attack={self._episode_under_attack_count}, "
                f"mitigated={self._episode_mitigated_under_attack_count}, "
                f"ratio={mitigated_ratio:.3f}\n"
            )

        # ── 7. Return RAW obs for next host ─────────────────────────────
        obs = self._get_output_slice(next_host_idx)

        return obs, reward, done, truncated, infos

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def current_host_idx(self) -> int:
        return self._current_host_idx

    @property
    def current_host_name(self) -> str:
        return self.env.hosts[self._current_host_idx].name

    @property
    def num_hosts(self) -> int:
        return self._num_hosts
