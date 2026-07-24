from typing import Any, Dict

import numpy as np

from reinforcement_learning.instant_state import InstantState as BaseInstantState, _serialize_complex_types
from utility.constants import NORMAL

from .constants import COORDINATOR, COORDINATOR_STATUS_ID_MAPPING, CommStrategy, HOST_STATUS_ID_MAPPING


class MarlPzInstantState(BaseInstantState):
    """
    InstantState extended for the marl_pz scenario.

    Adds per-agent messages, coordinator aggregate state, and link status.
    """

    def __init__(self, hosts, coordinator_state: np.ndarray, messages: dict):
        super().__init__(hosts)
        # Override parent's string-based default so host_statuses always contains
        # full traffic-field dicts (required by base_agent.manage_step_data).
        for host in hosts:
            self.host_statuses[host.name] = {
                'id': 0,
                'status': 'normal',
                'received_packets': 0.0,
                'received_packets_percentage_change': 0.0,
                'received_bytes': 0.0,
                'received_bytes_percentage_change': 0.0,
                'transmitted_packets': 0.0,
                'transmitted_packets_percentage_change': 0.0,
                'transmitted_bytes': 0.0,
                'transmitted_bytes_percentage_change': 0.0,
            }
        self.coordinator_state = coordinator_state       # shape (4,): [pkts, pkts_pct, bytes, bytes_pct]
        self.coordinator_status = {'id': 0, 'status': 'normal',
                                   'packets': 0.0, 'packets_percentage_change': 0.0,
                                   'bytes': 0.0, 'bytes_percentage_change': 0.0}
        self.messages = messages                         # {agent_id: {sender_id: int}}
        self.links_status = {}                           # {host_name: 1|0}

    # ------------------------------------------------------------------
    # Status update
    # ------------------------------------------------------------------

    def update_statuses(self, statuses: dict):
        """Update host and coordinator statuses from a {name: status_str} dict."""
        for name, status in statuses.items():
            if name == COORDINATOR:
                status_id = COORDINATOR_STATUS_ID_MAPPING.get(status, -1)
                self.coordinator_status = {
                    "id":                       status_id,
                    "status":                   status,
                    "packets":                  self.coordinator_state[0],
                    "packets_percentage_change": self.coordinator_state[1],
                    "bytes":                    self.coordinator_state[2],
                    "bytes_percentage_change":  self.coordinator_state[3],
                }
                self.status = {
                    "id":                       status_id,
                    "status":                   status,
                    "packets":                  self.packets,
                    "bytes":                    self.bytes,
                    "packets_percentage_change": self.packets_percentage_change,
                    "bytes_percentage_change":  self.bytes_percentage_change,
                }
            else:
                if name in self.host_states:
                    status_id = HOST_STATUS_ID_MAPPING.get(status, -1)
                    hs = self.host_states[name]
                    self.host_statuses[name] = {
                        "id":                                    status_id,
                        "status":                                status,
                        "received_packets":                      hs[0],
                        "received_packets_percentage_change":    hs[1],
                        "received_bytes":                        hs[2],
                        "received_bytes_percentage_change":      hs[3],
                        "transmitted_packets":                   hs[4],
                        "transmitted_packets_percentage_change": hs[5],
                        "transmitted_bytes":                     hs[6],
                        "transmitted_bytes_percentage_change":   hs[7],
                    }

    # ------------------------------------------------------------------
    # Coordinator helpers
    # ------------------------------------------------------------------

    def get_coordinator_state(self) -> np.ndarray:
        return self.coordinator_state

    def get_messages(self, agent_name: str) -> dict:
        return self.messages.get(agent_name, {})

    def set_message(self, agent_name: str, sender_name: str, message: int):
        if agent_name in self.messages:
            self.messages[agent_name][sender_name] = message
        else:
            self.messages[agent_name] = {sender_name: message}

    # ------------------------------------------------------------------
    # Live communication snapshot (alert family: NONE/NAIVE_BROADCAST/UAQ)
    # ------------------------------------------------------------------

    _UAQ_LABELS = {0: 'normal', 1: 'uncertain', 2: 'confident'}

    def get_comm_snapshot(self, comm_strategy: str) -> Dict[str, Any]:
        """Per-step communication snapshot for the alert family, read fresh off
        `self.messages` (never accumulated — safe to call every step).
        Returns {} for CommStrategy.NONE (no coordinator, nothing to show) and
        for the policy-coordination strategies (those are surfaced as discrete
        `commEvent` messages instead, not a per-step snapshot).
        """
        if comm_strategy not in (CommStrategy.NAIVE_BROADCAST, CommStrategy.UAQ):
            return {}

        coordinator_inbox = self.messages.get(COORDINATOR, {})
        host_alerts: Dict[str, Any] = {}
        coordinator_broadcast = None
        for host_name in self.host_statuses.keys():
            value = coordinator_inbox.get(host_name, 0)
            if comm_strategy == CommStrategy.UAQ:
                label = self._UAQ_LABELS.get(value, 'normal')
            else:
                label = 'alert' if value else 'normal'
            host_alerts[host_name] = {'value': value, 'label': label}
            if coordinator_broadcast is None:
                coordinator_broadcast = self.messages.get(host_name, {}).get(COORDINATOR)

        return {
            'family': 'alert',
            'strategy': comm_strategy,
            'hostAlerts': host_alerts,
            'coordinatorBroadcast': coordinator_broadcast,
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        data = {
            'totalPackets':              self.total_packets,
            'totalBytes':                self.total_bytes,
            'packets':                   self.packets,
            'bytes':                     self.bytes,
            'packetsPercentageChange':   self.packets_percentage_change,
            'bytesPercentageChange':     self.bytes_percentage_change,
            'hostStatesRaw':             self.host_states,
            'coordinatorStateRaw':       self.coordinator_state,
            'messagesRaw':               self.messages,
            'hostStatusesStructured':    self.host_statuses,
            'coordinatorStatus':         self.coordinator_status,
            'status':                    self.status,
        }
        return _serialize_complex_types(data)
