from typing import Any, Dict
import numpy as np
from utility.constants import ATTACK, NORMAL 

# --- Helper function to handle NumPy ndarray conversion ---
def _serialize_complex_types(data: Any) -> Any:
    """Recursively converts NumPy arrays and scalar types to standard Python lists and primitives."""
    if isinstance(data, np.ndarray):
        # Convert NumPy array to a list, which is JSON serializable
        return data.tolist()
    elif isinstance(data, np.generic):
        # Convert NumPy scalar types (like np.float32, np.int64) to standard Python types
        return data.item()
    elif isinstance(data, dict):
        # Recursively apply conversion to dictionary values
        return {k: _serialize_complex_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        # Recursively apply conversion to list elements
        return [_serialize_complex_types(item) for item in data]
    else:
        # Return simple types as is
        return data


class InstantState:
    """
    A data class to hold the state of all agents and the coordinator
    at a single time step, including any messages passed.
    """
    def __init__(self, hosts: dict):
        """
        Args:
            agent_states (dict): A dictionary mapping agent IDs to their local state.
            coordinator_state: The global state for the coordinator agent.
            messages (dict): A dictionary mapping agent IDs to a list of messages they received.
        """
        self.total_packets = 0
        self.total_bytes = 0
        self.packets = 0
        self.bytes = 0
        self.packets_percentage_change = 0
        self.bytes_percentage_change = 0  
        # Backward-compatibility vector used by NetworkEnv.get_current_state().
        self._state = np.array([0, 0, 0, 0], dtype=np.float32)
        self.host_states = {host.name: np.zeros(8, dtype=np.float32) for host in hosts}
        self.host_states_total = {host_id: np.zeros(4) for host_id, _ in self.host_states.items() }
        self.host_statuses = { host_id: NORMAL for host_id in self.host_states.keys()} #default status

        self.status = {"id" :-1, 
                       "status" : "idle",  
                       "packets" :0, 
                       "bytes" :0,
                       "packets_percentage_change" :0, 
                       "bytes_percentage_change" :0} #default status
        self.consecutive_corrects = 0
        
    def set_state(self, status: dict):

        self.total_packets += status["packets"]
        self.total_bytes += status["bytes"]
        self.packets = status["packets"]
        self.bytes = status["bytes"]
        self.packets_percentage_change = status["packetsPercentageChange"]
        self.bytes_percentage_change = status["bytesPercentageChange"]
        self._state = np.array(
            [
                self.packets,
                self.packets_percentage_change,
                self.bytes,
                self.bytes_percentage_change,
            ],
            dtype=np.float32,
        )

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        # Allow setting with either a raw state array (np.ndarray/list)
        # or with a structured status dict that contains hostStatusesStructured.
        if isinstance(value, (np.ndarray, list, tuple)):
            self._state = np.array(value, dtype=np.float32)
            return

        # If a dict-like status object is provided, update both the main
        # state vector and the per-host structures.
        if isinstance(value, dict):
            # If a flat state vector is embedded, try to read it
            if "packets" in value and "bytes" in value:
                try:
                    self.packets = value.get("packets", self.packets)
                    self.bytes = value.get("bytes", self.bytes)
                    self.packets_percentage_change = value.get("packetsPercentageChange", self.packets_percentage_change)
                    self.bytes_percentage_change = value.get("bytesPercentageChange", self.bytes_percentage_change)
                    self._state = np.array([
                        self.packets,
                        self.packets_percentage_change,
                        self.bytes,
                        self.bytes_percentage_change,
                    ], dtype=np.float32)
                except Exception:
                    # fallback: attempt to coerce any provided raw state
                    try:
                        self._state = np.array(value, dtype=np.float32)
                    except Exception:
                        pass

            # Update host-level information if present
            host_struct = value.get("hostStatusesStructured") or value.get("host_statuses_structured")
            if isinstance(host_struct, dict):
                for host_id, host_status in host_struct.items():
                    # update totals if present
                    if host_id in self.host_states_total:
                        self.host_states_total[host_id][0] += host_status.get("receivedPackets", 0)
                        self.host_states_total[host_id][1] += host_status.get("receivedBytes", 0)
                        self.host_states_total[host_id][2] += host_status.get("transmittedPackets", 0)
                        self.host_states_total[host_id][3] += host_status.get("transmittedBytes", 0)

                    # update current host state vector if the host exists
                    if host_id in self.host_states:
                        self.host_states[host_id][0] = host_status.get("receivedPackets", self.host_states[host_id][0])
                        self.host_states[host_id][1] = host_status.get("receivedPacketsPercentageChange", self.host_states[host_id][1])
                        self.host_states[host_id][2] = host_status.get("receivedBytes", self.host_states[host_id][2])
                        self.host_states[host_id][3] = host_status.get("receivedBytesPercentageChange", self.host_states[host_id][3])
                        self.host_states[host_id][4] = host_status.get("transmittedPackets", self.host_states[host_id][4])
                        self.host_states[host_id][5] = host_status.get("transmittedPacketsPercentageChange", self.host_states[host_id][5])
                        self.host_states[host_id][6] = host_status.get("transmittedBytes", self.host_states[host_id][6])
                        self.host_states[host_id][7] = host_status.get("transmittedBytesPercentageChange", self.host_states[host_id][7])

                    # keep a copy of the structured host status
                    self.host_statuses[host_id] = host_status

            # update top-level status field if present
            if "status" in value:
                self.status = value.get("status", self.status)
            return

        # Fallback: coerce whatever is given into the float32 state
        self._state = np.array(value, dtype=np.float32)
           
   
    

            
    def get_host_state(self, host_id: str):
        """
        Retrieves the local state for a specific agent.

        Args:
            agent_id (str): The ID of the agent (e.g., 'h1', 'h2').

        Returns:
            The local state for the specified agent.
        """
        return self.host_states.get(host_id, None)
    
        
    def get_host_status(self, host_id: str):
        """
        Retrieves the local state for a specific agent.

        Args:
            agent_id (str): The ID of the agent (e.g., 'h1', 'h2').

        Returns:
            The local state for the specified agent.
        """
        return self.host_statuses.get(host_id, None)   
    
    def get_network_traffic_status(self):
        """
        Retrieves the network traffic status for all agents.
        Returns:
            dict: A dictionary mapping agent IDs to their network traffic status.
        """
        ht = {}
        for host_name, host_status in self.host_statuses.items():
            ht[host_name] = {
                'id': host_status['id'],
                'status': host_status['status'],
                'receivedPackets': host_status['received_packets'],
                'receivedPacketsPercentageChange': host_status['received_packets_percentage_change'],
                'receivedBytes': host_status['received_bytes'],
                'receivedBytesPercentageChange': host_status['received_bytes_percentage_change'],
                'transmittedPackets': host_status['transmitted_packets'],
                'transmittedPacketsPercentageChange': host_status['transmitted_packets_percentage_change'],
                'transmittedBytes': host_status['transmitted_bytes'],
                'transmittedBytesPercentageChange': host_status['transmitted_bytes_percentage_change']  
                             }
        data = {
            "id": self.status["id"],
            "status" : self.status["status"],
            'packets': self.packets,
            'bytes': self.bytes,
            'packetsPercentageChange': self.packets_percentage_change,
            'bytesPercentageChange': self.bytes_percentage_change,
            'hostStatusesStructured': ht
        }
        
        return _serialize_complex_types(data) 
    
    def subtract(self, other_state: 'InstantState') -> 'InstantState':
        """
        Subtracts another InstantState from this one, returning a new InstantState
        representing the difference.

        Args:
            other_state (InstantState): The state to subtract from this one. 
        Returns:
            InstantState: A new InstantState representing the difference.
        """
        # Create a new InstantState instance without invoking __init__
        # (avoids expecting host objects with a .name attribute).
        new_state = object.__new__(InstantState)

        # Copy and compute numeric differences
        new_state.total_packets = self.total_packets - getattr(other_state, "total_packets", 0)
        new_state.total_bytes = self.total_bytes - getattr(other_state, "total_bytes", 0)
        new_state.packets = self.packets - getattr(other_state, "packets", 0)
        new_state.bytes = self.bytes - getattr(other_state, "bytes", 0)
        new_state.packets_percentage_change = self.packets_percentage_change - getattr(other_state, "packets_percentage_change", 0)
        new_state.bytes_percentage_change = self.bytes_percentage_change - getattr(other_state, "bytes_percentage_change", 0)

        # Compute state vector difference if available
        try:
            new_state._state = self._state - other_state._state
        except Exception:
            try:
                new_state._state = np.array([
                    new_state.packets,
                    new_state.packets_percentage_change,
                    new_state.bytes,
                    new_state.bytes_percentage_change,
                ], dtype=np.float32)
            except Exception:
                new_state._state = np.array([0, 0, 0, 0], dtype=np.float32)

        # Per-host differences
        new_state.host_states = {}
        for host_id, hs in self.host_states.items():
            other_hs = other_state.host_states.get(host_id) if hasattr(other_state, "host_states") else None
            if other_hs is not None:
                new_state.host_states[host_id] = hs - other_hs
            else:
                new_state.host_states[host_id] = hs.copy()

        # Totals per host
        new_state.host_states_total = {}
        for host_id, total in self.host_states_total.items():
            other_total = other_state.host_states_total.get(host_id) if hasattr(other_state, "host_states_total") else None
            if other_total is not None:
                new_state.host_states_total[host_id] = total - other_total
            else:
                new_state.host_states_total[host_id] = total.copy()

        # Shallow copy statuses and metadata
        new_state.host_statuses = {k: v for k, v in self.host_statuses.items()}
        new_state.status = dict(self.status) if isinstance(self.status, dict) else self.status
        new_state.consecutive_corrects = 0

        return new_state

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the InstantState instance into a JSON serializable dictionary.
        Note: agent_statuses and coordinator_status are already dictionaries 
        thanks to update_statuses, so they can be included directly.
        """
                # Build the initial dictionary
        data = {
            'totalPackets': self.total_packets,
            'totalBytes': self.total_bytes,
            'packets': self.packets,
            'bytes': self.bytes,
            'packetsPercentageChange': self.packets_percentage_change,
            'bytesPercentageChange': self.bytes_percentage_change,
            # The "raw" state fields are the most likely to contain ndarrays
            'hostStatesRaw': self.host_states,
            # These are usually already clean dictionaries, but we'll run the check anyway for safety
            'hostStatusesStructured': self.host_statuses,
            # The main status field is a simple string
            'status': self.status 
        }
        
        # Apply the recursive serialization helper to the entire data structure
        return _serialize_complex_types(data)