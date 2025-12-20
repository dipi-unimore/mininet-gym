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
        # self.status = {"id" : status["id"], 
        #                "status" : status["status"],  
        #                "packets" :status["packets"], 
        #                "bytes" :status["bytes"],
        #                "packets_percentage_change" :status["packetsPercentageChange"], 
        #                "bytes_percentage_change" :status["bytesPercentageChange"] 
        #                }
        self.total_packets += status["packets"]
        self.total_bytes += status["bytes"]
        self.packets = status["packets"]
        self.bytes = status["bytes"]
        self.packets_percentage_change = status["packetsPercentageChange"]
        self.bytes_percentage_change = status["bytesPercentageChange"]
        #self.host_statuses = status["hostStatusesStructured"]  
        for host_id, host_status in status["hostStatusesStructured"].items():
            if host_id in self.host_states_total:
                self.host_states_total[host_id][0] += host_status["receivedPackets"]
                self.host_states_total[host_id][1] += host_status["receivedBytes"]
                self.host_states_total[host_id][2] += host_status["transmittedPackets"]
                self.host_states_total[host_id][3] += host_status["transmittedBytes"]
                
            if host_id in self.host_states:
                self.host_states[host_id][0] = host_status["receivedPackets"]
                self.host_states[host_id][1] = host_status["receivedPacketsPercentageChange"]
                self.host_states[host_id][2] = host_status["receivedBytes"]
                self.host_states[host_id][3] = host_status["receivedBytesPercentageChange"]
                self.host_states[host_id][4] = host_status["transmittedPackets"]
                self.host_states[host_id][5] = host_status["transmittedPacketsPercentageChange"]
                self.host_states[host_id][6] = host_status["transmittedBytes"]
                self.host_states[host_id][7] = host_status["transmittedBytesPercentageChange"]   
           
   
    
    def get_state(self, agent_name: str = ""):
        """
        Retrieves the current global state of the network.
        """
        return [self.packets, self.packets_percentage_change, self.bytes, self.bytes_percentage_change]
    
            
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
        new_host_states = {}
        for host_id in self.host_states:
            if host_id in other_state.host_states:
                new_host_states[host_id] = self.host_states[host_id] - other_state.host_states[host_id]
            else:
                new_host_states[host_id] = self.host_states[host_id]
        
        new_state = InstantState(new_host_states)
        new_state.total_packets = self.total_packets
        new_state.total_bytes = self.total_bytes
        new_state.packets = self.packets - other_state.packets
        new_state.bytes = self.bytes - other_state.bytes
        new_state.packets_percentage_change = self.packets_percentage_change - other_state.packets_percentage_change
        new_state.bytes_percentage_change = self.bytes_percentage_change - other_state.bytes_percentage_change

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