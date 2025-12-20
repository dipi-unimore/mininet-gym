from typing import Any, Dict
#from reinforcement_learning.classification.constants import AGENT_STATUS_ID_MAPPING
from reinforcement_learning.instant_state import InstantState as BaseInstantState, _serialize_complex_types
import numpy as np
from utility.constants import NORMAL 
import inspect

class InstantState(BaseInstantState):
    """
    A data class to hold the state of all agents and the coordinator
    at a single time step, including any messages passed.
    """
    def __init__(self, hosts):
        """
        Args:
            agent_states (dict): A dictionary mapping agent IDs to their local state.
            coordinator_state: The global state for the coordinator agent.
            messages (dict): A dictionary mapping agent IDs to a list of messages they received.
        """
        super().__init__(hosts)


    def update_statuses(self, status: str, mapping: dict, host_statuses: dict):
        """
        Updates the internal states. This method can be expanded to include
        any logic needed to update the states based on new observations.
        """
        status_id = mapping.get(status, -1)
        self.status = {
            "id" : status_id,
            "status" : status,
            "packets" : self.packets, #self.current_packets,
            "bytes" : self.bytes, #self.current_bytes,
            "packets_percentage_change" : self.packets_percentage_change, #self.current_packets_percentage_change,
            "bytes_percentage_change" : self.bytes_percentage_change, #self.current_bytes_percentage_change
        }
        
        for host,status in host_statuses.items():     
            if host in self.host_states:
                # Retrieve the ID from the mapping, defaulting to -1 (idle) if the status string is unexpected
                status_id = mapping.get(status, -1)
                host_state = self.host_states[host]
                self.host_statuses[host] = {
                    "id" : status_id,
                    "status" : status,
                    "received_packets": host_state[0], 
                    "received_packets_percentage_change": host_state[1], 
                    "received_bytes": host_state[2], 
                    "received_bytes_percentage_change": host_state[3],
                    "transmitted_packets": host_state[4], 
                    "transmitted_packets_percentage_change": host_state[5], 
                    "transmitted_bytes": host_state[6],
                    "transmitted_bytes_percentage_change": host_state[7]
                }
    
