from typing import Any, Dict
from reinforcement_learning.marl.constants import COORDINATOR, HOST_STATUS_ID_MAPPING, COORDINATOR_STATUS_ID_MAPPING
from reinforcement_learning.instant_state import InstantState as BaseInstantState, _serialize_complex_types
import numpy as np
from utility.constants import NORMAL 
import inspect

class InstantState(BaseInstantState):
    """
    A data class to hold the state of all agents and the coordinator
    at a single time step, including any messages passed.
    """
    def __init__(self, hosts, coordinator_state, messages: dict):
        """
        Args:
            agent_states (dict): A dictionary mapping agent IDs to their local state.
            coordinator_state: The global state for the coordinator agent.
            messages (dict): A dictionary mapping agent IDs to a list of messages they received.
        """
        super().__init__(hosts)
        self.coordinator_state = coordinator_state
        self.coordinator_status = NORMAL #default status
        self.messages = messages #{'h1': 0, 'h2': 2, ...}
        self.links_status = {} #{'h1': 1, 'h2': 0, ...} 1 link on, 0 link off

    def update_statuses(self, statuses: dict):
        """
        Updates the internal states. This method can be expanded to include
        any logic needed to update the states based on new observations.
        """
              
        for host,status in statuses.items():
            if host == COORDINATOR:
                # Retrieve the ID from the mapping, defaulting to -1 (idle) if the status string is unexpected
                status_id = COORDINATOR_STATUS_ID_MAPPING.get(status, -1)
                self.coordinator_status = {
                    "id" : status_id,
                    "status" : status,
                    "packets" : self.coordinator_state[0], #self.current_packets,
                    "packets_percentage_change" : self.coordinator_state[1], #self.current_packets_percentage_change,
                    "bytes" : self.coordinator_state[2], #self.current_bytes,
                    "bytes_percentage_change" : self.coordinator_state[3], #self.current_bytes_percentage_change
                    #"message": self.coordinator_state[4]
                }
                self.status = {
                    "id" : status_id,
                    "status" : status,
                    "packets" : self.packets,
                    "bytes" : self.bytes, 
                    "packets_percentage_change" : self.packets_percentage_change, 
                    "bytes_percentage_change" : self.bytes_percentage_change, 
                }
                    
            else:
                if host in self.host_states:
                    # Retrieve the ID from the mapping, defaulting to -1 (idle) if the status string is unexpected
                    status_id = HOST_STATUS_ID_MAPPING.get(status, -1)
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
                        "transmitted_bytes_percentage_change": host_state[7],
                        #"message": host_state[8] 
                    }
                
                   
                else:
                    print(f"Warning: Host {host} not found in agent states.")
    
    def get_state(self, agent_name: str = ""):
        """
        Retrieves the current global state of the network.
        """
        #TODO message must be customized by agent variant
        
        return [self.packets, self.packets_percentage_change, self.bytes, self.bytes_percentage_change, 0]

    def get_coordinator_state(self):
        """
        Retrieves the global state for the coordinator.

        Returns:
            The coordinator's global state.
        """
        return self.coordinator_state

    def get_messages(self, agent_name: str):
        """
        Retrieves the list of messages for a specific agent.

        Args:
            agent_name (str): The name of the agent.

        Returns:
            list: A list of messages.
        """
        return self.messages[agent_name] if agent_name in self.messages else {}
    
    def set_message(self, agent_name: str, host_name: str, message: int):
        """
        Sets the message for a specific agent from a specific host.

        Args:
            agent_name (str): The name of the agent.
            host_name (str): The name of the host sending the message.
            message (int): The message to be set.
        """
        if agent_name in self.messages:
            self.messages[agent_name][host_name] = message
        else:
            self.messages[agent_name] = {host_name: message}    
    
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
            'coordinatorStateRaw': self.coordinator_state,
            'messagesRaw': self.messages,
            
            # These are usually already clean dictionaries, but we'll run the check anyway for safety
            'hostStatusesStructured': self.host_statuses,
            'coordinatorStatusStructured': self.coordinator_status,
            
            # The main status field is a simple string
            'status': self.status 
        }
        
        # Apply the recursive serialization helper to the entire data structure
        return _serialize_complex_types(data)