
from typing import List
import numpy as np
from .constants import AGENT_STATUS_ID_MAPPING, HOST_STATUS_ID_MAPPING
from reinforcement_learning.instant_state import InstantState as BaseInstantState, _serialize_complex_types
from utility.constants import ATTACK, NORMAL, HostStatus

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
    
    
    @property
    def state(self):
        _state = []
        for host_state in self.host_states.values():
            for i in range(len(host_state)):
                _state.append(host_state[i].item())
        return _state

    def update_statuses(self,  host_statuses: dict):
        """
        Updates the internal states. This method can be expanded to include
        any logic needed to update the states based on new observations.
        """        
        status_hosts = list(host_statuses.values())
        status_ids = [HOST_STATUS_ID_MAPPING.get(s, -1) for s in status_hosts]
        self.status = {
            "id" : status_ids,
            "status" : status_hosts,
            "packets" : self.packets, #self.current_packets,
            "bytes" : self.bytes, #self.current_bytes,
            "packets_percentage_change" : self.packets_percentage_change, #self.current_packets_percentage_change,
            "bytes_percentage_change" : self.bytes_percentage_change, #self.current_bytes_percentage_change
        }
        
        for host,status in host_statuses.items():     
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
                    "transmitted_bytes_percentage_change": host_state[7]
                }
                
    
    def set_host_statuses(self):
        #This is useful for attack env, for classification are always as default,NORMAL
        for host_name, id in self.host_states.items():
            if id==0:
                self.host_statuses[host_name] = HostStatus.NORMAL
            elif id==1:
                self.host_statuses[host_name] = HostStatus.UNDER_ATTACK  
            elif id==2:
                self.host_statuses[host_name] = HostStatus.ATTACKING

    # def get_state(self):
    #     """
    #     Retrieves the current global state of the network.
    #     """
    #     return [self.packets, self.packets_percentage_change, self.bytes, self.bytes_percentage_change]
                
    
