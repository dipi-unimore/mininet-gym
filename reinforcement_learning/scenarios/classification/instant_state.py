from typing import Any, Dict
#from .constants import AGENT_STATUS_ID_MAPPING
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
        self.received_packets  = 0
        self.received_bytes  = 0
        self.transmitted_packets  = 0
        self.transmitted_bytes  = 0


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
        self.received_packets  = 0
        self.received_bytes  = 0
        self.transmitted_packets  = 0
        self.transmitted_bytes  = 0        
        for host,status in host_statuses.items():     
            if host in self.host_states:
                # Retrieve the ID from the mapping, defaulting to -1 (idle) if the status string is unexpected
                status_id = mapping.get(status, -1)
                host_state = self.host_states[host]
                self.received_packets += int(host_state[0])
                self.received_bytes += int(host_state[2])
                self.transmitted_packets += int(host_state[4])
                self.transmitted_bytes += int(host_state[6])               
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
                
    def get_state(self):
            """
            Retrieves the current global state of the network.
            """
            return [self.received_packets, self.transmitted_packets, self.received_bytes,  self.transmitted_bytes]
                    
    def set_state(self, status: dict):
            self.received_packets = status["receivedPackets"]
            self.received_bytes = status["receivedBytes"]
            self.transmitted_packets = status["transmittedPackets"]
            self.transmitted_bytes = status["transmittedBytes"]
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
 
                            
        
