"""Custom Network Environment for Reinforcement Learning using Gymnasium."""
import copy
import inspect
import traceback
from utility import constants
from utility.network_configurator import create_host, create_network, format_bytes, stop
from utility.network_flows import check_and_reset_if_needed, get_data_flow
from utility.my_log import notify_client, set_log_level, information, debug, error
from utility.params import Params
from utility.constants import LONG_ATTACK, NORMAL, SHORT_ATTACK, HostStatus, GYM_TYPE, ATTACKS, SystemLevels
from colorama import Fore
import json as jsonlib, time, threading, gymnasium as gym, numpy as np
from gymnasium import spaces
from abc import ABC, abstractmethod


class NetworkEnv(gym.Env, ABC):     
    """Custom Environment that follows gym interface."""
    metadata = {"render_modes": ["human"], "render_fps": 30}    
           
    def __init__(self, params, server_user = 'server_user', existing_net=None):
        super(NetworkEnv, self).__init__()
        self.flow_reset_check_interval = 100  # Check every 100 reads
        self.read_count = 0
        self.reading_history = []
                
        self.data_traffic_file = params.data_traffic_file
        self.set_gym_type(params.gym_type)
        self.stop_update_event = threading.Event()
        
        # Network creation
        self.net = existing_net
        
        # New conditional logic
        if self.net is None:
            from_dataset = [constants.GYM_TYPE[constants.CLASSIFICATION_FROM_DATASET],
                            constants.GYM_TYPE[constants.ATTACKS_FROM_DATASET],
                            constants.GYM_TYPE[constants.MARL_ATTACKS_FROM_DATASET],
                            constants.GYM_TYPE[constants.MARL_PZ_FROM_DATASET]]
            self.net = create_network(params.net_params, server_user) if self.gym_type not in from_dataset else self.create_empty(params.net_params)
        self.n_hosts = len(self.net.hosts)
        self.generated_traffic_type = -1
        
        # Define action and observation space, gym.spaces objects
        self.actions_number = params.actions_number
        self.action_space = spaces.Discrete(self.actions_number) #number of actions
          

        # Simulation parameters
        self.max_steps = params.max_steps  # Define the maximum number of steps
        self.max_reward = 1
        self.steps_min_percentage = params.steps_min_percentage
        self.min_accuracy = params.accuracy_min        
        self.early_exit = False
              
        # Status
        self.is_state_normalized = False #and not discretized
        self.initial_state = [np.zeros(self.actions_number, dtype=np.float32)] #initial state with 0 packets and bytes
        self.prev_state = self.state = self.initial_state #in self.state always continuos values, no discretized
        self.host_tasks = None         
  
        # Storages
        self.initialize_storage()          

    def create_empty(self, params):
        net = type('', (), {})()
        net.traffic_types = params.traffic_types
        net.hosts = []
        for i in range(params.num_hosts):
            host = create_host(name=f'h{i+1}')
            net.hosts.append(host)
        for i in range(params.num_iots):
            host = create_host(name=f'iot{i+1}')
            net.hosts.append(host)
        return net
     
    def initialize_storage(self):
        self.data_traffic = {key: {'p_r': [], 'p_t': [], 'b_r': [], 'b_t': []} for key in  self.net.traffic_types}
        self.statuses = [] 
        
    def stop(self):
        """Stop the environment and clean up resources."""
        if getattr(self, '_stop_called', False):
            return
        self._stop_called = True
        try:
            if hasattr(self, 'stop_event'):
                self.stop_event.set()   
            if hasattr(self, 'update_state_thread_instance'):
                self.update_state_thread_instance.join()      #wait for all thread stopped   
            if hasattr(self, 'host_threads'):   
                for host_thread in self.host_threads:
                    host_thread.join()
            stop(self.net)
        except Exception as e:
            error(Fore.RED+f"Error stopping environment: {e}\n{traceback.format_exc()}\n"+Fore.WHITE)   
   
     
    def get_current_state(self, is_discretized_state = False, is_real_state= False):
        # Compatibility: some InstantState variants expose `state`, others only `get_state()`.
        current_state = None
        if hasattr(self.global_state, 'state'):
            current_state = self.global_state.state
        elif hasattr(self.global_state, 'get_state') and callable(self.global_state.get_state):
            current_state = self.global_state.get_state()

        if current_state is None:
            raise AttributeError("global_state has neither 'state' nor callable 'get_state()'")

        self.real_state = self.state = np.array(current_state, dtype=np.float32)
        if is_real_state: #only for prediction
            return self.real_state
        if is_discretized_state:
            return self.get_discretized_state(self.state)
        return get_normalized_state(self.state, self.low, self.high) 
                  
    def reset(self, seed = None, options={"is_discretized_state": False, "is_real_state": False}):
        """Reset environment at the beginning of an episode."""
        super().reset(seed=seed)
        if "is_discretized_state" not in options:
            options["is_discretized_state"] = False
        if "is_real_state" not in options:
            options["is_real_state"] = False
        #update status, in some gym_type generate traffic too
        self.update_state()
        self.state = self.get_current_state(options["is_discretized_state"], options["is_real_state"]) 
        
        return  self.state, {}   
        
    @abstractmethod
    def step(self):
        """
        Placeholder for the `step` method. Must be implemented by derived classes.
        """
        pass   
       
    @abstractmethod
    def check_if_done_or_truncated(self):
        """
        Placeholder for the `check_if_done_or_truncated` method. Must be implemented by derived classes.
        """
        pass    

    @abstractmethod
    def execute_action(self, action):
        """
        Placeholder for the `execute_action` method. Must be implemented by derived classes.
        """
        pass    
     
    @abstractmethod
    def calculate_reward(self, action):
        """
        Placeholder for the `calculate_reward` method. Must be implemented by derived classes.
        """
        pass 
    
    def update_state_thread(self):
        """
        update environment state every N=params.wait_after_read seconds, only gym_type=6, reading from switch
        """
        time.sleep(0.5) 
        while (not hasattr(self,'stop_event') or not self.stop_event.is_set()) and (not hasattr(self,'stop_update_status_event') or not self.stop_update_status_event.is_set()):   
            if self.stop_update_event and self.stop_update_event.is_set():
                break
            while hasattr(self,'pause_event') and self.pause_event.is_set():
                time.sleep(0.01)
                continue
            time.sleep(self.params.wait_after_read+self.params.wait_after_read*0.05*self.n_hosts)
            self.update_state() 
        debug("Update state thread finished")  

    def read_from_network(self) -> bool: #read from switch
        """to retrieve new observation
        """
        
        self.read_count += 1
        
        # Periodic check for counter overflow (every 100 reads)
        if self.read_count % self.flow_reset_check_interval == 0:
            if check_and_reset_if_needed(
                self.net, 
                bridge_name="s1",
                threshold_packets=3_500_000_000,  # Reset at 3.5B (before 4.2B limit)
                threshold_bytes=900_000_000_000   # Reset at 900GB (before 1TB)
            ):
                # Reset was performed - adjust previous state to avoid false differences
                information("Resetting previous state after flow counter reset")
                self.reading_history.append({
                    "total_packets": self.global_prev_state.total_packets,
                    "total_bytes": self.global_prev_state.total_bytes,
                    "host_states_total": copy.deepcopy(self.global_prev_state.host_states_total)
                })
                self.global_prev_state.total_packets = 0
                self.global_prev_state.total_bytes = 0
                for host_name in self.global_prev_state.host_states_total.keys():
                    self.global_prev_state.host_states_total[host_name] = np.zeros(4, dtype=np.float32)

        
        start_get_time = time.time() 
        #save previous state
        self.global_prev_state = copy.copy(self.global_state)
        
        while True:
            flows = get_data_flow(self.net)
            if flows:
                break
            debug("No flows data retrieved, retrying...")
            
        total_packets = flows['packets']['transmitted']
        total_bytes = flows['bytes']['transmitted']

        if self.gym_type is not GYM_TYPE[constants.CLASSIFICATION]:
            if total_packets == 0 and total_bytes == 0:
                debug("No traffic detected in the network")
                return False

            if total_packets == self.global_prev_state.total_packets and total_bytes == self.global_prev_state.total_bytes:
                debug("No NEW traffic detected in the network")
                end_get_time = time.time()
                get_time = end_get_time - start_get_time
                debug(f"Time {get_time}\n")   
                return False
            
        self.global_state.total_packets = total_packets
        self.global_state.total_bytes = total_bytes
        
        # Check and handle global counter restart
        if self.global_state.total_packets < self.global_prev_state.total_packets:
            # Only reset global packets if the total counter went backwards
            self.global_prev_state.total_packets = 0
            debug("Global packet counter has restarted.")
            
        
        if self.global_state.total_bytes < self.global_prev_state.total_bytes:
            self.global_prev_state.total_bytes = 0
            debug("Global byte counter has restarted.")
        
        self.global_state.packets = self.global_state.total_packets - self.global_prev_state.total_packets
        self.global_state.bytes = self.global_state.total_bytes - self.global_prev_state.total_bytes
        # Ensure calculated difference is non-negative (just in case)
        self.global_state.packets = max(0, self.global_state.packets)
        self.global_state.bytes = max(0, self.global_state.bytes)
        
        self.global_state.packets_percentage_change  = self.eval_percentage_change(self.global_state.packets, self.global_prev_state.packets, self.threshold_var_packets)
        self.global_state.bytes_percentage_change = self.eval_percentage_change(self.global_state.bytes, self.global_prev_state.bytes, self.threshold_var_bytes)
        
        self.global_state.host_states = {}
        self.global_state.host_states_total = {} # Resetting or initializing the total state for current reading

        # for host in self.net.hosts:
        #     total_transmitted_packets = 0
        #     total_transmitted_bytes = 0
        #     total_received_packets = 0
        #     total_received_bytes = 0

        #     for flow in flows['flows']:
        #         if flow.get('src_name') == host.name:
        #             total_transmitted_packets += flow.get('packets', 0)
        #             total_transmitted_bytes += flow.get('bytes', 0)

        #         elif flow.get('dst_name') == host.name:
        #             total_received_packets += flow.get('packets', 0)
        #             total_received_bytes += flow.get('bytes', 0)
        
        for host in self.net.hosts:
            total_transmitted_packets = 0
            total_transmitted_bytes = 0
            total_received_packets = 0
            total_received_bytes = 0

            for flow in flows['flows']:
                # Check if this flow involves the current host
                host_in_flow = (flow.get('src_name') == host.name or 
                                flow.get('dst_name') == host.name)
                
                if not host_in_flow:
                    continue
                
                # Get direction field (only present in port statistics)
                direction = flow.get('direction', None)
                
                if direction == 'TX':
                    # TX from switch perspective = packets going TO host (host receives)
                    total_received_packets += flow.get('packets', 0)
                    total_received_bytes += flow.get('bytes', 0)
                elif direction == 'RX':
                    # RX from switch perspective = packets coming FROM host (host transmits)
                    total_transmitted_packets += flow.get('packets', 0)
                    total_transmitted_bytes += flow.get('bytes', 0)
                else:
                    # Legacy OpenFlow table logic (when direction is not present)
                    if flow.get('src_name') == host.name:
                        total_transmitted_packets += flow.get('packets', 0)
                        total_transmitted_bytes += flow.get('bytes', 0)
                    elif flow.get('dst_name') == host.name:
                        total_received_packets += flow.get('packets', 0)
                        total_received_bytes += flow.get('bytes', 0)
                    
            # 2. Retrieve previous totals and increments (for percentage change calculation)            
            # Indices for agent_states_total (assumed: [RX_PKT_TOTAL, RX_BYTE_TOTAL, TX_PKT_TOTAL, TX_BYTE_TOTAL])
            IDX_RX_PKT_TOTAL = 0
            IDX_RX_BYTE_TOTAL = 1
            IDX_TX_PKT_TOTAL = 2
            IDX_TX_BYTE_TOTAL = 3

            # Indices for agent_states (assumed to hold the previous INCREMENTS, not totals)
            # [RX_PKT_INC, RX_PKT_PCT_CHG, RX_BYTE_INC, RX_BYTE_PCT_CHG, 0, TX_PKT_INC, TX_PKT_PCT_CHG, TX_BYTE_INC, TX_BYTE_PCT_CHG]
            IDX_PREV_RX_PKT_INC = 0
            IDX_PREV_RX_BYTE_INC = 2
            IDX_PREV_TX_PKT_INC = 5
            IDX_PREV_TX_BYTE_INC = 7 
            
             # --- RECEIVED PACKETS ---
            HOST_TOTAL_STATE_SIZE = 4
            HOST_STATE_SIZE = 8
            total_prev_received_packets = self.global_prev_state.host_states_total.get(host.name, np.zeros(HOST_TOTAL_STATE_SIZE))[IDX_RX_PKT_TOTAL]
            prev_received_packets = self.global_prev_state.host_states.get(host.name, np.zeros(HOST_STATE_SIZE))[IDX_PREV_RX_PKT_INC]
            
            # Check for counter reset on received packets
            if total_received_packets < total_prev_received_packets:
                debug(f"Host {host.name}: Received packet counter reset detected.")
                total_prev_received_packets = 0 # Reset the previous total to calculate the full current total as the difference
            
            received_packets = total_received_packets - total_prev_received_packets
            received_packets_percentage_change = self.eval_percentage_change(received_packets, prev_received_packets, self.threshold_var_packets)

            # --- RECEIVED BYTES ---
            total_prev_received_bytes = self.global_prev_state.host_states_total.get(host.name, np.zeros(HOST_TOTAL_STATE_SIZE))[IDX_RX_BYTE_TOTAL]
            prev_received_bytes = self.global_prev_state.host_states.get(host.name, np.zeros(HOST_STATE_SIZE))[IDX_PREV_RX_BYTE_INC]
            
            # Check for counter reset on received bytes
            if total_received_bytes < total_prev_received_bytes:
                debug(f"Host {host.name}: Received byte counter reset detected.")
                total_prev_received_bytes = 0 
            
            received_bytes = total_received_bytes - total_prev_received_bytes
            received_bytes_percentage_change = self.eval_percentage_change(received_bytes, prev_received_bytes, self.threshold_var_bytes)

            # --- TRANSMITTED PACKETS ---
            total_prev_transmitted_packets = self.global_prev_state.host_states_total.get(host.name, np.zeros(HOST_TOTAL_STATE_SIZE))[IDX_TX_PKT_TOTAL]
            prev_transmitted_packets = self.global_prev_state.host_states.get(host.name, np.zeros(HOST_STATE_SIZE))[IDX_PREV_TX_PKT_INC]
            
            # Check for counter reset on transmitted packets
            if total_transmitted_packets < total_prev_transmitted_packets:
                debug(f"Host {host.name}: Transmitted packet counter reset detected.")
                total_prev_transmitted_packets = 0
            
            transmitted_packets = total_transmitted_packets - total_prev_transmitted_packets
            transmitted_packets_percentage_change = self.eval_percentage_change(transmitted_packets, prev_transmitted_packets, self.threshold_var_packets)

            # --- TRANSMITTED BYTES ---
            total_prev_transmitted_bytes = self.global_prev_state.host_states_total.get(host.name, np.zeros(HOST_TOTAL_STATE_SIZE))[IDX_TX_BYTE_TOTAL]
            prev_transmitted_bytes = self.global_prev_state.host_states.get(host.name, np.zeros(HOST_STATE_SIZE))[IDX_PREV_TX_BYTE_INC]
            
            # Check for counter reset on transmitted bytes
            if total_transmitted_bytes < total_prev_transmitted_bytes:
                debug(f"Host {host.name}: Transmitted byte counter reset detected.")
                total_prev_transmitted_bytes = 0
                
            transmitted_bytes = total_transmitted_bytes - total_prev_transmitted_bytes
            transmitted_bytes_percentage_change = self.eval_percentage_change(transmitted_bytes, prev_transmitted_bytes, self.threshold_var_bytes)
                                       
            # 3. Assemble new state arrays
            # Zero percentage changes when no traffic flowed this step — avoids
            # showing -100% for hosts that simply had no activity this step.
            host_state = np.array([
                max(0, received_packets),
                received_packets_percentage_change   if received_packets   > 0 else 0.0,
                max(0, received_bytes),
                received_bytes_percentage_change     if received_bytes     > 0 else 0.0,
                max(0, transmitted_packets),
                transmitted_packets_percentage_change if transmitted_packets > 0 else 0.0,
                max(0, transmitted_bytes),
                transmitted_bytes_percentage_change  if transmitted_bytes  > 0 else 0.0,
            ], dtype=np.float32)
            
            host_state_total = np.array([
                total_received_packets, 
                total_received_bytes, 
                total_transmitted_packets, 
                total_transmitted_bytes
            ], dtype=np.float32)
                            
            self.global_state.host_states[host.name] = host_state
            self.global_state.host_states_total[host.name] = host_state_total

        end_get_time = time.time()
        get_time = end_get_time - start_get_time
        debug(f"Time {get_time}\n")   
        return True
     
    
    def eval_percentage_change(self, now, before, threshold_percentage_change=None, threshold_percentage_change_multiple = 10):
        """
        Evaluate the percentage change from 'before' to 'now'.
        Never returns ±inf: when before==0 and now>0, returns the clamped maximum.
        """
        if before == 0:
            if now == 0:
                return 0.0
            # new traffic from zero: return clamped max to avoid ±inf in state features
            max_val = (threshold_percentage_change_multiple * threshold_percentage_change
                       if threshold_percentage_change is not None else 100.0)
            return float(max_val) if now > 0 else float(-max_val)
        else:
            percentage_change = ((now - before) / before) * 100

        if threshold_percentage_change is not None and abs(percentage_change) > threshold_percentage_change_multiple * threshold_percentage_change:
            if percentage_change > 0:
                percentage_change = threshold_percentage_change_multiple * threshold_percentage_change
            else:
                percentage_change = -threshold_percentage_change_multiple * threshold_percentage_change

        return percentage_change


    
    def update_hosts_status(self):
        """
        Analyze the current tasks in self.host_tasks and return the status of each host
        in the network, including the coordinator, based on active attack/normal tasks.

        Returns:
            dict: {hostname: status_string} where status is "normal", "under_attack",
                "attacking", "both attacking/underattack", or "attack" (for coordinator).
        """
        if self.host_tasks is None:
            # Assuming self.host_envs contains all hosts in the network
            statuses = {host.name: "idle" for host in self.hosts}
            return statuses, False  
        host_tasks = {}
        task_items = list(self.host_tasks.items())
        for host_name, host_task in task_items:
            # Remove expired tasks
            host_tasks[host_name]={
                "taskType": host_task["task_type"], 
                "trafficType": host_task["traffic_type"],              
                "end_time": host_task["end_time"],
                "destination": host_task.get("destination", None),
                "linkStatus": self.global_state.links_status.get(host_name, 1) if hasattr(self.global_state, 'links_status') else 1  # Default to link ON (1) if not found
            }
        notify_client(level=SystemLevels.DATA, host_tasks = host_tasks )
        
        # Sets to track hosts currently involved in an attack
        attacking_hosts = set()
        under_attack_hosts = set()

        # Iterate through the tasks of all hosts
        current_time = time.time()
        for host_name, task_info in task_items:
            # Check if the task is still active
            #if task_info["end_time"] is None or task_info["end_time"] > current_time:
            task_type = task_info["task_type"]                
            # Only process if the task is an attack
            if task_type in (SHORT_ATTACK, LONG_ATTACK):
                dest_host_name = task_info.get("destination")
                
                # The source host is attacking
                attacking_hosts.add(host_name)
                
                # The destination host is under attack
                if dest_host_name:
                    under_attack_hosts.add(dest_host_name)
        
        # --- Phase 2: Convert roles into final status string for individual hosts ---
        final_statuses = {}
        is_any_host_under_attack_or_attacking = False


        for host in self.hosts:
            host_name = host.name
            is_attacking = host_name in attacking_hosts
            is_under_attack = host_name in under_attack_hosts

            if is_attacking and is_under_attack:
                status = HostStatus.WAR
                is_any_host_under_attack_or_attacking = True
            elif is_attacking:
                status = HostStatus.ATTACKING
                is_any_host_under_attack_or_attacking = True
            elif is_under_attack:
                status = HostStatus.UNDER_ATTACK
                is_any_host_under_attack_or_attacking = True
            else:
                status = NORMAL

            final_statuses[host_name] = status

        return final_statuses, is_any_host_under_attack_or_attacking   

    
    def set_gym_type(self, gym_type):
        self.gym_type = -1
        if gym_type in constants.GYM_TYPE:
            self.gym_type = constants.GYM_TYPE[gym_type]
        else:
           error(f"Set correctly 'gym_type' on config/default.yaml") 
           raise Exception("Set correctly 'gym_type' on config/default.yaml")
       
        if self.gym_type == constants.GYM_TYPE[constants.CLASSIFICATION]: 
            time.sleep(2) #wait 2 seconds for controller starts properly, before to read first time
        
            #read initial traffic
            self.sync_time = 1 #self.synchronize_controller()
            self.read_time = self.sync_time * 0.6

    def clean_network_state(self):
        """Clean residual state from previous agent training: kill traffic processes, remove drop rules."""
        debug("Cleaning network state ...")

        # Kill any lingering iperf/traffic processes on all hosts
        for host in self.net.hosts:
            try:
                host.cmd("pkill -9 iperf 2>/dev/null; pkill -9 ping 2>/dev/null; pkill -9 nmap 2>/dev/null")
            except Exception as e:
                debug(f"Could not kill processes on {host.name}: {e}")

        # Remove any leftover OVS drop rules
        if hasattr(self.net, 'blocked_hosts') and len(self.net.blocked_hosts)>0:
            try:
                from utility.network_configurator import unblock_flow_delete
                for host in self.net.hosts:
                    try:
                        unblock_flow_delete(self.net, host.name)
                    except Exception:
                        pass
            except Exception as e:
                debug(f"Could not remove drop rules: {e}")

        # Reset link status (all links up)
        if hasattr(self, 'status_links'):
            self.status_links = [True] * len(self.net.hosts)

        information("Network state cleaned.")

    def close(self):
        """Close the environment (optional)."""
        information("Environment closed.")
        
    def render(self, mode="human"):
        """Render the environment (optional)."""
        information(f"State: {self.state}, Ground Truth: {self.generated_traffic_type}")

    @abstractmethod
    def get_discretized_state(self, state):
        pass 
    
def get_agent_name(host_name=None) -> str:
    #usare inspect to get the agent that has called    
    result = find_in_stack(target_type=None, target_function="train")
    if result:
        #print(f"Found: {result.f_code.co_name}, self={result.f_locals.get('self')}")
        agent=result.f_locals.get('self')
        if host_name is not None:
            #remove string host_name from agent name
            agent_name = agent.name.replace(f"_{host_name}", "")
            return agent_name
        return agent.name
    return ""
  

def find_in_stack(target_type=None, target_function=None):
    frame = inspect.currentframe().f_back
    while frame:
        caller_self = frame.f_locals.get('self', None)

        if target_type and isinstance(caller_self, target_type):
            return frame
        if target_function and frame.f_code.co_name == target_function:
            return frame

        frame = frame.f_back  # vai indietro
    return None

  
  
def get_normalized_state(state, low_to_normalize, high_to_normalize):
    tmp_state = np.array(state, dtype=np.float32)  # ensure it's a NumPy array
    clipped_state = np.clip(tmp_state, low_to_normalize, high_to_normalize) #limit the values in an array.
    normalized = (clipped_state - low_to_normalize) / (high_to_normalize - low_to_normalize + 1e-8)
    return np.array(normalized, dtype=np.float32)
    
def get_linear_bin_index( val, low, high, n_bins):
        # Create dynamic bins based on the range of observation space
        if np.isinf(high):  # If upper bound is infinity
            # Handle this with a reasonable max value based on observation data
            bins = np.linspace(0, np.percentile(val, 95), n_bins)                
        else:
            bins = np.linspace(low, high, n_bins) #create equal ranges with n=n_bins threshold

        # Now digitize the state
        bin_index = np.digitize(val, bins) - 1  # Get the range index (from 1 to n_bins if val > high) for val. Minus one to rebase from 0 to n_bins-1
        # Clamp the bin index to valid range [0, n_bins - 1]
        # bin_index = max(0, min(bin_index, n_bins - 1))
        return bin_index
    
def get_log_bin_index( val, low, high, n_bins):
        # Create dynamic bins based on the range of observation space
        if np.isinf(high):  # If upper bound is infinity
            # Handle this with a reasonable max value based on observation data
            bins = np.linspace(0, np.percentile(val, 95), n_bins)                
        else:
            bins = np.logspace(low, np.log10(high), n_bins) #create equal ranges with n=n_bins threshold

        # Now digitize the state
        bin_index = np.digitize(val, bins) - 1  # Get the range index (from 1 to n_bins if val > high) for val. Minus one to rebase from 0 to n_bins-1
        # Clamp the bin index to valid range [0, n_bins - 1]
        # bin_index = max(0, min(bin_index, n_bins - 1))
        return bin_index  
    
def get_custom_bin_index( val, low, high, n_bins):
        # Create dynamic bins based on the range of observation space
        bin_edges = create_adaptive_bins(high*0.7, high, n_middle_bins=n_bins-3, very_low_max=3)
        # Now digitize the state
        bin_index = np.digitize(val, bin_edges) - 1  # Get the range index (from 1 to n_bins if val > high) for val. Minus one to rebase from 0 to n_bins-1
        # Clamp the bin index to valid range [0, n_bins - 1]
        # bin_index = max(0, min(bin_index, n_bins - 1))
        return bin_index
    
def create_adaptive_bins(low_threshold, high_threshold, n_middle_bins=1, very_low_max=0.001):
    """
    Create custom bins with flexible subdivision of the middle range
    
    Args:
        low_threshold: Upper bound for "under threshold" zone
        high_threshold: Lower bound for "up threshold" zone
        n_middle_bins: Number of bins to create in the "big range" (bin 1)
        very_low_max: Maximum value for "very low" bin
    
    Returns:
        bin_edges: Array of bin edges
    """
    # Fixed bins
    bin_edges = [0, very_low_max]
    
    # Subdivide the middle range (very_low_max to low_threshold)
    if n_middle_bins == 1:
        bin_edges.append(low_threshold)
    else:
        # You can choose linear or logarithmic subdivision here
        middle_edges = np.linspace(very_low_max, low_threshold, n_middle_bins + 1)[1:]
        # OR: middle_edges = np.logspace(np.log10(very_low_max), np.log10(low_threshold), n_middle_bins + 1)[1:]
        bin_edges.extend(middle_edges)
    
    # Fixed upper bins
    bin_edges.extend([high_threshold, np.inf])
    
    return np.array(bin_edges) 


if __name__ == '__main__':
    set_log_level('info')
    env_params = {
        'net_params' : { 
            'num_hosts':3,
            'num_switches':1,
            'num_iots':0,
            'controller': {
                'ip':'192.168.1.226',
                'port':6633,
                'usr':'admin',
                'pwd':'admin'
            }
        },
        'K_steps':2,
        'steps_min_percentage':0.9,
        'accuracy_min':0.9        
    } 
    env_params = jsonlib.loads(jsonlib.dumps(env_params), object_hook=Params) 
    env = NetworkEnv(env_params)
    observation, info = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action,3)
        if done:
            break
    env.close()
