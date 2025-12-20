"""Custom Network Environment for Reinforcement Learning using Gymnasium."""
import copy
import inspect
import traceback
from utility import constants
from utility.network_configurator import create_host, create_network, format_bytes, stop
from utility.network_flows import get_data_flow
from utility.my_log import notify_client, set_log_level, information, debug, error
from utility.params import Params
from utility.constants import LONG_ATTACK, NORMAL, SHORT_ATTACK, HostStatus, GYM_TYPE, ATTACKS, SystemLevels
from colorama import Fore
import json as jsonlib, time, gymnasium as gym, numpy as np
from gymnasium import spaces
from abc import ABC, abstractmethod


class NetworkEnv(gym.Env, ABC):     
    """Custom Environment that follows gym interface."""
    metadata = {"render_modes": ["human"], "render_fps": 30}    
           
    def __init__(self, params, server_user = 'server_user', existing_net=None):
        super(NetworkEnv, self).__init__()
                
        self.data_traffic_file = params.data_traffic_file
        self.set_gym_type(params.gym_type)
        
        # Network creation
        self.net = existing_net
        
        # New conditional logic
        if self.net is None:  
            from_dataset = [constants.GYM_TYPE[constants.CLASSIFICATION_FROM_DATASET],
                            constants.GYM_TYPE[constants.ATTACKS_FROM_DATASET],
                            constants.GYM_TYPE[constants.MARL_ATTACKS_FROM_DATASET]]      
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
        for i in range(params.num_iot):
            host = create_host(name=f'iot{i+1}')
            net.hosts.append(host)
        return net
     
    def initialize_storage(self):
        self.data_traffic = {key: {'p_r': [], 'p_t': [], 'b_r': [], 'b_t': []} for key in  self.net.traffic_types}
        self.statuses = [] 
        
    def stop(self):
        """Stop the environment and clean up resources."""
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
        self.real_state = self.state = self.global_state.get_state()
        if is_real_state: #only for prediction
            return self.real_state
        if is_discretized_state:
            return self.get_discretized_state(self.state)
        return get_normalize_state(self.state, self.low, self.high) 
                  
    def reset(self, seed = None, options=None, is_discretized_state = False, is_real_state = False):
        """Reset environment at the beginning of an episode."""
        super().reset(seed=seed)

        #update status, in some gym_type generate traffic too
        self.update_state()
        self.state = self.get_current_state(is_discretized_state, is_real_state) 
        #self.status = self.global_state.status #initial status
        
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
            while hasattr(self,'pause_event') and self.pause_event.is_set():
                time.sleep(1)
                continue
            time.sleep(self.params.wait_after_read)
            self.update_state() 
        debug("Update state thread finished")  

    def read_from_network(self) -> bool: #read from switch
        """to retrieve new observation
        """
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

        if total_packets == 0 and total_bytes == 0:
            debug("No traffic detected in the network")

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

        for host in self.net.hosts:
            total_transmitted_packets = 0
            total_transmitted_bytes = 0
            total_received_packets = 0
            total_received_bytes = 0

            for flow in flows['flows']:
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
            host_state = np.array([
                max(0, received_packets),             # 0: received_packets (INC)
                received_packets_percentage_change,   # 1: received_packets_percentage_change
                max(0, received_bytes),               # 2: received_bytes (INC)
                received_bytes_percentage_change,     # 3: received_bytes_percentage_change
                max(0, transmitted_packets),          # 4: transmitted_packets (INC)
                transmitted_packets_percentage_change,# 5: transmitted_packets_percentage_change
                max(0, transmitted_bytes),            # 6: transmitted_bytes (INC)
                transmitted_bytes_percentage_change,  # 7: transmitted_bytes_percentage_change
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
     
    def show_network_status(self):       
        agent_name = get_agent_name() #impossible to get agent name directly here    
        state = self.global_state.get_state(agent_name)
        t_q_v_p = self.threshold_var_packets * self.threshold_var_packets #square threshold_var_packets
        t_q_v_b = self.threshold_var_bytes * self.threshold_var_bytes #square threshold_var_bytes
        
        color1 = Fore.BLUE if state[1] * state[1] < t_q_v_p else Fore.WHITE
        color3 = Fore.BLUE if state[3] * state[3] < t_q_v_b else Fore.WHITE
        information(
            Fore.BLUE + f"Packet {int(state[0])} " +
            color1 + f"{int(state[1])}%" +
            Fore.BLUE + f" - {format_bytes(int(state[2]))}B" +
            color3 + f" {int(state[3])}%" +
            #Impossible to show the message for the right agent here
            #Fore.BLUE + f" - Message {int(state[4])}\n" if len(state) == 5 else "" +
            Fore.WHITE
        )
        #status by host
        for host in self.global_state.host_states.keys():
            host_state = self.global_state.get_host_state(host)
            host_status = self.global_state.get_host_status(host)
            if host_status is not None:
                if host_status['status'] in ("under_attack", "attacking", "attacking/underattack"):
                    debug(Fore.WHITE + 
                                f"{host} " + Fore.RED + f"{host_status['status']} "+
                                f"- RP {int(host_state[0])} {int(host_state[1])}% " +
                                f"- RB {int(host_state[2])} {int(host_state[3])}% " +
                                f"- TP {int(host_state[4])} {int(host_state[5])}% " +
                                f"- TB {int(host_state[6])} {int(host_state[7])}%\n" + Fore.WHITE)
                else:
                    debug(Fore.WHITE + 
                                f"{host} "+Fore.GREEN +
                                f"- RP {int(host_state[0])} {int(host_state[1])}% " +
                                f"- RB {int(host_state[2])} {int(host_state[3])}% " +
                                f"- TP {int(host_state[4])} {int(host_state[5])}% " +
                                f"- TB {int(host_state[6])} {int(host_state[7])}%\n" + Fore.WHITE)
                 
    
    def eval_percentage_change(self, now, before, threshold_percentage_change=None, threshold_percentage_change_multiple = 10):
        """
        Evaluate the percentage change from 'before' to 'now'.
        Args:
            now (float): The current value.
            before (float): The previous value.
            threshold_percentage_change (float, optional): If provided, the percentage change
                will be truncated to this threshold.
        Returns:
            float: The calculated percentage change.
        """
        if before == 0:
            if now == 0:
                percentage_change = 0.0
            elif now > 0:
                # From zero to a positive number: infinite change. 
                # We can choose to return a conventional maximum value or signal infinity.
                # We use float('inf') to represent positive infinity
                percentage_change = float('inf')
            else: # now < 0
                # From zero to a negative number: negative infinite change
                percentage_change = float('-inf')
        else:
            # Standard formula for percentage change
            percentage_change = ((now - before) / before) * 100
        
        # Optional application of the threshold (value truncation)
        if threshold_percentage_change is not None and abs(percentage_change) > threshold_percentage_change_multiple * threshold_percentage_change:
            # Truncate the value to the maximum/minimum of the specified threshold
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
        for host_name, host_task in self.host_tasks.items():
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
        for host_name, task_info in self.host_tasks.items():
            # Check if the task is still active
            if task_info["end_time"] is None or task_info["end_time"] > current_time:
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
           error(f"Set correctly 'gym_type' on config.yaml") 
           raise Exception("Set correctly 'gym_type' on config.yaml")
       
        if self.gym_type == constants.GYM_TYPE[constants.CLASSIFICATION]: 
            time.sleep(2) #wait 2 seconds for controller starts properly, before to read first time
        
            #read initial traffic
            self.sync_time = 1 #self.synchronize_controller()
            self.read_time = self.sync_time * 0.6
    
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

  
  
def get_normalize_state(state, low_to_normalize, high_to_normalize):
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


if __name__ == '__main__':
    set_log_level('info')
    env_params = {
        'net_params' : { 
            'num_hosts':3,
            'num_switches':1,
            'num_iot':0,
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
