from reinforcement_learning.adversarial_agent import  generate_random_traffic, print_traffic
from utility.network_configurator import create_network,  get_data_controller_flow, comunicate_no_traffic_detected, comunicate_ping_traffic_detected, comunicate_udp_traffic_detected, comunicate_tcp_traffic_detected, get_host_by_name
from utility.my_log import set_log_level, information, debug, error, notify_client
from utility.params import Params
from colorama import Fore, Back, Style
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json as jsonlib
import threading, random, time
import pandas as pd
from abc import ABC, abstractmethod
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class NetworkEnv(gym.Env, ABC):     
    """Custom Environment that follows gym interface."""
    metadata = {"render_modes": ["human"], "render_fps": 30}    
           
    def __init__(self, params = {
                    'net_params' : { 
                        'num_hosts':10,
                        'num_switches':1,
                        'num_iot':1,
                        'controller': {
                            'ip':'192.168.1.226',
                            'port':6633,
                            'usr':'admin',
                            'pwd':'admin'
                        }
                    },
                    'gym_type':'classification_without_syncronize',
                    'number_of_actions':4,
                    'K_steps':2,
                    'steps_min_percentage':0.9,
                    'accuracy_min':0.9         
                }, server_user = 'server_user'):
        super(NetworkEnv, self).__init__()
                
        self.csv_file = params.csv_file
        self.set_gym_type(params.gym_type)
        
        # Network creation
        self.net = create_network(params.net_params, server_user) if self.gym_type>0 and self.gym_type<5 else self.create_empty(params.net_params)
        self.n_hosts = len(self.net.hosts) 
        self.generated_traffic_type = -1
        #self.threshold = params.threshold
        
        # Define action and observation space, gym.spaces objects
        self.num_actions = params.number_of_actions
        self.action_space = spaces.Discrete(self.num_actions) #number of actions
          

        # Simulation parameters
        self.max_steps = params.max_steps  # Define the maximum number of steps
        self.max_reward = 1
        self.min_percentage_steps = params.steps_min_percentage
        self.min_accuracy = params.accuracy_min        
        self.early_exit = False
              
        # Status
        self.is_state_normalized = False #and not discretized
        self.initial_state = [0,0,0,0]
        self.prev_state = self.state = self.initial_state #in self.state always continuos values, no discretized
        self.status = {"id" :-1, "text" : "idle",  "variation_packet" :0, "variation_byte" :0,  "packets" :0, "bytes" :0}
        self.host_tasks = None         
  
        # Storages
        self.initialize_storage()          

    def create_empty(self, params):
        net = type('', (), {})()
        net.traffic_types = params.traffic_types
        net.hosts= range(params.num_hosts + params.num_iot)
        return net
     
    def initialize_storage(self):
        self.data_traffic = {key: {'p_r': [], 'p_t': [], 'b_r': [], 'b_t': []} for key in  self.net.traffic_types}
        self.statuses = [] 
        
    def reset(self, seed = None, options=None, is_discretized_state = False, is_real_state = False):
        """Reset environment at the beginning of an episode."""
        super().reset(seed=seed)

        #update status, in some gym_type generate traffic too
        self.update_state()
        state = self.get_current_state(is_discretized_state, is_real_state) 
        return state, {}       
     
    def get_current_state(self, is_discretized_state = False, is_real_state= False):
        self.real_state = self.state
        # if self.is_state_normalized:
        #     normalized_state = (self.state - self.low_to_normalize) / (self.high_to_normalize - self.low_to_normalize)
        #     return normalized_state
        # return self.get_discretized_state(self.state) if is_discretized_state else np.float32(self.state)
        if is_discretized_state:
            return self.get_discretized_state(self.state)
        elif is_real_state:
            return self.real_state
        else:
            return self.get_normalize_state(self.state)
            
     
    def step(self, action, is_discretized_state = False, is_real_state= False, current_step=-1, correct_predictions=0):

        # Calculate reward
        reward = self.calculate_reward(action)  
           
        ground_truth_step = np.zeros(self.num_actions)
        predicted_step = np.zeros(self.num_actions)
        # if self.gym_type == 4 or self.gym_type == 5:
        #     self.generated_traffic_type = 1 if self.status["id"]>0 else 0
        ground_truth_step[self.generated_traffic_type] = 1
        predicted_step[action] = 1            

        action_correct = self.generated_traffic_type == action
        if action_correct:
            correct_predictions+=1
        percentage_correct_predictions = correct_predictions/current_step

        debug(Fore.CYAN + f"Environment reward {reward}"+Fore.WHITE )          
           
        # Check if the episode is done
        done, truncated = self.check_if_done_or_truncated(current_step, percentage_correct_predictions) 
        # Update state here
        if not done and not truncated:
            if self.gym_type==4:
                time.sleep(1)
            else:
                self.update_state()      
        state = self.get_current_state(is_discretized_state=is_discretized_state, is_real_state= is_real_state ) 
            
        return state , reward, done, truncated, {'action_correct': action_correct, 'TimeLimit.truncated': truncated, 'Ground_truth_step': ground_truth_step, 'Predicted_step': predicted_step}  
       
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
    
    def set_gym_type(self, gym_type):
        self.gym_type = -1
        if gym_type == 'classification_from_dataset':
            self.gym_type = 0             
        elif gym_type == 'classification_with_syncronize':
            self.gym_type = 1 
        elif gym_type == 'classification':
            self.gym_type = 2
        elif gym_type == 'real_time': 
            self.gym_type = 3
        elif gym_type == 'attacks': #real time with legitimic traffic and 2 kinds of DOS attacks (shor 1-5 seconds and long about 60 seconds)
            self.gym_type = 4
        elif gym_type == 'attacks_from_dataset':
            self.gym_type = 5             
        else:
           error(f"Set correctly 'gym_type' on config.yaml") 
           raise Exception("Set correctly 'gym_type' on config.yaml")
       
        if self.gym_type ==1:
            time.sleep(9) #wait 7 seconds for controller starts properly, before to read first time
        
            #read initial traffic
            self.sync_time = self.synchronize_controller()
            self.read_time = self.sync_time * 0.6

            if self.gym_type==3: #real time
                #update status reading from controller every second
                update_thread = threading.Thread(target=self.read_from_network)
                # Start the thread 
                update_thread.start()   
        
    def synchronize_controller(self, packet_received_threshold=0,byte_received_threshold=0):
        start_sync_time = time.time()        
        debug(f"\nController syncronization\n")
        self.read_time = 4
        self.read_from_network()
        
        states = [self.state]
        while self.state[0] > self.n_hosts or self.state[2] > self.n_hosts: #packet received and byte received are 0
            self.read_time = self.read_time * 0.75
            self.read_from_network()
            states.append(self.state)
       
        final_state = self.sum_states(states)
        debug(f"final_state after syncronization {[int(x) for x in final_state]}\n")
        end_sync_time = time.time()
        sync_time = end_sync_time - start_sync_time + 1
        debug(f"Controller syncronization time {sync_time} s\n")
        return sync_time
        
    def time_sleep(self):
        time.sleep(self.read_time) 
        
    def get_normalize_state(self, state):
        tmp_state = np.array(state, dtype=np.float32)  # ensure it's a NumPy array
        clipped_state = np.clip(tmp_state, self.low_to_normalize, self.high_to_normalize) #limit the values in an array.
        normalized = (clipped_state - self.low_to_normalize) / (self.high_to_normalize - self.low_to_normalize + 1e-8)
        return np.array(normalized, dtype=np.float32)

    def get_discretized_state(self, state):
        if state is None:
            return np.array(np.zeros(4) , dtype=np.float32)
        #from globals
        n_bins = self.n_bins
        bins = self.bins
        low = self.low
        high = self.high
        
        # Ensure the state is not a tuple
        if isinstance(state, tuple):
            state = state[0]
            
        # Initialize the discrete_state list
        discrete_state = []

        for i, val in enumerate(state):
            if (i==1 or i==3) and self.gym_type>=4:
                bin_index = self.get_linear_bin_index(val, low[i], high[i], n_bins-1)+1
            else:       
                bin_index = self.get_linear_bin_index(val, low[i], high[i], n_bins)
            discrete_state.append(bin_index)
        # return np.array(discrete_state, dtype=np.int32)
        return tuple(discrete_state)  
    
    def get_linear_bin_index(self, val, low, high, n_bins):
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
    
    def get_log_bin_index(self, val, low, high, n_bins):
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
        
    def update_network_status(self):
        """
        Analyze the current tasks in self.host_tasks and return the overall network status.
        Returns:
            str: One of the statuses - "normal", "attack_short", "attack_long", or "both".
        """
        if self.host_tasks is None:
            return
        
        # Initialize flags for different task types
        has_normal_traffic = False
        has_short_attack = False
        has_long_attack = False

        # Iterate through the tasks of all hosts
        current_time = time.time()
        for host_name, task_info in self.host_tasks.items():
            # Check if the task is still active
            if task_info["end_time"] > current_time:
                task_type = task_info["task_type"]
                if task_type == "normal":
                    has_normal_traffic = True
                elif task_type == "short_attack":
                    has_short_attack = True
                elif task_type == "long_attack":
                    has_long_attack = True

        #evaluate % variation packet
        now = self.state[0] 
        self.status["packets"] = now
        before =  self.prev_state[0]
        if (now == before):
            self.status["variation_packet"]=0
        elif (now>before and before>0):
            self.status["variation_packet"]=(now/before-1)*100
        elif (now>before and before==0):
            self.status["variation_packet"]=self.threshold_var_packets-1
        elif (now<before and now>0):
            self.status["variation_packet"]=-(before/now-1)*100
        else:
            self.status["variation_packet"]=-self.threshold_var_packets+1

        #evaluate % variation byte
        now = self.state[2] 
        self.status["bytes"]= now
        before =  self.prev_state[2] 
        if (now == before):
            self.status["variation_byte"]=0
        elif (now>before and before>0):
            self.status["variation_byte"]=(now/before-1)*100
        elif (now>before and before==0):
            self.status["variation_byte"]=self.threshold_var_bytes-1
        elif (now<before and now>0):
            self.status["variation_byte"]=-(before/now-1)*100
        else:
            self.status["variation_byte"]=-self.threshold_var_bytes+1

        # Determine the overall network status based on the flags
        if has_short_attack or has_long_attack:
            if self.status["packets"]<self.threshold_packets*0.5 and self.status["bytes"]<self.threshold_bytes*0.5: 
                self.status["id"] = -2
                self.status["text"] = Fore.LIGHTYELLOW_EX+"failed_attack"+Fore.WHITE  
            elif self.status["id"]<1 and (self.status["variation_packet"]<self.threshold_var_packets or self.status["variation_byte"]<self.threshold_var_bytes):
                self.status["id"] = 1
                self.status["text"] = Fore.YELLOW+"init_attack"+Fore.WHITE  
            elif has_short_attack and has_long_attack:
                self.status["id"] = 4
                self.status["text"] = Fore.RED+"both"+Fore.WHITE  # Both short and long attacks are occurring
            elif has_short_attack:
                self.status["id"]  = 2
                self.status["text"]  = Fore.LIGHTRED_EX+"attack_short"+Fore.WHITE
            elif has_long_attack:
                self.status["id"]  = 3
                self.status["text"]  = Fore.RED+"attack_long"+Fore.WHITE
        else:
            if self.status["id"]>1 and self.status["id"]<6 and (self.status["variation_packet"]>-100 and self.status["variation_byte"]>-100):
                self.status["id"] = 5
                self.status["text"]  = Fore.MAGENTA+"trail_attack"+Fore.WHITE      
            elif self.status["packets"]>self.threshold_packets and self.status["bytes"]>self.threshold_bytes:
                self.status["id"] = 6
                self.status["text"]  = Fore.LIGHTMAGENTA_EX+"abnormal_high_traffic"+Fore.WHITE                                
            elif has_normal_traffic:    
                self.status["id"]  = 0
                self.status["text"]  = Fore.GREEN+"normal"+Fore.WHITE
            else:
                self.status["id"]  = -1
                self.status["text"]  = "idle"  # No tasks are currently active        
                                      

    def get_all_traffic_generated(self): #read from switch, until 0 packet and byte are found  
        start_sync_time = time.time()        
        debug(f"Reading traffic\n")
        self.prev_state = self.state
        self.read_time = self.sync_time * 0.75
        prev_total_packets_received = self.net.total_packets_received
        prev_total_bytes_received = self.net.total_bytes_received
        prev_total_packets_transmitted = self.net.total_packets_transmitted
        prev_total_bytes_transmitted = self.net.total_bytes_transmitted
        current_packets_received = 0 
        current_packets_transmitted = 0 
        current_bytes_received = 0 
        current_bytes_transmitted = 0 
        time.sleep(3)
        while(True):
            time.sleep(1)
            self.read_from_network()
            if self.net.total_packets_received == 0 and self.net.total_bytes_received == 0 and self.net.total_packets_transmitted == 0 and self.net.total_bytes_transmitted == 0:
                self.net.total_packets_received = current_packets_received 
                self.net.total_packets_transmitted = current_packets_transmitted
                self.net.total_bytes_received = current_bytes_received
                self.net.total_bytes_transmitted = current_bytes_transmitted
                self.net.prev_total_packets_received = prev_total_packets_received
                self.net.prev_total_bytes_received = prev_total_bytes_received
                self.net.prev_total_packets_transmitted = prev_total_packets_transmitted
                self.net.prev_total_bytes_transmitted = prev_total_bytes_transmitted
                self.state = np.array([current_packets_received, current_packets_transmitted, current_bytes_received, current_bytes_transmitted], dtype=np.float32) 
                break
            
            current_packets_received += self.net.total_packets_received 
            current_packets_transmitted += self.net.total_packets_transmitted 
            current_bytes_received += self.net.total_bytes_received 
            current_bytes_transmitted += self.net.total_bytes_transmitted 
        self.show_status_classification()
        
        self.update_data_traffic(self.state)  
        end_sync_time = time.time()
        sync_time = end_sync_time - start_sync_time + 1
        debug(f"Traffic read time {sync_time} s\n")

    def update_data_traffic(self, state):
        traffic_label = self.net.traffic_types[self.generated_traffic_type]
        self.data_traffic[traffic_label]["p_r"].append(state[0])
        self.data_traffic[traffic_label]["p_t"].append(state[1])
        self.data_traffic[traffic_label]["b_r"].append(state[2])
        self.data_traffic[traffic_label]["b_t"].append(state[3])          
         
    def sum_states(self, states):
        # Calculate the final cumulative sums for each component across states
        total_packets_received = sum([s[0] for s in states])
        total_packets_transmitted = sum([s[1] for s in states])
        total_bytes_received = sum([s[2] for s in states])
        total_bytes_transmitted = sum([s[3] for s in states])

        # Log the cumulative state after ending the loop
        return np.array([
            total_packets_received,
            total_packets_transmitted,
            total_bytes_received,
            total_bytes_transmitted
        ], dtype=np.float32)
    
    def read_from_network(self): #read from switch
        """to retrieve new observation
        """
        start_get_time = time.time() 

        self.net.prev_total_packets_received = self.net.total_packets_received
        self.net.prev_total_bytes_received = self.net.total_bytes_received
        self.net.prev_total_packets_transmitted = self.net.total_packets_transmitted
        self.net.prev_total_bytes_transmitted = self.net.total_bytes_transmitted
        
        self.net.total_packets_received = 0
        self.net.total_bytes_received = 0
        self.net.total_packets_transmitted = 0
        self.net.total_bytes_transmitted = 0        

        #new method with os dump flow
        flows = get_data_controller_flow(self.net)
        debug(f"Flows are {flows}")
        self.net.total_packets_received = flows['packets']['received']
        self.net.total_bytes_received = flows['bytes']['received']
        self.net.total_packets_transmitted = flows['packets']['transmitted']
        self.net.total_bytes_transmitted = flows['bytes']['transmitted']
        
        #old method with restconf
        #get_data_controller(self.net.controller, self.net.switches)        
        # for sw in self.net.switches:
        #     data = sw.data
        #     # print(f"id {data.id}")
        #     for nc in data.node_connector:
        #         self.net.total_packets_received+=nc.statistics.packets.received
        #         self.net.total_bytes_received+=nc.statistics.bytes.received
        #         self.net.total_packets_transmitted+=nc.statistics.packets.transmitted
        #         self.net.total_bytes_transmitted+=nc.statistics.bytes.transmitted

        state = np.array([self.net.total_packets_received, self.net.total_packets_transmitted, self.net.total_bytes_received, self.net.total_bytes_transmitted], dtype=np.float32) 
        
        end_get_time = time.time()
        get_time = end_get_time - start_get_time
        debug(f"Time {get_time}\n") 
        debug(f"State  {[int(x) for x in state]}\n")        
 
        self.state = state   
        if self.gym_type==4:
            self.update_network_status() #update the status: normal or attack?
            self.statuses.append(dict(self.status))
            self.state[1] = self.status['variation_packet']
            self.state[3] = self.status['variation_byte']
            self.show_status_attack() 
        # else: 
        #     self.show_status_classification()
        
    def show_status_classification(self):            
        state = self.state
        information(Fore.BLUE + 
                    f"Packet {int(state[0])} - Byte {int(state[2])}" +
                    Fore.WHITE + f" - Traffic type {self.net.traffic_types[self.generated_traffic_type]}" +
                    "\n" + Fore.WHITE)
    
    def show_status_attack(self):            
        state = self.state
        t_q_v_p = self.threshold_var_packets * self.threshold_var_packets #square threshold_var_packets
        t_q_v_b = self.threshold_var_bytes * self.threshold_var_bytes #square threshold_var_bytes
        #prev_state = self.prev_state
        #information(Fore.CYAN +f"Packet {int(prev_state[0])} {int(prev_state[1])}% - Byte {int(prev_state[2])} {int(prev_state[3])}%\n")
        color1 = Fore.BLUE if state[1] * state[1] < t_q_v_p else Fore.WHITE
        color3 = Fore.BLUE if state[3] * state[3] < t_q_v_b else Fore.WHITE
        information(
            Fore.BLUE + f"Packet {int(state[0])} " +
            color1 + f"{int(state[1])}%" +
            Fore.BLUE + f" - Byte {int(state[2])}" +
            color3 + f" {int(state[3])}%" +
            Fore.BLUE + f" - Status {self.status['text']}\n" +
            Fore.WHITE
        )            
    #TODO separate e move the single parts to the extensions    
    def update_state(self, episode = None):
        """
        update state Evalueting gym type
        """                  
        if self.gym_type == 4 and self.state is not None:  #attacks
            self.prev_state = self.state
            self.read_from_network() #the traffic is generated continuosly by adversarial_agent
        elif self.gym_type == 1:  #classification_with_syncronize
            self.synchronize_controller()            
            self.generated_traffic_type, self.src_host, self.dst_host = generate_random_traffic(self.net)  
            self.read_from_network()
        elif self.gym_type==2:  #classification_without_syncronize
            self.generated_traffic_type, self.src_host, self.dst_host = generate_random_traffic(self.net)  
            self.get_all_traffic_generated() 
        elif self.gym_type == 0: #classification_from_dataset
            try: 
                if not hasattr(self, 'df'):
                    self.state = self.initial_state
                elif len(self.df) > 0:
                    status=self.df.pop(0)
                    self.state = np.array([status["packets_received"],status["packets_transmitted"], status["bytes_received"], status["bytes_transmitted"]]) 
                    self.generated_traffic_type = status["traffic_type"]
                    self.status = status
                    self.show_status_classification()
                else:
                    error(Fore.RED+"Missing dataset row: no status read\n")
            except : 
                error(Fore.RED+"Reading status error\n")            
            
            
            # if episode is not None:
            #     data_episode=self.df._values[episode]
            # else:
            #     n = len(self.df._values)
            #     i = random.randint(0, n-1)
            #     data_episode=self.df._values[i]   
                         
            # self.state = np.array([data_episode[0],data_episode[2], data_episode[1], data_episode[3]]) 
            # self.generated_traffic_type = data_episode[4]
            # self.update_data_traffic(self.state)  
                      
            # if self.generated_traffic_type > 0:
            #     self.src_host=get_host_by_name(self.net, data_episode[5])
            #     self.dst_host=get_host_by_name(self.net, data_episode[6])
            # else:
            #     self.src_host = self.dst_host = None  
            #print_traffic(self.net.traffic_types[self.generated_traffic_type], data_episode[5], data_episode[6])                 
        elif self.gym_type == 5: #attack_from_dataset
            try: 
                if not hasattr(self, 'df'):
                    self.state = self.initial_state
                elif len(self.df) > 0:
                    status=self.df.pop(0)
                    self.state = np.array([status["packets"],status["variation_packet"], status["bytes"], status["variation_byte"]]) 
                    self.status = status
                    self.show_status_attack()
                else:
                    error(Fore.RED+"Missing dataset row: no status read\n")
            except : 
                error(Fore.RED+"Reading status error\n")
    

    def show_actions_choosen(self, count_actions_by_type, generated_traffic_type=-1, no_color = False):
        for num, count in count_actions_by_type.items(): 
            if not no_color and generated_traffic_type>-1:
                information(Fore.GREEN if num==generated_traffic_type else Fore.RED)  
            else:
                information(Fore.WHITE)    
            information(f"Action {self.execute_action(num)} appeared {count} times\n"+Fore.WHITE)
            
 
    # @abstractmethod
    # def evaluate_episode(self, num_episodes):
    #     """
    #     Placeholder for the `evaluate_episode` method. Must be implemented by derived classes.
    #     """
    #     pass   
            
    def evaluate_test(self, ground_truth,predicted):
        # Calculate and store metrics at the end of the episode with library sklearn
        accuracy_episode = accuracy_score(ground_truth,predicted)
        precision_episode, recall_episode, f1_score_episode, _ = precision_recall_fscore_support(ground_truth, predicted, average='weighted', zero_division=0.0)
        return accuracy_episode, precision_episode, recall_episode, f1_score_episode        
                          
    def close(self):
        """Close the environment (optional)."""
        information("Environment closed.")
        
    def render(self, mode="human"):
        """Render the environment (optional)."""
        information(f"State: {self.state}, Ground Truth: {self.generated_traffic_type}")

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
        'number_of_actions':4,
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
