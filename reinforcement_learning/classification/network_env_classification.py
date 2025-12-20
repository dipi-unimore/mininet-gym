import threading
from reinforcement_learning.adversarial_agent import generate_random_traffic
from reinforcement_learning.classification.constants import AGENT_ACTIONS, REWARDS
from reinforcement_learning.classification.instant_state import InstantState
from reinforcement_learning.network_env import NetworkEnv, get_agent_name
from utility.constants import  GYM_TYPE, CLASSIFICATION_FROM_DATASET, CLASSIFICATION, NORMAL, TRAFFIC_TYPE_ID_MAPPING, SystemLevels, SystemModes, SystemStatus, TrafficTypes
from utility.my_log import debug, information, notify_client
from utility.network_configurator import comunicate_no_traffic_detected, comunicate_ping_traffic_detected, comunicate_tcp_traffic_detected, comunicate_udp_traffic_detected, format_bytes
from utility.params import Params
from gymnasium import spaces
from colorama import Fore
from os import error
import copy, time,  numpy as np, json as jsonlib


class NetworkEnvClassification(NetworkEnv):     
    """Custom Environment that follows gym interface.
    This is a simple env where an agent must classify the network traffic type.
    The agent can choose between four actions: NO traffic, PING, UDP, TCP.
    The environment provides a reward based on the correctness of the classification.
    The network topology is a simple star topology with one switch and multiple hosts, 
    but the number of hosts don't affect the environment behavior, because only a traffic type is generated randomly.   
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 30}    
           
    def __init__(self, params, server_user = 'server_user', existing_net=None):        
        params.actions_number = AGENT_ACTIONS.NUMBER
        super().__init__(params, server_user, existing_net=None)
        self.params = params
        self.statuses = []
        self.hosts = self.net.hosts         
        
         # Network params
        self.threshold_packets = 1e4
        self.threshold_var_packets = 100
        self.threshold_bytes = 1e7 
        self.threshold_var_bytes = 100      
        
        # Define action and observation space, gym.spaces objects
        self.low = np.array([0,-self.threshold_var_packets,0,-self.threshold_var_bytes])        
        self.high = np.array([self.threshold_packets, self.threshold_var_packets, self.threshold_bytes, self.threshold_var_bytes])        #fixed for every network
        self.observation_space = spaces.Box(low=0, high=self.high, shape=(len(self.low),), dtype=np.float64)
        
        # Define the number of discrete bins for each observation dimension
        self.n_bins = params.n_bins
        self.low_to_normalize = self.low
        self.high_to_normalize = self.high
        low = self.low 
        high = np.floor(np.log10(self.high)).astype(int)
        self.bins = [np.logspace(low[i], high[i], self.n_bins) for i in range(self.observation_space.shape[0])]
        
        self.sync_time = 3 #seconds
        
        self.global_prev_state = self.global_state = InstantState(self.hosts)
    
        if self.gym_type == GYM_TYPE[CLASSIFICATION]:   
            self.update_state_thread_instance = threading.Thread(target=self.update_state_thread)
            self.update_state_thread_instance.start()
            
        self.reset()
    
    def update_hosts_status(self, generated_traffic_type_text, src_host, dst_host):
        """
        Updates the host statuses based on the generated traffic type.
        Overrides the base method to adapt to classification environment.
        Args:
            generated_traffic_type_text (str): The type of traffic generated ("NONE", "PING", "UDP", "TCP").
            src_host: The source host of the generated traffic.
            dst_host: The destination host of the generated traffic.
        Returns:
            dict: A dictionary mapping host names to their updated statuses.    
        """
        statuses = {}
        host_tasks = {}
        for host in self.hosts:
            if src_host is not None and host.name == src_host.name:
                statuses[host.name] = generated_traffic_type_text
                host_tasks[host.name] = {'taskType': NORMAL, 'trafficType': generated_traffic_type_text, 'destination': dst_host.name}
            elif dst_host is not None and  host.name == dst_host.name:
                statuses[host.name] = generated_traffic_type_text
                host_tasks[host.name] = {'taskType': NORMAL, 'trafficType': TrafficTypes.NONE} 
            else:
                statuses[host.name] = TrafficTypes.NONE
                host_tasks[host.name] = {'taskType': NORMAL, 'trafficType': TrafficTypes.NONE}        
            host_tasks[host.name]["linkStatus"] = 1  # link always ON 
        notify_client(level=SystemLevels.DATA, host_tasks = host_tasks )
        return statuses
                
    def update_state(self, episode = None):
        """
        update state Evalueting gym type
        """                 
        if self.gym_type == GYM_TYPE[CLASSIFICATION] and self.state is not None:    #classification
            # self.synchronize_controller()            
            self.generated_traffic_type_text, self.src_host, self.dst_host = generate_random_traffic(self.net)  
            self.generated_traffic_type = TRAFFIC_TYPE_ID_MAPPING[self.generated_traffic_type_text]
            # wait for traffic complete generation
            time.sleep(self.sync_time) 
            if self.read_from_network(): 
                self.evaluate_traffic() 
        # elif self.gym_type==GYM_TYPE[CLASSIFICATION_WITHOUT_SYNCRONIZE]:  #classification_without_syncronize
        #     self.generated_traffic_type, self.src_host, self.dst_host = generate_random_traffic(self.net)  
        #     # try to read all traffic: slower than the other but sure to read all traffic
        #     # to be used to record traffic to create dataset  
        #     self.get_all_traffic_generated() 
        #     self.evaluate_traffic()              
        elif self.gym_type == GYM_TYPE[CLASSIFICATION_FROM_DATASET]: #classification_from_dataset
            try: 
                if not hasattr(self, 'df'):
                    self.state = self.initial_state
                elif len(self.df) > 0:
                    state=self.df.pop(0)
                    self.global_prev_state = copy.copy(self.global_state)
                    self.global_state.set_state(state)
                    self.generated_traffic_type = state["id"]
                    self.generated_traffic_type_text = state["status"]
                    self.src_host = state["src_host"]
                    self.dst_host = state["dst_host"]
                    self.state = self.global_state.get_state() #this is real state from dataset
                    # self.show_status_classification()
                    # traffic_data = self.global_state.get_network_traffic_status()
                    # notify_client(level=SystemLevels.DATA, traffic_data = traffic_data)
                    self.evaluate_traffic() 
                else:
                    error(Fore.RED+"Missing dataset row: no status read\n")
            except : 
                error(Fore.RED+"Reading status error\n") 

    def evaluate_traffic(self):
        self.global_state.state = np.array([
                    self.global_state.packets, 
                    self.global_state.packets_percentage_change, 
                    self.global_state.bytes, 
                    self.global_state.bytes_percentage_change,
                ], dtype=np.float32) 
        statuses = self.update_hosts_status(self.generated_traffic_type_text, self.src_host, self.dst_host) #update the status: normal or attack?
        self.global_state.update_statuses(self.generated_traffic_type_text, TRAFFIC_TYPE_ID_MAPPING, statuses)
        traffic_data = self.global_state.get_network_traffic_status()
        notify_client(level=SystemLevels.DATA, traffic_data = traffic_data)
        traffic_data["src_host"] = self.src_host.name if self.src_host is not None else None
        traffic_data["dst_host"] = self.dst_host.name if self.dst_host is not None else None
        self.statuses.append(traffic_data)
        self.show_network_status()

    def get_all_traffic_generated(self): #read from switch, until 0 packet and byte are found  
        start_sync_time = time.time()        
        debug(f"Reading traffic\n")
        self.read_time = self.sync_time * 0.75

        old_global_state = copy.copy(self.global_state)
        while(True):
            time.sleep(1)
            if not self.read_from_network():
                break
            #the new global_state is self.prev_global_state - old_global_state
            self.global_state = self.global_prev_state.substract(old_global_state)

        end_sync_time = time.time()
        sync_time = end_sync_time - start_sync_time + 1
        debug(f"Traffic read time {sync_time} s\n")
      
            
    def show_status_classification(self):            
        state = self.global_state.get_state()
        information(Fore.BLUE + 
                    f"Packet {int(state[0])} - {format_bytes(int(state[2]))}B" +
                    Fore.WHITE + f" - Traffic type {self.net.traffic_types[self.generated_traffic_type]}" +
                    "\n" + Fore.WHITE)       
    
             
    def step(self, action, is_discretized_state = False, is_real_state= False, current_step=-1, correct_predictions=0, show_action=False, name = None):
        while hasattr(self,"pause_event") and self.pause_event.is_set():
            notify_client(level=SystemLevels.STATUS, status=SystemStatus.PAUSED, message="Paused training agents...", mode=SystemModes.TRAINING)
            time.sleep(1)
            continue   
        # Calculate reward
        action_correct = self.generated_traffic_type
        text_action_correct = self.generated_traffic_type_text
        reward = self.calculate_reward(action, action_correct)  
        self.execute_action(action, show_action=show_action, reward=reward)
        is_action_correct = action_correct == action
        if is_action_correct:
            correct_predictions+=1
        percentage_correct_predictions = correct_predictions/current_step
           
        ground_truth_step = np.zeros(self.actions_number)
        predicted_step = np.zeros(self.actions_number)
        ground_truth_step[self.generated_traffic_type] = 1
        predicted_step[action] = 1            

        debug(Fore.CYAN + f"Environment reward {reward}"+Fore.WHITE )          
           
        # Check if the episode is done
        done, truncated = self.check_if_done_or_truncated(current_step, percentage_correct_predictions) 
        # Update state here
        if not done and not truncated:
            if self.gym_type==GYM_TYPE[CLASSIFICATION]:
                time.sleep(self.sync_time)
            else:
                self.update_state()     
        next_state = self.get_current_state(is_discretized_state=is_discretized_state, is_real_state= is_real_state ) 
            
        return next_state, reward, done, truncated, {'action_correct': action_correct, 
                                                     'text_action_correct': text_action_correct, 
                                                     'is_correct_action':is_action_correct, 
                                                     'TimeLimit.truncated': truncated, 
                                                     'Ground_truth_step': ground_truth_step, 
                                                     'Predicted_step': predicted_step}  
   
    def check_if_done_or_truncated(self, current_step, percentage_correct_predictions):
        """
        Check if the episode is done.
        The episode is done if:
        1. The maximum number of steps is reached.
        2. Temporary accuracy over {self.min_accuracy}, after at least {self.steps_min_percentage} of steps.
        """ 
        # if action == self.generated_traffic_type and self.is_exploitation:
        #     return True  # only for Sarsa and QLearning
        
        # if action == self.generated_traffic_type and self.early_exit:
        #      return False, True  # for DQN
        
        # #probably to realize better graphs, use all max_steps, so skip next two if
        # if  current_step >= 100 and sum(self.rewards[-100:]) == 100:
        #     return False, True     
        
        if current_step >= self.max_steps*self.steps_min_percentage and percentage_correct_predictions>self.min_accuracy :
            return False, True # accuracy greater than param accuracy_min
        
        # End the episode if maximum steps are reached
        if current_step >= self.max_steps:
            return True, False       

        return False, False   
    
    def execute_action(self, action, show_action=False, name = None, reward = 0):
        # Handle the action logic
        if action == AGENT_ACTIONS.ACTIONS[TrafficTypes.NONE]:
            # Action 0: Do nothing
            msg =  comunicate_no_traffic_detected()
        if action == AGENT_ACTIONS.ACTIONS[TrafficTypes.PING]:
            # Action 1: Comunicate a ping traffic
            msg =  comunicate_ping_traffic_detected()
        if action == AGENT_ACTIONS.ACTIONS[TrafficTypes.UDP]:
            # Action 2: Comunicate a UDP traffic
            msg =  comunicate_udp_traffic_detected()  
        if action == AGENT_ACTIONS.ACTIONS[TrafficTypes.TCP]:
            # Action 3: Comunicate a TCP traffic
            msg =  comunicate_tcp_traffic_detected()
        if show_action:
            if name is None:
                agent_name = get_agent_name()
            else:
                agent_name = name   
            information(f"{msg} R: {reward}\n", agent_name)    
        return msg   
    
    def calculate_reward(self, action: int, action_correct: int) -> float:
        """
        This modified function provides informative feedback, 
        rewarding the agent based on how close it was to the correct answer.
        """
        
        if action == action_correct:
            return REWARDS.CORRECT_TRAFFIC  # Max reward for correct prediction
        if abs(action - action_correct) == 1:
            return REWARDS.CLOSE  # Partial penalty for close predictions
        if abs(action - action_correct) == 2:
            return REWARDS.FARTHER  # Higher penalty for farther predictions
        return REWARDS.COMPLETELY_INCORRECT  # -0.9 Max penalty for completely incorrect predictions
     
    def get_discretized_state(self, state):
        if state is None:
            return np.array(np.zeros(len(self.low)) , dtype=np.float32)
        
        # Ensure the state is not a tuple
        if isinstance(state, tuple):
            state = state[0]
            
        # Initialize the discrete_state list
        discrete_state = []

        for i, val in enumerate(state):
            bin_index = np.digitize(val+1, self.bins[i]) - 1
            discrete_state.append(bin_index)
        return tuple(discrete_state)         

    # def evaluate_episode(self, episode, cumulative_reward, exploration_count=0, exploitation_count=0, ground_truth = None, predicted= None):
    #     # Calculate and store metrics at the end of the episode with library sklearn
    #     if ground_truth is None:
    #         ground_truth = self.ground_truth
    #     if predicted is None:
    #         predicted = self.predicted            
    #     accuracy_episode = accuracy_score(ground_truth, predicted)
    #     precision_episode, recall_episode, f1_score_episode, _ = precision_recall_fscore_support(ground_truth, predicted, average='weighted', zero_division=0.0)
    #     self.metrics['accuracy'].append(accuracy_episode)
    #     self.metrics['precision'].append(precision_episode)
    #     self.metrics['recall'].append(recall_episode)
    #     self.metrics['f1_score'].append(f1_score_episode)
        
    #     # Calculate and store train type (exploration/exploitation) at the end of the episode
    #     if exploration_count > 0 or exploitation_count > 0 :
    #         self.train_types['explorations'].append(exploration_count)
    #         self.train_types['exploitations'].append(exploitation_count)
    #         self.train_types['steps'].append(self.current_step)
        
    #     if (episode > 0):
    #         if (self.src_host is not None):
    #             i_src_host = self.src_host.name #int(self.src_host.name.replace('h','').replace('iot',''))
    #         else:
    #             i_src_host = '0'
    #         if (self.dst_host is not None):
    #             i_dst_host = self.dst_host.name #int(self.dst_host.name.replace('h','').replace('iot',''))
    #         else:
    #             i_dst_host = '0'

    #         self.indicators.append({
    #             'episode': episode,
    #             'steps': self.current_step,
    #             'correct_predictions': self.correct_predictions,
    #             'packets_received': self.state[0].item(),
    #             'bytes_received': self.state[2].item(),
    #             'packets_transmitted': self.state[1].item(),
    #             'bytes_transmitted': self.state[3].item(),
    #             'cumulative_reward': cumulative_reward,
    #             'traffic_type': self.generated_traffic_type,
    #             'i_src_host': i_src_host,
    #             'i_dst_host': i_dst_host
    #         })

    
    
    # def synchronize_controller(self, packet_received_threshold=0,byte_received_threshold=0):
    #     start_sync_time = time.time()        
    #     debug(f"\nController syncronization\n")
    #     self.read_time = 4
    #     self.read_from_network()
        
    #     states = [self.state]
    #     while self.state[0] > self.n_hosts or self.state[2] > self.n_hosts: #packet received and byte received are 0
    #         self.read_time = self.read_time * 0.75
    #         self.read_from_network()
    #         states.append(self.state)
       
    #     final_state = self.sum_states(states)
    #     debug(f"final_state after syncronization {[int(x) for x in final_state]}\n")
    #     end_sync_time = time.time()
    #     sync_time = end_sync_time - start_sync_time + 1
    #     debug(f"Controller syncronization time {sync_time} s\n")
    #     return sync_time       

if __name__ == '__main__':
    #setLogLevel('info')
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
