
import copy
from colorama import Fore
from reinforcement_learning.attack_detect.constants import AGENT_ACTIONS, REWARDS
from reinforcement_learning.attack_detect.instant_state import InstantState
from reinforcement_learning.network_env import NetworkEnv, get_agent_name, get_linear_bin_index
from utility.my_log import debug, information, error, notify_client
from utility.network_configurator import comunicate_normal_traffic_detected, comunicate_attack_detected
from utility.params import Params
from utility.constants import *
import json as jsonlib
from gymnasium import spaces
import numpy as np, time, threading

class NetworkEnvAttackDetect(NetworkEnv):     
    """Custom Environment that follows gym interface.
    This is a simple env where an agent must detect if there is an attack or normal traffic.
    The agent can choose between two actions: NORMAL_TRAFFIC, ATTACK.
    The environment provides a reward based on the correctness of the detection.
    The network topology is a simple star topology with one switch and multiple hosts.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 30}    
           
    def __init__(self, params, server_user, existing_net=None):
        params.actions_number = AGENT_ACTIONS.NUMBER
        super().__init__(params, server_user, existing_net)
        self.params = params
        self.statuses = []
        self.hosts = self.net.hosts # Access hosts from the parent class's network
        
        # Network params
        self.threshold_packets = params.attack_thresholds.packets
        self.threshold_var_packets = params.attack_thresholds.var_packets
        self.threshold_bytes = params.attack_thresholds.bytes
        self.threshold_var_bytes = params.attack_thresholds.var_bytes
        # Define action and observation space, gym.spaces objects
        self.low = np.array([0,-self.threshold_var_packets,0,-self.threshold_var_bytes])   
        self.high = np.array([self.threshold_packets*len(self.net.hosts),self.threshold_var_packets,self.threshold_bytes*len(self.net.hosts),self.threshold_var_bytes])
        self.observation_space = spaces.Box(low=self.low, high=self.high, shape=(len(self.low),), dtype=np.float32)
        
        # Define the number of discrete bins for each observation dimension
        self.n_bins = params.n_bins
        self.low_to_normalize = self.low #np.array([0,-self.threshold_var_packets,0,-self.threshold_var_bytes]) #np.floor(np.log10(self.low)).astype(int)
        self.high_to_normalize = self.high #np.array([self.threshold_packets*len(self.net.hosts),self.threshold_var_packets,self.threshold_bytes*len(self.net.hosts),self.threshold_var_bytes]) #np.inf

        self.global_prev_state = self.global_state = InstantState(self.hosts)
    
        self.last_short_attack_timestamp = time.time()
        self.last_long_attack_timestamp = time.time()       
        
  
        if self.gym_type == GYM_TYPE[ATTACKS]:
            self.attack_likely = self.init_attack_likely = params.attack_likely
            self.update_state_thread_instance = threading.Thread(target=self.update_state_thread)
            self.update_state_thread_instance.start()
        
        self.reset()
    

    def update_state(self):
        """
        update state Evalueting gym type
        """                 
        if self.gym_type == GYM_TYPE[ATTACKS] and self.state is not None:  
            if self.read_from_network(): #the traffic is generated continuosly by adversarial_agent          
                self.evaluate_traffic() 
        elif self.gym_type == GYM_TYPE[ATTACKS_FROM_DATASET]: 
            try: 
                if not hasattr(self, 'df'):
                    self.state = self.initial_state
                elif len(self.df) > 0:
                    status=self.df.pop(0)
                    self.global_prev_state = copy.copy(self.global_state)
                    self.global_state.set_state(status)                                        
                    self.state = self.global_state.get_state() 
                    self.host_tasks = {}
                    for host_name, host_status in status["hostStatusesStructured"].items():
                        self.host_tasks[host_name] = {
                            'traffic_type': host_status['trafficType'],
                            'task_type': host_status['taskType'],
                            'destination': None if host_status['trafficType'] == TrafficTypes.NONE else host_status.get('destination', None),
                            'end_time': None
                        }
                    self.evaluate_traffic()
                else:
                    error(Fore.RED+"Missing dataset row: no status read\n")
            except Exception as e: 
                error(Fore.RED+"Reading status error\n")

    def evaluate_traffic(self):
        self.global_state.state = np.array([
                    self.global_state.packets, 
                    self.global_state.packets_percentage_change, 
                    self.global_state.bytes, 
                    self.global_state.bytes_percentage_change,
                ], dtype=np.float32) 
                
        statuses, self.is_attack = self.update_hosts_status() #update the status: normal or attack?
        status = ATTACK if self.is_attack else NORMAL
        self.global_state.update_statuses(status, statuses)
        traffic_data = self.global_state.get_network_traffic_status()
        notify_client(level=SystemLevels.DATA, traffic_data = traffic_data)
        if self.gym_type == GYM_TYPE[ATTACKS] and hasattr(self, 'host_tasks') and self.host_tasks is not None:
            for host_name, host_task in self.host_tasks.items():
                traffic_data["hostStatusesStructured"][host_name].update(
                    {'trafficType': host_task['traffic_type'], 
                    'taskType': host_task['task_type'],
                    'destination': host_task['destination']})
            self.statuses.append(traffic_data)
            self.show_network_status()
 
   
    def initialize_storage(self):
        pass   
         
        
    def step(self, action, is_discretized_state = False, is_real_state= False, current_step=-1, correct_predictions=0, show_action=False, name = None):
        while hasattr(self,"pause_event") and self.pause_event.is_set():
            notify_client(level=SystemLevels.STATUS, status=SystemStatus.PAUSED, message="Paused training agents...", mode=SystemModes.TRAINING)
            time.sleep(1)
            continue  
              
        # Calculate reward
        action_correct = self.global_state.status["id"]
        text_action_correct = self.global_state.status["status"]
        reward = self.calculate_reward(action)  
        self.execute_action(action, show_action=show_action, reward=reward)
        is_action_correct = self.generated_traffic_type == action
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
            if self.gym_type==GYM_TYPE[ATTACKS]:
                time.sleep(1)
            else:
                self.update_state()      
        
        next_state = self.get_current_state(is_discretized_state=is_discretized_state, is_real_state= is_real_state ) 
            
        return next_state , reward, done, truncated, {'action_correct': action_correct, 
                                                     'text_action_correct': text_action_correct, 
                                                     'is_correct_action': is_action_correct, 
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
        
        if current_step >= self.max_steps*self.steps_min_percentage and percentage_correct_predictions>self.min_accuracy :
            return False, True # accuracy greater than param accuracy_min
        
        # End the episode if maximum steps are reached
        if current_step >= self.max_steps:
            return True, False       

        return False, False   
    
    def execute_action(self, action, show_action=False, name = None, reward = 0):
        # Handle the action logic
        if action == AGENT_ACTIONS.NORMAL_TRAFFIC:
            # Action 0: 
            msg = comunicate_normal_traffic_detected()
        if action == AGENT_ACTIONS.ATTACK:
            # Action 1: 
            msg = comunicate_attack_detected()
        if show_action:
            if name is None:
                agent_name = get_agent_name()
            else:
                agent_name = name   
            information(f"{msg} R: {reward}\n", agent_name)   
        return msg         
    
    def calculate_reward(self, action: int) -> float:
        """
        This function provides informative feedback, 
        rewarding the agent based on how close it was to the correct answer.
        """
        is_attack = True if self.global_state.status["id"] > 0 else False
        self.generated_traffic_type = 0 if not is_attack else 1
        # if self.generated_traffic_type == AGENT_ACTIONS.IDLE:
        #     return 0.0
        
        # Scenario 1: Correctly detected attack
        if action == AGENT_ACTIONS.ATTACK and is_attack:
            return 1.0  # 1 reward for correct attack detected
        # Scenario 2: Correctly detected normal traffic
        if action == AGENT_ACTIONS.NORMAL_TRAFFIC and not is_attack:
            return REWARDS.CORRECT_NORMAL_TRAFFIC  
        # Scenario 3: Misses an attack
        if action == AGENT_ACTIONS.NORMAL_TRAFFIC and is_attack:
            return REWARDS.FALSE_NEGATIVE            
        # Scenario 4: False alarm
        if action == AGENT_ACTIONS.ATTACK and not is_attack:
            return REWARDS.FALSE_POSITIVE
        
        # Default case for other actions or scenarios
        return 0.0 
     
    def get_discretized_state(self, state):
        if state is None:
            return np.array(np.zeros(4) , dtype=np.float32)
        #from globals
        n_bins = self.n_bins
        low = self.low
        high = self.high
        
        # Ensure the state is not a tuple
        if isinstance(state, tuple):
            state = state[0]
            
        # Initialize the discrete_state list
        discrete_state = []

        for i, val in enumerate(state):
            if (i==1 or i==3) : #variation packet and byte for attacks gym_type
                bin_index = get_linear_bin_index(val, low[i], high[i], n_bins-1)+1
            else: #for attacks gym_type use linear discretization
                bin_index = get_linear_bin_index(val, low[i], high[i], n_bins)
            discrete_state.append(bin_index)
        return tuple(discrete_state)           
            

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
