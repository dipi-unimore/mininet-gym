from reinforcement_learning.network_env import NetworkEnv
from utility.my_log import debug
from utility.network_configurator import comunicate_normal_traffic_detected, comunicate_attack_detected
from utility.params import Params
#from utility.my_log import setLogLevel, information, debug, error, notify_client
#from colorama import Fore, Back, Style
#from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json as jsonlib
from gymnasium import spaces
import numpy as np, time, threading

class NetworkEnvAttackDetect(NetworkEnv):     
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
                    'K_steps':2,
                    'steps_min_percentage':0.9,
                    'accuracy_min':0.9         
                }, server_user = 'server_user'):
        params.actions_number = 2
        super().__init__(params, server_user)
        
        # Network creation
        self.threshold_packets = params.threshold.packets
        self.threshold_var_packets = params.threshold.var_packets
        self.threshold_bytes = params.threshold.bytes
        self.threshold_var_bytes = params.threshold.var_bytes
        # Define action and observation space, gym.spaces objects
        self.low = np.array([0,-self.threshold_var_packets,0,-self.threshold_var_bytes])   
        self.high = np.array([self.threshold_packets*len(self.net.hosts),self.threshold_var_packets,self.threshold_bytes*len(self.net.hosts),self.threshold_var_bytes])
        self.observation_space = spaces.Box(low=0, high=self.high, shape=(len(self.state),), dtype=np.float32)
        
        # Define the number of discrete bins for each observation dimension
        self.n_bins = params.n_bins
        self.low_to_normalize = self.low #np.array([0,-self.threshold_var_packets,0,-self.threshold_var_bytes]) #np.floor(np.log10(self.low)).astype(int)
        self.high_to_normalize = self.high #np.array([self.threshold_packets*len(self.net.hosts),self.threshold_var_packets,self.threshold_bytes*len(self.net.hosts),self.threshold_var_bytes]) #np.inf
        low = self.low #np.floor(np.log10(self.low)).astype(int)
        high = np.floor(np.log10(self.high)).astype(int)
        self.bins = [np.logspace(low[i], high[i], self.n_bins) for i in range(self.observation_space.shape[0])]                
    
        self.last_short_attack_timestamp = time.time()
        self.last_long_attack_timestamp = time.time()
        
        if self.gym_type == 4:
            self.attack_probability = self.init_attack_probability = params.attack_probability
            self.stop_update_status_event = threading.Event()            
            self.update_state_thread_instance = threading.Thread(target=self.update_state_thread, args=(self.stop_update_status_event,))
            self.update_state_thread_instance.start()
    
    def update_state_thread(self, stop_event):
        """
        update environment state every N=0.5 seconds, only gym_type=4, reading from switch
        """
        while not stop_event.is_set() and not self.stop_update_status_event.is_set():   
            time.sleep(0.5)
            self.update_state() 
        debug("Update state thread finished") 
        
    def check_if_done_or_truncated(self, current_step, percentage_correct_predictions):
        """
        Check if the episode is done.
        The episode is done if:
        1. The maximum number of steps is reached.
        2. Temporary accuracy over {self.min_accuracy}, after at least {self.min_percentage_steps} of steps.
        """ 
        
        if current_step >= self.max_steps*self.min_percentage_steps and percentage_correct_predictions>self.min_accuracy :
            return False, True # accuracy greater than param accuracy_min
        
        # End the episode if maximum steps are reached
        if current_step >= self.max_steps:
            return True, False       

        return False, False   
    
    def execute_action(self,action):
        # Handle the action logic
        if action == 0:
            # Action 0: 
            return comunicate_normal_traffic_detected()
        if action == 1:
            # Action 1: 
            return comunicate_attack_detected()
    
    def calculate_reward(self, action: int) -> float:
        """
        This modified function provides informative feedback, 
        rewarding the agent based on how close it was to the correct answer.
        """
        traffic_type = 1 if self.status["id"] > 0 else 0
        self.generated_traffic_type = traffic_type
        # if action == traffic_type:
        #     return 1.0  # 1 reward for correct prediction
        #return -1.0  # -1 Max penalty for wrong prediction
        if action == traffic_type and traffic_type == 1:
            return 1.0  # 1 reward for correct attack detected
        elif action == traffic_type:
            return 1  # 0.5 reward for correct normal traffic prediction
        elif traffic_type == 0:
            return -1  # -0.5 penalty for false alarm
        return -1.0  # -1 Max penalty for attack not detected    
            

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
