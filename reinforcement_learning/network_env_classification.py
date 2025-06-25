from reinforcement_learning.network_env import NetworkEnv
#from reinforcement_learning.adversarial_agent import launch_dos_attack, generate_random_traffic, print_traffic
from utility.network_configurator import create_network, get_data_controller, get_data_controller_flow, comunicate_no_traffic_detected, comunicate_ping_traffic_detected, comunicate_udp_traffic_detected, comunicate_tcp_traffic_detected, get_host_by_name
#from utility.my_log import setLogLevel, information, debug, error, notify_client
from utility.params import Params
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json as jsonlib
from gymnasium import spaces
import numpy as np
# from colorama import Fore, Back, Style
# import threading, random, time
# import pandas as pd
# import gymnasium as gym

class NetworkEnvClassification(NetworkEnv):     
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
                    'number_of_actions':4,
                    'max_steps':2,
                    'steps_min_percentage':0.9,
                    'accuracy_min':0.9         
                }, server_user = 'server_user'):
        params.numbers_of_actions = 4
        # Network creation
        super().__init__(params, server_user)
        
        
        # Define action and observation space, gym.spaces objects

        self.low = np.array([0,0,0,0])      
        self.high = np.array([1e6, 1e6, 1e11, 1e11])        #fixed for every network
        self.observation_space = spaces.Box(low=0, high=self.high, shape=(len(self.state),), dtype=np.float32)
        
        # Define the number of discrete bins for each observation dimension
        self.n_bins = params.n_bins
        # self.low = np.array([1, len(self.net.hosts), 1e2, 1e3]) #np.inf
        # self.high = np.array([len(self.net.hosts), 1e2, 1e6, 1e7]) #np.inf
        self.low_to_normalize = np.array(np.zeros(self.num_actions)) #np.inf
        self.high_to_normalize = np.array([1e6,1e6,1e11,1e11]) #fixed for every network
        low = self.low #np.floor(np.log10(self.low)).astype(int)
        high = np.floor(np.log10(self.high)).astype(int)
        self.bins = [np.logspace(low[i], high[i], self.n_bins) for i in range(self.observation_space.shape[0])]
        self.sync_time = 10 #seconds
        
        
    def check_if_done_or_truncated(self, current_step, percentage_correct_predictions):
        """
        Check if the episode is done.
        The episode is done if:
        1. The maximum number of steps is reached.
        2. Temporary accuracy over {self.min_accuracy}, after at least {self.min_percentage_steps} of steps.
        """ 
        # if action == self.generated_traffic_type and self.is_exploitation:
        #     return True  # only for Sarsa and QLearning
        
        # if action == self.generated_traffic_type and self.early_exit:
        #      return False, True  # for DQN
        
        # #probably to realize better graphs, use all max_steps, so skip next two if
        # if  current_step >= 100 and sum(self.rewards[-100:]) == 100:
        #     return False, True     
        
        if current_step >= self.max_steps*self.min_percentage_steps and percentage_correct_predictions>self.min_accuracy :
            return False, True # accuracy greater than param accuracy_min
        
        # End the episode if maximum steps are reached
        if current_step >= self.max_steps:
            return True, False       

        return False, False   
    
    def execute_action(self,action):
        # Handle the action logic
        if action == 0:
            # Action 0: Do nothing
            return comunicate_no_traffic_detected()
        if action == 1:
            # Action 1: Comunicate a ping traffic
            return comunicate_ping_traffic_detected()
        if action == 2:
            # Action 2: Comunicate a UDP traffic
            return comunicate_udp_traffic_detected()  
        if action == 3:
            # Action 3: Comunicate a TCP traffic
            return comunicate_tcp_traffic_detected()
    
    def calculate_reward(self, action: int) -> float:
        """
        This modified function provides informative feedback, 
        rewarding the agent based on how close it was to the correct answer.
        """
        traffic_type = self.generated_traffic_type
        if action == traffic_type:
#            self.correct_predictions += 1
            return 1.0  # Max reward for correct prediction
        if abs(action - traffic_type) == 1:
            return -1.0  # Partial penalty for close predictions
        if abs(action - traffic_type) == 2:
            return -1.0  # Higher penalty for farther predictions
        return -1.0  # -0.9 Max penalty for completely incorrect predictions
        

    def evaluate_episode(self, episode, cumulative_reward, exploration_count=0, exploitation_count=0, ground_truth = None, predicted= None):
        # Calculate and store metrics at the end of the episode with library sklearn
        if ground_truth is None:
            ground_truth = self.ground_truth
        if predicted is None:
            predicted = self.predicted            
        accuracy_episode = accuracy_score(ground_truth, predicted)
        precision_episode, recall_episode, f1_score_episode, _ = precision_recall_fscore_support(ground_truth, predicted, average='weighted', zero_division=0.0)
        self.metrics['accuracy'].append(accuracy_episode)
        self.metrics['precision'].append(precision_episode)
        self.metrics['recall'].append(recall_episode)
        self.metrics['f1_score'].append(f1_score_episode)
        
        # Calculate and store train type (exploration/exploitation) at the end of the episode
        if exploration_count > 0 or exploitation_count > 0 :
            self.train_types['explorations'].append(exploration_count)
            self.train_types['exploitations'].append(exploitation_count)
            self.train_types['steps'].append(self.current_step)
        
        if (episode > 0):
            if (self.src_host is not None):
                i_src_host = self.src_host.name #int(self.src_host.name.replace('h','').replace('iot',''))
            else:
                i_src_host = '0'
            if (self.dst_host is not None):
                i_dst_host = self.dst_host.name #int(self.dst_host.name.replace('h','').replace('iot',''))
            else:
                i_dst_host = '0'

            self.indicators.append({
                'episode': episode,
                'steps': self.current_step,
                'correct_predictions': self.correct_predictions,
                'packets_received': self.state[0].item(),
                'bytes_received': self.state[2].item(),
                'packets_transmitted': self.state[1].item(),
                'bytes_transmitted': self.state[3].item(),
                'cumulative_reward': cumulative_reward,
                'traffic_type': self.generated_traffic_type,
                'i_src_host': i_src_host,
                'i_dst_host': i_dst_host
            })

            

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
