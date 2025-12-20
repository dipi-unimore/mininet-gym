from abc import ABC, abstractmethod
import time
import numpy as np
import json
from reinforcement_learning.marl.constants import COORDINATOR
from reinforcement_learning.network_env import NetworkEnv
from utility.constants import SystemLevels
from utility.my_log import information, debug, notify_client
from colorama import Fore
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class BaseAgent(ABC):
    def __init__(self, env: NetworkEnv, params):
        """
        Base class for RL agents. Handles initialization, saving, and loading of models.
        """
        self.env = env
        self.learning_rate = params.learning_rate
        self.discount_factor = params.discount_factor
        self.exploration_rate = params.exploration_rate
        self.exploration_decay = params.exploration_decay
        self.is_exploitation = False

        # Initialize Q-table dimensions based on action space and number of discrete states
        self.q_table = np.zeros((self.env.n_bins,) * self.env.observation_space.shape[0] + (self.env.action_space.n, ), dtype=float)
        
        #
        self.train_types = {'explorations': [], 'exploitations': [], 'steps' :[]}
        self.indicators = []
        
        # Metrics tracking
        self.metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}
        self.rewards = []
        self.ground_truth = []
        self.predicted = []     
        self.current_step = 0
        self.correct_predictions = 0
        
        self.name = params.name
        self.show_action = params.show_action
                
    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            self.exploration_count += 1  # Track exploration
            self.is_exploitation = False            
            action =  self.env.action_space.sample().item()
        else:
            self.exploitation_count += 1  # Track exploitation
            self.is_exploitation = True   
            action = np.argmax(self.q_table[state]).item()
            # max_val = np.max(self.q_table[state,:])
            # find_max_val = np.where(self.q_table[state, :] == max_val)
            # action = np.random.choice(find_max_val[0]).item()
        debug(Fore.GREEN + f"Agent Action {action} in " + ("Exploitation" if self.is_exploitation else "Exploration") +"\n")
        return action       

    def learn(self, episodes = 100, stop_event = None, t = None):
        self.stop_event = stop_event
        for episode in range(episodes):
            self.episode=episode+1
            information(f"************* Episode {self.episode} *************\n", self.name)
            #learn
            cumulative_reward, count_actions_by_type = self.train()
            if self.stop_event.is_set():
                break
            #get learning results
            exploration_count = self.exploration_count
            exploitation_count = self.exploitation_count
            #evaluating episode results
            if exploration_count > 0 or exploitation_count > 0:    
                information(Fore.YELLOW+f"Exploration used {exploration_count} times\n"+Fore.WHITE, self.name)
                information(Fore.BLUE+f"Exploitation used {exploitation_count} times\n"+Fore.WHITE, self.name)
            #the environment evaluates episode metrics, indicators etc
            self.evaluate_episode(self.episode, cumulative_reward, exploration_count, exploitation_count)    
            self.print_metrics(cumulative_reward)

            information(f"*** Episode {self.episode} finished ***\n", self.name) 
            if t is not None:
                time.sleep(t)  
    
    def get_metrics(self):
        return self.metrics['accuracy'][-1], self.metrics['precision'][-1], self.metrics['recall'][-1], self.metrics['f1_score'][-1]
       
       
    def manage_step_data(self, action, reward, infos, status):
        status['action_choosen'] = action
        status['traffic_type'] = infos['action_correct']
        self.episode_statuses.append(status)            
        self.rewards.append(reward)
        self.ground_truth.append(infos['Ground_truth_step'])
        self.predicted.append(infos['Predicted_step']) 
        if infos['is_correct_action']:
            self.correct_predictions+=1
        if self.name.endswith("coordinator"):
            debug(Fore.MAGENTA + f"Coordinator correct predictions so far: {self.correct_predictions}\n"+Fore.WHITE)
        step_data = {
            'episode': self.episode,
            'step': self.current_step,
            'status': {'id': infos['action_correct'], 'text': infos["text_action_correct"]},
            'action': {'choosen': action, 'isCorrect': infos['is_correct_action']},  
            'correctPredictions': self.correct_predictions,         
            'reward': reward
        }
        if hasattr(self, 'is_team_member') and hasattr(self, 'is_team_coordinator'):
            if self.is_team_member and not self.is_team_coordinator:
                step_data['host'] = self.name.split("_").pop()
            elif self.is_team_coordinator and self.is_team_member:
                step_data['host'] = COORDINATOR
        else:
            step_data['host'] = "single_agent" 
        if 'packets' in status:
            step_data.update({
                'packets': int(status['packets']),
                'bytes': int(status['bytes']),
                'packetsPercentageChange': float(status['packets_percentage_change']),
                'bytesPercentageChange': float(status['bytes_percentage_change'])
                })
        else:
            step_data.update({
                'receivedPackets': int(status['received_packets']),
                'receivedPacketsPercentageChange': float(status['received_packets_percentage_change']),
                'receivedBytes': int(status['received_bytes']),
                'receivedBytesPercentageChange': float(status['received_bytes_percentage_change']),
                'transmittedPackets': int(status['transmitted_packets']),
                'transmittedPacketsPercentageChange': float(status['transmitted_packets_percentage_change']),
                'transmittedBytes': int(status['transmitted_bytes']),
                'transmittedBytesPercentageChange': float(status['transmitted_bytes_percentage_change'])
            })
        notify_client(level=SystemLevels.DATA, agent_name = self.name, step_data = step_data)
        # if not hasattr(self, 'steps_data'):
        #     self.steps_data = []
        # self.steps_data.append(step_data)


    
    def evaluate_episode(self, episode, cumulative_reward, exploration_count=0, exploitation_count=0):
        # Calculate and store metrics at the end of the episode with library sklearn
        ground_truth = [next((i for i, val in enumerate(item) if val == 1), -1) for item in self.ground_truth]   
        predicted = [next((i for i, val in enumerate(item) if val == 1), -1) for item in self.predicted]                          
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
            self.indicators.append({
                'episode': episode,
                'steps': self.current_step,
                'correct_predictions': self.correct_predictions,
                'episode_statuses': self.episode_statuses,
                'cumulative_reward': cumulative_reward,
            })             
            try :  
                metrics={
                        'episode': episode,
                        'steps': self.current_step,
                        'correctPredictions': self.correct_predictions,                        
                        'accuracy': accuracy_episode,
                        'precision': precision_episode,
                        'recall': recall_episode,
                        'f1Score': f1_score_episode,
                        'cumulativeReward': cumulative_reward}
                notify_client(level=SystemLevels.DATA, agent_name = self.name, metrics = metrics)   
            except Exception as e:
                debug(Fore.RED + f"Error notifying client with metrics: {e}\n"+Fore.WHITE)
    
    def print_metrics(self, cumulative_reward):
        accuracy = self.metrics['accuracy'][-1]
        precision = self.metrics['precision'][-1]
        recall = self.metrics['recall'][-1]
        f1_score = self.metrics['f1_score'][-1]
        if cumulative_reward is None:
            information(f"Accuracy {accuracy * 100 :.2f}%\nPrecision {precision * 100 :.2f}%\nRecall {recall * 100 :.2f}%\nF1-score {f1_score * 100 :.2f}%\n")    
        else:
            information(f"\n\t\tAccuracy {accuracy * 100 :.2f}%\n\t\tPrecision {precision * 100 :.2f}%\n\t\tRecall {recall * 100 :.2f}%\n\t\tF1-score {f1_score * 100 :.2f}%\n\t\tCumulative reward {cumulative_reward}\n", self.name)

        
    def episode_reset(self, is_discretized_state):
        self.ground_truth=[]
        self.predicted=[]
        self.rewards=[]
        self.current_step=0
        self.correct_predictions=0
        cumulative_reward=0
        done = truncated = False
        count_actions_by_type =  {i: 0 for i in range(self.env.action_space.n)}        
        self.episode_statuses = []
        self.exploration_count,  self.exploitation_count = 0, 0  # Reset counters for each episode
        state, _ = self.env.reset(is_discretized_state = is_discretized_state)  
        return  cumulative_reward, state, done, truncated, count_actions_by_type
    
    
    @abstractmethod
    def train(self, num_episodes):
        """
        Placeholder for the `train` method. Must be implemented by derived classes.
        """
        pass
    
    def predict(self, state): 
        """
        receives normal state, discretizes it, and predicts action
        """                  
        discretized_state = self.env.get_discretized_state(state)
        return np.argmax(self.q_table[discretized_state]).item()

    def save(self, filename):
        """
        Save the Q-table and parameters to a file.
        """
        data = {
            'q_table': self.q_table.tolist(),  # Convert to list for JSON compatibility
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'exploration_rate': self.exploration_rate,
            'exploration_decay': self.exploration_decay
        }
        #drop_privileges("salvo")
        with open(filename + ".json", 'w') as file:
            json.dump(data, file)
        
        #regain_root()
        information(f"Model saved to {filename}\n",self.name)

    def load(self, filename):
        """
        Load the Q-table and parameters from a file.
        """
        if filename[-5:]!='.json':
            filename += ".json"
        with open(filename, 'r') as file:
            data = json.load(file)
        self.q_table = np.array(data['q_table'])
        self.learning_rate = data['learning_rate']
        self.discount_factor = data['discount_factor']
        self.exploration_rate = data['exploration_rate']
        self.exploration_decay = data['exploration_decay']
        information(f"Model loaded from {filename}\n",self.name)

    def discretize_state(self, state):
        """
        Convert a continuous state into discrete bins.
        """
        discrete_state = []
        for i, val in enumerate(state):
            bin_index = np.digitize(val, self.bins[i]) - 1
            bin_index = max(0, min(bin_index, self.n_bins - 1))
            discrete_state.append(bin_index)
        return tuple(discrete_state)
    
    def track_metrics(self, generated_traffic_type, predicted_traffic_type):
        ground_truth = [0, 0, 0, 0]
        predicted = [0, 0, 0, 0]

        ground_truth[generated_traffic_type] = 1
        predicted[predicted_traffic_type] = 1

        for gt, pred in zip(ground_truth, predicted):
            if gt == 1 and pred == 1:
                self.TP += 1  # True Positive: correctly predicted traffic
            elif gt == 0 and pred == 0:
                self.TN += 1  # True Negative: correctly predicted no traffic
            elif gt == 0 and pred == 1:
                self.FP += 1  # False Positive: predicted traffic is different
            elif gt == 1 and pred == 0:
                self.FN += 1  # False Negative: missed the actual traffic

   
    def calculate_metrics(self):
        TP, FP, TN, FN = self.TP, self.FP, self.TN, self.FN
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        return accuracy, precision, recall, f1_score    
