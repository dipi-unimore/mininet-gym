import time
from colorama import Fore
from stable_baselines3.common.callbacks import BaseCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from reinforcement_learning.marl.constants import COORDINATOR
from utility.constants import SystemLevels
from utility.my_log import debug, notify_client, information

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """    
    def __init__(self, env = None, agent_param= None, name=None,
                 training_start=None, training_end=None,
                 rollout_start=None, rollout_end=None,
                 step_start=None, step_end=None,
                 verbose: int = 0):
        #super(CustomCallback, self).__init__(verbose)
        super().__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]   
             
        #custom variables
        self.env = env
        self.episodes = agent_param.episodes
        self.show_action = agent_param.show_action
        if name is not None:
            self.name = f"{agent_param.name}_{name}" 
        else:  
            self.name = agent_param.name 
            
        self.train_types = {'explorations': [], 'exploitations': [], 'steps' :[]}
        self.indicators = []
        
        # Metrics tracking
        self.metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}
        self.rewards = []
        self.ground_truth = []
        self.predicted = []     
        self.current_step = 0
        self.cumulative_reward = 0  
        self.episode_rewards = []
        self.episode = 0 
        self.episode_statuses = []
        self.correct_predictions = 0
         
        self.rollout_start = rollout_start
        self.step_start = step_start
        self.step_end = step_end
        self.rollout_end = rollout_end        
        self.training_start = training_start
        self.training_end = training_end
        self.rollout = 0

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """     
        # if hasattr(self.model, 'exploration_rate'):
        #     self.model.exploration_rate =  self.model.exploration_rate * 0.9
        #     self.model.exploration_fraction = self.model.exploration_fraction * 0.9
        #     self.model.exploration_initial_eps = self.model.exploration_initial_eps * 0.9 #Start with exploration rate = 1
        #     self.model.exploration_final_eps = self.model.exploration_final_eps * 0.9 
        #     self.model.gamma = self.model.gamma * 0.9
            
        if self.training_start:
            self.training_start(self)

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction using the current policy.
        This event is triggered before collecting new samples.
        for DQN is before every 4 steps
        """       

        if self.rollout_start:
            self.rollout_start(self)
    
    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.
        :return: If the callback returns False, training is aborted early.
        """
        if hasattr(self.env, 'stop_event') and self.env.stop_event.is_set():
            return False     # Stop training if the stop event is set
        while hasattr(self.env, 'pause_event') and self.env.pause_event.is_set():
            time.sleep(0.1)  # Pause training if the pause event is set
            
        obs = self.locals.get("new_obs")  # Observation after the last action
        #print(obs)
        # # Convert observation to PyTorch tensor
        # if obs is not None:
        #     obs_tensor = torch.tensor(obs, dtype=torch.float32)
        #     # Compute Q-values using the model's policy
        #     q_values = self.model.policy.q_net(obs_tensor)
        #     # Log the Q-values for debugging
        #     #information(f"Q-values: {q_values}")        
        
        # Store indicators and data for metrics
        action = self.locals["actions"][0]
        self.count_actions_by_type[action] += 1        
        self.current_step += 1 #self.locals["num_collected_steps"]
        reward = self.locals["rewards"][0]  # assuming a single environment
        self.episode_rewards.append(reward)
        self.cumulative_reward += reward
        #next_status = dict(self.env.global_state.status) #to detach to memory
        self.manage_step_data(action, reward, self.locals["infos"][0])
 
        # Custom function after each step
        if self.step_end:
            self.step_end(self)
            
        if self.current_step == self.env.max_steps:
            return False           
        
        return True  # Continue training, False stop training   

    def manage_step_data(self, action, reward, infos):
        status = infos["status"]
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
            'action': {'choosen': int(action), 'isCorrect': bool(infos['is_correct_action'])},  
            'correctPredictions': self.correct_predictions,         
            'reward': round(float(reward),1)
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
        #print(Fore.CYAN + f"E/S {self.episode}/{self.current_step} - Status {status['status']} - Action chosen: {action} - Reward: {reward} - Is correct: {infos['is_correct_action']}"+Fore.WHITE)
        #print(Fore.CYAN + f"Packets: {status.get('packets', 0)} - Bytes: {status.get('bytes',0)}"+Fore.WHITE)
        
    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy. 
        for DQN is after 4 steps
        """
        #print("_on_rollout_end")
        if self.rollout_end:
            self.rollout_end(self)

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """  
        # This function is called at the end of training
        if self.training_end:
            self.training_end(self)
            
    def evaluate_episode(self, exploration_count=0, exploitation_count=0):
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
        
        if (self.episode > 0):
            self.indicators.append({
                'episode': self.episode,
                'steps': self.current_step,
                'correct_predictions': self.correct_predictions,
                'episode_statuses': self.episode_statuses,
                'cumulative_reward':  float(self.cumulative_reward),
            })             
            try :  
                metrics={
                        'episode': self.episode,
                        'steps': self.current_step,
                        'correctPredictions': self.correct_predictions,                        
                        'accuracy': accuracy_episode,
                        'precision': precision_episode,
                        'recall': recall_episode,
                        'f1Score': f1_score_episode,
                        'cumulativeReward': float(self.cumulative_reward)}
                notify_client(level=SystemLevels.DATA, agent_name = self.name, metrics = metrics)   
            except Exception as e:
                debug(Fore.RED + f"Error notifying client with metrics: {e}\n"+Fore.WHITE)
    
    def print_metrics(self):
        accuracy = self.metrics['accuracy'][-1]
        precision = self.metrics['precision'][-1]
        recall = self.metrics['recall'][-1]
        f1_score = self.metrics['f1_score'][-1]
        if self.cumulative_reward is None:
            information(f"Accuracy {accuracy * 100 :.2f}%\nPrecision {precision * 100 :.2f}%\nRecall {recall * 100 :.2f}%\nF1-score {f1_score * 100 :.2f}%\n")    
        else:
            information(f"\n\t\tAccuracy {accuracy * 100 :.2f}%\n\t\tPrecision {precision * 100 :.2f}%\n\t\tRecall {recall * 100 :.2f}%\n\t\tF1-score {f1_score * 100 :.2f}%\n\t\tCumulative reward {self.cumulative_reward}\n", self.name)

        
    def episode_reset(self):
        self.ground_truth=[]
        self.predicted=[]
        self.rewards=[]
        self.current_step=0
        self.correct_predictions=0
        self.cumulative_reward=0
        self.done = self.truncated = False
        self.count_actions_by_type =  {i: 0 for i in range(self.env.action_space.n)}        
        self.episode_statuses = []
        self.exploration_count,  self.exploitation_count = 0, 0  # Reset counters for each episode
        #state, _ = self.env.reset()         
            
    def get_metrics(self):
        return self.metrics['accuracy'][-1], self.metrics['precision'][-1], self.metrics['recall'][-1], self.metrics['f1_score'][-1]

    def before_episode(self, episode):
        self.episode = episode               
        information(f"************* Episode {self.episode} *************\n", self.name)         
        self.episode_reset()    
        
    def after_episode(self):
        self.evaluate_episode()
        self.print_metrics()
        information(f"*** Episode {self.episode} finished ***\n", self.name)   
     
                