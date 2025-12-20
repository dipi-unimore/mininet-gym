from stable_baselines3.common.callbacks import BaseCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utility.constants import SystemLevels
from utility.my_log import notify_client, set_log_level, information
import numpy as np
import torch

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """    
    def __init__(self, 
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
             
        #
        self.train_types = {'explorations': [], 'exploitations': [], 'steps' :[]}
        self.indicators = []
        
        # Metrics tracking
        self.metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}
        self.rewards = []
        self.ground_truth = []
        self.predicted = []     
        #self.current_step = 0
        self.correct_predictions = 0
        self.episode_statuses = []
         
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
        self.number_actions = self.env.actions_number            
        self.count_actions = {i: 0 for i in range(self.number_actions)}
        self.exploration_count,  self.exploitation_count = 0, 0  # Reset counters for each episode
        self.ground_truth = []
        self.predicted = []
        self.correct_predictions = 0

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
        
        obs = self.locals.get("new_obs")  # Observation after the last action
        
        # # Convert observation to PyTorch tensor
        # if obs is not None:
        #     obs_tensor = torch.tensor(obs, dtype=torch.float32)
        #     # Compute Q-values using the model's policy
        #     q_values = self.model.policy.q_net(obs_tensor)
        #     # Log the Q-values for debugging
        #     #information(f"Q-values: {q_values}")        
        
        # Store indicators and data for metrics
        #self.training_env.envs[0].env custom_env
        action = self.locals["actions"][0]
        self.count_actions[action] += 1        
        ground_truth_step = self.locals["infos"][0]["Ground_truth_step"]
        predicted_step = self.locals["infos"][0]["Predicted_step"]
        self.action_correct = self.locals["infos"][0]["action_correct"]
        self.truncated = self.locals["infos"][0]["TimeLimit.truncated"]
        self.done = self.locals["dones"]
        self.correct_predictions += 1 if self.done else 0
        #self.current_step = self.locals["num_collected_steps"]
        self.ground_truth.append(ground_truth_step)
        self.predicted.append(predicted_step) 
        reward = self.locals["rewards"][0]  # assuming a single environment
        self.episode_rewards.append(reward)
        status = dict(self.env.global_state.status) #to detach to memory
        if bool(self.action_correct):
           self.correct_predictions+= 1
        self.episode_statuses.append(status)
        step_data = {
            'episode': self.episode,
            'step': self.current_step,
            'status': {'id': status["id"], 'text': status["status"]},
            'receivedPackets': status['received_packets'],
            'receivedPacketsPercentageChange': status['received_packets_percentage_change'],
            'receivedBytes': status['received_bytes'],
            'receivedBytesPercentageChange': status['received_bytes_percentage_change'],
            'transmittedPackets': status['transmitted_packets'],
            'transmittedPacketsPercentageChange': status['transmitted_packets_percentage_change'],
            'transmittedBytes': status['transmitted_bytes'],
            'transmittedBytesPercentageChange': status['transmitted_bytes_percentage_change'],
            'action': {'choosen': status["action_choosen"], 'isCorrect': status["action_correct"]},               
            'correctPredictions': self.correct_predictions,        
            'reward': reward
        }
        #name = self.locals['tb_log_name']
        notify_client(level=SystemLevels.DATA, agent_name = self.name, step_data = step_data)
        # if not hasattr(self, 'steps_data'):
        #     self.steps_data = []
        # self.steps_data.append(step_data)
        
        
        self.env.execute_action(action, show_action = self.show_action, name = self.name, reward = reward)            
       
        # Custom function after each step
        if self.step_end:
            self.step_end(self)
        return True  # Continue training, False top training   
  
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
            
    def evaluate_episode(self, episode, cumulative_reward):
        # Calculate and store metrics at the end of the episode with library sklearn
        # ground_truth = [1 if item[0] == 0 else 0 for item in self.ground_truth]   
        # predicted = [1 if item[0] == 0 else 0 for item in self.predicted]   
        ground_truth = [next((i for i, val in enumerate(item) if val == 1), -1) for item in self.ground_truth]   
        predicted = [next((i for i, val in enumerate(item) if val == 1), -1) for item in self.predicted]                                
        accuracy_episode = accuracy_score(ground_truth, predicted)
        precision_episode, recall_episode, f1_score_episode, _ = precision_recall_fscore_support(ground_truth, predicted, average='weighted', zero_division=0.0)
        self.metrics['accuracy'].append(accuracy_episode)
        self.metrics['precision'].append(precision_episode)
        self.metrics['recall'].append(recall_episode)
        self.metrics['f1_score'].append(f1_score_episode)
        
        
        if (episode > 0):
            self.indicators.append({
                'episode': episode,
                'steps': self.num_timesteps,
                'correct_predictions': self.correct_predictions,
                'episode_statuses': self.episode_statuses,
                'cumulative_reward': cumulative_reward,
            })    
    
    def print_metrics(self, cumulative_reward):
        accuracy = self.metrics['accuracy'][-1]
        precision = self.metrics['precision'][-1]
        recall = self.metrics['recall'][-1]
        f1_score = self.metrics['f1_score'][-1]
        if cumulative_reward is None:
            information(f"Accuracy {accuracy * 100 :.2f}%\nPrecision {precision * 100 :.2f}%\nRecall {recall * 100 :.2f}%\nF1-score {f1_score * 100 :.2f}%\n")    
        else:
            information(f"\n\t\tAccuracy {accuracy * 100 :.2f}%\n\t\tPrecision {precision * 100 :.2f}%\n\t\tRecall {recall * 100 :.2f}%\n\t\tF1-score {f1_score * 100 :.2f}%\n\t\tCumulative reward {cumulative_reward}\n", self.locals['tb_log_name'])
    
    
    def get_metrics(self):
        return self.metrics['accuracy'][-1], self.metrics['precision'][-1], self.metrics['recall'][-1], self.metrics['f1_score'][-1]


    # def before_episode(self):
    #     self.episode_rewards=[]
    #     self.episode_statuses=[]
    #     self.episode += 1         
    #     self.ground_truth=[]
    #     self.predicted=[]
    #     # callback_self.data_count_actions = {key: {0: 0, 1: 0, 2: 0, 3: 0} for key in  range(self.env.actions_number)}
    #     # if hasattr(self, 'model'):
    #     #     callback_self.count_actions = {action: 0 for action in range(self.training_env.action_space.n)}          
    #     self.env.early_exit = True
    #     #self.env.max_steps = callback_self.model.total_timesteps
    #     information(f"************* Episode {self.episode} *************\n", self.locals['tb_log_name'])         
    #     #return self.env.actions_number       
        
    def after_episode(self):
        cumulative_reward = sum(self.episode_rewards)
        self.evaluate_episode(self.episode, cumulative_reward)
        self.print_metrics(cumulative_reward)
        # information(f"Evalueting...")     
        # mean_reward, std_reward = evaluate_policy(callback_self.model, self.env, n_eval_episodes=1)
        # information(f"{mean_reward} {std_reward}")
        information(f"*** Episode {self.episode} finished ***\n", self.locals['tb_log_name'])             
                