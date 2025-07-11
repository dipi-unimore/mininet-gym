import os
from reinforcement_learning.custom_policy import CustomDQNPolicy
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_checker import check_env
from reinforcement_learning.qlearning_agent import QLearningAgent
from reinforcement_learning.sarsa_agent import SARSAAgent
from reinforcement_learning.network_env import NetworkEnv
from reinforcement_learning.custom_callback import CustomCallback
from utility.my_files import find_latest_file
from utility.my_log import set_log_level, information, debug, error, notify_client
from supervised_agent import SupervisedAgent

from colorama import Fore, Back, Style
import numpy as np
import torch.nn as nn
import threading, time

class AgentManager:
    def __init__(self, env : NetworkEnv, config): 
        self.env = env
        self.test_episodes = config.test_episodes
        self.agents_params = [agent for agent in config.agents if agent.enabled]
        self.csv_file = config.env_params.csv_file
        self.gym_type = config.env_params.gym_type
        execution_dir = os.getcwd()
        self.training_directory = os.path.join(execution_dir, config.training_directory)      
        self.net_config_filter = config.net_config_filter
        self.episode = 0
        
        information("Check environment\n")
        check_env(env, warn=True)
        env.initialize_storage()
        information("Check environment finished\n")
        
        if len(self.agents_params)>0:
            self.create_agents()
        else:
            raise ValueError(f"In config.yaml insert at least one agent and its params")
            
    def create_agent(self, agent_param ):
        env = self.env
        algorithm = agent_param.algorithm
        name = agent_param.name
        model = None
        
        if algorithm == "Q-learning":
            model =  QLearningAgent(env, agent_param)
            try:            
                if agent_param.load:
                    path = agent_param.load_dir if agent_param.load_dir is not None else find_latest_file(self.training_directory,name,'json',self.net_config_filter)
                    model.load(self.training_directory+"/"+path)
            except:
                debug(f"No {name} model file\n")              
            return model, {}, True
        elif algorithm == "Sarsa":
            model =  SARSAAgent(env, agent_param)
            try:
                if agent_param.load:                
                    path = agent_param.load_dir if agent_param.load_dir is not None else  find_latest_file(self.training_directory,name,'json',self.net_config_filter)
                    model.load(self.training_directory+"/"+path)
            except:
                debug(f"No {name} model file\n")              
            return model, {}, True        
        elif algorithm == "Supervised":
            if self.gym_type.startswith("attacks"): #attack detect with or not with dataset
                return SupervisedAgent(), {}, False 
            return SupervisedAgent(self.csv_file), {}, False        
        elif algorithm == "PPO":
            try:
                model = self.load_agent_model(agent_param)
            except Exception as e: 
                debug(f"No {algorithm} model file\n")
            if model is None:
                model =  PPO('MlpPolicy', env, 
                        policy_kwargs = dict(net_arch=agent_param.net_arch),                             
                        n_steps = agent_param.n_steps, 
                        learning_rate=agent_param.learning_rate, 
                        gamma=agent_param.gamma, 
                        ent_coef=agent_param.ent_coef, 
                        verbose=1)
            cc = CustomCallback(training_start=self.before_episode, 
                                training_end=self.after_episode)
            cc.net_env = self.env
            return model, cc, False
            
        elif algorithm == "DQN":
            try:                
                model = self.load_agent_model(agent_param)
            except:
                debug(f"No {algorithm} model file\n")
            if model is None:
                model =  DQN(policy=CustomDQNPolicy,
                        #'MlpPolicy', 
                        env=env, 
                       policy_kwargs=dict(
                            net_arch=agent_param.net_arch,
                            activation_fn=nn.ReLU
                        ),                       
                       learning_rate=agent_param.learning_rate, 
                       gamma=agent_param.gamma, 
                       exploration_fraction=agent_param.exploration_fraction,  
                       exploration_initial_eps=agent_param.exploration_initial_eps, 
                       exploration_final_eps=agent_param.exploration_final_eps, 
                       buffer_size=agent_param.buffer_size,
                       learning_starts=agent_param.learning_starts,
                       batch_size=agent_param.batch_size,
                       target_update_interval=agent_param.target_update_interval,
                       verbose=1)
            else:
                model.learning_rate=agent_param.learning_rate 
                model.gamma=agent_param.gamma 
                model.exploration_fraction=agent_param.exploration_fraction  
                model.exploration_initial_eps=agent_param.exploration_initial_eps
                model.exploration_final_eps=agent_param.exploration_final_eps
                model.buffer_size=agent_param.buffer_size
                model.learning_starts=agent_param.learning_starts
                model.batch_size=agent_param.batch_size
                model.target_update_interval=agent_param.target_update_interval
                model.verbose=1                
            cc = CustomCallback(training_start=self.before_episode, 
                                training_end=self.after_episode)
            cc.net_env = self.env
            return model, cc, False
            
        elif algorithm == "A2C":
            try:
                model = self.load_agent_model(agent_param)
            except:
                debug(f"No {algorithm} model file\n")            

            if model is None:
                model = A2C('MlpPolicy', env,
                       policy_kwargs = dict(net_arch=agent_param.net_arch), 
                       n_steps = agent_param.n_steps, 
                       learning_rate=agent_param.learning_rate, 
                       gamma=agent_param.gamma, 
                       ent_coef=agent_param.ent_coef, 
                       verbose=1)
            cc = CustomCallback(training_start=self.before_episode, 
                                training_end=self.after_episode)
            cc.net_env = self.env
            return model, cc, False
            
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")      
        
    
    def create_agents(self):
        for agent_param in self.agents_params:
            agent_param.instance, agent_param.custom_callback, agent_param.is_custom_agent = self.create_agent(agent_param)
            agent_param.max_steps = self.env.max_steps
    
    def load_agent_model(self, agent_param):
        """
        Load an RL model (DQN, PPO, A2C) for a given agent.

        Args:
            agent: Object containing 'load' and 'load_dir' attributes.
            algorithm_name (str): One of 'DQN', 'PPO', 'A2C'.
            env: The environment to bind to the model.
            training_directory (str): Path to the directory with saved models.
            net_config_filter (str): Optional filter for file selection.

        Returns:
            model: Loaded RL model ready to use.
        """
        # Determine path to model
        algorithm = agent_param.algorithm
        env = self.env
        training_directory = self.training_directory 
        net_config_filter = self.net_config_filter
        # Check if the algorithm is supported
        # and set the path to the model        
        if algorithm not in ['DQN', 'PPO', 'A2C']:
            raise ValueError(f"Unsupported algorithm: {algorithm}")    
        if agent_param.load:
            if agent_param.load_dir == 'None':
                path = find_latest_file(training_directory, agent_param.name, 'zip', net_config_filter)
                agent_param.load_dir = path[:-4]  # Strip .zip
            else:
                path = self.training_directory+"/"+agent_param.load_dir

            # Load appropriate model
            if algorithm == 'DQN':
                model = DQN.load(path, env=env)
            elif algorithm == 'PPO':
                model = PPO.load(path, env=env)
            elif algorithm == 'A2C':
                model = A2C.load(path, env=env)
            return model
        else:
            return None
    
    def before_episode(self, callback):
        callback.episode_rewards=[]
        callback.episode_statuses=[]
        callback.ground_truth=[]
        callback.predicted=[]        
        self.env.early_exit = True
        information(f"************* Episode {callback.episode} *************\n", callback.locals['tb_log_name'])         
   
    def after_episode(self, callback):
        cumulative_reward = sum(callback.episode_rewards)
        callback.evaluate_episode(callback.episode, cumulative_reward)
        callback.print_metrics(cumulative_reward)
        # information(f"Evalueting...")     
        # mean_reward, std_reward = evaluate_policy(callback_self.model, self.env, n_eval_episodes=1)
        # information(f"{mean_reward} {std_reward}")
        information(f"*** Episode {callback.episode} finished ***\n", callback.locals['tb_log_name'])             
