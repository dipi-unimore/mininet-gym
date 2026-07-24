import os, time
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, DQN, A2C
from reinforcement_learning.agents.custom_policy import CustomDQNPolicy
from reinforcement_learning.agents.qlearning_agent import QLearningAgent
from reinforcement_learning.agents.sarsa_agent import SARSAAgent
from reinforcement_learning.agents.custom_callback import CustomCallback
from reinforcement_learning.agents.supervised_agent import SupervisedAgent
from reinforcement_learning.scenarios.marl.constants import COORDINATOR
from reinforcement_learning.scenarios.marl.network_env_marl_attack_detect import NetworkEnvMarlAttackDetect
from reinforcement_learning.network_env import NetworkEnv
from utility.constants import *
from utility.my_files import find_latest_file
from utility.my_log import information, debug

import torch.nn as nn


class AgentManager:
    def __init__(self, env : NetworkEnv, config): 
        self.env = env
        
        self.agents_params = [agent for agent in config.agents if agent.enabled]
        self.data_traffic_file = config.env_params.data_traffic_file
        self.gym_type = config.env_params.gym_type
        execution_dir = os.getcwd()
        self.training_directory = os.path.join(execution_dir, config.training_directory)      
        self.net_config_filter = config.net_config_filter
        self.episode = 0
        self.episodes = config.env_params.episodes
        self.test_episodes = config.env_params.test_episodes
        
        information("Check environment\n")
        if self.env is None:
            raise ValueError("The environment can't be None. Create environment")
        elif isinstance(self.env, NetworkEnvMarlAttackDetect):
            information("Marl environment\n")
            for name,env in self.env.host_envs.items():
                information(f"Host {name} environment\n")
                if config.env_params.must_check_env:
                    check_env(env, warn=True)
                else:
                    time.sleep(1.5)
                break
            information(f"Coordinator environment\n")
            if config.env_params.must_check_env:
                check_env(self.env.coordinator_env, warn=True)
            else:
                time.sleep(1.5)
        elif self.gym_type.startswith(MARL_PZ):
            # parallel_api_test + check_env(SingleAgentView) run in marl_pz_main
            # after the scenario is loaded so env.reset() has data to work with.
            information("Marl PZ environment — PZ/SB3 checks deferred to after scenario load\n")
            time.sleep(1.5)
        else:
            if config.env_params.must_check_env:
                check_env(env, warn=True)
            else:
                time.sleep(1.5)
        self.env.initialize_storage()
        information("Check environment finished\n")
        
        if len(self.agents_params)>0:
            self.create_agents()
        else:
            raise Exception(f"In config/default.yaml insert at least one agent and its params")
    
    def create_marl_agent(self, agent_param ):
        agent_param.instances = {}
        for host_name,env in self.env.host_envs.items():
            instance, custum_callback, is_custom_agent =  self.create_agent(agent_param, env, name = host_name )
            instance.is_team_member = True
            instance.is_team_coordinator = False

            agent_param.instances[host_name] = {
                'instance': instance,
                'custom_callback': custum_callback,
                'is_custom_agent': is_custom_agent,
                'max_steps' : self.env.max_steps
            }
        instance, custum_callback, is_custom_agent =  self.create_agent(agent_param, self.env.coordinator_env, name = COORDINATOR )
        instance.is_team_member = True
        instance.is_team_coordinator = True
        agent_param.instances[COORDINATOR] = {
                'instance': instance,
                'custom_callback': custum_callback,
                'is_custom_agent': is_custom_agent,
                'max_steps' : self.env.max_steps
        }          
        
            
    def create_agent(self, agent_param, env=None, name = None): 
        if env is None:
            env = self.env
        algorithm = agent_param.algorithm
        if not hasattr(agent_param, 'episodes') or agent_param.episodes is None:
            agent_param.episodes = self.episodes
        model = None
        #use lower case for algorithm names check
        if algorithm.lower() == ALGO_Q_LEARNING:
            if env.observation_space.shape[0] > 64:
                raise Exception(f"Custom agents with Q-Learning or SARSA algorithms are not supported in this env, the observation is too big for Q-Table (max 64, so max 8 hosts * 8 features, but it's better to avoid and use Deep algorithms).")
            return self.create_custom_agent(QLearningAgent, env, agent_param, name), {}, True
        elif algorithm.lower() == ALGO_SARSA:
            if env.observation_space.shape[0] > 64:
                raise Exception(f"Custom agents with Q-Learning or SARSA algorithms are not supported in this env, the observation is too big for Q-Table (max 64, so max 8 hosts * 8 features, but it's better to avoid and use Deep algorithms).")
            return self.create_custom_agent(SARSAAgent, env, agent_param, name), {}, True       
        elif algorithm.lower() == ALGO_SUPERVISED:
            split_ratio = getattr(agent_param, 'train_test_split_ratio', 0.20)
            agent_name = agent_param.name if name is None else f"{agent_param.name}_{name}"
            return SupervisedAgent(env.gym_type, self.data_traffic_file,
                                   train_test_split_ratio=split_ratio,
                                   name=agent_name), {}, False        
        elif algorithm.lower() == ALGO_PPO:
            try:
                if agent_param.load:
                    model = self.load_agent_model(agent_param, name)
            except Exception as e: 
                debug(f"No {algorithm} model file\n")
            if model is None:
                model =  PPO('MlpPolicy', env, device='cpu', 
                        policy_kwargs = dict(net_arch=agent_param.net_arch),                             
                        n_steps = agent_param.n_steps, 
                        learning_rate=agent_param.learning_rate, 
                        gamma=agent_param.gamma, 
                        ent_coef=agent_param.ent_coef, 
                        verbose=1)
            cc = CustomCallback(env = env, agent_param=agent_param, name=name)
            model.name = cc.name               
            return model, cc, False
            
        elif algorithm.lower() == ALGO_DQN:
            try: 
                if agent_param.load:               
                    model = self.load_agent_model(agent_param, name)
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
            cc = CustomCallback(env = env, agent_param=agent_param, name=name)
            model.name = cc.name            
            return model, cc, False
            
        elif algorithm.lower() == ALGO_A2C:
            try:
                if agent_param.load:
                    model = self.load_agent_model(agent_param, name)
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
            cc = CustomCallback(env = env, agent_param=agent_param, name=name)
            model.name = cc.name               
            return model, cc, False
            
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")      
        
    def create_custom_agent(self, agent_class, env, agent_param, host_name):
        """
        Create a custom RL agent (Q-Learning, SARSA) for a given host.

        For the ATTACKS_HO scenario the environment is a PerHostScanWrapper
        that decomposes each real step into num_hosts micro-steps.  Each
        micro-step calls exploration_rate *= exploration_decay, so the decay
        configured in the yaml (which the user thinks about in terms of real
        steps) would be applied num_hosts times per real step — causing
        exploration to collapse num_hosts times faster than expected.

        We correct this automatically:
            adjusted_decay = nominal_decay ^ (1 / num_hosts)
        so that after one full round of num_hosts micro-steps the product is:
            adjusted_decay ^ num_hosts = nominal_decay
        exactly matching the user's intent.

        The original nominal value is logged so the user can see what happened.

        Args:
            agent_class: QLearningAgent or SARSAAgent class.
            env:         The environment (may be a PerHostScanWrapper).
            agent_param: Agent configuration params.
            host_name:   Host name for multi-agent scenarios, None otherwise.

        Returns:
            model: Instantiated and optionally loaded agent.
        """
        from reinforcement_learning.scenarios.attack_detect_host_observable.per_host_scan_wrapper import (
            PerHostScanWrapper,
        )
        from utility.my_log import information
        from colorama import Fore

        # Adjust exploration_decay for per-host micro-step scenarios.
        # We mutate a copy of the param so other agents are not affected.
        if isinstance(env, PerHostScanWrapper) and env.num_hosts > 1:
            nominal_decay = agent_param.exploration_decay
            num_hosts     = env.num_hosts
            adjusted_decay = nominal_decay ** (1.0 / num_hosts)
            # Temporarily override for this agent's construction
            agent_param.exploration_decay = adjusted_decay
            information(
                f"{Fore.CYAN}[AgentManager] {agent_param.name}: "
                f"exploration_decay adjusted from {nominal_decay:.6f} "
                f"to {adjusted_decay:.6f} "
                f"(num_hosts={num_hosts}, "
                f"effective per-round decay = {adjusted_decay**num_hosts:.6f})"
                f"{Fore.WHITE}\n"
            )

        model = agent_class(env, agent_param)

        # Restore the nominal value in agent_param so config dumps/saves
        # still show the user-facing value, not the internal adjusted one.
        if isinstance(env, PerHostScanWrapper) and env.num_hosts > 1:
            agent_param.exploration_decay = nominal_decay

        try:
            if agent_param.load:
                path = (
                    agent_param.load_dir
                    if agent_param.load_dir is not None
                    else find_latest_file(
                        self.training_directory, host_name, 'json',
                        self.net_config_filter
                    )
                )
                if host_name is not None:
                    filename = (
                        f"{self.training_directory}/{path}"
                        f"/{agent_param.name}_{host_name}.json"
                    )
                else:
                    filename = f"{self.training_directory}/{path}"
                model.load(filename)
        except Exception as e:
            debug(f"No {agent_param.name} model file\n")

        if host_name is not None:
            model.name = f"{agent_param.name}_{host_name}"

        return model
    
    def create_marl_pz_agent(self, agent_param):
        """
        Create per-agent instances for the marl_pz scenario.

        Each agent in possible_agents (hosts + coordinator) gets its own
        SingleAgentView and an associated model instance.
        """
        from reinforcement_learning.scenarios.marl_pz.marl_pz_env import (
            MarlPzEnv, SingleAgentView,
        )
        from reinforcement_learning.scenarios.marl_pz.constants import (
            COORDINATOR as MARL_PZ_COORDINATOR,
            NORMALIZED, RAW,
        )

        env: MarlPzEnv = self.env
        agent_param.instances = {}

        for agent_id in env.possible_agents:
            is_coordinator = (agent_id == MARL_PZ_COORDINATOR)

            # Determine state_mode: tabular agents work with RAW obs (they
            # discretize manually); deep agents need NORMALIZED.
            algo = agent_param.algorithm.lower()
            is_tabular = algo in (ALGO_Q_LEARNING, ALGO_SARSA)
            state_mode = RAW if is_tabular else NORMALIZED

            single_view = SingleAgentView(env, agent_id, state_mode=state_mode)

            instance, custom_callback, is_custom_agent = self.create_agent(
                agent_param, env=single_view, name=agent_id
            )
            instance.is_team_member      = True
            instance.is_team_coordinator = is_coordinator

            agent_param.instances[agent_id] = {
                'instance':        instance,
                'custom_callback': custom_callback,
                'is_custom_agent': is_custom_agent,
                'max_steps':       env.max_steps,
                'single_view':     single_view,
            }

    def create_agents(self):
        for agent_param in self.agents_params:
            if self.gym_type.startswith(MARL_PZ):
                self.create_marl_pz_agent(agent_param)
            elif self.gym_type.startswith("marl"):
                self.create_marl_agent(agent_param)
            else:
                agent_param.instance, agent_param.custom_callback, agent_param.is_custom_agent = self.create_agent(agent_param)
                agent_param.max_steps = self.env.max_steps
    
    def load_agent_model(self, agent_param, name=None):
        """
        Load an RL model (DQN, PPO, A2C) for a given agent.

        Args:
            agent: Object containing 'load' and 'load_dir' attributes.
            algorithm_name (str): One of 'DQN', 'PPO', 'A2C'.
            env: The environment to bind to the model.
            training_directory (str): Path to the directory with saved models.
            net_config_filter (str): Optional filter for file selection.
            name (str, optional): Host name only for multi-agent scenarios.

        Returns:
            model: Loaded RL model ready to use.
        """
        # Determine path to model
        algorithm = agent_param.algorithm
        env = self.env
        training_directory = self.training_directory 
        net_config_filter = self.net_config_filter
        # Check if the algorithm is supported and set the path to the model        
        if algorithm not in [ALGO_DQN, ALGO_PPO, ALGO_A2C]:
            raise ValueError(f"Unsupported algorithm: {algorithm}")    
        
        try:
            if agent_param.load_dir == 'None':
                #it's bettr to show an error if the model saved file has not beeen specified, instead of looking for the latest file and maybe loading a wrong model
                raise Exception(f"Model file not specified for loading. Please set load_dir in config/default.yaml")
                # path = find_latest_file(training_directory, agent_param.name, 'zip', net_config_filter)
                # agent_param.load_dir = path[:-4]  # Strip .zip
            else:
                path = f"{self.training_directory}/{agent_param.load_dir}"
                if name is not None:
                    path = f"{path}/{agent_param.name}_{name}.zip"

            assert os.path.isfile(path), f"Model file not found at {path}"
        except AssertionError as e:
            debug(str(e))
            return None
        
        # Load appropriate model
        if algorithm == ALGO_DQN:
            model = DQN.load(path, env=env)
        elif algorithm == ALGO_PPO:
            model = PPO.load(path, env=env)
        elif algorithm == ALGO_A2C:
            model = A2C.load(path, env=env)
        return model

    
    # def before_episode(self, callback):
    #     callback.episode_rewards=[]
    #     callback.episode_statuses=[]
    #     callback.ground_truth=[]
    #     callback.predicted=[]        
    #     self.env.early_exit = True
    #     information(f"************* Episode {callback.episode} *************\n", callback.locals['tb_log_name'])         
   
    # def after_episode(self, callback):
    #     cumulative_reward = sum(callback.episode_rewards)
    #     callback.evaluate_episode(callback.episode, cumulative_reward)
    #     callback.print_metrics(cumulative_reward)
    #     # information(f"Evalueting...")     
    #     # mean_reward, std_reward = evaluate_policy(callback_self.model, self.env, n_eval_episodes=1)
    #     # information(f"{mean_reward} {std_reward}")
    #     information(f"*** Episode {callback.episode} finished ***\n", callback.locals['tb_log_name'])             
