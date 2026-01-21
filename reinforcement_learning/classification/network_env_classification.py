
from reinforcement_learning.agents.adversarial_agent import generate_random_traffic
from reinforcement_learning.classification.constants import AGENT_ACTIONS, REWARDS
from reinforcement_learning.classification.instant_state import InstantState
from reinforcement_learning.network_env import NetworkEnv, get_agent_name, get_normalized_state
from utility.constants import  GYM_TYPE, CLASSIFICATION_FROM_DATASET, CLASSIFICATION, NORMAL, TRAFFIC_TYPE_ID_MAPPING, SystemLevels, SystemModes, SystemStatus, TrafficTypes
from utility.my_files import save_data_to_file
from utility.my_log import debug, information, notify_client
from utility.network_configurator import comunicate_no_traffic_detected, comunicate_ping_traffic_detected, comunicate_tcp_traffic_detected, comunicate_udp_traffic_detected, format_bytes
from utility.params import Params, read_config_file
from gymnasium import spaces
from colorama import Fore
from os import error
import copy, time,  numpy as np


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
        self.threshold_packets = params.classification.thresholds.packets
        self.threshold_bytes = params.classification.thresholds.bytes
        self.threshold_var_packets = params.classification.thresholds.var_packets
        self.threshold_var_bytes = params.classification.thresholds.var_bytes    
        
        # Define action and observation space, gym.spaces objects
        #self.low = np.array([0,-self.threshold_var_packets,0,-self.threshold_var_bytes])        
        #self.high = np.array([self.threshold_packets, self.threshold_var_packets, self.threshold_bytes, self.threshold_var_bytes])        #fixed for every network
        self.low = np.array([0,0,0,0])        
        self.high = np.array([self.threshold_packets, self.threshold_packets, self.threshold_bytes, self.threshold_bytes]) 
        self.observation_space = spaces.Box(low=0, high=self.high, shape=(len(self.low),), dtype=np.float64)
        
        # Define the number of discrete bins for each observation dimension
        self.n_bins = params.n_bins
        self.low_to_normalize = self.low
        self.high_to_normalize = self.high
        low = self.low 
        high = np.floor(np.log10(self.high)).astype(int)
        self.bins = [np.logspace(low[i], high[i], self.n_bins) for i in range(self.observation_space.shape[0])]
        
        self.global_prev_state = self.global_state = InstantState(self.hosts)   
            
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
        src_name = src_host.name if type(src_host) is not str and src_host is not None else src_host
        dst_name = dst_host.name if type(dst_host) is not str and dst_host is not None else dst_host
        try:
            for host in self.hosts:
                if src_host is not None and host.name == src_name:
                    statuses[host.name] = generated_traffic_type_text
                    host_tasks[host.name] = {'taskType': NORMAL, 'trafficType': generated_traffic_type_text, 'destination': dst_name}
                elif dst_host is not None and  host.name == dst_name:
                    statuses[host.name] = generated_traffic_type_text
                    host_tasks[host.name] = {'taskType': NORMAL, 'trafficType': TrafficTypes.NONE} 
                else:
                    statuses[host.name] = TrafficTypes.NONE
                    host_tasks[host.name] = {'taskType': NORMAL, 'trafficType': TrafficTypes.NONE}        
                host_tasks[host.name]["linkStatus"] = 1  # link always ON 
            notify_client(level=SystemLevels.DATA, host_tasks = host_tasks )
        except Exception as e:
            error(Fore.RED + f"Error updating host statuses: {e}\n" + Fore.WHITE)
        return statuses
                
    def update_state(self, episode = None):
        """
        update state Evalueting gym type
        """                 
        if self.gym_type == GYM_TYPE[CLASSIFICATION] and self.state is not None:    #classification
  
            self.generated_traffic_type_text, self.src_host, self.dst_host = generate_random_traffic(self.net)  
            self.generated_traffic_type = TRAFFIC_TYPE_ID_MAPPING[self.generated_traffic_type_text]
            # wait for traffic complete generation
            if self.generated_traffic_type != TrafficTypes.NONE:
                time.sleep((self.generated_traffic_type+0.5))
            if self.read_from_network(): 
                self.evaluate_traffic() 
            
        elif self.gym_type == GYM_TYPE[CLASSIFICATION_FROM_DATASET]: #classification_from_dataset
            try: 
                if not hasattr(self, 'df'):
                    self.state = self.initial_state
                elif len(self.df) > 0:
                    status=self.df.pop(0)
                    self.global_prev_state = copy.copy(self.global_state)
                    self.global_state.set_state(status)
                    self.generated_traffic_type = status["id"]
                    self.generated_traffic_type_text = status["status"]
                    self.src_host = status["src_host"]
                    self.dst_host = status["dst_host"]
                    if self.generated_traffic_type > 0:
                        information(Fore.WHITE + f"{self.src_host} {self.generated_traffic_type_text} {self.dst_host}\n")  
                    self.state = self.global_state.get_state() #this is real state from dataset
                    self.evaluate_traffic() 
                else:
                    error(Fore.RED+"Missing dataset row: no status read\n")
            except : 
                error(Fore.RED+"Reading status error\n") 

    def evaluate_traffic(self):
        self.global_state.state = np.array([
                    self.global_state.received_packets, 
                    self.global_state.transmitted_packets, 
                    self.global_state.received_bytes, 
                    self.global_state.transmitted_bytes,
                ], dtype=np.float32) 
        statuses = self.update_hosts_status(self.generated_traffic_type_text, self.src_host, self.dst_host) #update the status: normal or attack?
        self.global_state.update_statuses(self.generated_traffic_type_text, TRAFFIC_TYPE_ID_MAPPING, statuses)
        traffic_data = self.global_state.get_network_traffic_status()
        traffic_data.update({
            "receivedPackets": self.global_state.received_packets,
            "transmittedPackets": self.global_state.transmitted_packets,
            "receivedBytes": self.global_state.received_bytes,
            "transmittedBytes": self.global_state.transmitted_bytes
        })
        notify_client(level=SystemLevels.DATA, traffic_data = traffic_data)
        traffic_data["src_host"] = self.src_host.name if type(self.src_host) is not str and self.src_host is not None else None
        traffic_data["dst_host"] = self.dst_host.name if type(self.dst_host) is not str and self.dst_host is not None else None
        self.statuses.append(traffic_data)
        self.show_network_status()
                  
    def step(self, action, options={"is_discretized_state": False, "is_real_state": False, "current_step": -1, "correct_predictions": 0, "show_action": False, "name": None}):
        while hasattr(self,"pause_event") and self.pause_event.is_set():
            notify_client(level=SystemLevels.STATUS, status=SystemStatus.PAUSED, message="Paused training agents...", mode=SystemModes.TRAINING)
            time.sleep(1)
            continue   
        # Calculate reward
        status = self.global_state.status.copy()
        action_correct = self.generated_traffic_type
        text_action_correct = self.generated_traffic_type_text
        reward = self.calculate_reward(action, action_correct)  
        self.execute_action(action, show_action=options["show_action"], reward=reward)
        is_action_correct = action_correct == action
        if is_action_correct:
            options["correct_predictions"]+=1
        percentage_correct_predictions = options["correct_predictions"]/options["current_step"] if options["current_step"]>0 else 0
           
        ground_truth_step = np.zeros(self.actions_number)
        predicted_step = np.zeros(self.actions_number)
        ground_truth_step[self.generated_traffic_type] = 1
        predicted_step[action] = 1            

        debug(Fore.CYAN + f"Environment reward {reward}"+Fore.WHITE )          
           
        # Check if the episode is done
        done, truncated = self.check_if_done_or_truncated(options["current_step"], percentage_correct_predictions) 
        # Update state here
        if not done and not truncated:
            self.update_state()     
        next_state = self.get_current_state(is_discretized_state=options["is_discretized_state"]) 
            
        return next_state, reward, done, truncated, {'action_correct': action_correct, 
                                                     'text_action_correct': text_action_correct, 
                                                     'status': status,
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
        return REWARDS.COMPLETELY_INCORRECT  # -1 Max penalty for completely incorrect predictions
     
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

      
    def show_network_status(self):       
        state = self.global_state.get_state()
        discrete_state = self.get_discretized_state(state)
        normalized_state = get_normalized_state(state, self.low_to_normalize, self.high_to_normalize) 

        information(
            Fore.BLUE + f"{self.generated_traffic_type_text} Packet {int(state[0])}/{int(state[1])}" +
            Fore.BLUE + f" - {format_bytes(int(state[2]))}B/{format_bytes(int(state[3]))}B" +
            Fore.CYAN + f" - {int(discrete_state[0])} {int(discrete_state[1])} {int(discrete_state[2])} {int(discrete_state[3])}" +
            Fore.MAGENTA + f" - {float(normalized_state[0]):.3f} {float(normalized_state[1]):.3f} {float(normalized_state[2]):.5f} {float(normalized_state[3]):.5f}\n" +
            Fore.WHITE
        )
        #status by host
        for host in self.global_state.host_states.keys():
            host_state = self.global_state.get_host_state(host)
            host_status = self.global_state.get_host_status(host)
            if host_status is not None:
                debug(Fore.WHITE + f"{host} "+Fore.GREEN +
                            f"- Packets {int(host_state[0])}/{int(host_state[4])} " +
                            f"- {format_bytes(int(host_state[2]))}B/{format_bytes(int(host_state[6]))}B\n" + Fore.WHITE)   
  

if __name__ == '__main__':
    #setLogLevel('info')
    config,config_dict = read_config_file('config.yaml')
    env = NetworkEnvClassification(config.env_params, server_user = config.server_user)
    exit = False
    while not exit:
        env.update_state()
        
    statuses = list(env.statuses)
    save_data_to_file(statuses, config.training_execution_directory, "statuses")
    
    observation, info = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action,3)
        if done:
            break
    env.close()
