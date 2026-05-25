from colorama import Fore
from mininet.net import Mininet
from .instant_state import InstantState
from .constants import COORDINATOR, REWARDS, COORDINATOR_ACTIONS
from reinforcement_learning.network_env import get_agent_name, get_custom_bin_index, get_linear_bin_index,  get_normalized_state
from utility.constants import SystemLevels, SystemModes, SystemStatus, GYM_TYPE, MARL_ATTACKS
from utility.my_log import  information, debug, notify_client
import gymnasium as gym
from gymnasium import spaces
import numpy as np, time

from utility.network_configurator import comunicate_attack_detected, comunicate_normal_traffic_detected
from utility.params import Params

class CoordinatorEnv(gym.Env):
    """Custom Environment for the Coordinator Agent."""
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, params: Params, net:Mininet, global_state: InstantState, gym_type: int):
        self.params = params
        self.max_steps = params.max_steps
        self.steps_min_percentage  = params.steps_min_percentage 
        self.accuracy_min = params.accuracy_min  
        self.gym_type = gym_type
        self.net = net # Reference to the single Mininet network
        
        # Define action and observation space
        self.actions_number = COORDINATOR_ACTIONS.NUMBER
        self.action_space = spaces.Discrete(self.actions_number)
        self.threshold_packets = params.attacks.thresholds.packets
        self.threshold_var_packets = params.attacks.thresholds.var_packets
        self.threshold_bytes = params.attacks.thresholds.bytes
        self.threshold_var_bytes = params.attacks.thresholds.var_bytes
        # Define the global state space.
        # This will need to be populated with appropriate values for your network.
        self.global_state = global_state
        # State: [total_packets, var_packets, total_bytes, var_bytes, receving_attack_message]
        self.low = np.array([0, -self.threshold_var_packets, -self.threshold_var_bytes,0, 0])
        self.high = np.array([self.threshold_packets*len(self.net.hosts), 
                              self.threshold_var_packets, 
                              self.threshold_bytes*len(self.net.hosts), 
                              self.threshold_var_bytes,
                              len(self.net.hosts)])
        self.observation_space = spaces.Box(low=self.low, high=self.high, shape=(len(self.low),), dtype=np.float32)

        # Define the number of discrete bins for each observation dimension
        self.n_bins = params.n_bins
        
        # This will be updated with the global state at each step
        #self.reset()
        
    def reset(self, seed=None, options={"is_discretized_state": False, "is_real_state": False}):
        # The CoordinatorEnv does not reset the network, only gets a fresh state.
        self.state = self.get_current_state(is_discretized_state=options["is_discretized_state"], is_real_state=options["is_real_state"])        
        self.status = self.global_state.coordinator_status #initial status
        return self.state, {}

    def step(self, action: int, options={"is_discretized_state": False, "current_step": -1, "correct_predictions": 0, "show_action": False, "name": None}):
        while hasattr(self,"pause_event") and self.pause_event.is_set():
            #sent notify to client only in coordinator env. Not necessary in host envs. 
            notify_client(level=SystemLevels.STATUS, status=SystemStatus.PAUSED, message="Paused training agents...", mode=SystemModes.TRAINING)
            time.sleep(1)
            continue
        # Execute the coordinator's action (e.g., sending an alert)
        # This part of the logic will be handled by the main training loop
        # when it interacts with the communication bus.
        
        status = self.global_state.status.copy()
        #print(f"Env Step {options['current_step']} - Current Status: {status} - State: {self.state} - Action taken: {action}")
        action_correct = status["id"]
        text_action_correct = status["status"]
        
        reward = self.calculate_reward(action) 
        self.execute_action(action, options["show_action"], options["name"], reward)  
        is_action_correct = self.generated_traffic_type == action
        if is_action_correct:
            options["correct_predictions"]+=1
            self.global_state.consecutive_corrects+=1
            if REWARDS.TEAM_SUCCESSFUL > 0:
                cc = self.global_state.consecutive_corrects
                nh = len(self.global_state.host_states)+1
                if cc >= nh:
                    reward+=REWARDS.TEAM_SUCCESSFUL #all predicted correctly
                elif cc >= nh*0.5:
                    reward+=REWARDS.TEAM_SUCCESSFUL*0.5 #at least 50% predicted correctly
        else:
            self.global_state.consecutive_corrects=0 
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
            if self.gym_type==GYM_TYPE[MARL_ATTACKS]:
                time.sleep(1)
        #     else:
        #TODO verify if we need to update in from dataset
        #         self.update_state() 
             
        next_state = self.get_current_state(is_discretized_state=options["is_discretized_state"]) 
        
        return next_state, reward, done, truncated, {'action_correct': action_correct, 
                                                     'text_action_correct': text_action_correct, 
                                                     'status': status,
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
        
        if current_step >= self.max_steps*self.steps_min_percentage  and percentage_correct_predictions>self.accuracy_min :
            return False, True # accuracy greater than param accuracy_min
        
        # End the episode if maximum steps are reached
        if current_step >= self.max_steps:
            return True, False       

        return False, False
    
    def execute_action(self, action, show_action=False, name = "", reward = 0):
        # Handle the action logic
        if action == COORDINATOR_ACTIONS.NORMAL_TRAFFIC:
            # Action 0: 
            msg = comunicate_normal_traffic_detected()
        if action == COORDINATOR_ACTIONS.ATTACK:
            # Action 1: 
            msg = comunicate_attack_detected()
        # message  customized by agent variant
        agent_name = get_agent_name( host_name = COORDINATOR)
        self.global_state.set_message(agent_name, COORDINATOR, action)
        if show_action:
            information(f"{COORDINATOR} {msg} R: {reward}\n", f"{agent_name}")

    def calculate_reward(self, action: int) -> float:
        """
        Calculates the reward for the coordinator based on its action and the global network state.

        Args:
            action (int): The action taken by the coordinator.

        Returns:
            float: The reward value.
        """
        self.generated_traffic_type = self.global_state.coordinator_status["id"]
        if self.generated_traffic_type == COORDINATOR_ACTIONS.IDLE:
            return 0.0
        
        is_attack = True if self.generated_traffic_type > 0 else False
        
        # Scenario 1: Coordinator correctly broadcasts an alert
        if action == COORDINATOR_ACTIONS.ATTACK and is_attack:
            return REWARDS.COORDINATOR_CORRECT_ALERT
            
        # Scenario 2: Coordinator misses an attack
        if action == COORDINATOR_ACTIONS.NORMAL_TRAFFIC and is_attack:
            return REWARDS.COORDINATOR_MISSED_ALERT
            
        # Scenario 3: Coordinator makes a false alarm
        if action == COORDINATOR_ACTIONS.ATTACK and not is_attack:
            return REWARDS.COORDINATOR_FALSE_ALERT
            
        # Scenario 4: Coordinator correctly does nothing when there is no attack
        if action == COORDINATOR_ACTIONS.NORMAL_TRAFFIC and not is_attack:
            return REWARDS.CORRECT_NORMAL_TRAFFIC # Using a positive reward for stability
        
        # Default case for other actions or scenarios
        return 0.0

    ############# STATE RELATED FUNCTIONS #####################
    
    def get_current_state(self, is_discretized_state=False, is_real_state=False):
        """
        Retrieves the current global state of the network.
        """
        state = self.global_state.get_coordinator_state()
        #message by agent variant
        agent_name = get_agent_name( host_name = COORDINATOR)
        agent_messages = self.global_state.get_messages(agent_name)
        message_state = sum(1 for k, v in agent_messages.items() if v > 0 and k != COORDINATOR) # 5: Message
        state = np.append(state, message_state)
        self.status = self.global_state.coordinator_status
        
        if is_real_state:
            return state
        if is_discretized_state:
            return self.get_discretized_state(state)
        return get_normalized_state(state, self.low, self.high)



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
            elif i==0 : #for packets in attacks gym_type use linear discretization
                bin_index = get_custom_bin_index(val, low[i], high[i], n_bins)
            else: #for bytes in attacks gym_type use log discretization
                bin_index = get_linear_bin_index(val, low[i], high[i], n_bins)
            discrete_state.append(bin_index)
        return tuple(discrete_state) 