from mininet.net import Mininet
from .instant_state import InstantState  
from .constants import REWARDS, AGENT_ACTIONS, AgentActions
from reinforcement_learning.network_env import get_agent_name, get_custom_bin_index, get_linear_bin_index, get_normalized_state
from utility import constants
from utility.network_configurator import block_flow_drop, comunicate_normal_traffic_detected, comunicate_in_attack_detected, comunicate_out_attack_detected, detach_link, unblock_flow_delete
from utility.my_log import  information, debug 
from utility.params import Params
from colorama import Fore
import gymnasium as gym
from gymnasium import spaces
import numpy as np, time

class HostAgentEnv(gym.Env):
    """Custom Environment that follows gym interface."""
    metadata = {"render_modes": ["human"], "render_fps": 30}       
    def __init__(self, params: Params, net: Mininet, global_state: InstantState, host_name, gym_type: int):
        self.params = params
        self.max_steps = params.max_steps
        self.steps_min_percentage  = params.steps_min_percentage 
        self.accuracy_min = params.accuracy_min          
        self.gym_type = gym_type
        self.net = net # Reference to the single Mininet network
        self.host_name = host_name
        # Network params
        self.threshold_packets = params.attacks.thresholds.packets
        self.threshold_var_packets = params.attacks.thresholds.var_packets
        self.threshold_bytes = params.attacks.thresholds.bytes
        self.threshold_var_bytes = params.attacks.thresholds.var_bytes
        # Define action and observation space, gym.spaces objects
        self.actions_number = AGENT_ACTIONS.NUMBER
        self.action_space = spaces.Discrete(self.actions_number) #number of actions
        # Define action and observation space, gym.spaces objects
        # State: 
        # [packets_sent, var_packets_sent, bytes_sent, var_bytes_sent, packets_received, var_packets_received, bytes_received, var_bytes_received, receving_attack_message] 
        # first four for undestanding if there is an attack outgoing, sent by this host 
        # last five for understanding if there is an attack incoming, received by this host
        self.global_state = global_state
        self.set_link_status(True)
        
        self.low = np.array([0,
                             -self.threshold_var_packets,
                             0,
                             -self.threshold_var_bytes,
                             0,
                             -self.threshold_var_packets,
                             0,
                             -self.threshold_var_bytes,
                             0])
        # Each agent observes only its own host traffic, so high is per-host threshold
        # (not multiplied by n_hosts, which was causing attack and normal states to
        # collapse into the same Q-table bin, making the agent unable to learn)
        self.high = np.array([self.threshold_packets,
                              self.threshold_var_packets,
                              self.threshold_bytes,
                              self.threshold_var_bytes,
                              self.threshold_packets,
                              self.threshold_var_packets,
                              self.threshold_bytes,
                              self.threshold_var_bytes,
                              len(self.net.hosts)])
        self.observation_space = spaces.Box(low=0, high=self.high, shape=(len(self.low),), dtype=np.float32)
        
        # Define the number of discrete bins for each observation dimension
        self.n_bins = params.n_bins
        
        #self.reset()
        

    def reset(self, seed=None, options={"is_discretized_state": False, "is_real_state": False}):
        # Get the initial state for this specific host        
        self.state = self.get_current_state(options["is_discretized_state"], options["is_real_state"])
        self.status = self.global_state.get_host_status(self.host_name) #initial status
        return self.state, {}

    def step(self, action, options={"is_discretized_state": False, "current_step": -1, "correct_predictions": 0, "show_action": False, "name": None}):
        while hasattr(self,"pause_event") and self.pause_event.is_set():
            time.sleep(1)
            continue                
        
        status = self.global_state.host_statuses[self.host_name].copy()
        #print(f"Env Step {options['current_step']} - Current Status: {status} - State: {self.state} - Action taken: {action}")
        action_correct = status["id"]
        text_action_correct = status["status"]
        # Calculate reward
        reward = self.calculate_reward(action)  
        # Execute the action for this host within the main network
        self.execute_action(action, show_action = options["show_action"], name = options["name"], reward = reward) 
        # Check if the action was correct
        is_action_correct = self.generated_traffic_type == action
        if is_action_correct:
            options["correct_predictions"]+=1
            self.global_state.consecutive_corrects+=1
            if REWARDS.TEAM_SUCCESSFUL > 0:
                cc = self.global_state.consecutive_corrects
                nh = len(self.global_state.host_states) + 1
                if cc >= nh:
                    reward+=REWARDS.TEAM_SUCCESSFUL #all predicted correctly
                elif cc >= nh*0.5:
                    reward+=REWARDS.TEAM_SUCCESSFUL*0.5 #at least 50% predicted correctly
        else:
            self.global_state.consecutive_corrects=0
           
        ground_truth_step = np.zeros(self.actions_number)
        predicted_step = np.zeros(self.actions_number)
        
        ground_truth_step[self.generated_traffic_type] = 1
        predicted_step[action] = 1            

        percentage_correct_predictions = options["correct_predictions"]/options["current_step"] if options["current_step"]>0 else 0

        debug(Fore.CYAN + f"Environment reward {reward}"+Fore.WHITE )          
           
        # Check if the episode is done
        done, truncated = self.check_if_done_or_truncated(options["current_step"], percentage_correct_predictions) 
        # Update state here
        if not done and not truncated:
            if self.gym_type==constants.GYM_TYPE[constants.MARL_ATTACKS]:
                time.sleep(1)
        #     else:
        #         self.update_state()      
        next_state = self.get_current_state(is_discretized_state=options["is_discretized_state"] ) 
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
    
    def set_link_status(self, is_on: bool):
        self.is_link_on = is_on
        self.global_state.links_status[self.host_name] = 1 if is_on else 0
    
    def execute_action(self,action, show_action=False, name = "", reward = 0):
        apply_drop_rules = getattr(self.params.attacks, 'apply_drop_rules', True)
        # Handle the action logic
        if action == AGENT_ACTIONS.NORMAL_TRAFFIC:
            # Action 0:
            msg = comunicate_normal_traffic_detected()
            #attach link if stopped
            if apply_drop_rules and not self.is_link_on and unblock_flow_delete(self.net, self.host_name):
                time.sleep(0.2)
                self.set_link_status(True)
        if action == AGENT_ACTIONS.INCOMING_ATTACK:
            # Action 1:
            msg = comunicate_in_attack_detected()
            if apply_drop_rules and not self.is_link_on and unblock_flow_delete(self.net, self.host_name):
                time.sleep(0.2)
                self.set_link_status(True)
        if action == AGENT_ACTIONS.OUTGOING_ATTACK:
            # Action 2:
            msg = comunicate_out_attack_detected()
            #stop link
            if apply_drop_rules and self.is_link_on and block_flow_drop(self.net, self.host_name):
                time.sleep(0.2)
                self.set_link_status(False)
        agent_name=get_agent_name(host_name = self.host_name)                  
        #TODO verify if 0,1,2 is it good as message or 
        # if is enough 0 [0 and 1 actions] and 1 [action 2]
        self.global_state.set_message(agent_name, self.host_name, action)
        if show_action:
            information(f"{self.host_name} {msg} R: {reward}\n", agent_name) 
    
    def calculate_reward(self, action: int) -> float:
        """
        Calculates the reward for an agent based on its action and the ground truth.

        Args:
            action (int): The action taken by the agent (0, 1, or 2).

        Returns:
            float: The reward value for the current step.
        """
        self.generated_traffic_type = self.status["id"]
        if self.generated_traffic_type == AgentActions.IDLE:
            return 0.0
#TODO go on working from here        
        # if self.is_link_on:
        #     if self.generated_traffic_type == AgentActions.OUTGOING_ATTACK and action == AGENT_ACTIONS.NORMAL_TRAFFIC :
        #         #the agent should block an outgoing attack

        #         return REWARDS.LINK_OFF  #the agent successfully blocked an outgoing attack  
        
        
        if not self.is_link_on :  
            return REWARDS.LINK_OFF
        
        # Determine if the ground truth is normal traffic or attack
        is_normal = True if self.generated_traffic_type == AgentActions.NORMAL_TRAFFIC else False
        
        # Scenario 1: Agent correctly detects normal traffic
        if action == AGENT_ACTIONS.NORMAL_TRAFFIC and is_normal:
            return REWARDS.CORRECT_NORMAL_TRAFFIC  # Reward for correct normal traffic detection
        # Scenario 2: Agent makes a False Negative
        if action == AGENT_ACTIONS.NORMAL_TRAFFIC and not is_normal:
            return REWARDS.FALSE_NEGATIVE  # Penalty for missing an attack

        # Scenario 3: Agent makes a False Positive
        if action is not AGENT_ACTIONS.NORMAL_TRAFFIC and is_normal:
            return REWARDS.FALSE_POSITIVE  # Penalty for false alarm
        
        is_incoming_attack = True if self.generated_traffic_type == AgentActions.INCOMING_ATTACK else False
        is_outgoing_attack = not is_incoming_attack

    
        # if is_outgoing_attack and not self.is_link_on and action == AGENT_ACTIONS.NORMAL_TRAFFIC :  
        #     return REWARDS.LINK_OFF  #the agent successfully blocked an outgoing attack       
        
            
        # Scenario 4: Scenario: Differentiate between incoming and outgoing attacks
        if action == AGENT_ACTIONS.INCOMING_ATTACK and is_incoming_attack:
            return REWARDS.CORRECT_ATTACK_DETECTION  #the agent should allow an incoming attack
        if action == AGENT_ACTIONS.INCOMING_ATTACK and is_outgoing_attack:
            return REWARDS.WRONG_ATTACK_DIRECTION_DETECTED  #the agent should not allow an outgoing attack
        
        if action == AGENT_ACTIONS.OUTGOING_ATTACK and is_outgoing_attack:
            return REWARDS.CORRECT_ATTACK_DETECTION
        if action == AGENT_ACTIONS.OUTGOING_ATTACK and is_incoming_attack:
            return REWARDS.WRONG_ATTACK_DIRECTION_DETECTED  #the agent should not block an incoming attack   
                
        # Default case for any other action, should not be reached with this design
        return 0.0   
    
    def close(self):
        """Close the environment (optional)."""
        information("Environment closed.")
        
    def render(self, mode="human"):
        """Render the environment (optional)."""
        information(f"State: {self.state}, Ground Truth: {self.generated_traffic_type}")
    
    ############# STATE RELATED FUNCTIONS #####################        
    def get_current_state(self, is_discretized_state = False, is_real_state = False):
        state = self.global_state.get_host_state(self.host_name)
        #message customized by agent variant
        agent_name = get_agent_name(host_name = self.host_name)        
        agent_messages = self.global_state.get_messages(agent_name)
        message_state = sum(1 for k, v in agent_messages.items() if v > 0 and k != self.host_name) # 8: Message
        state = np.append(state, message_state) 
        self.status = self.global_state.get_host_status(self.host_name) #update status
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
    
          