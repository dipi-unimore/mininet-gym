from colorama import Fore
from reinforcement_learning.network_env import NetworkEnv, get_normalized_state
from reinforcement_learning.marl.instant_state import InstantState 
from reinforcement_learning.marl.host_agent_env import HostAgentEnv
from reinforcement_learning.marl.coordinator_env import CoordinatorEnv 
from reinforcement_learning.marl.constants import *
from utility.my_log import debug, information, notify_client
from utility.network_configurator import format_bytes, stop
from utility.params import Params
from utility.constants import *
import json as jsonlib
import numpy as np, time, threading

class NetworkEnvMarlAttackDetect(NetworkEnv):     
    """Custom Environment that follows gym interface."""
    metadata = {"render_modes": ["human"], "render_fps": 30}    
           
    def __init__(self, params, server_user = 'server_user'):
        params.actions_number = 2 #to avoid issues, but not used
        super().__init__(params, server_user)
        self.params = params
        # After super().__init__ is called, self.net is available
        self.host_envs = {}
        self.coordinator_env = None
        self.hosts = self.net.hosts # Access hosts from the parent class's network
        self.statuses = []

        # Create the sub-environments, passing a reference to the main network
        self._create_agent_envs(params)

        self.last_short_attack_timestamp = time.time()
        self.last_long_attack_timestamp = time.time()
        
        if self.gym_type == GYM_TYPE[MARL_ATTACKS]:
            self.attack_likely = self.init_attack_likely = params.attacks.likely
            self.update_state_thread_instance = threading.Thread(target=self.update_state_thread, args=())            
            self.update_state_thread_instance.start()
    
    
    def _create_agent_envs(self, params):      
        # Network creation
        self.threshold_packets = params.attacks.thresholds.packets
        self.threshold_var_packets = params.attacks.thresholds.var_packets
        self.threshold_bytes = params.attacks.thresholds.bytes
        self.threshold_var_bytes = params.attacks.thresholds.var_bytes
        # Define action and observation space, gym.spaces objects
        self.hosts_observation_space_size = 9
        self.coordinator_observation_space_size = 5

        #initial state general and hosts
        #size obs minus one because no message, to generalize, messages will added when given to the agent
        self.global_prev_state = self.global_state = InstantState(self.hosts, np.zeros(self.coordinator_observation_space_size-1, dtype=np.float32), {})
        time.sleep(0.5) #wait to have a new state
        self.update_state()
        """Creates the gym environments for each agent, reusing the network."""
        for host in self.hosts:
            # Create a lightweight environment for each host
            self.host_envs[host.name] = HostAgentEnv(self.params, self.net, self.global_state, host.name, self.gym_type)

        # Create the coordinator's environment
        self.coordinator_env = CoordinatorEnv(self.params, self.net, self.global_state, self.gym_type) 
  
    
    def reset(self, seed=None, options={"is_discretized_state": False, "is_real_state": False}):
        """Reset the environment to an initial state and returns an initial observation.
        """
        time.sleep(0.5) #wait to have a new state
        if "is_discretized_state" not in options:
            options["is_discretized_state"] = False
        if "is_real_state" not in options:
            options["is_real_state"] = False
        self.update_state()       
        
        # Reset each host environment and collect their initial observations
        states = {}
        for host_name, host_env in self.host_envs.items():
            obs, _ = host_env.reset(seed=seed, options=options)
            states[host_name] = obs

        # Reset the coordinator environment and collect its initial observation
        states[COORDINATOR], _ = self.coordinator_env.reset(seed=seed, options=options)
        info = {}  # Additional info can be added here if needed

        return states, info   
    
        
    def update_state(self):  
        """to update the state of the environment, only gym_type=6
        """
        if self.read_from_network(): #read from switch
            self.global_state.coordinator_state = np.array([
                self.global_state.packets, 
                self.global_state.packets_percentage_change, 
                self.global_state.bytes, 
                self.global_state.bytes_percentage_change,
            ], dtype=np.float32) 
            
            statuses, is_any_host_under_attack_or_attacking = self.update_hosts_status()
            # Coordinator status is "attack" if any individual host status is not "normal"
            if is_any_host_under_attack_or_attacking:
                coordinator_status = ATTACK
            else:
                coordinator_status = NORMAL
            # Add the coordinator status to the results
            statuses['coordinator'] = coordinator_status       
            self.global_state.update_statuses(statuses)            
            traffic_data = self.global_state.get_network_traffic_status()
            notify_client(level=SystemLevels.DATA, traffic_data = traffic_data)
            self.statuses.append(traffic_data)
            self.show_network_status()

    def show_network_status(self):
        try:         
            state = self.global_state.get_state()
            discrete_state = self.get_discretized_state(state)
            normalized_state = get_normalized_state(state, self.coordinator_env.low, self.coordinator_env.high)         
            t_q_v_p = self.coordinator_env.high[1] * self.coordinator_env.high[1] #square threshold_var_packets
            t_q_v_b = self.coordinator_env.high[3] * self.coordinator_env.high[3] #square threshold_var_bytes
            color0 = Fore.BLUE if state[0] < self.coordinator_env.high[0] else Fore.WHITE
            color2 = Fore.BLUE if state[2] < self.coordinator_env.high[2] else Fore.WHITE
            color1 = Fore.BLUE if state[1] * state[1] < t_q_v_p else Fore.WHITE
            color3 = Fore.BLUE if state[3] * state[3] < t_q_v_b else Fore.WHITE
            colorstatus = Fore.GREEN if self.global_state.status['status']=="normal" else Fore.RED
            information(
                colorstatus +  f"{self.global_state.status['status']} " +
                Fore.BLUE +  f"Packet "+
                color0 + f"{int(state[0])} " +
                color1 + f"{int(state[1])}%" +
                color2 + f" - {format_bytes(int(state[2]))}"+Fore.BLUE +"B" +
                color3 + f" {int(state[3])}%" +
                Fore.CYAN + f" - {int(discrete_state[0])} {int(discrete_state[1])} {int(discrete_state[2])} {int(discrete_state[3])}" +
                Fore.MAGENTA + f" - {float(normalized_state[0]):.3f} {float(normalized_state[1]):.3f} {float(normalized_state[2]):.5f} {float(normalized_state[3]):.5f}\n" +

                Fore.WHITE
            )
            level_function = debug
            if self.show_complete_network_status:
                level_function = information
            #status by host
            for host in self.global_state.host_states.keys():
                host_state = self.global_state.get_host_state(host)
                host_status = self.global_state.get_host_status(host)
                if host_status is not None:
                    #print(f"{host}: {self.global_state.get_host_status(host)} - {self.global_state.get_host_state(host)} ")
                    if host_status['status'] in (HostStatus.ATTACKING, HostStatus.WAR):
                        level_function(Fore.WHITE + 
                                    f"{host} " + Fore.RED + f"{host_status['status']} {self.host_tasks[host].get('attack_subtype', '').upper()}"+
                                    f"- RX Pkt {int(host_state[0])} {int(host_state[1])}% " +
                                    f"- {format_bytes(int(host_state[2]))}B {int(host_state[3])}% " +
                                    f"- TX Pkt {int(host_state[4])} {int(host_state[5])}% " +
                                    f"- {format_bytes(int(host_state[6]))}B {int(host_state[7])}%\n" + Fore.WHITE)
                    elif host_status['status'] in (HostStatus.UNDER_ATTACK):
                        level_function(Fore.WHITE + 
                                    f"{host} " + Fore.YELLOW + f"{host_status['status']}-{self.host_tasks[host]['traffic_type'].upper()}"+
                                    f"- RX Pkt {int(host_state[0])} {int(host_state[1])}% " +
                                    f"- {format_bytes(int(host_state[2]))}B {int(host_state[3])}% " +
                                    f"- TX Pkt {int(host_state[4])} {int(host_state[5])}% " +
                                    f"- {format_bytes(int(host_state[6]))}B {int(host_state[7])}%\n" + Fore.WHITE)                    
                    else:
                        level_function(Fore.WHITE + 
                                    f"{host} "+Fore.GREEN + f"{host_status['status']}-{self.host_tasks[host]['traffic_type'].upper()} to {self.host_tasks[host]['destination']} "+
                                    f"- RX Pkt {int(host_state[0])} {int(host_state[1])}% " +
                                    f"- {format_bytes(int(host_state[2]))}B {int(host_state[3])}% " +
                                    f"- TX Pkt {int(host_state[4])} {int(host_state[5])}% " +
                                    f"- {format_bytes(int(host_state[6]))}B {int(host_state[7])}%\n" + Fore.WHITE)         
        except Exception as e:
            debug(f"Error showing network status: {e}") 
    
    def initialize_storage(self):
        pass            
  
    #this is not th real env, this is only the container fo the envs of agents and coordinator
    def get_discretized_state(self, state):
        pass  

    def step(self, options=None):
        pass  

    def check_if_done_or_truncated(self, current_step, percentage_correct_predictions):
        pass  
    
    def execute_action(self,action):
        pass
    
    def calculate_reward(self, action: int) -> float:
        """
        This modified function provides informative feedback, 
        rewarding the agent based on how close it was to the correct answer.
        """
        pass
        


        
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
        observation, reward, done, _ = env.step(action)
        if done:
            break
    env.close()
