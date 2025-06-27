import gymnasium as gym
import time, random, os, csv, orjson
from collections import Counter
import numpy as np

from reinforcement_learning.network_env_attack_detect import NetworkEnvAttackDetect
from reinforcement_learning.network_env_classification import NetworkEnvClassification
from reinforcement_learning.network_env import NetworkEnv
from reinforcement_learning.qlearning_agent import QLearningAgent
from reinforcement_learning.adversarial_agent import  generate_random_traffic, generate_traffic
from utility.my_log import set_log_level, information, debug, error, notify_client
from utility.params import Params, read_config_file
from utility.my_statistics import plot_agent_execution_confusion_matrix, plot_radar_chart, plot_combined_performance_over_time, plot_comparison_bar_charts, plot_metrics, read_data_file, plot_net_metrics
from utility.my_files import read_data_file, save_data_to_file
import numpy as np
import pandas as pd
from collections import defaultdict

def test_env():
    env = gym.make("CartPole-v1")
    observation, info = env.reset(seed=42)
    print(f"create {info}")
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
    env.close()


def test_controller_delay(net_env : NetworkEnv):
    timestr = time.strftime("%Y%m%d-%H%M%S.%f")
    print(f"\n{timestr} : {net_env.prev_state}") 
    print(f"{timestr} : {net_env.state}") 
    while True:
        net_env.generated_traffic_type,net_env.src_host,net_env.dst_host=generate_random_traffic(net_env.net)
        if net_env.generated_traffic_type > 0:
            break
    start_time = time.time()
    while True:
        net_env.state = net_env.read_from_network()
        # timestr = time.strftime("%Y%m%d-%H%M%S.%f")
        # print(f"{timestr} : {net_env.prev_state}") 
        # print(f"{timestr} : {net_env.state}") 
        if net_env.state[0]>0 and net_env.state[1]>0:
            break
    end_time = time.time()
        # Calculate the difference in milliseconds
    elapsed_time_ms = (end_time - start_time) * 1000
    timestr = time.strftime("%Y%m%d-%H%M%S.%f")
    print(f"{timestr} : controller delay {elapsed_time_ms} ms") 
    return elapsed_time_ms

def test_controller_traffic_audit(net_env : NetworkEnv):    
    data =  {'p_r': [], 'p_t': [], 'b_r': [], 'b_t': []}
    store =  {'none': data, 'ping': data, 'udp': data, 'tcp': data}
    for traffic_type in net_env.net.traffic_types:
        for i in range(random.randint(0,5)):
            net_env.synchronize_controller()
            net_env.generated_traffic_type,net_env.src_host,net_env.dst_host=generate_traffic(net_env.net, traffic_type)
            net_env.state = net_env.read_from_network()        
        store[traffic_type]=net_env.data_traffic
    return net_env.data_traffic

def test_create_traffic(net_env : NetworkEnv, episodes):    
    store =  []
    for episode in range(episodes):
        net_env.synchronize_controller()
        net_env.generated_traffic_type,net_env.src_host,net_env.dst_host=generate_random_traffic(net_env.net)
        net_env.state = net_env.read_from_network()     
        if (net_env.src_host is not None):
            i_src_host = net_env.src_host.name #int(self.src_host.name.replace('h','').replace('iot',''))
        else:
            i_src_host = '0'
        if (net_env.dst_host is not None):
            i_dst_host = net_env.dst_host.name #int(self.dst_host.name.replace('h','').replace('iot',''))
        else:
            i_dst_host = '0'   
        store.append({
                #'episode': episode,
                'packets_received': net_env.state[0].item(),
                'bytes_received': net_env.state[2].item(),
                'packets_transmitted': net_env.state[1].item(),
                'bytes_transmitted': net_env.state[3].item(),
                'traffic_type': net_env.generated_traffic_type,
                'i_src_host': i_src_host,
                'i_dst_host': i_dst_host
            })
    return store

def test_create_traffic_no_synchronize(net_env : NetworkEnv, episodes):    
    store =  []
    for episode in range(episodes):
        episode += 1
        information(f"Episode {episode}\n")
        net_env.generated_traffic_type,net_env.src_host,net_env.dst_host=generate_random_traffic(net_env.net)
        net_env.get_all_traffic_generated()     
        if (net_env.src_host is not None):
            i_src_host = net_env.src_host.name #int(self.src_host.name.replace('h','').replace('iot',''))
        else:
            i_src_host = '0'
        if (net_env.dst_host is not None):
            i_dst_host = net_env.dst_host.name #int(self.dst_host.name.replace('h','').replace('iot',''))
        else:
            i_dst_host = '0'   
        store.append({
                #'episode': episode,
                'packets_received': net_env.state[0].item(),
                'bytes_received': net_env.state[2].item(),
                'packets_transmitted': net_env.state[1].item(),
                'bytes_transmitted': net_env.state[3].item(),
                'traffic_type': net_env.generated_traffic_type,
                'i_src_host': i_src_host,
                'i_dst_host': i_dst_host
            })
    return store

def create_network_from_config():
    #read configuration
    config, config_dict = read_config_file('config.yaml')
    data = type('', (), {})()
    data.config = config_dict
    if config.random_seed > 0:
        random.seed(config.random_seed)
        
    # Step 1: Create Network
    if config.env_params.gym_type.startswith("attacks"):
        information("Creating network and environment for attack detection")
        net_env = NetworkEnvAttackDetect(config.env_params, config.server_user)
    else:
        information("Creating network and environment for classification")
        net_env = NetworkEnvClassification(config.env_params, config.server_user)
    return config, net_env

def create_traffic_classification_env( config, net_env):        
    #test create traffic  
    traffic = test_create_traffic_no_synchronize(net_env, config.test_episodes)
    keys = list(traffic[0].keys())
    with open(config.env_params.csv_file, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(traffic)
    save_data_to_file(traffic, config.training_directory , file_name="traffic") 
    #store = indicators_to_net_metrics(indicators)
    #plot_net_metrics(store, '..', '') 

def create_traffic_csv_from_json(csv_file):      
    # put togheter different traffic generated
    jsons = [f for f in os.listdir('..') if f.endswith('.json') and f.startswith('traffic')]    
    store = []
    for json in jsons:
        indicators = read_data_file(f"../{json.replace('.json','')}")
        if type(indicators) is dict: #if hasattr(indicators, 'none'):
            continue            
        store = store + indicators
    save_data_to_file(store, '..', file_name="traffic") 
    
    keys = list(store[0].keys())
    with open(csv_file, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(store)
    #store = indicators_to_net_metrics(store)
    plot_net_metrics(store, '..', '')       

def test_traffic_csv_with_qLearning(csv_file, net_env, qlearning_agent_params):
    df = pd.read_csv(csv_file)     
    q_agent = QLearningAgent(net_env, qlearning_agent_params)   
    net_env.observation_space.low = np.array([0, 0, 0, 0]) #np.inf
    net_env.observation_space.high = np.array([len(net_env.net.hosts), 1e2, 1e6, 1e7]) #np.inf
    q_agent.n_bins = 4
    low = net_env.observation_space.low #np.floor(np.log10(net_env.observation_space.low)).astype(int)
    high = np.floor(np.log10(net_env.observation_space.high)).astype(int)
    q_agent.bins = [
            np.logspace(low[i], high[i], q_agent.n_bins)
            for i in range(net_env.observation_space.shape[0])
        ]
    print(q_agent.bins)
    #df = df.sort_values("traffic_type")
    df.head()
    data_traffic = {key: {} for key in  range(4)}
    for data_episode in df._values:
        observation = np.array([data_episode[0],data_episode[2], data_episode[1], data_episode[3]]) 
        generated_traffic_type = data_episode[4]
        print(generated_traffic_type)
        state_discrete = q_agent.discretize_state(observation)
        print(f"discretized {state_discrete} for {observation}")
        # data_traffic[generated_traffic_type][state_discrete] = defaultdict(int)
        # data_traffic[generated_traffic_type][state_discrete]+=1
        data_traffic[generated_traffic_type][state_discrete] = data_traffic[generated_traffic_type].get(state_discrete, 0) + 1
    
    for key, values in data_traffic.items():    
        print(key)
        for k,v in values.items():
            print(f"{k} - {v}")
  
import gymnasium as gym
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
            
def test_deep_learning(config, net_env):
    vec_env = DummyVecEnv([lambda: net_env])

    # Choose your algorithm: DQN, PPO, or A2C
    # Model configurations can be customized as needed
    model_dqn = DQN("MlpPolicy", vec_env, verbose=1, learning_rate=1e-3, buffer_size=50000, exploration_fraction=0.1)
    model_ppo = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=3e-4, n_steps=128)
    model_a2c = A2C("MlpPolicy", vec_env, verbose=1, learning_rate=7e-4)

    # Train the model
    print("Training DQN...")
    model_dqn.learn(total_timesteps=10000)

    print("Training PPO...")
    model_ppo.learn(total_timesteps=10000)

    print("Training A2C...")
    model_a2c.learn(total_timesteps=10000)

    # Evaluate the model
    print("Evaluating DQN...")
    mean_reward, std_reward = evaluate_policy(model_dqn, vec_env, n_eval_episodes=10)
    print(f"DQN Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Predict using PPO (as an example)
    obs = vec_env.reset()
    for _ in range(10):
        action, _states = model_ppo.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        if dones:
            break

def test_results_from_saved_data(env_type, execution_dir, agent_name):
    directory_name = f"_training/{env_type}/{execution_dir}/{agent_name}"
    if not os.path.exists(directory_name):
        error(f"Directory {directory_name} does not exist")
        return
    information(f"Testing results from saved data in {directory_name}")
    data = read_data_file(directory_name + "/data")
    if data is None:
        error(f"Data file not found in {directory_name}")
        return
    if 'train_indicators' in data:
        #information(f"Train metrics: {data['train_indicators']}")
        plot_agent_execution_confusion_matrix(data['train_indicators'], directory_name)
    if 'train_metrics' in data:
        #information(f"Train metrics: {data['train_metrics']}")
        #plot_metrics(data['train_metrics'], directory_name, agent_name + " Train metrics") 
        plot_combined_performance_over_time(data['train_metrics'], directory_name, agent_name + " Combined performance over time")
        return data['train_metrics']  

if __name__ == '__main__':
    set_log_level('info')
    
    testResultsFromSavedData = True
    createTrafficClassificationEnv = False
    createCSVTrafficFromJson = False
    testCSVTrafficWithQLearning = False
    deepLearning = False
    
    if testResultsFromSavedData:
        #test_results_from_saved_data('classification_from_dataset','20250625-121343_1_10_1','Q-learning_1')
        agents_metrics=[]
        env_type = 'attacks'
        execution_dir = '20250307-095840_1_10_1'
        directory_name = f"_training/{env_type}/{execution_dir}"
        if not os.path.exists(directory_name):
            error(f"Directory {directory_name} does not exist")
            exit(-1)
        #read directory to find the agents trained ( the agent are all directory contained expet Test)
        agents = [f for f in os.listdir(directory_name) if os.path.isdir(os.path.join(directory_name, f)) and not f.startswith('TEST')] 
        agents_metrics = defaultdict(list)
        for agent in agents:
            agents_metrics[agent]=test_results_from_saved_data(env_type,execution_dir,agent)
        plot_comparison_bar_charts(directory_name, agents_metrics)
        plot_radar_chart(directory_name, agents_metrics)
        exit(0)
    
    config,net_env = create_network_from_config()     
    if createTrafficClassificationEnv:
        create_traffic_classification_env(config, net_env)
    
    
    if createCSVTrafficFromJson:
       create_traffic_csv_from_json(config.env_params.csv_file)
    
    
    
    # statuses = []
    # # read a big file   
    # with open(config.training_directory+"/20250318-130111_1_10_1/statuses.json", 'r') as file:
    #     statuses = orjson.loads(file.read())     
    
    # count_second_with_attack=0
    # count_second_with_attack_outliers=0
    # count_second_with_normal_outliers=0
    # count_attack=0
    # discretized_states = []
    # is_attack = False
    # states = []
    # for status in statuses:
    #     state = (status["packets"],status["variation_packet"],status["bytes"],status["variation_byte"])
    #     discretized_state = net_env.get_discretized_state(np.array(state))
    #     if status["id"]>0:
    #         count_second_with_attack+=1
    #         if discretized_state[0] != 3 and discretized_state[2] != 3:
    #             print(f"{status['id']} {state} {discretized_state}")
    #             count_second_with_attack_outliers+=1
    #         if not is_attack:
    #             is_attack = True
    #             count_attack+=1
    #     else:
    #         is_attack=False
    #         if discretized_state[0] == 3 or discretized_state[2] == 3:
    #             print(f"{status['id']} {state} {discretized_state}")
    #             count_second_with_normal_outliers+=1
            
            
    #     states.append(state)
    #     discretized_states.append(discretized_state)
    
    # columns = list(zip(*discretized_states))

    # counts_per_column = [Counter(col) for col in columns]


    # for i, counts in enumerate(counts_per_column):
    #     print(f"Colonna {i}: {dict(counts)}")
    # print(f"Attack : {count_attack}")
    # print(f"Attack seconds : {count_second_with_attack}")
    # print(f"Attack seconds outliers: {count_second_with_attack_outliers}")
    # print(f"Normal seconds outliers: {count_second_with_normal_outliers}")
    
    # #if createTraffic:
    #     #create_traffic(config,net_env)
        
    #     #test controller delay    
    #     # while True:    
    #     #     if test_controller_delay(net_env) < 500:
    #     #         break
        
    #     #test controller traffic audit   
    #     # store = test_controller_traffic_audit(net_env)
    #     # save_data_to_file(store, '..')
    #     # plot_net_metrics(store, '..', '')
        

    
    # if testCSVTrafficWithQLearning:
    #     test_traffic_csv_with_qLearning(config.env_params.csv_file, net_env, config.agents[4])
   
    # if deepLearning:
    #     test_deep_learning(config, net_env)