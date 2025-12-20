from reinforcement_learning.adversarial_agent import continuous_traffic_generation
from reinforcement_learning.classification.network_env_classification import NetworkEnvClassification
from reinforcement_learning.attack_detect.network_env_attack_detect import NetworkEnvAttackDetect
from reinforcement_learning.marl.network_env_marl_attack_detect import NetworkEnvMarlAttackDetect
from reinforcement_learning.classification.traffic_classification import traffic_classification_main
from reinforcement_learning.attack_detect.attack_detect import attack_detect_main
from reinforcement_learning.marl.marl_attack_detect import marl_attack_detect_main
from utility.constants import  ATTACKS, FROM_DATASET, MARL_ATTACKS, SystemLevels
from utility.my_files import copy_config_file_to_training_dir, create_directory_training_execution, drop_privileges, regain_root
from utility.params import Params, read_config_file
from utility.my_log import initialize_client_notifier, notify_client, set_is_from_dataset, set_log_level, set_log_file, information
from app.socket_handler import get_socketio_instance, register_handlers, send_live_data, send_status
import random, threading
from app.app_api import start_api
import json as js

def start_experiment(config_dict, pause_event=None, stop_event=None):
    config= js.loads(js.dumps(config_dict), object_hook=Params)
    
    set_log_level(config.log_level)
    server_user = config.server_user
    if config.random_seed > 0:
        random.seed(config.random_seed)
    config.net_config_filter = f"{config.env_params.net_params.num_switches}_{config.env_params.net_params.num_hosts}_{config.env_params.net_params.num_iot}"
    config_dict["net_config_filter"] = config.net_config_filter

    drop_privileges(server_user) #to create dir without root privileges
    training_execution_directory = create_directory_training_execution(config)
    set_log_file(f"{training_execution_directory}/log.txt")
    config.training_execution_directory = training_execution_directory
    config_dict["training_execution_directory"] = training_execution_directory
    if copy_config_file_to_training_dir(training_execution_directory):
        information(f"Config file copied to {training_execution_directory}/config.yaml")
    regain_root() #to create mininet topology
    
    #config.net_config_filter = f"{config.env_params.net_params.num_switches}_{config.env_params.net_params.num_hosts}_{config.env_params.net_params.num_iot}"
    # Create Network
    information("Creating network and environment")
    isMultiAgent = False
    config.env_params.data_traffic_file = config.training_directory + f"/statuses_{config.env_params.gym_type.replace(f'_{FROM_DATASET}', '')}.json"
    if config.env_params.gym_type.startswith(ATTACKS):
        env = NetworkEnvAttackDetect(config.env_params, server_user)  
    elif config.env_params.gym_type.startswith(MARL_ATTACKS):
        isMultiAgent = True
        raise NotImplementedError("MARL_ATTACKS environment is not yet supported in this version.")
    else:
        env = NetworkEnvClassification(config.env_params, server_user)
    if config.env_params.gym_type.endswith(FROM_DATASET):
        set_is_from_dataset(True)
    
    if hasattr(start_experiment, "env"):
        start_experiment.env = env

    #here no more root user needed    
    drop_privileges(server_user)
    
    if config.enable_web_interface:
        socketio = get_socketio_instance()
        register_handlers(socketio)   
        initialize_client_notifier(send_live_data, send_status)
        cfg = { "hosts" : [h.name for h in env.hosts],
                "agents": [agent.name for agent in config.agents if agent.enabled],
                "isMultiAgent": isMultiAgent
        }
        notify_client(level=SystemLevels.CONFIG, config=cfg)
        socketio.cfg = cfg
    env.pause_event = pause_event if pause_event else threading.Event()
    env.stop_event = stop_event if stop_event else threading.Event()
    if hasattr(env, "host_envs"):
        for host in env.hosts:
            env.host_envs[host.name].pause_event = env.pause_event 
        env.host_envs[host.name].stop_event = env.stop_event 
        env.coordinator_env.pause_event = env.pause_event 
        env.coordinator_env.stop_event = env.stop_event  
    
    
    #Start the traffic generation thread if not from dataset
    if not config.env_params.gym_type.endswith(FROM_DATASET) and not config.env_params.gym_type.startswith("classification"):
        continuous_traffic_generation(env, config.env_params.show_normal_traffic)  
    
    # while True:
    #     time.sleep(5)
    
    # create agents, training, evaluation metrics, test        
    if config.env_params.gym_type.startswith(ATTACKS):
        attack_detect_main(config, env)        
    elif config.env_params.gym_type.startswith(MARL_ATTACKS):
        marl_attack_detect_main(config, env)
    else:
        traffic_classification_main(config, env)

def main():
    #read configuration
    config,config_dict = read_config_file('config.yaml')
    if config.enable_web_interface:
        # Create a thread for the Flask app
        start_experiment.env = {"param": 123}  # Pass the env to the function
        api_thread = threading.Thread(target=start_api, args=(start_experiment,config_dict,))
        # Start the thread (this will run the Flask app without blocking the main thread)
        api_thread.start()    
        api_thread.join()
    else:        
        start_experiment(config_dict)          
        
if __name__ == '__main__':
    main()
 