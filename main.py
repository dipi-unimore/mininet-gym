# main.py
from reinforcement_learning.network_env import NetworkEnv #, agent_training, attack_monitor
from reinforcement_learning.network_env_classification import NetworkEnvClassification
from reinforcement_learning.network_env_attack_detect import NetworkEnvAttackDetect
from traffic_classification import traffic_classification_main
from attack_detect import attack_detect_main
#from utility.my_statistics import plot_statutes
from utility.my_files import copy_config_file_to_training_dir, create_directory_training_execution, save_data_to_file, drop_privileges, regain_root
from utility.params import read_config_file
from utility.my_log import set_log_level, set_log_file, information, debug, error, notify_client
import time, random
from colorama import Fore, Back, Style
from app.app_api import start_api, change_config
from app.app_socket import start_socket
import threading
import socket
import requests



def main():
    #read configuration
    config,config_dict = read_config_file('config.yaml')
    if config.random_seed > 0:
        random.seed(config.random_seed)
    config.net_config_filter = f"{config.env_params.net_params.num_switches}_{config.env_params.net_params.num_hosts}_{config.env_params.net_params.num_iot}"
    server_user = config.server_user
    drop_privileges(server_user)
    training_execution_directory = create_directory_training_execution(config)
    set_log_file(f"{training_execution_directory}/log.txt")
    config.training_execution_directory = training_execution_directory
    if copy_config_file_to_training_dir(training_execution_directory):
        information(f"Config file copied to {training_execution_directory}/config.yaml")
    #save_data_to_file(config_dict, training_execution_directory, "config")    
    regain_root()
        
    if (False):
        # Create a thread for the Flask app
        socket_thread = threading.Thread(target=start_socket, args=(config_dict,))
        # Start the thread (this will run the Flask app without blocking the main thread)
        socket_thread.start()    
        #wait_for_socket_start()
        api_thread = threading.Thread(target=start_api, args=(config_dict,))
        # Start the thread (this will run the Flask app without blocking the main thread)
        api_thread.start()    
        #wait_for_api_start()
        while True:
            time.sleep(2)
            traffic_type = random.choice(config.env_params.traffic_types)
            information(f"{traffic_type}")
    
    if config.env_params.gym_type.startswith("attacks"):
        # Create Network
        information("Creating network and environment")
        net_env = NetworkEnvAttackDetect(config.env_params, server_user)
        
        #create agents, training, evaluation metrics, test, plotting
        attack_detect_main(config, net_env)        

    else:
        #Create Network
        information("Creating network and environment")
        net_env = NetworkEnvClassification(config.env_params, server_user)
        
        # create agents, training, evaluation metrics, test         
        traffic_classification_main(config, net_env)
        
def wait_for_api_start():
    url = "http://127.0.0.1:5000/status"  # An endpoint that Flask exposes
    while True:
        try:
            response = requests.get(url)
            if response.status_code == 200 or response.status_code == 404:
                break
        except requests.exceptions.ConnectionError:
            print("Waiting for WebApi server to start...")
            time.sleep(1)
            
def wait_for_socket_start():
    # The server address and port
    server_address = '127.0.0.1'
    port = 8001

    # Keep trying to connect to the WebSocket port
    while True:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((server_address, port))
        if result == 0:
            print(f"WebSocket server is available on port {port}")
            sock.close()
            break
        else:
            print("Waiting for WebSocket server to start...")
            time.sleep(1)          

if __name__ == '__main__':
    set_log_level('info')
    main()
 