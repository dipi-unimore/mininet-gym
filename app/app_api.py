import datetime, json, shutil, numpy as np, yaml, os
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, Blueprint
from threading import Thread, Event

from reinforcement_learning.marl.constants import COORDINATOR
from utility.constants import ALGO_A2C, ALGO_DQN, ALGO_PPO, ALGO_Q_LEARNING, ALGO_SARSA, ALGO_SUPERVISED, ATTACKS, ATTACKS_FROM_DATASET, CLASSIFICATION, CLASSIFICATION_FROM_DATASET, FROM_DATASET, MARL_ATTACKS, MARL_ATTACKS_FROM_DATASET, SystemModes, SystemStatus
from utility.my_files import read_data_file
from utility.network_configurator import get_host_agents_by_network_config
from utility.params import read_config_file
from .socket_handler import init_socketio

# --- Config and Global Variables ---
# Consider that CONFIG_PATH points to the config.yaml file one level above the app/ directory
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'config.yaml')
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Configura Flask per trovare i template e i file statici
app = Flask(__name__,
            static_folder=os.path.join(APP_ROOT, 'static'),
            template_folder=os.path.join(APP_ROOT, 'templates'))


# # blueprint for "extra" section with different static folder name
# extra_bp = Blueprint(
#     'extra', __name__, 
#     static_url_path='/static-training', # URL 
#     static_folder="../_training" # Name of the folder
# )
# #Advantage: can call from dir in jinja2 using: {{ url_for('extra.static', filename='image.png') }}.

# app.register_blueprint(extra_bp)


current_config = {}
main_ref = None 
socketio_instance = None
training_thread = None 
pause_event = Event()  
stop_event = Event() 

@app.route('/')
def index():
    """Render the configuration page."""
    # Pass the current configuration as a JSON string to the template
    if socketio_instance and hasattr(socketio_instance, 'cfg'):
        current_config['cfg'] = socketio_instance.cfg
    return render_template('index.html', config=json.dumps(current_config))


# Route to serve files from the training directory
@app.route('/static-training/<path:filename>')
def training_static(filename):
    # Get the training directory from the current configuration
    global current_config
    folder_name = current_config.get("training_directory", "_training")
    
    # Build the absolute path to the training directory
    base_path = os.path.abspath(os.path.join(app.root_path, "..", folder_name))
    
    return send_from_directory(base_path, filename)


@app.route('/get_config', methods=['GET'])
def get_config():
    """API to get the current configuration."""
    return jsonify(current_config)

@app.route('/update_config', methods=['POST'])
def update_config():
    """API to update configuration."""
    global current_config
    
    new_config_data = request.json
    
    # Update in-memory configuration
    current_config = new_config_data
    
    try:
        return jsonify({"status": "success", "message": "Configuration updated"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error updating config in memory: {e}"}), 500

@app.route('/save_config', methods=['POST'])
def save_config():
    """API to save updated configuration."""
    global current_config
    
    new_config_data = request.json
    config_file = "config.yaml"
    # Update in-memory configuration
    current_config = new_config_data
    
    # Save to YAML file
    try:
        #rename the yaml file with time stamp prefexing YYYYMMDD_HHMMSS_        
        timeStamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_")    
        new_config_file  = f"{timeStamp}{config_file}"
        #save file    
        config_file_path = CONFIG_PATH.replace(config_file,new_config_file)
        with open(config_file_path, 'w') as f:
            yaml.dump(current_config, f, sort_keys=False)
        #get the file from disk to download it
        return send_file(config_file_path, download_name=new_config_file, as_attachment=True)            
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error saving file config.yaml: {e}"}), 500
    
@app.route('/download_results', methods=['POST'])
def download_results():
    """API to save updated configuration."""
    global current_config
    

    
    # Zip experiment results and send the file to download
    try:
        directory_to_zip = current_config.get("training_execution_directory", None)
        if directory_to_zip is None or not os.path.isdir(directory_to_zip):
            return jsonify({"status": "error", "message": "Invalid training execution directory"}), 400
        #zip the directory
        zip_filename = os.path.basename(directory_to_zip.rstrip('/'))  # Get the directory name
        zip_filepath = os.path.join(os.path.dirname(directory_to_zip), f"{zip_filename}.zip")
        shutil.make_archive(base_name=zip_filepath.replace('.zip',''), format='zip', root_dir=directory_to_zip)

        #get the file from disk to download it
        return send_file(f'../{zip_filepath}', download_name=f"{zip_filename}.zip", as_attachment=True)            
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error saving file zip: {e}"}), 500


@app.route('/start_training', methods=['POST'])
def start_training():
    global current_config
    
    #new_config_data = request.json
    
    # Aggiorna la configurazione in memoria
    #current_config = new_config_data    
    """API to start training in a separate thread."""
    global training_thread, pause_event, stop_event
    if pause_event.is_set():
        pause_event.clear()
        print("Resumed")
        return jsonify({"status": SystemStatus.RESUMED, "message": "Resumed...", }), 200
    if training_thread is not None and training_thread.is_alive():
        return jsonify({"status": SystemStatus.ERROR, "message": "Just running"}), 409    
    if main_ref is None:
        return jsonify({"status": SystemStatus.ERROR, "message": "The main function has not been referenced"}), 500
        
    print("Server API: received request to start training...")

    # Clear previous events
    pause_event.clear() 
    stop_event.clear()  
        
    # Start the training in a new thread
    training_thread = Thread(target=main_ref, 
                             args=(current_config, pause_event, stop_event))
    training_thread.start()
    
    return jsonify({"status": SystemStatus.STARTING, "message": "Training starting..."}), 200


@app.route('/get_results_dir_list', methods=['GET'])
def get_results_dir_list():
    results_dir_list = []
    expected_files = ["log.txt", "metrics_comparison.png", "radar_chart.png", "statuses.json"]
    data_file_in_agent_folder = "data.json"
    test_dir = "TEST"
    expected_files_in_agent_folder_print_training_chart_enabled = [ "matrix.png", "metrics.png", "metrics_combined.png", "rewards.png"]    
    training_directory = current_config.get("training_directory", None)
    if training_directory and os.path.isdir(training_directory):
        for gym_type in [MARL_ATTACKS, ATTACKS, CLASSIFICATION, MARL_ATTACKS_FROM_DATASET, ATTACKS_FROM_DATASET, CLASSIFICATION_FROM_DATASET]:
            path = os.path.join(training_directory, gym_type)
            if os.path.isdir(path):
                datas_gym_type = []
                for d in os.listdir(path):
                    dir_path = os.path.join(path, d)
                    if os.path.isdir(dir_path):
                        str_date = d.split('_')[0]
                        #if date is in format YYYYMMDD_HHMMSS else skip
                        try:
                            date_time = datetime.datetime.strptime(str_date, "%Y%m%d-%H%M%S")
                        except ValueError:
                            continue
                        #now control if there is at least an agent folder inside with the data.json with an accuracy, then test forder exists,
                        # the config.yaml file exists, log.txt exists, metrics_comparison.png exists, radar_chart.png exists, statuses.json exists 
                        # we do this looking for for all files and folders in dir_path, and controlling one by one the existence
                        # from config.yaml, read the gym_type, network_config and agent_names
                        
                        _, config_yaml = read_config_file(os.path.join(dir_path, "config.yaml"))
                        env_params = config_yaml.get("env_params", None)
                        if env_params is None:
                            continue
                        gym_type_in_config = env_params.get("gym_type", "")
                        if gym_type_in_config != gym_type:
                            continue
                        print_training_chart = env_params.get("print_training_chart", False)
                        net_params = env_params.get("net_params", None)
                        if net_params is None:
                            continue
                        training_episodes = env_params.get("episodes", 0)
                        max_steps = env_params.get("max_steps", 0)
                        test_episodes = env_params.get("test_episodes", 0)
                        network_config = f"{net_params.get("num_switches", "")}_{net_params.get("num_hosts", "")}_{net_params.get("num_iot", "")}"
                        agents = config_yaml.get("agents", [])
                        agent_names = [agent.get("name", "") for agent in agents if 
                                       agent.get("enabled", False) and 
                                       not agent.get("skip_learn", True) and 
                                       not agent.get("name", "").startswith(ALGO_SUPERVISED)]
                        data_gym_type={
                            "network_config": network_config,
                            "datetime": str_date,
                            "training_episodes": training_episodes,
                            "max_steps": max_steps,
                            "test_episodes": test_episodes
                        }
                        #check agents directories
                        agents_ok = True
                        agents_data = []
                        for agent_name in agent_names:
                            all_accuracies = []
                            agent_path = os.path.join(dir_path, agent_name)
                            if not os.path.isdir(agent_path):
                                agents_ok = False
                                break
                            #check the files expected exist
                            if print_training_chart and not gym_type_in_config.startswith( MARL_ATTACKS):
                                for ef in expected_files_in_agent_folder_print_training_chart_enabled :
                                    if not os.path.isfile(os.path.join(agent_path, ef)):
                                        agents_ok = False
                                        break
                            #check the data.json file exists
                            if not os.path.isfile(os.path.join(agent_path, data_file_in_agent_folder)):
                                agents_ok = False
                                break

                            data = read_data_file(os.path.join(agent_path, data_file_in_agent_folder))
                            min_accuracy = 0
                            name_min_accuracy = ""
                            max_accuracy = 0
                            name_max_accuracy = ""
                            if gym_type_in_config.startswith( MARL_ATTACKS):
                                accuracy = []               
                                for _,metrics in data.get("train_metrics", {}).items():
                                    accuracy.append(float(np.mean(metrics.get("accuracy", 0))))
                                accuracy = float(np.mean(accuracy)) if accuracy else 0
                                training_episodes = len(data.get("ground_truth", {}).get(COORDINATOR, [])) if "ground_truth" in data else training_episodes
                                max_steps = data["train_indicators"][COORDINATOR][0]['steps']
                            else:
                                max_steps = data["train_indicators"][0]['steps']                                
                                training_episodes = len(data.get("ground_truth", {})) if "ground_truth" in data else training_episodes
                                accuracy = float(np.mean(data.get("train_metrics", {}).get("accuracy", 0))) if data else 0
                            if accuracy == 0:
                                agents_ok = False
                                break
                            #in charts we have the name of all the png files in the agent folder
                            charts = []
                            if print_training_chart:
                                for f in os.listdir(agent_path):
                                    if f.endswith(".png"):
                                        charts.append(f)
                            agents_data.append({
                                "agent_name": agent_name,
                                "accuracy": accuracy,
                                "charts": charts,
                                "print_training_chart": print_training_chart
                            })
                            if accuracy > min_accuracy:
                                min_accuracy = accuracy
                                name_min_accuracy = agent_name
                            if accuracy < max_accuracy:
                                max_accuracy = accuracy
                                name_max_accuracy = agent_name
                            all_accuracies.append(accuracy)
                        if not agents_ok:
                            continue
                        mean_accuracy = float(np.mean(all_accuracies)) if all_accuracies else 0
                        data_gym_type["mean_accuracy"] = mean_accuracy
                        #i have to check if we can access variable min_accuracy and max_accuracy
                        try:
                            data_gym_type["min_accuracy"] = min_accuracy
                            data_gym_type["name_min_accuracy"] = name_min_accuracy
                        except NameError:
                            min_accuracy = 0
                            name_min_accuracy = ""
                        try:
                            data_gym_type["max_accuracy"] = max_accuracy
                            data_gym_type["name_max_accuracy"] = name_max_accuracy
                        except NameError:
                            max_accuracy = 0
                            name_max_accuracy = ""
                        data_gym_type["agents_data"] = agents_data
                        
                        #check the test folder exists
                        if not os.path.isdir(os.path.join(dir_path, test_dir)):
                            continue   
                        #check and read test.json file
                        test_file_path = os.path.join(dir_path, test_dir, "test.json")
                        if not os.path.isfile(test_file_path):
                            continue
                        test_data = read_data_file(test_file_path)
                        data_gym_type["test_scores"] = test_data["score"] if "score" in test_data else None
                        if data_gym_type["test_scores"] is None:
                            continue
                        min_score = test_episodes
                        name_min_score = ""
                        max_score = 0
                        name_max_score = ""
                        mean_score = []   
                        for agent_name,agent_score in test_data["score"].items():                        
                            if gym_type_in_config.startswith( MARL_ATTACKS):
                                mean_agent_score = []            
                                for host_name,host_score in agent_score.items():
                                    mean_agent_score.append(host_score)
                                mean_agent_score = float(np.mean(mean_agent_score)) if mean_agent_score else 0
                            else:
                                mean_agent_score = agent_score
                            if mean_agent_score < min_score:
                                min_score = mean_agent_score
                                name_min_score = agent_name
                            if mean_agent_score > max_score:
                                max_score = mean_agent_score
                                name_max_score = agent_name
                            mean_score.append(mean_agent_score)
                        mean_score = float(np.mean(mean_score)) if mean_score else 0
                        if gym_type_in_config.startswith( MARL_ATTACKS):
                            test_episodes = len(test_data["ground_truth"][COORDINATOR]) if "ground_truth" in test_data else test_episodes
                        else:
                            test_episodes = len(test_data["ground_truth"]) if "ground_truth" in test_data else test_episodes
                        data_gym_type["test_episodes"] = test_episodes
                        data_gym_type["mean_score"] = mean_score
                        data_gym_type["min_score"] = min_score
                        data_gym_type["name_min_score"] = name_min_score
                        data_gym_type["name_max_score"] = name_max_score
                        data_gym_type["max_score"] = max_score
                        
                        #get all png filese in test folder
                        test_charts = []
                        for f in os.listdir(os.path.join(dir_path, test_dir)):
                            if f.endswith(".png"):
                                test_charts.append(f)
                        data_gym_type["test_charts"] = test_charts                       
                                       
                        #check the expected files in dir_path
                        files_ok = True
                        for ef in expected_files:
                            if not os.path.isfile(os.path.join(dir_path, ef)):
                                files_ok = False
                                print(f"Missing file: {ef} in {dir_path}")
                                break
                        if not files_ok:
                            continue
                        
                        data_gym_type["files"] = expected_files
                        data_gym_type["path"] = dir_path.replace(f"{training_directory}/", '')
                        #if everything is ok, append the data                    
                        datas_gym_type.append(data_gym_type)                               

                        
                results_dir_list.append({"gym_type": gym_type, "data": datas_gym_type})        
                       

    return jsonify({"results_dir_list": results_dir_list})


@app.route('/get_load_dir_list', methods=['GET'])
def get_load_dir_list():
    gym_type = request.args.get('gym_type', '')
    network_config = request.args.get('network_config', '')
    agent_name = request.args.get('agent_name', '')
    if agent_name.lower().startswith(ALGO_DQN) or agent_name.lower().startswith(ALGO_A2C) or agent_name.lower().startswith(ALGO_PPO):
        extension = "zip"
    elif agent_name.lower().startswith(ALGO_Q_LEARNING) or agent_name.lower().startswith(ALGO_SARSA) :                    
        extension = "json"    
    
    # Logic to get the list of load directories based on the parameters
    # This is a placeholder implementation; replace it with actual logic
    load_dir_list = []
    gym_type_training_directories = []
    training_directory = current_config.get("training_directory", None)
    gym_type_training_directory = training_directory + f"/{gym_type}"
    gym_type_training_directories.append(gym_type_training_directory)
    if gym_type.endswith(FROM_DATASET):
        gym_type_training_directories.append(gym_type_training_directory.replace(f"_{FROM_DATASET}", ''))
    else :
        gym_type_training_directories.append(gym_type_training_directory + f"_{FROM_DATASET}")
    
    dir_list =  []
    for dir in gym_type_training_directories:
        if os.path.isdir(dir):
            dir_list.extend([os.path.join(dir, d) for d in os.listdir(dir) if d.__contains__(network_config) and os.path.isdir(os.path.join(dir, d))])
    
    for d in dir_list:    
        for da in os.listdir(d):  
            path = os.path.join(d, da)  
            if da == agent_name and os.path.isdir(path):
                if gym_type.startswith(MARL_ATTACKS):
                    #in marl we have multiple agents saved in subfolders
                    host_agents = get_host_agents_by_network_config(network_config)
                    host_agents.append(COORDINATOR)
                    for host_agent in host_agents:
                        isOk = True
                        if not os.path.isfile(f"{path}/{agent_name}_{host_agent}.{extension}"):
                            isOk = False
                            break
                    if not isOk or not os.path.isfile(f"{path}/data.json"):
                        continue
                    data = read_data_file(f"{path}/data.json")
                    complete_path = path.replace(f"{training_directory}/", '')
                    accuracy = []               
                    for _,metrics in data.get("train_metrics", {}).items():
                        accuracy.append(float(np.mean(metrics.get("accuracy", 0))))
                        
                    load_dir_list.append({
                        "accuracy": float(np.mean(accuracy)) if accuracy else 0,
                        "datetime": path.split('/')[2].split('_')[0],
                        "path": complete_path
                    })  
                else:
                    if not os.path.isfile(f"{path}/{agent_name}.{extension}") or not os.path.isfile(f"{path}/data.json"):
                        continue
                    data = read_data_file(f"{path}/data.json")
                    complete_path = path.replace(f"{training_directory}/", '')+f"/{agent_name}.{extension}"               
                    load_dir_list.append({
                        "accuracy": float(np.mean(data.get("train_metrics", {}).get("accuracy", 0))) if data else 0,
                        "datetime": path.split('/')[2].split('_')[0],
                        "path": complete_path
                    })            
      
    return jsonify({"load_dir_list": load_dir_list})

@app.route('/pause_training', methods=['POST'])
def pause_training():
    global pause_event
    if training_thread is not None and training_thread.is_alive():
        print("Training paused")
        pause_event.set() # Set the pause event to pause training
        return jsonify({"status": SystemStatus.PAUSED, "message": "Training in pause."}), 200
    return jsonify({"status": SystemStatus.ERROR, "message": "No training to pause."}), 404

# @app.route('/resume_training', methods=['POST'])
# def resume_training():
#     #TODO at the moment we are using start, let's decide what do to do
#     global pause_event
#     if training_thread is not None and training_thread.is_alive():
#         pause_event.clear() 
#         return jsonify({"status": SystemStatus.RESUME, "message": "Training resumed."}), 200
#     return jsonify({"status": SystemStatus.ERROR, "message": "No training to resume."}), 404

@app.route('/stop_training', methods=['POST'])
def stop_training():
    global stop_event
    if training_thread is not None and training_thread.is_alive():
        if hasattr(main_ref, "env"):
            main_ref.env.stop()
        else:
            stop_event.set() 
        training_thread.join()  # Wait for the training thread to finish
        print("Training posted")
        return jsonify({"status": SystemStatus.STOPPED, "message": "STOP signal sent"}), 200
    return jsonify({"status": SystemStatus.ERROR, "message": "No training to STOP"}), 200

def start_api(main_training_func, loaded_config, host='0.0.0.0', port=5000):
    """Start server Flask + SocketIO in a separate thread."""
    global main_ref
    global current_config
    global socketio_instance
    main_ref = main_training_func 
    current_config = loaded_config 
    socketio_instance = init_socketio(app)
    print("Server Web API + SocketIO started...")
    
    socketio_instance.run(app, host=host, port=port, debug=False)