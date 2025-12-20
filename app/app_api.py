import datetime
import json
import shutil
import yaml
import os
from flask import Flask, render_template, request, jsonify, send_file
from threading import Thread, Event

from utility.constants import SystemModes, SystemStatus
from .socket_handler import init_socketio

# --- Config and Global Variables ---
# Consider that CONFIG_PATH points to the config.yaml file one level above the app/ directory
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'config.yaml')
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Configura Flask per trovare i template e i file statici
app = Flask(__name__,
            static_folder=os.path.join(APP_ROOT, 'static'),
            template_folder=os.path.join(APP_ROOT, 'templates'))

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
    

    
    # Zip expreiment results and send the file to download
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