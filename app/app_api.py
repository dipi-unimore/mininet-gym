from flask import Flask, jsonify, request, g, render_template
from flask_cors import CORS  # Import CORS
import time

app = Flask(__name__,
    template_folder='templates',
    static_folder='static')
CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:8000"}})  # Enable CORS for port 5000

# Initialize the config globally
config = {
    "param1": 10,
    "param2": "default",
}

# Example function to measure time
def measure_execution_time():
    start_time = time.time()
    time.sleep(1)
    end_time = time.time()
    elapsed_time_ms = (end_time - start_time) * 1000
    return elapsed_time_ms

@app.route('/execute', methods=['GET'])
def execute_function():
    elapsed_time = measure_execution_time()
    return jsonify({"elapsed_time_ms": elapsed_time})

# Route to serve index.html
@app.route('/')
def index():
    return render_template('index.html')  # This looks inside the 'templates' directory

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify(True)

@app.route('/params', methods=['GET'])
def get_params():
    # Return the updated config
    return jsonify(config)

@app.route('/params', methods=['POST'])
def set_params():
    # Get JSON data from the request body
    data = request.json
    for key, value in data.items():
        if key in config:
            config[key] = value
    return jsonify(config)  #TODO send ok

def change_config(static_config):
    global config
    config = static_config  # Set the global config to the passed config
    

# Create a function to run the app and set the initial config
def start_api(static_config):
    global config
    config = static_config  # Set the global config to the passed config
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=5000)

# This allows us to run app.py directly too
if __name__ == '__main__':
    default_config = {
        "param1": 10,
        "param2": "default",
    }
    start_api(default_config)
