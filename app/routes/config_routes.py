import json
import datetime
import os
import shutil

import yaml
from flask import Blueprint, jsonify, render_template, request, send_file, send_from_directory

from utility.config_sanitizer import clean_load_dir
from utility.scenario_params import load_scenario_env_params

ALGO_DEFAULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'reinforcement_learning', 'agents', 'defaults'
)


def load_algo_defaults():
    """Load all algorithm default YAML files and return a dict keyed by lowercase algo name."""
    defaults = {}
    if not os.path.isdir(ALGO_DEFAULTS_DIR):
        return defaults
    for filename in os.listdir(ALGO_DEFAULTS_DIR):
        if not filename.endswith('.yaml'):
            continue
        filepath = os.path.join(ALGO_DEFAULTS_DIR, filename)
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        if data and 'algorithm' in data:
            defaults[data['algorithm'].lower()] = data
    return defaults


def create_config_blueprint(state, config_path, app_root):
    bp = Blueprint("config_routes", __name__)

    @bp.route('/')
    def index():
        if state.get('socketio_instance') and hasattr(state['socketio_instance'], 'cfg'):
            state['current_config']['cfg'] = state['socketio_instance'].cfg
        return render_template('index.html', config=json.dumps(state['current_config']))

    @bp.route('/static-training/<path:filename>')
    def training_static(filename):
        folder_name = state['current_config'].get('training_directory', '_training')
        base_path = os.path.abspath(os.path.join(app_root, '..', folder_name))
        return send_from_directory(base_path, filename)

    @bp.route('/get_config', methods=['GET'])
    def get_config():
        return jsonify(state['current_config'])

    @bp.route('/update_config', methods=['POST'])
    def update_config():
        state['current_config'] = clean_load_dir(request.json)
        return jsonify({"status": "success", "message": "Configuration updated"}), 200

    @bp.route('/save_config', methods=['POST'])
    def save_config():
        state['current_config'] = clean_load_dir(request.json)
        config_file = 'config/default.yaml'

        try:
            time_stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_')
            new_config_file = f'{time_stamp}config.yaml'
            config_file_path = config_path.replace("default.yaml", new_config_file)
            with open(config_file_path, 'w') as file_handle:
                yaml.dump(state['current_config'], file_handle, sort_keys=False)
            return send_file(config_file_path, download_name=new_config_file, as_attachment=True)
        except Exception as exc:
            return jsonify({"status": "error", "message": f"Error saving file config default.yaml: {exc}"}), 500

    @bp.route('/get_algo_defaults', methods=['GET'])
    def get_algo_defaults():
        try:
            defaults = load_algo_defaults()
            return jsonify(defaults)
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

    @bp.route('/get_scenario_env_params', methods=['GET'])
    def get_scenario_env_params():
        gym_type = request.args.get('gym_type', '')
        if not gym_type:
            return jsonify({"status": "error", "message": "gym_type is required"}), 400
        params = load_scenario_env_params(gym_type)
        return jsonify(params)

    @bp.route('/download_results', methods=['POST'])
    def download_results():
        try:
            directory_to_zip = state['current_config'].get('training_execution_directory', None)
            if directory_to_zip is None or not os.path.isdir(directory_to_zip):
                return jsonify({"status": "error", "message": "Invalid training execution directory"}), 400

            zip_filename = os.path.basename(directory_to_zip.rstrip('/'))
            zip_filepath = os.path.join(os.path.dirname(directory_to_zip), f'{zip_filename}.zip')
            shutil.make_archive(base_name=zip_filepath.replace('.zip', ''), format='zip', root_dir=directory_to_zip)
            return send_file(zip_filepath, download_name=f'{zip_filename}.zip', as_attachment=True)
        except Exception as exc:
            return jsonify({"status": "error", "message": f"Error saving file zip: {exc}"}), 500

    return bp
