from flask import Blueprint, jsonify, request
import traceback

from utility.config_sanitizer import clean_load_dir
from utility.my_log import error, information

from ..services import (
    build_dataset_list,
    build_load_dir_list,
    build_result_comm_stats_preview,
    build_result_pdf_response,
    build_results_dir_list,
    build_saved_configs_list,
    build_scenario_list,
    build_result_statuses_preview,
    build_test_scenario_preview,
    delete_result_dir,
    delete_result_dirs,
    load_saved_config_by_relative_path,
    reprint_result_charts,
)


def create_results_blueprint(state):
    bp = Blueprint("results_routes", __name__)

    @bp.route('/reprint_result_charts', methods=['POST'])
    def reprint_result_charts_route():
        payload = request.get_json(silent=True) or {}
        gym_type = payload.get('gym_type', '')
        path = payload.get('path', '')
        try:
            result, status_code = reprint_result_charts(state['current_config'], gym_type, path)
            return jsonify(result), status_code
        except Exception as exc:
            error(f"[RESULTS] Error in reprint_result_charts: {exc}")
            error(f"[RESULTS] Traceback: {traceback.format_exc()}")
            return jsonify({"status": "error", "message": str(exc)}), 500

    @bp.route('/download_result_pdf', methods=['POST'])
    def download_result_pdf():
        payload = request.get_json(silent=True) or {}
        return build_result_pdf_response(
            state['current_config'],
            payload.get('gym_type', ''),
            payload.get('path', ''),
            payload.get('data', {}) if isinstance(payload.get('data', {}), dict) else {},
        )

    @bp.route('/preview_result_statuses', methods=['POST'])
    def preview_result_statuses():
        payload = request.get_json(silent=True) or {}
        result_path = payload.get('path', '')
        sample_size = payload.get('sample_size', 20)
        try:
            preview, status_code = build_result_statuses_preview(state['current_config'], result_path, sample_size=sample_size)
            return jsonify(preview), status_code
        except Exception as exc:
            error(f"[RESULTS] Error in preview_result_statuses: {exc}")
            error(f"[RESULTS] Traceback: {traceback.format_exc()}")
            return jsonify({"status": "error", "message": str(exc)}), 500

    @bp.route('/preview_result_comm_stats', methods=['POST'])
    def preview_result_comm_stats():
        payload = request.get_json(silent=True) or {}
        result_path = payload.get('path', '')
        try:
            preview, status_code = build_result_comm_stats_preview(state['current_config'], result_path)
            return jsonify(preview), status_code
        except Exception as exc:
            error(f"[RESULTS] Error in preview_result_comm_stats: {exc}")
            error(f"[RESULTS] Traceback: {traceback.format_exc()}")
            return jsonify({"status": "error", "message": str(exc)}), 500

    @bp.route('/get_results_dir_list', methods=['GET'])
    def get_results_dir_list():
        try:
            information("[RESULTS] Fetching results dir list...")
            result = build_results_dir_list(state['current_config'])
            information("[RESULTS] Successfully built results dir list")
            return jsonify({"results_dir_list": result})
        except KeyError as e:
            error(f"[RESULTS] KeyError - Missing key in state: {str(e)}")
            error(f"[RESULTS] Traceback: {traceback.format_exc()}")
            return jsonify({"error": f"Configuration error: {str(e)}"}), 400
        except Exception as e:
            error(f"[RESULTS] Error in get_results_dir_list: {str(e)}")
            error(f"[RESULTS] Traceback: {traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500

    @bp.route('/delete_result_dir', methods=['POST'])
    def delete_result_dir_route():
        payload = request.get_json(silent=True) or {}
        path = payload.get('path', '')
        response, status_code = delete_result_dir(state['current_config'], path)
        return jsonify(response), status_code

    @bp.route('/delete_result_dirs', methods=['POST'])
    def delete_result_dirs_route():
        payload = request.get_json(silent=True) or {}
        paths = payload.get('paths', [])
        response, status_code = delete_result_dirs(state['current_config'], paths)
        return jsonify(response), status_code

    @bp.route('/get_load_dir_list', methods=['GET'])
    def get_load_dir_list():
        gym_type = request.args.get('gym_type', '')
        network_config = request.args.get('network_config', '')
        agent_name = request.args.get('agent_name', '')
        return jsonify({"load_dir_list": build_load_dir_list(state['current_config'], gym_type, network_config, agent_name)})

    @bp.route('/get_dataset_list', methods=['GET'])
    def get_dataset_list():
        gym_type = request.args.get('gym_type', '')
        network_config = request.args.get('network_config', '')
        return jsonify({"dataset_list": build_dataset_list(state['current_config'], gym_type, network_config)})

    @bp.route('/get_scenario_list', methods=['GET'])
    def get_scenario_list():
        gym_type = request.args.get('gym_type', '')
        network_config = request.args.get('network_config', '')
        try:
            information(f"[SCENARIO] Fetching scenario list for gym_type={gym_type}, net={network_config}")
            scenario_list = build_scenario_list(state['current_config'], gym_type, network_config)
            information(f"[SCENARIO] Scenario list ready: {len(scenario_list)} item(s)")
            return jsonify({"scenario_list": scenario_list})
        except KeyError as exc:
            error(f"[SCENARIO] Missing key while fetching scenario list: {exc}")
            error(f"[SCENARIO] Traceback: {traceback.format_exc()}")
            return jsonify({"message": f"Configuration error: {exc}"}), 400
        except Exception as exc:
            error(f"[SCENARIO] Error in get_scenario_list: {exc}")
            error(f"[SCENARIO] Traceback: {traceback.format_exc()}")
            return jsonify({"message": str(exc)}), 500

    @bp.route('/preview_test_scenario', methods=['POST'])
    def preview_test_scenario():
        payload = request.get_json(silent=True) or {}
        cfg = payload.get('config', state['current_config'])
        if isinstance(cfg, dict):
            state['current_config'] = clean_load_dir(cfg)

        try:
            preview = build_test_scenario_preview(state['current_config'])
            return jsonify(preview), 200
        except ValueError as exc:
            return jsonify({"message": str(exc)}), 400
        except Exception as exc:
            error(f"[SCENARIO] Error in preview_test_scenario: {exc}")
            error(f"[SCENARIO] Traceback: {traceback.format_exc()}")
            return jsonify({"message": str(exc)}), 500

    @bp.route('/get_scenario_details', methods=['POST'])
    def get_scenario_details():
        import json
        import os
        payload = request.get_json(silent=True) or {}
        scenario_path = payload.get('scenario_path', '')
        
        if not scenario_path:
            return jsonify({"message": "No scenario path provided"}), 400
        
        try:
            # Normalize path - handle both relative and absolute paths
            workspace_root = os.path.abspath(os.path.join(state['app_root'], '..'))
            if not os.path.isabs(scenario_path):
                abs_scenario_path = os.path.join(workspace_root, scenario_path)
            else:
                abs_scenario_path = scenario_path
            
            abs_scenario_path = os.path.abspath(abs_scenario_path)
            
            # Security check: ensure path is within workspace
            if not abs_scenario_path.startswith(workspace_root + os.sep) and abs_scenario_path != workspace_root:
                return jsonify({"message": "Invalid scenario path"}), 403
            
            # Scenario might be a directory path, check for scenario.json inside
            if os.path.isdir(abs_scenario_path):
                scenario_file = os.path.join(abs_scenario_path, "scenario.json")
            else:
                scenario_file = abs_scenario_path
            
            if not os.path.isfile(scenario_file):
                return jsonify({"message": f"Scenario file not found at {scenario_file}"}), 404
            
            with open(scenario_file, 'r') as f:
                scenario_data = json.load(f)
            
            # Extract summary and statistics in the same format as build_test_scenario_preview
            training_stats = scenario_data.get("statistics", {}).get("training", {})
            evaluation_stats = scenario_data.get("statistics", {}).get("evaluation", {})
            response = {
                "summary": {
                    "train_episodes": training_stats.get("episodes", 0),
                    "train_max_steps": training_stats.get("max_steps", 0),
                    "train_steps": training_stats.get("total_steps", 0),
                    "eval_episodes": evaluation_stats.get("episodes", 0),
                    "eval_steps": evaluation_stats.get("total_steps", 0),
                    "attack_likely_used": training_stats.get("attack_likely_used", 0),
                },
                "statistics": scenario_data.get("statistics", {}),
                "storage": "file",
                "scenario_file": scenario_file,
            }
            
            return jsonify(response), 200
            
        except json.JSONDecodeError:
            return jsonify({"message": "Invalid scenario.json file format"}), 400
        except Exception as exc:
            error(f"[SCENARIO] Error in get_scenario_details: {exc}")
            error(f"[SCENARIO] Traceback: {traceback.format_exc()}")
            return jsonify({"message": str(exc)}), 500

    @bp.route('/get_saved_configs_list', methods=['GET'])
    def get_saved_configs_list():
        return jsonify({"config_list": build_saved_configs_list(state['app_root'])})

    @bp.route('/load_saved_config', methods=['POST'])
    def load_saved_config():
        payload = request.get_json(silent=True) or {}
        relative_path = payload.get('path', '')
        try:
            loaded_cfg = load_saved_config_by_relative_path(state['app_root'], relative_path)
            state['current_config'] = clean_load_dir(loaded_cfg)
            return jsonify({"status": "success", "config": state['current_config']}), 200
        except FileNotFoundError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 404
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        except Exception as exc:
            return jsonify({"status": "error", "message": f"Unable to load config: {exc}"}), 500

    return bp
