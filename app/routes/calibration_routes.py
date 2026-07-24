import re
import threading
import time

from flask import Blueprint, jsonify, request

from utility.constants import SystemStatus

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')


def create_calibration_blueprint(state):
    """
    'Calibrate Thresholds' button backend: runs test_tools/calibrate_thresholds.py's
    run_calibration() in a background thread (same in-process pattern as
    /start_training — Mininet needs root, so this must run as the main app
    process, not a subprocess), using the *live* net_params currently set in
    the system configuration panel (state['current_config']), not necessarily
    what's saved on disk.

    Mutually exclusive with training: both build a Mininet topology on bridge
    's1' via mininet.clean.cleanup(), so running them concurrently would tear
    down whichever network started first.
    """
    bp = Blueprint("calibration_routes", __name__)

    state.setdefault('calibration_thread', None)
    state.setdefault('calibration_status', SystemStatus.IDLE)
    state.setdefault('calibration_log', [])
    state.setdefault('calibration_result', None)
    state.setdefault('calibration_error', None)

    def _log_line(line):
        for sub_line in str(line).splitlines() or ['']:
            state['calibration_log'].append(_ANSI_RE.sub('', sub_line))

    def _run(overrides):
        from test_tools.calibrate_thresholds import run_calibration
        state['calibration_status'] = SystemStatus.RUNNING
        state['calibration_log'] = []
        state['calibration_result'] = None
        state['calibration_error'] = None
        try:
            result = run_calibration(log=_log_line, **overrides)
            state['calibration_result'] = result
            state['calibration_status'] = SystemStatus.FINISHED
        except Exception as exc:
            state['calibration_error'] = str(exc)
            state['calibration_status'] = SystemStatus.ERROR
            _log_line(f"ERROR: {exc}")

    @bp.route('/calibrate_thresholds', methods=['POST'])
    def calibrate_thresholds():
        if state.get('training_thread') is not None and state['training_thread'].is_alive():
            return jsonify({"status": "error",
                            "message": "Cannot calibrate while training is running."}), 409
        if state.get('calibration_thread') is not None and state['calibration_thread'].is_alive():
            return jsonify({"status": "error", "message": "Calibration already running."}), 409

        payload = request.get_json(silent=True) or {}
        config = state.get('current_config') or {}
        env_params = config.get('env_params', {}) if isinstance(config, dict) else {}
        net_params = env_params.get('net_params', {}) if isinstance(env_params, dict) else {}

        overrides = dict(
            gym_type=env_params.get('gym_type'),
            num_hosts=net_params.get('num_hosts'),
            num_iots=net_params.get('num_iots'),
            link_bw_mbps=net_params.get('link_bw_mbps'),
            normal_bw_mbps=net_params.get('normal_traffic_bw_mbps'),
            normal_types=[t for t in net_params.get('traffic_types', []) if t != 'none'] or None,
            duration=env_params.get('wait_after_read'),
            n_bins=env_params.get('n_bins'),
            rounds=int(payload.get('rounds', 6)),
            attack_types=payload.get('attack_types') or ['udp_flood'],
        )

        state['calibration_thread'] = threading.Thread(
            target=_run, args=(overrides,), daemon=True,
        )
        state['calibration_thread'].start()
        return jsonify({"status": SystemStatus.STARTING,
                        "message": "Calibration started..."}), 200

    @bp.route('/calibration_status', methods=['GET'])
    def calibration_status():
        return jsonify({
            "status": state.get('calibration_status', SystemStatus.IDLE),
            "log": state.get('calibration_log', []),
            "result": state.get('calibration_result'),
            "error": state.get('calibration_error'),
        }), 200

    @bp.route('/save_scenario_thresholds', methods=['POST'])
    def save_scenario_thresholds():
        """
        Persist attacks/classification.thresholds.packets|bytes currently held in
        state['current_config'] (the in-memory config edited in the panel — may or
        may not come from a calibration run, e.g. the user might have hand-edited
        the Scenario tab) to the selected scenario's scenario_env_param.yaml.

        This is the "Save" half of Apply (memory) -> Save (disk): Apply only ever
        updates state['current_config'] via /update_config, never the YAML file;
        this endpoint is the only one that writes to disk.
        """
        from utility.scenario_params import (
            gym_type_to_scenario_folder, SCENARIOS_BASE_DIR, SCENARIO_PARAM_FILE,
        )
        import os

        config = state.get('current_config') or {}
        env_params = config.get('env_params', {}) if isinstance(config, dict) else {}
        gym_type = env_params.get('gym_type') if isinstance(env_params, dict) else None
        if not gym_type:
            return jsonify({"status": "error",
                            "message": "No gym_type set in the current configuration."}), 400

        folder = gym_type_to_scenario_folder(gym_type)
        if folder is None:
            return jsonify({"status": "error",
                            "message": f"Cannot map gym_type '{gym_type}' to a scenario folder."}), 400
        yaml_path = os.path.join(SCENARIOS_BASE_DIR, folder, SCENARIO_PARAM_FILE)
        if not os.path.exists(yaml_path):
            return jsonify({"status": "error", "message": f"Scenario YAML not found: {yaml_path}"}), 404

        section = env_params.get('attacks')
        if not isinstance(section, dict) or 'thresholds' not in section:
            section = env_params.get('classification')
        thresholds = section.get('thresholds') if isinstance(section, dict) else None
        if not isinstance(thresholds, dict) or 'packets' not in thresholds or 'bytes' not in thresholds:
            return jsonify({"status": "error",
                            "message": "No thresholds.packets/bytes found in the current configuration."}), 400

        from test_tools.calibrate_thresholds import write_recommendations

        timestr = time.strftime("%Y-%m-%d")
        try:
            backup = write_recommendations(
                yaml_path,
                packets=thresholds['packets'],
                bytes_=thresholds['bytes'],
                source_note=f"saved {timestr} from web UI (Environment > Network)",
            )
            return jsonify({
                "status": "success",
                "message": f"Thresholds saved to {yaml_path}",
                "yaml_path": yaml_path,
                "backup": backup,
            }), 200
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

    return bp
