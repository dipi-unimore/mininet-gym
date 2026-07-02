from threading import Thread

from flask import Blueprint, jsonify

from utility.constants import SystemStatus
from utility.my_log import set_drop_rule_message_visibility, get_agent_summaries, get_current_training_agent


def create_training_blueprint(state):
    bp = Blueprint("training_routes", __name__)

    @bp.route('/start_training', methods=['POST'])
    def start_training():
        attacks_cfg = state.get('current_config', {}).get('env_params', {}).get('attacks', {})
        if isinstance(attacks_cfg, dict):
            apply_drop_rules = bool(attacks_cfg.get('apply_drop_rules', True))
        else:
            apply_drop_rules = bool(getattr(attacks_cfg, 'apply_drop_rules', True))
        set_drop_rule_message_visibility(apply_drop_rules)

        if state['pause_event'].is_set():
            state['pause_event'].clear()
            return jsonify({"status": SystemStatus.RESUMED, "message": "Resumed..."}), 200

        if state.get('training_thread') is not None and state['training_thread'].is_alive():
            return jsonify({"status": SystemStatus.ERROR, "message": "Just running"}), 409

        if state.get('main_ref') is None:
            return jsonify({"status": SystemStatus.ERROR, "message": "The main function has not been referenced"}), 500

        state['pause_event'].clear()
        state['stop_event'].clear()

        state['training_thread'] = Thread(
            target=state['main_ref'],
            args=(state['current_config'], state['pause_event'], state['stop_event']),
        )
        state['training_thread'].start()

        return jsonify({"status": SystemStatus.STARTING, "message": "Training starting..."}), 200

    @bp.route('/pause_training', methods=['POST'])
    def pause_training():
        if state.get('training_thread') is not None and state['training_thread'].is_alive():
            state['pause_event'].set()
            return jsonify({"status": SystemStatus.PAUSED, "message": "Training in pause."}), 200
        return jsonify({"status": SystemStatus.ERROR, "message": "No training to pause."}), 404

    @bp.route('/stop_training', methods=['POST'])
    def stop_training():
        if state.get('training_thread') is not None and state['training_thread'].is_alive():
            state['stop_event'].set()
            state['training_thread'].join()
            return jsonify({"status": SystemStatus.STOPPED, "message": "STOP signal sent"}), 200
        return jsonify({"status": SystemStatus.ERROR, "message": "No training to STOP"}), 200

    @bp.route('/get_training_status', methods=['GET'])
    def get_training_status():
        """Get current training status to recover state after page reload/reconnect."""
        try:
            is_training = state.get('training_thread') is not None and state['training_thread'].is_alive()
            pause_set = state.get('pause_event') and state['pause_event'].is_set()
            stop_set = state.get('stop_event') and state['stop_event'].is_set()
            
            if is_training:
                if pause_set:
                    status = SystemStatus.PAUSED
                    message = "Experiment is paused"
                elif stop_set:
                    status = SystemStatus.STOPPED
                    message = "Experiment is stopping"
                else:
                    status = SystemStatus.RUNNING
                    message = "Experiment is running"
            else:
                status = SystemStatus.IDLE
                message = "No active experiments"
            
            # --- AGENT DATA FOR CHARTS & BUTTONS ---
            agent_chart_data = {}
            agent_button_state = {}
            # Prova a estrarre agent_manager da state, fallback su config solo se non presente
            try:
                agent_manager = state.get('agent_manager')
                if agent_manager is None:
                    config = state.get('current_config', {})
                    agent_manager = config.get('agent_manager') if isinstance(config, dict) else None
                if agent_manager and hasattr(agent_manager, 'agents_params'):
                    for agent_param in agent_manager.agents_params:
                        agent_name = getattr(agent_param, 'name', None)
                        if not agent_name:
                            continue
                        # Metrics (accuracy, reward). Single-agent scenarios expose
                        # them directly on `.instance` (flat list, one point per
                        # episode). marl_pz/marl multi-host scenarios expose them
                        # per host under `.instances[host]['custom_callback']` —
                        # each host trains independently (sequential per-instance
                        # for on-policy PPO/A2C), so history is kept per host and
                        # returned as {host: [...]} instead of a single flat list.
                        try:
                            single_instance = getattr(agent_param, 'instance', None)
                            instances = getattr(agent_param, 'instances', None) or {}
                            if instances:
                                accuracy_by_host = {}
                                reward_by_host = {}
                                for host_name, inst_info in instances.items():
                                    candidate = (
                                        inst_info.get('custom_callback')
                                        or inst_info.get('instance')
                                    )
                                    host_metrics = getattr(candidate, 'metrics', None)
                                    if host_metrics and host_metrics.get('accuracy'):
                                        accuracy_by_host[host_name] = host_metrics['accuracy']
                                    host_indicators = getattr(candidate, 'indicators', None)
                                    if host_indicators:
                                        reward_by_host[host_name] = [
                                            indicator.get('cumulative_reward')
                                            for indicator in host_indicators
                                        ]
                                agent_chart_data[agent_name] = {
                                    'accuracy': accuracy_by_host or None,
                                    'reward': reward_by_host or None,
                                }
                            elif single_instance is not None:
                                metrics = getattr(single_instance, 'metrics', None)
                                indicators = getattr(single_instance, 'indicators', None)
                                agent_chart_data[agent_name] = {
                                    'accuracy': metrics['accuracy'] if metrics and 'accuracy' in metrics else None,
                                    'reward': [i['cumulative_reward'] for i in indicators] if indicators else None,
                                }
                            else:
                                agent_chart_data[agent_name] = {'accuracy': None, 'reward': None}
                        except Exception:
                            agent_chart_data[agent_name] = {'accuracy': None, 'reward': None}
                        # Button state: actual summary data for reconnect recovery
                        cached = get_agent_summaries().get(agent_name, {})
                        agent_button_state[agent_name] = {
                            'training_summary': cached.get('training'),
                            'evaluation_summary': cached.get('evaluation'),
                        }
            except Exception as e:
                print(f"[get_training_status] Could not extract agent chart/button data: {e}")

            current_agent_info = get_current_training_agent() if is_training else {}

            return jsonify({
                "status": status,
                "message": message,
                "is_training": is_training,
                "is_paused": pause_set,
                "is_stopping": stop_set,
                "agent_chart_data": agent_chart_data,
                "agent_button_state": agent_button_state,
                "current_agent": current_agent_info.get('name'),
                "current_agent_mode": current_agent_info.get('mode'),
            }), 200
        except Exception as e:
            print(f"Error in get_training_status: {e}")
            return jsonify({"status": SystemStatus.ERROR, "message": "Error retrieving training status"}), 500

    return bp
