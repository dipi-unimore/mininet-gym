import os
import sys
from colorama import Fore
from reinforcement_learning.agent_manager import AgentManager
from reinforcement_learning.agents.adversarial_agent import continuous_traffic_generation
from reinforcement_learning.scenarios.attack_detect_host_observable.attack_detect_ho import attack_detect_ho_main
from reinforcement_learning.scenarios.attack_detect_host_observable.network_env_attack_detect_per_host_observable import NetworkEnvAttackDetectPerHostObservable
from reinforcement_learning.scenarios.attack_detect_host_observable.per_host_scan_wrapper import PerHostScanWrapper
from reinforcement_learning.scenarios.classification.network_env_classification import NetworkEnvClassification
from reinforcement_learning.scenarios.attack_detect.network_env_attack_detect import NetworkEnvAttackDetect
from reinforcement_learning.scenarios.marl.network_env_marl_attack_detect import NetworkEnvMarlAttackDetect
from reinforcement_learning.scenarios.classification.traffic_classification import traffic_classification_main
from reinforcement_learning.scenarios.attack_detect.attack_detect import attack_detect_main
from reinforcement_learning.scenarios.marl.marl_attack_detect import marl_attack_detect_main
from utility.constants import ATTACKS, ATTACKS_HO, CLASSIFICATION, FROM_DATASET, GYM_TYPE, MARL_ATTACKS, SystemLevels
from utility.config_sanitizer import clean_load_dir
from utility.my_files import copy_config_file_to_training_dir, create_directory_training_execution, drop_privileges, regain_root, resolve_data_file_path
from utility.params import Params, read_config_file
from utility.my_log import initialize_client_notifier, notify_client, set_is_from_dataset, set_log_level, set_log_file, information, error
from app.socket_handler import get_socketio_instance, register_handlers, send_live_data, send_status
import random, threading
from app.app_api import start_api
import json as js


def start_experiment(config_dict, pause_event=None, stop_event=None):
    config = js.loads(js.dumps(config_dict), object_hook=Params)

    set_log_level(config.log_level)
    server_user = config.server_user
    if config.random_seed > 0:
        random.seed(config.random_seed)
    config.net_config_filter = (
        f"{config.env_params.net_params.num_switches}_"
        f"{config.env_params.net_params.num_hosts}_"
        f"{config.env_params.net_params.num_iots}"
    )
    config_dict["net_config_filter"] = config.net_config_filter

    drop_privileges(server_user)  # create dirs without root privileges
    training_execution_directory = create_directory_training_execution(config)
    set_log_file(f"{training_execution_directory}/log.txt")
    information(f"Experiment started in {training_execution_directory}")  # force log file creation
    config.training_execution_directory = training_execution_directory
    config_dict["training_execution_directory"] = training_execution_directory
    if copy_config_file_to_training_dir(training_execution_directory, config_dict):
        information(f"Config file saved to {training_execution_directory}/config.yaml")
    regain_root()  # needed to create mininet topology

    # ── Create base environment ────────────────────────────────────────
    # The base env (env) is always created here and owns all network-level
    # state: Mininet topology, threads, pause/stop events, traffic generation.
    # For ATTACKS_HO a PerHostScanWrapper is applied later, after all
    # network setup is complete, so env keeps its events and threads intact.
    information("Creating network and environment")
    isMultiAgent = False
    current_dataset_path = getattr(config.env_params, 'data_traffic_file', None)
    if current_dataset_path in (None, '', 'None'):
        config.env_params.data_traffic_file = resolve_data_file_path(
            os.path.join(
                config.training_directory,
                f"statuses_{config.env_params.gym_type.replace(f'_{FROM_DATASET}', '')}.json",
            )
        )
    else:
        config.env_params.data_traffic_file = resolve_data_file_path(current_dataset_path)

    if config.env_params.gym_type.startswith(ATTACKS_HO):
        env = NetworkEnvAttackDetectPerHostObservable(config.env_params, server_user)
    elif config.env_params.gym_type.startswith(ATTACKS):
        env = NetworkEnvAttackDetect(config.env_params, server_user)
    elif config.env_params.gym_type.startswith(MARL_ATTACKS):
        env = NetworkEnvMarlAttackDetect(config.env_params, server_user)
        isMultiAgent = True
    else:
        env = NetworkEnvClassification(config.env_params, server_user)

    if config.env_params.gym_type.endswith(FROM_DATASET):
        set_is_from_dataset(True)

    if hasattr(start_experiment, "env"):
        start_experiment.env = env

    # ── Network-level setup — always on base env ───────────────────────
    drop_privileges(server_user)

    if config.enable_web_interface:
        socketio = get_socketio_instance()
        register_handlers(socketio)
        initialize_client_notifier(send_live_data, send_status)
        cfg = {
            "hosts":        [h.name for h in env.hosts],
            "agents":       [agent.name for agent in config.agents if agent.enabled],
            "isMultiAgent": isMultiAgent,
        }
        notify_client(level=SystemLevels.CONFIG, config=cfg)
        socketio.cfg = cfg

    # Assign pause/stop events directly on base env
    env.pause_event = pause_event if pause_event else threading.Event()
    env.stop_event  = stop_event  if stop_event  else threading.Event()


    if hasattr(env, "host_envs"):
        for host in env.hosts:
            env.host_envs[host.name].pause_event = env.pause_event
        env.host_envs[host.name].stop_event = env.stop_event
        env.coordinator_env.pause_event = env.pause_event
        env.coordinator_env.stop_event  = env.stop_event

    # Start traffic generation thread (not needed for dataset-based scenarios
    # or for ATTACKS_HO sequential mode — in that case _apply_scenario_step
    # drives traffic directly, one step at a time, without background threads)
    if (not config.env_params.gym_type.endswith(FROM_DATASET)
            and not config.env_params.gym_type.startswith(CLASSIFICATION)
            and not config.env_params.gym_type.startswith(ATTACKS_HO)):
        continuous_traffic_generation(
            env, options={"show_normal_traffic": config.env_params.show_normal_traffic}
        )

    # ── Agents, training, evaluation, test ────────────────────────────
    try:
        # get the global state if exists (for web API)
        global state
        try:
            from app.app_api import state as web_state
        except ImportError:
            web_state = None            

        def set_agent_manager(am):
            # Save the instance in the state for API access,
            # so that we can retrieve agent data and training 
            # status in the /status endpoint
            if 'state' in globals() and isinstance(state, dict):
                state['agent_manager'] = am
                state['pause_event'] = am.env.pause_event
                state['stop_event'] = am.env.stop_event
            elif web_state is not None:
                web_state['agent_manager'] = am
                web_state['pause_event'] = am.env.pause_event
                web_state['stop_event'] = am.env.stop_event

        if config.env_params.gym_type.startswith(ATTACKS_HO):
            wrapped_env = PerHostScanWrapper(env)
            am = AgentManager(wrapped_env, config)
            set_agent_manager(am)
            attack_detect_ho_main(config, am, wrapped_env)

        elif config.env_params.gym_type.startswith(ATTACKS):
            am = AgentManager(env, config)
            set_agent_manager(am)
            attack_detect_main(config, am, env)

        elif config.env_params.gym_type.startswith(MARL_ATTACKS):
            am = AgentManager(env, config)
            set_agent_manager(am)
            marl_attack_detect_main(config, am, env)

        else:
            am = AgentManager(env, config)
            set_agent_manager(am)
            traffic_classification_main(config, am, env)

    except Exception as e:
        error(Fore.RED + f"{e}\n" + Fore.WHITE)
        env.stop()


def main(port=None):
    config, config_dict = read_config_file(
        'config/default.yaml',
        config_cleaner=lambda cfg: clean_load_dir(cfg, GYM_TYPE.keys())
    )

    # Use provided port, config port, or default
    web_port = port or getattr(config, 'web_server_port', 5000)

    if config.enable_web_interface:
        start_experiment.env = {"param": 123}
        api_thread = threading.Thread(
            target=start_api, args=(start_experiment, config_dict, '0.0.0.0', web_port)
        )
        api_thread.start()
        api_thread.join()
    else:
        start_experiment(config_dict)


if __name__ == '__main__':
    port = None
    if '--port' in sys.argv:
        try:
            port_idx = sys.argv.index('--port')
            port = int(sys.argv[port_idx + 1])
        except (ValueError, IndexError):
            print("Invalid port argument. Usage: python main.py [--port PORT_NUMBER]", file=sys.stderr)
            sys.exit(1)
    main(port=port)