import csv
import os
import random
import time

import gymnasium as gym
import numpy as np
import pandas as pd

from reinforcement_learning.agents.adversarial_agent import (
    continuous_traffic_generation,
    generate_random_traffic,
    generate_traffic,
)
from reinforcement_learning.agents.qlearning_agent import QLearningAgent
from reinforcement_learning.scenarios.attack_detect.network_env_attack_detect import (
    NetworkEnvAttackDetect,
)
from reinforcement_learning.scenarios.classification.network_env_classification import (
    NetworkEnvClassification,
)
from reinforcement_learning.network_env import NetworkEnv
from utility import constants
from utility.my_files import read_data_file, save_data_to_file
from utility.my_log import error, information
from utility.my_statistics import (
    plot_agent_execution_confusion_matrix,
    plot_combined_performance_over_time,
    plot_comparison_bar_charts,
    plot_net_metrics,
    plot_radar_chart,
)
from utility.params import read_config_file


def test_env():
    env = gym.make("CartPole-v1")
    observation, info = env.reset(seed=42)
    print(f"create {info}")
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()
    env.close()


def test_controller_delay(net_env: NetworkEnv):
    timestr = time.strftime("%Y%m%d-%H%M%S.%f")
    print(f"\n{timestr} : {net_env.prev_state}")
    print(f"{timestr} : {net_env.state}")
    while True:
        net_env.generated_traffic_type, net_env.src_host, net_env.dst_host = generate_random_traffic(net_env.net)
        if net_env.generated_traffic_type > 0:
            break
    start_time = time.time()
    while True:
        net_env.state = net_env.read_from_network()
        if net_env.state[0] > 0 and net_env.state[1] > 0:
            break
    end_time = time.time()
    elapsed_time_ms = (end_time - start_time) * 1000
    timestr = time.strftime("%Y%m%d-%H%M%S.%f")
    print(f"{timestr} : controller delay {elapsed_time_ms} ms")
    return elapsed_time_ms


def test_controller_traffic_audit(net_env: NetworkEnv):
    data = {"p_r": [], "p_t": [], "b_r": [], "b_t": []}
    store = {"none": data, "ping": data, "udp": data, "tcp": data}
    for traffic_type in net_env.net.traffic_types:
        for _ in range(random.randint(0, 5)):
            net_env.synchronize_controller()
            net_env.generated_traffic_type, net_env.src_host, net_env.dst_host = generate_traffic(net_env.net, traffic_type)
            net_env.state = net_env.read_from_network()
        store[traffic_type] = net_env.data_traffic
    return net_env.data_traffic


def test_create_traffic(net_env: NetworkEnv, episodes):
    store = []
    for _ in range(episodes):
        net_env.synchronize_controller()
        net_env.generated_traffic_type, net_env.src_host, net_env.dst_host = generate_random_traffic(net_env.net)
        net_env.state = net_env.read_from_network()
        i_src_host = net_env.src_host.name if net_env.src_host is not None else "0"
        i_dst_host = net_env.dst_host.name if net_env.dst_host is not None else "0"
        store.append({
            "packets_received": net_env.state[0].item(),
            "bytes_received": net_env.state[2].item(),
            "packets_transmitted": net_env.state[1].item(),
            "bytes_transmitted": net_env.state[3].item(),
            "traffic_type": net_env.generated_traffic_type,
            "i_src_host": i_src_host,
            "i_dst_host": i_dst_host,
        })
    return store


def test_create_traffic_no_synchronize(net_env: NetworkEnv, episodes):
    store = []
    for episode in range(episodes):
        information(f"Episode {episode + 1}\n")
        net_env.generated_traffic_type, net_env.src_host, net_env.dst_host = generate_random_traffic(net_env.net)
        net_env.get_all_traffic_generated()
        i_src_host = net_env.src_host.name if net_env.src_host is not None else "0"
        i_dst_host = net_env.dst_host.name if net_env.dst_host is not None else "0"
        store.append({
            "packets_received": net_env.state[0].item(),
            "bytes_received": net_env.state[2].item(),
            "packets_transmitted": net_env.state[1].item(),
            "bytes_transmitted": net_env.state[3].item(),
            "traffic_type": net_env.generated_traffic_type,
            "i_src_host": i_src_host,
            "i_dst_host": i_dst_host,
        })
    return store


def create_network_from_config(config_file="config/default.yaml"):
    config, config_dict = read_config_file(config_file)
    if config.random_seed > 0:
        random.seed(config.random_seed)

    if config.env_params.gym_type.startswith(constants.ATTACKS):
        information("Creating network and environment for attack detection")
        net_env = NetworkEnvAttackDetect(config.env_params, config.server_user)
    else:
        information("Creating network and environment for classification")
        net_env = NetworkEnvClassification(config.env_params, config.server_user)
    return config, net_env


def create_traffic_classification_env(config, net_env):
    traffic = test_create_traffic_no_synchronize(net_env, config.env_params.test_episodes)
    keys = list(traffic[0].keys())
    with open(config.env_params.csv_file, "w", newline="") as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(traffic)
    save_data_to_file(traffic, config.training_directory, file_name="traffic")


def create_traffic_csv_from_json(csv_file):
    jsons = [f for f in os.listdir("..") if f.endswith(".json") and f.startswith("traffic")]
    store = []
    for json in jsons:
        indicators = read_data_file(f"../{json.replace('.json', '')}")
        if isinstance(indicators, dict):
            continue
        store = store + indicators
    save_data_to_file(store, "..", file_name="traffic")

    keys = list(store[0].keys())
    with open(csv_file, "w", newline="") as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(store)
    plot_net_metrics(store, "..", "")


def test_traffic_csv_with_qlearning(csv_file, net_env, qlearning_agent_params):
    df = pd.read_csv(csv_file)
    q_agent = QLearningAgent(net_env, qlearning_agent_params)
    net_env.observation_space.low = np.array([0, 0, 0, 0])
    net_env.observation_space.high = np.array([len(net_env.net.hosts), 1e2, 1e6, 1e7])
    q_agent.n_bins = 4
    low = net_env.observation_space.low
    high = np.floor(np.log10(net_env.observation_space.high)).astype(int)
    q_agent.bins = [
        np.logspace(low[i], high[i], q_agent.n_bins)
        for i in range(net_env.observation_space.shape[0])
    ]
    print(q_agent.bins)
    df.head()
    data_traffic = {key: {} for key in range(4)}
    for data_episode in df._values:
        observation = np.array([data_episode[0], data_episode[2], data_episode[1], data_episode[3]])
        generated_traffic_type = data_episode[4]
        print(generated_traffic_type)
        state_discrete = q_agent.discretize_state(observation)
        print(f"discretized {state_discrete} for {observation}")
        data_traffic[generated_traffic_type][state_discrete] = data_traffic[generated_traffic_type].get(state_discrete, 0) + 1

    for key, values in data_traffic.items():
        print(key)
        for state, count in values.items():
            print(f"{state} - {count}")


def test_deep_learning(config, net_env):
    from stable_baselines3 import A2C, DQN, PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.vec_env import DummyVecEnv

    vec_env = DummyVecEnv([lambda: net_env])
    model_dqn = DQN("MlpPolicy", vec_env, verbose=1, learning_rate=1e-3, buffer_size=50000, exploration_fraction=0.1)
    model_ppo = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=3e-4, n_steps=128)
    model_a2c = A2C("MlpPolicy", vec_env, verbose=1, learning_rate=7e-4)

    print("Training DQN...")
    model_dqn.learn(total_timesteps=10000)
    print("Training PPO...")
    model_ppo.learn(total_timesteps=10000)
    print("Training A2C...")
    model_a2c.learn(total_timesteps=10000)

    print("Evaluating DQN...")
    mean_reward, std_reward = evaluate_policy(model_dqn, vec_env, n_eval_episodes=10)
    print(f"DQN Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    obs = vec_env.reset()
    for _ in range(10):
        action, _states = model_ppo.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        if dones:
            break


def test_results_from_saved_data(
    env_type,
    execution_dir,
    agent_name,
    print_indicators=True,
    print_metrics=True,
    print_comparisons=True,
    agents_metrics=None,
):
    directory_name = f"_training/{env_type}/{execution_dir}/{agent_name}"
    if not os.path.exists(directory_name):
        error(f"Directory {directory_name} does not exist")
        return
    data = read_data_file(directory_name + "/data")
    if data is None:
        error(f"Data file not found in {directory_name}")
        return
    if "train_indicators" in data:
        data["train_confusion_matrix"] = plot_agent_execution_confusion_matrix(
            data["train_indicators"], directory_name, must_print=print_indicators
        )
    if "train_metrics" in data:
        if print_metrics:
            plot_combined_performance_over_time(
                data["train_metrics"],
                directory_name,
                agent_name + " Combined performance over time",
            )
        if print_comparisons and agents_metrics is not None:
            plot_comparison_bar_charts(directory_name, agents_metrics)
            plot_radar_chart(directory_name, agents_metrics)
    return data


def test_generate_classification_traffic(config):
    config.env_params.data_traffic_file = config.training_directory + f"/statuses_{config.env_params.gym_type.replace(f'_{constants.FROM_DATASET}', '')}.json"

    env = NetworkEnvClassification(config.env_params, server_user=config.server_user)
    do_exit = False
    for _ in range(5000):
        try:
            if do_exit:
                break
            env.update_state()
        except Exception as exc:
            print(f"Exception {exc}")

    statuses = list(env.statuses)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_data_to_file(statuses, config.training_directory + "/classification", f"{timestr}_statuses")


def test_generate_attacks(config):
    config.env_params.data_traffic_file = config.training_directory + f"/statuses_{config.env_params.gym_type.replace(f'_{constants.FROM_DATASET}', '')}.json"

    env = NetworkEnvAttackDetect(config.env_params, server_user=config.server_user)
    env.show_complete_network_status = True
    continuous_traffic_generation(
        env,
        options={
            "show_normal_traffic": False,
            "send_only_attacks": False,
            "only_one_can_attack": False,
            "send_only_normal_traffic": False,
            "send_only_tcp_traffic": False,
            "send_only_udp_traffic": False,
        },
    )
    do_exit = False
    for _ in range(5000):
        try:
            time.sleep(1)
            if do_exit:
                env.stop()
                break
        except Exception as exc:
            print(f"Exception {exc}")

    statuses = list(env.statuses)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_data_to_file(statuses, config.training_directory + "/attacks", f"{timestr}_statuses")


# Backward-compatibility alias for existing callers.
test_traffic_csv_with_qLearning = test_traffic_csv_with_qlearning


def plot_host_states_evolution(statuses_file_path, output_dir="_training/_debug", title="Host States Evolution"):
    """
    Visualize the evolution of host states over time during an experiment.
    
    Shows how each host's state changes across timesteps:
    - State 0: Normal traffic
    - State 1: Host attacking (outgoing attack)
    - State 2: Host under attack (incoming attack detected)
    - State 3: Incoming attack blocked (attack mitigated/blocked)
    - State 4: Outgoing attack blocked (host's own attack blocked)
    
    Args:
        statuses_file_path: Path to statuses.json file from experiment
        output_dir: Directory to save generated visualization
        title: Title for the visualization
        
    Returns:
        Path to saved image or None if error
    """
    import json
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    def _normalize_host_state(raw_state):
        if isinstance(raw_state, (int, float, np.integer, np.floating)):
            state_id = int(raw_state)
            return state_id if 0 <= state_id <= 4 else 0

        status_text = str(raw_state).strip().lower()
        mapping = {
            "normal": 0,
            "attacking": 1,
            "attack_out": 1,
            "outgoing_attack": 1,
            "under_attack": 2,
            "attack_in": 2,
            "incoming_attack": 2,
            "under_attack/attacking": 2,
            "incoming_blocked_attack": 3,
            "incoming_blocked": 3,
            "out_attack_blocked": 4,
            "outgoing_blocked": 4,
        }
        return mapping.get(status_text, 0)
    
    try:
        # Load statuses
        with open(statuses_file_path, 'r') as f:
            statuses = json.load(f)
        
        if not isinstance(statuses, list) or len(statuses) == 0:
            error(f"Invalid statuses format: expected list, got {type(statuses)}")
            return None
        
        # Extract host states over time
        host_names = None
        timesteps_data = []  # List of {host_name: state} dicts
        
        for timestep_idx, status_entry in enumerate(statuses):
            if not isinstance(status_entry, dict):
                continue
                
            hosts_struct = status_entry.get("hostStatusesStructured", {})
            if not hosts_struct:
                continue
            
            # Initialize host names on first entry
            if host_names is None:
                host_names = sorted(list(hosts_struct.keys()))
            
            # Extract status for each host
            timestep_states = {}
            for host_name in host_names:
                host_status = hosts_struct.get(host_name, {})
                # Status code: 0=normal, 1=attacking, 2=under_attack, 3=incoming_blocked, 4=outgoing_blocked
                timestep_states[host_name] = _normalize_host_state(host_status.get("status", 0))
            
            timesteps_data.append(timestep_states)
        
        if not host_names or not timesteps_data:
            error("No host data found in statuses file")
            return None

        n_hosts = len(host_names)
        n_timesteps = len(timesteps_data)

        # Create visualization as a per-host timeline.
        fig_height = max(4.5, n_hosts * 0.55)
        fig, ax = plt.subplots(figsize=(16, fig_height), dpi=100)

        state_colors = {
            0: "#FFFFFF",  # normal
            1: "#4F86C6",  # cold blue
            2: "#E76F51",  # warm muted orange-red
            3: "#3CB371",  # green
            4: "#8FD19E",  # lighter green
        }

        def _compress_state_runs(host_states):
            runs = []
            if not host_states:
                return runs

            run_start = 0
            run_state = host_states[0]
            for idx in range(1, len(host_states)):
                if host_states[idx] != run_state:
                    runs.append((run_start, idx - run_start, run_state))
                    run_start = idx
                    run_state = host_states[idx]
            runs.append((run_start, len(host_states) - run_start, run_state))
            return runs

        for host_idx, host_name in enumerate(host_names):
            host_states = [timestep_states.get(host_name, 0) for timestep_states in timesteps_data]
            for start, duration, state in _compress_state_runs(host_states):
                ax.broken_barh(
                    [(start, duration)],
                    (host_idx - 0.26, 0.52),
                    facecolors=state_colors.get(state, state_colors[0]),
                    edgecolors="none",
                    alpha=0.95,
                )

            ax.hlines(host_idx, 0, max(1, n_timesteps - 1), colors="#D9DEE5", linewidth=0.8, zorder=0)

        ax.set_yticks(range(n_hosts))
        ax.set_yticklabels(host_names, fontsize=9)
        ax.set_xlabel("Timestep", fontsize=11, fontweight='bold')
        ax.set_ylabel("Host", fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=20)

        ax.set_xlim(0, max(1, n_timesteps - 1))
        ax.set_ylim(-0.75, n_hosts - 0.25)

        # Sample ticks so the timeline stays readable without gridlines.
        step = max(1, n_timesteps // 12)
        ax.set_xticks(range(0, n_timesteps, step))
        ax.set_xticklabels(range(0, n_timesteps, step), fontsize=8)

        legend_patches = [
            mpatches.Patch(facecolor=state_colors[0], edgecolor='black', label='0: Normal'),
            mpatches.Patch(facecolor=state_colors[1], edgecolor='black', label='1: Attacking (Out)'),
            mpatches.Patch(facecolor=state_colors[2], edgecolor='black', label='2: Under Attack (In)'),
            mpatches.Patch(facecolor=state_colors[3], edgecolor='black', label='3: Incoming Blocked'),
            mpatches.Patch(facecolor=state_colors[4], edgecolor='black', label='4: Outgoing Blocked'),
        ]
        ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.01, 1),
                  fontsize=9, title='Host State')

        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        
        # Save image
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "host_states_evolution.png")
        try:
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
        except PermissionError:
            fallback_output_path = os.path.join(
                output_dir,
                f"host_states_evolution_{time.strftime('%Y%m%d-%H%M%S')}.png",
            )
            information(
                f"Default output is not writable, saving host states visualization to: {fallback_output_path}"
            )
            output_path = fallback_output_path
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        information(f"Host states evolution visualization saved: {output_path}")
        return output_path
        
    except Exception as e:
        error(f"Error visualizing host states evolution: {e}")
        import traceback
        error(traceback.format_exc())
        return None


def plot_host_traffic_evolution(statuses_file_path, output_dir="_training/_debug", metric="packets", title=None):
    """
    Visualize traffic (packets or bytes) evolution for each host over time.
    
    Shows how transmitted and received traffic evolves across timesteps.
    Useful for identifying when attacks start/stop and traffic patterns.
    
    Args:
        statuses_file_path: Path to statuses.json file from experiment
        output_dir: Directory to save generated visualization
        metric: "packets" or "bytes" - which metric to plot
        title: Title for the visualization (auto-generated if None)
        
    Returns:
        Path to saved image or None if error
    """
    import json
    import matplotlib.pyplot as plt
    
    try:
        # Load statuses
        with open(statuses_file_path, 'r') as f:
            statuses = json.load(f)
        
        if not isinstance(statuses, list) or len(statuses) == 0:
            error(f"Invalid statuses format: expected list, got {type(statuses)}")
            return None
        
        # Extract host traffic over time
        host_names = None
        traffic_data = {}  # {host_name: {received: [...], transmitted: [...]}}
        
        metric_keys = {
            "packets": ("receivedPackets", "transmittedPackets"),
            "bytes": ("receivedBytes", "transmittedBytes")
        }
        
        if metric not in metric_keys:
            error(f"Unknown metric: {metric}. Use 'packets' or 'bytes'")
            return None
        
        received_key, transmitted_key = metric_keys[metric]
        
        for timestep_idx, status_entry in enumerate(statuses):
            if not isinstance(status_entry, dict):
                continue
                
            hosts_struct = status_entry.get("hostStatusesStructured", {})
            if not hosts_struct:
                continue
            
            # Initialize host names on first entry
            if host_names is None:
                host_names = sorted(list(hosts_struct.keys()))
                for h in host_names:
                    traffic_data[h] = {"received": [], "transmitted": []}
            
            # Extract traffic for each host
            for host_name in host_names:
                host_status = hosts_struct.get(host_name, {})
                received = host_status.get(received_key, 0)
                transmitted = host_status.get(transmitted_key, 0)
                
                traffic_data[host_name]["received"].append(float(received))
                traffic_data[host_name]["transmitted"].append(float(transmitted))
        
        if not host_names or not traffic_data:
            error("No host data found in statuses file")
            return None
        
        # Create visualization
        n_hosts = len(host_names)
        fig, axes = plt.subplots(n_hosts, 1, figsize=(14, max(6, n_hosts * 2.5)), dpi=100, sharex=True)
        
        if n_hosts == 1:
            axes = [axes]
        
        if title is None:
            title = f"Host Traffic Evolution ({metric.capitalize()})"
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        
        for host_idx, host_name in enumerate(host_names):
            ax = axes[host_idx]
            t_data = traffic_data[host_name]
            timesteps = range(len(t_data["received"]))
            
            ax.plot(timesteps, t_data["received"], label="Received", linewidth=1.5, marker='o', markersize=3, color='#0066CC', alpha=0.7)
            ax.plot(timesteps, t_data["transmitted"], label="Transmitted", linewidth=1.5, marker='s', markersize=3, color='#CC0000', alpha=0.7)
            
            ax.set_ylabel(host_name, fontweight='bold', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            if host_idx == 0:
                ax.legend(loc='upper left', fontsize=9)
            
            # Add some statistics on the plot
            recv_max = max(t_data["received"]) if t_data["received"] else 0
            trans_max = max(t_data["transmitted"]) if t_data["transmitted"] else 0
            ax.set_ylim(bottom=0)
        
        axes[-1].set_xlabel("Timestep", fontweight='bold', fontsize=11)
        fig.text(0.04, 0.5, metric.capitalize(), va='center', rotation='vertical', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        
        # Save image
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"host_traffic_evolution_{metric}.png")
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        information(f"Host traffic evolution visualization saved: {output_path}")
        return output_path
        
    except Exception as e:
        error(f"Error visualizing host traffic evolution: {e}")
        import traceback
        error(traceback.format_exc())
        return None


