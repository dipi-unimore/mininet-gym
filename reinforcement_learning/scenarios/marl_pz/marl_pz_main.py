"""
marl_pz_main — training and evaluation entry point for the MARL PZ scenario.

Execution flow
--------------
Phase 1 — SCENARIO
  Generate (or load) a deterministic scenario with training + evaluation
  sequences.  Same generator as the HO scenario.

Phase 2/3 — TRAIN + EVALUATE (per agent, sequentially)
  Each agent is trained independently via its SingleAgentView:
  - Tabular (Q-Learning / SARSA): agent.instance.learn(episodes)
  - SB3 (DQN / PPO / A2C):       per-episode callback loop
  All other agents submit action 0 (NORMAL_TRAFFIC) during peer training.
  Evaluation runs all agents together on the env (coordinated inference).

Phase 4 — SUMMARY
  Comparative table + plots.

MARL_PZ_FROM_DATASET
  Load scenario from the data_traffic_file directory.
"""
import os
import time
import traceback

import numpy as np
from colorama import Fore
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from reinforcement_learning.agent_manager import AgentManager
from reinforcement_learning.agents.qlearning_agent import QLearningAgent
from reinforcement_learning.agents.sarsa_agent import SARSAAgent
from reinforcement_learning.agents.supervised_agent import SupervisedAgent
from utility.constants import (
    ALGO_SUPERVISED,
    GYM_TYPE,
    MARL_PZ,
    MARL_PZ_FROM_DATASET,
    SystemLevels,
    SystemModes,
    SystemStatus,
)
from utility.evaluation_summary import build_agent_evaluation_summary
from utility.training_summary import build_agent_training_summary
from utility.my_files import create_directory_training_execution, save_data_to_file
from utility.my_ho_statistics import (
    map_ho_status_id_to_class,
    plot_discrete_feature_bin_coverage,
    plot_ho_agent_execution_confusion_matrix,
    plot_ho_agent_execution_statuses,
    plot_ho_attack_mitigation_stats,
    plot_ho_agent_test,
    plot_ho_agent_test_errors,
    plot_ho_enviroment_execution_statutes,
    plot_ho_test_confusion_matrix,
    plot_qtable_coverage,
    plot_policy_exploration,
)
from utility.my_log import debug, error, information, notify_client
from utility.my_statistics import (
    plot_agent_cumulative_rewards,
    plot_combined_performance_over_time,
    plot_comparison_bar_charts,
    plot_metrics,
    plot_radar_chart,
)
from utility.scenario_generator import (
    DEFAULT_ATTACK_LIKELY_EVAL,
    DEFAULT_ATTACK_LIKELY_TRAIN,
    generate_and_save_scenario,
    load_scenario,
)
from utility.utils import ndarray_to_list

from .constants import AGENT_ACTIONS, COORDINATOR, COORDINATOR_ACTIONS
from .marl_pz_env import MarlPzEnv, SingleAgentView


# ────────────────────────────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────────────────────────────

def marl_pz_main(config, am: AgentManager, env: MarlPzEnv):
    """Entry point called from main.py."""
    try:
        # ── Phase 1: get or generate scenario ─────────────────────────
        scenario = _get_scenario(config, env)
        scenario_training   = list(scenario["training"])
        scenario_evaluation = list(scenario["evaluation"])
        del scenario

        # Stop background update_state thread (sequential replay drives its own updates)
        if hasattr(env, 'stop_update_event') and env.stop_update_event:
            env.stop_update_event.set()
        if hasattr(env, 'update_state_thread_instance'):
            env.update_state_thread_instance.join(timeout=5.0)

        # ── Phase 2 + 3: train and evaluate each agent ─────────────────
        agents_metrics = {}
        agents_summary = {}
        all_statuses   = []

        trainable = [
            agent for agent in am.agents_params
            if not (isinstance(getattr(agent, 'instance', None), SupervisedAgent)
                    or (getattr(agent, 'skip_learn', False)
                        and not getattr(agent, 'load', False)
                        and not getattr(agent, 'load_dir', None)))
        ]

        for agent in trainable:
            if env.stop_event.is_set():
                break

            if agent != trainable[0]:
                env.clean_network_state()

            if not getattr(agent, 'skip_learn', False):
                information(
                    f"\n{'='*60}\n"
                    f"  Agent: {agent.name}\n"
                    f"{'='*60}\n"
                )
                notify_client(
                    level=SystemLevels.STATUS,
                    status=SystemStatus.RUNNING,
                    mode=SystemModes.TRAINING,
                    message=f"Training {agent.name}...",
                )

                # ── Train ──────────────────────────────────────────────
                env.statuses   = []
                env.df         = list(scenario_training)
                env.min_accuracy = 2.0      # disable early termination
                env._env_initialized = False  # re-init on next reset

                _train_marl_agent(agent, env, scenario_training)
                env.min_accuracy = config.env_params.accuracy_min

                if env.statuses:
                    agent_dir = create_directory_training_execution(
                        config, agent_name=agent.name
                    )
                    save_data_to_file(list(env.statuses), agent_dir, "train_statuses")
                    all_statuses.extend(env.statuses)

                if config.env_params.print_training_chart:
                    notify_client(
                        level=SystemLevels.STATUS,
                        status=SystemStatus.RUNNING,
                        mode=SystemModes.PLOTTING,
                        message=f"Plotting training data for {agent.name}...",
                    )
                    data = _plot_and_save_agent(agent, config)
                    if data is not None:
                        agents_metrics[agent.name] = _get_agent_metrics(agent)

            # ── Evaluate ───────────────────────────────────────────────
            if not getattr(agent, 'skip_learn', False) or (
                    getattr(agent, 'load', False) and getattr(agent, 'load_dir', None)):
                notify_client(
                    level=SystemLevels.STATUS,
                    status=SystemStatus.RUNNING,
                    mode=SystemModes.EVALUATION,
                    message=f"Evaluating {agent.name}...",
                )
                env.statuses = []
                env.df = list(scenario_evaluation)
                env._env_initialized = False

                score, gt, pred, mitigation = _evaluate_marl_agent(
                    agent, env,
                    config.env_params.test_episodes,
                    config.env_params.max_steps,
                )

                if len(gt) == 0:
                    information(f"[{agent.name}] No evaluation data. Skipping.\n")
                    continue

                acc = accuracy_score(gt, pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    gt, pred, average='weighted', zero_division=0.0
                )
                agents_summary[agent.name] = {
                    'score':     score,
                    'accuracy':  acc,
                    'precision': precision,
                    'recall':    recall,
                    'f1_score':  f1,
                }

                directory_name = create_directory_training_execution(
                    config, "TEST_" + agent.name
                )
                try:
                    plot_ho_test_confusion_matrix(directory_name, gt, pred, agent.name)
                    plot_ho_agent_test(
                        {"ground_truth": gt, "predicted": {agent.name: pred}},
                        directory_name,
                    )
                    plot_ho_agent_test_errors(
                        {"ground_truth": gt, "predicted": {agent.name: pred}},
                        directory_name,
                        title=f"{agent.name} Evaluation Errors",
                    )
                    plot_ho_attack_mitigation_stats(
                        mitigation, directory_name,
                        title=f"{agent.name} Attack Mitigation",
                    )
                except Exception as exc:
                    error(Fore.RED
                          + f"Plotting test for {agent.name} failed: {exc}\n"
                          + traceback.format_exc() + Fore.WHITE)

                save_data_to_file(
                    {"ground_truth": gt, "predicted": {agent.name: pred},
                     "metrics": agents_summary[agent.name],
                     "mitigation_history": mitigation},
                    directory_name, "test",
                )
                try:
                    notify_client(
                        level=SystemLevels.DATA,
                        agent_name=agent.name,
                        agent_evaluation_summary=build_agent_evaluation_summary(
                            config=config,
                            agent_name=agent.name,
                            directory_name=directory_name,
                            metrics=agents_summary[agent.name],
                            score=score,
                            test_episodes=config.env_params.test_episodes,
                            mitigation_history=mitigation,
                            shared_directory=False,
                        ),
                    )
                except Exception as exc:
                    debug(Fore.YELLOW
                          + f"Unable to notify eval summary for {agent.name}: {exc}\n"
                          + Fore.WHITE)

                information(
                    f"\n[{agent.name}] "
                    f"Acc={acc*100:.1f}%  "
                    f"P={precision*100:.1f}%  "
                    f"R={recall*100:.1f}%  "
                    f"F1={f1*100:.1f}%\n"
                )
                if env.statuses:
                    all_statuses.extend(list(env.statuses))

        # ── Phase 4: summary ──────────────────────────────────────────
        if agents_metrics:
            try:
                plot_comparison_bar_charts(
                    config.training_execution_directory, agents_metrics
                )
                plot_radar_chart(
                    config.training_execution_directory, agents_metrics
                )
            except Exception as exc:
                debug(f"Comparison plots failed: {exc}\n")

        _print_summary(agents_summary)

        if agents_summary:
            try:
                notify_client(
                    level=SystemLevels.DATA,
                    final_data=ndarray_to_list(agents_summary),
                )
            except Exception:
                pass

        env.stop()

        if env.statuses:
            all_statuses.extend(list(env.statuses))
        if len(all_statuses) > 2:
            save_data_to_file(
                all_statuses, config.training_execution_directory, "statuses"
            )
            try:
                plot_ho_enviroment_execution_statutes(
                    all_statuses, config.training_execution_directory, "Statuses"
                )
            except Exception as exc:
                error(Fore.RED
                      + f"Error plotting env statuses: {exc}\n"
                      + traceback.format_exc() + Fore.WHITE)

    except Exception as exc:
        error(Fore.RED
              + f"marl_pz_main error: {exc}\n"
              + traceback.format_exc() + Fore.WHITE)
        env.stop()
        return
    finally:
        information(Fore.WHITE)
        notify_client(
            level=SystemLevels.STATUS,
            status=SystemStatus.FINISHED,
            mode=SystemModes.PLOTTING,
            message="Finished. Ready to start again.",
        )


# ────────────────────────────────────────────────────────────────────────────
# Scenario helpers
# ────────────────────────────────────────────────────────────────────────────

def _get_scenario(config, env: MarlPzEnv) -> dict:
    gym_type = config.env_params.gym_type

    if gym_type == MARL_PZ_FROM_DATASET:
        base_dir = os.path.dirname(config.env_params.data_traffic_file)
        scenario_path = os.path.join(base_dir, "scenario.json")
        return load_scenario(scenario_path)

    scenario_source = str(getattr(config.env_params, "scenario_source", "generate")).strip().lower()
    configured_scenario = str(getattr(config.env_params, "scenario_file", "") or "").strip()
    if scenario_source == "load" and configured_scenario:
        if not os.path.isabs(configured_scenario):
            configured_scenario = os.path.join(os.getcwd(), configured_scenario)
        return load_scenario(configured_scenario)

    scenario_path = os.path.join(config.training_execution_directory, "scenario.json")
    notify_client(
        level=SystemLevels.STATUS,
        status=SystemStatus.RUNNING,
        mode=SystemModes.TRAINING,
        message="Generating traffic scenario...",
    )

    attacks_config = getattr(config.env_params, "attacks", {})
    if isinstance(attacks_config, dict):
        train_likely = attacks_config.get("likely_train", DEFAULT_ATTACK_LIKELY_TRAIN)
        eval_likely  = attacks_config.get("likely_eval",  DEFAULT_ATTACK_LIKELY_EVAL)
    else:
        train_likely = getattr(attacks_config, "likely_train", DEFAULT_ATTACK_LIKELY_TRAIN)
        eval_likely  = getattr(attacks_config, "likely_eval",  DEFAULT_ATTACK_LIKELY_EVAL)

    return generate_and_save_scenario(
        env, config, scenario_path,
        train_attack_likely=train_likely,
        eval_attack_likely=eval_likely,
    )


# ────────────────────────────────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────────────────────────────────

def _train_marl_agent(agent, env: MarlPzEnv, scenario_training: list):
    """
    Train all instances of one agent.

    For tabular agents (Q-Learning / SARSA): joint-step training where every
    instance observes, acts, and updates its Q-table together each step, so all
    host gauges in the dashboard update simultaneously.

    For SB3 agents (DQN / PPO / A2C): sequential training via model.learn() per
    instance (SB3 manages its own rollout buffer internally).
    """
    from utility.constants import ALGO_Q_LEARNING, ALGO_SARSA

    start = time.time()
    algo = getattr(agent, 'algorithm', '').lower()
    is_tabular = algo in (ALGO_Q_LEARNING, ALGO_SARSA)

    try:
        if is_tabular:
            information(f"Starting joint training (all agents step together)\n", agent.name)
            _train_marl_joint_tabular(agent, env, scenario_training)
        else:
            information(f"Starting training (sequential per instance)\n", agent.name)
            for agent_id, inst_info in agent.instances.items():
                if env.stop_event.is_set():
                    break

                information(f"  Training agent instance: {agent_id}\n", agent.name)

                env.df = list(scenario_training)
                env._env_initialized = False

                model = inst_info['instance']
                for episode in range(agent.episodes):
                    if env.stop_event.is_set():
                        break
                    agent_cb = inst_info.get('custom_callback', {})
                    if hasattr(agent_cb, 'before_episode'):
                        agent_cb.before_episode(episode + 1)
                    model.learn(
                        total_timesteps=inst_info.get('max_steps', env.max_steps),
                        callback=agent_cb,
                        progress_bar=getattr(agent, 'progress_bar', False),
                    )
                    if hasattr(agent_cb, 'after_episode'):
                        agent_cb.after_episode()
    except Exception as exc:
        error(f"[{agent.name}] training error: {exc}\n{traceback.format_exc()}")

    agent.elapsed_time = time.time() - start
    information(f"Training completed in {agent.elapsed_time:.1f}s\n", agent.name)


def _train_marl_joint_tabular(agent, env: MarlPzEnv, scenario_training: list):
    """
    Joint-step training for tabular (Q-Learning / SARSA) marl_pz agents.

    All instances step the environment together each step:
      1. Every agent discretizes its observation and chooses an action.
      2. The env advances once with all actions simultaneously.
      3. Each agent updates its own Q-table from its individual (s, a, r, s') tuple.
      4. manage_step_data() is called for every agent → all gauges refresh each step.

    The scenario df is consumed exactly once across all episodes
    (episodes × max_steps pops total).
    """
    from reinforcement_learning.agents.qlearning_agent import QLearningAgent
    from reinforcement_learning.agents.sarsa_agent import SARSAAgent

    instances  = agent.instances
    stop_event = env.stop_event

    # Load the scenario once; each step() call pops one entry via update_state()
    env.df = list(scenario_training)
    env._env_initialized = False

    for episode in range(agent.episodes):
        if stop_event.is_set():
            break

        information(f"  Episode {episode + 1}/{agent.episodes}\n", agent.name)

        # Reset the env once for ALL agents
        obs_dict, _ = env.reset()

        # Initialise per-agent episode accumulators
        for inst_info in instances.values():
            m = inst_info['instance']
            m.episode            = episode + 1
            m.current_step       = 0
            m.correct_predictions = 0
            m.rewards            = []
            m.ground_truth       = []
            m.predicted          = []
            m.episode_statuses   = []
            m.exploration_count  = 0
            m.exploitation_count = 0

        # Discretise initial observations
        disc_states: dict = {}
        for agent_id, inst_info in instances.items():
            disc_states[agent_id] = env.get_discretized_state_for_agent(
                agent_id, obs_dict[agent_id]
            )

        # SARSA: pre-select the first action before entering the step loop
        sarsa_next: dict = {}
        for agent_id, inst_info in instances.items():
            m = inst_info['instance']
            if isinstance(m, SARSAAgent):
                sarsa_next[agent_id] = m.choose_action(disc_states[agent_id])

        done = truncated = False

        while not done and not truncated and not stop_event.is_set():

            # ── Choose actions ────────────────────────────────────────────
            actions: dict = {}
            for agent_id, inst_info in instances.items():
                m = inst_info['instance']
                m.current_step += 1
                if isinstance(m, SARSAAgent):
                    action = sarsa_next[agent_id]   # already chosen
                elif isinstance(m, QLearningAgent):
                    action = m.choose_action(disc_states[agent_id])
                else:
                    action = 0
                actions[agent_id] = action

            # ── Single env step with all agents' actions ──────────────────
            next_obs_dict, rewards, terms, truncs, infos = env.step(actions)
            done      = any(terms.values())
            truncated = any(truncs.values())

            # ── Per-agent Q-table update and gauge notification ───────────
            for agent_id, inst_info in instances.items():
                m = inst_info['instance']

                next_state = env.get_discretized_state_for_agent(
                    agent_id, next_obs_dict[agent_id]
                )

                if isinstance(m, QLearningAgent):
                    m.update_qtable(disc_states[agent_id], actions[agent_id],
                                    rewards[agent_id], next_state)
                elif isinstance(m, SARSAAgent):
                    next_action = m.choose_action(next_state)
                    m.update_qtable(disc_states[agent_id], actions[agent_id],
                                    rewards[agent_id], next_state, next_action)
                    sarsa_next[agent_id] = next_action

                m.exploration_rate  *= m.exploration_decay
                disc_states[agent_id] = next_state

                # Each agent gets its own copy — manage_step_data mutates the dict
                agent_status = dict(env.global_state.status)

                # Send step data → updates this agent's gauge in the dashboard
                m.manage_step_data(
                    actions[agent_id], rewards[agent_id], infos[agent_id], agent_status
                )

            obs_dict = next_obs_dict

        # ── End-of-episode metrics for each agent ────────────────────────
        for inst_info in instances.values():
            m = inst_info['instance']
            m.evaluate_episode(
                m.episode, sum(m.rewards),
                m.exploration_count, m.exploitation_count,
            )
            m.print_metrics(sum(m.rewards))


# ────────────────────────────────────────────────────────────────────────────
# Evaluation (coordinated — all agents act together)
# ────────────────────────────────────────────────────────────────────────────

def _evaluate_marl_agent(agent, env: MarlPzEnv,
                          test_episodes: int, max_steps: int):
    """
    Evaluate one agent configuration with all instances acting in coordination.

    Returns (score, ground_truth_list, predicted_list, mitigation_history).
    All predictions are per-host (coordinator is tracked but excluded from
    per-host metrics to stay comparable with the HO scenario).
    """
    score = 0
    ground_truth = []
    predicted    = []
    mitigation_history = []

    instances: dict = agent.instances   # {agent_id: {instance, is_custom_agent, single_view, ...}}

    for episode in range(test_episodes):
        if env.stop_event.is_set():
            break

        information(
            f"\n*** Eval episode {episode+1}/{test_episodes} [{agent.name}] ***\n"
        )

        obs_dict, _ = env.reset()

        # Advance to almost done so each eval episode is 1 round
        env._step_count = env.max_steps - 1

        # Collect host ground truths BEFORE stepping
        host_agents = [a for a in env.possible_agents if a != COORDINATOR]
        for host_id in host_agents:
            host_idx = next(
                (i for i, h in enumerate(env.hosts) if h.name == host_id), 0
            )
            gt_hs = env.global_state.host_statuses.get(host_id, {})
            raw_gt_id = gt_hs.get('id', 0) if isinstance(gt_hs, dict) else 0
            gt_class = map_ho_status_id_to_class(raw_gt_id)
            ground_truth.append(gt_class)

        # Build coordinated actions
        actions: dict = {}
        for agent_id, inst_info in instances.items():
            model = inst_info['instance']
            single_view: SingleAgentView = inst_info['single_view']
            raw_obs = env._get_obs_raw(agent_id)

            if inst_info['is_custom_agent']:
                action = int(model.predict(raw_obs))
            else:
                norm_obs = env._get_obs_normalized(agent_id)
                action, _ = model.predict(norm_obs, deterministic=True)
                action = int(action)

            actions[agent_id] = action

        # Record host predictions
        for host_id in host_agents:
            action = actions.get(host_id, 0)
            gt_class = ground_truth[-(len(host_agents)) + host_agents.index(host_id)]
            predicted.append(action)
            if action == gt_class:
                score += 1

            color = Fore.GREEN if action == gt_class else Fore.RED
            information(
                f"  {agent.name} | {host_id}"
                + color
                + f" pred={action} gt={gt_class}"
                + Fore.WHITE + "\n"
            )

        # Step with coordinated actions
        obs_dict, rewards, terms, truncs, infos = env.step(actions)

        # Track mitigation
        under_attack = 0
        mitigated    = 0
        for host_id in host_agents:
            gt_hs = env.global_state.host_statuses.get(host_id, {})
            gt_id = gt_hs.get('id', 0) if isinstance(gt_hs, dict) else 0
            action = actions.get(host_id, 0)
            if gt_id == 1:   # under_attack
                under_attack += 1
                if action == AGENT_ACTIONS.ATTACK_IN:
                    mitigated += 1
            elif gt_id == 2: # attacking
                under_attack += 1
                if action == AGENT_ACTIONS.ATTACK_OUT:
                    mitigated += 1

        mitigation_history.append({
            "episode": episode + 1,
            "under_attack_count": under_attack,
            "mitigated_under_attack_count": mitigated,
            "mitigated_under_attack_ratio": mitigated / max(1, under_attack),
        })

    return score, ground_truth, predicted, mitigation_history


# ────────────────────────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────────────────────────

def _print_summary(agents_summary: dict):
    if not agents_summary:
        return
    line = "─" * 70
    information(f"\n{line}\n  MARL_PZ EXPERIMENT SUMMARY\n{line}\n")
    header = f"{'Agent':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}"
    information(header + "\n" + line + "\n")
    for name, m in agents_summary.items():
        row = (
            f"{name:<25} "
            f"{m['accuracy']*100:>8.1f}% "
            f"{m['precision']*100:>9.1f}% "
            f"{m['recall']*100:>7.1f}% "
            f"{m['f1_score']*100:>7.1f}%"
        )
        information(row + "\n")
    information(line + "\n")
    if agents_summary:
        best_acc = max(agents_summary, key=lambda n: agents_summary[n]['accuracy'])
        best_f1  = max(agents_summary, key=lambda n: agents_summary[n]['f1_score'])
        information(f"  Best accuracy: {best_acc}\n")
        information(f"  Best F1-score: {best_f1}\n")
        information(line + "\n")


# ────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ────────────────────────────────────────────────────────────────────────────

def _get_agent_metrics(agent):
    # marl_pz: metrics live on each instance inside agent.instances
    if hasattr(agent, 'instances') and agent.instances:
        for inst_info in agent.instances.values():
            m = inst_info.get('instance')
            if m and hasattr(m, 'metrics') and m.metrics.get('accuracy'):
                return m.metrics
        return {}
    if not getattr(agent, 'is_custom_agent', True):
        return getattr(agent, 'custom_callback', {}).metrics if hasattr(agent, 'custom_callback') else {}
    return getattr(agent, 'instance', type('', (), {'metrics': {}})()).metrics


def _plot_and_save_agent(agent, config):
    """Plot and save training data for one agent. Returns data object or None."""
    import gc

    # Resolve the primary model instance (first non-coordinator for marl_pz)
    if hasattr(agent, 'instances') and agent.instances:
        # marl_pz: pick first instance with metrics
        primary = None
        for inst_info in agent.instances.values():
            m = inst_info.get('instance')
            if m and hasattr(m, 'metrics') and m.metrics.get('accuracy'):
                primary = m
                break
        if primary is None:
            error(Fore.RED + f"Agent {agent.name} has no metrics, skipping.\n" + Fore.WHITE)
            return None
        is_custom = True
        instance  = primary
    elif not getattr(agent, 'is_custom_agent', True):
        cb = getattr(agent, 'custom_callback', None)
        if cb is None or len(cb.metrics.get('accuracy', [])) == 0:
            error(Fore.RED + f"Agent {agent.name} has no metrics, skipping.\n" + Fore.WHITE)
            return None
        agent.instance.metrics    = cb.metrics
        agent.instance.indicators = cb.indicators
        is_custom = False
        instance  = agent.instance
    else:
        instance = getattr(agent, 'instance', None)
        if instance is None or not hasattr(instance, 'metrics') or \
                len(instance.metrics.get('accuracy', [])) == 0:
            error(Fore.RED + f"Agent {agent.name} has no metrics, skipping.\n" + Fore.WHITE)
            return None
        is_custom = True

    data = type('', (), {})()
    data.train_execution_time = agent.elapsed_time
    data.train_metrics        = instance.metrics
    data.train_indicators     = instance.indicators
    if hasattr(instance, 'train_types'):
        data.train_types = instance.train_types

    directory_name = create_directory_training_execution(config, agent_name=agent.name)

    if getattr(agent, 'save', False) and hasattr(instance, 'save'):
        instance.save(directory_name + "/" + agent.name)

    information("Plotting training data\n", agent.name)

    if len(data.train_indicators) > 2:
        for fn, label in [
            (plot_agent_cumulative_rewards,            "cumulative rewards"),
            (plot_ho_agent_execution_confusion_matrix, "confusion matrix"),
            (plot_ho_agent_execution_statuses,         "episode statuses"),
        ]:
            try:
                fn(data.train_indicators, directory_name,
                   *([agent.name] if fn in (plot_agent_cumulative_rewards,
                                            plot_ho_agent_execution_statuses) else []))
            except Exception as exc:
                error(Fore.RED + f"Error plotting {label} for {agent.name}: {exc}\n" + Fore.WHITE)

    if is_custom \
            and hasattr(instance, 'qtable_coverage_history') \
            and len(instance.qtable_coverage_history) > 0:
        try:
            plot_qtable_coverage(
                data.train_indicators,
                instance.qtable_coverage_history,
                directory_name, agent.name,
            )
            plot_discrete_feature_bin_coverage(
                data.train_indicators, directory_name,
                agent_name=agent.name,
                n_bins=getattr(config.env_params, "n_bins", 4),
            )
        except Exception as exc:
            error(Fore.RED + f"Q-table coverage plot error for {agent.name}: {exc}\n" + Fore.WHITE)

    if not is_custom \
            and hasattr(agent, 'custom_callback') \
            and hasattr(agent.custom_callback, 'exploration_metric_history') \
            and len(agent.custom_callback.exploration_metric_history) > 0:
        try:
            plot_policy_exploration(
                data.train_indicators,
                agent.custom_callback.exploration_metric_history,
                directory_name, agent.name,
            )
        except Exception as exc:
            error(Fore.RED + f"Policy exploration plot error for {agent.name}: {exc}\n" + Fore.WHITE)

    for fn, label in [
        (plot_combined_performance_over_time, "combined performance"),
        (plot_metrics,                        "train metrics"),
    ]:
        try:
            fn(data.train_metrics, directory_name,
               agent.name + (" Combined performance over time"
                             if fn == plot_combined_performance_over_time
                             else " Train metrics"))
        except Exception as exc:
            error(Fore.RED + f"Error plotting {label} for {agent.name}: {exc}\n" + Fore.WHITE)

    save_data_to_file(data.__dict__, directory_name)

    for ind in data.train_indicators:
        ind['episode_statuses'] = None
    gc.collect()

    try:
        notify_client(
            level=SystemLevels.DATA,
            agent_name=agent.name,
            agent_training_summary=build_agent_training_summary(
                config=config,
                agent_name=agent.name,
                directory_name=directory_name,
                train_metrics=data.train_metrics,
                train_indicators=data.train_indicators,
                train_execution_time=data.train_execution_time,
            ),
        )
    except Exception as exc:
        debug(Fore.YELLOW
              + f"Unable to notify training summary for {agent.name}: {exc}\n"
              + Fore.WHITE)

    information(f"Data saved\n", agent.name)
    return data
