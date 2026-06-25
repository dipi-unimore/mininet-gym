# attack_detect_ho.py
"""
Entry point for the per-host observable attack detection scenario (ATTACKS_HO).

Execution flow
--------------
The scenario always runs in SEQUENTIAL mode:

  Phase 1 — GENERATE SCENARIO
    A synthetic sequence of host task assignments is generated using the same
    probabilistic logic as the adversarial agent (choose_task_type) but without
    executing real traffic.  Two sequences are produced:
      • training:   episodes * max_steps steps
            • evaluation: test_episodes * 1 step (mono-step episodes)
    Both are saved immediately to <training_execution_directory>/scenario.json.

  Phase 2 — TRAIN each agent in sequence
    For each enabled RL agent (one at a time):
      • Load the training sequence into the env
      • The env replays each step by executing the planned task on the real
        Mininet network, waiting ~1 s, then reading OVS counters as usual
      • The RL agent observes and learns via the PerHostScanWrapper
      • done-prematurely is disabled: every agent always runs all steps

  Phase 3 — EVALUATE each agent in sequence
    For each agent, replay the evaluation sequence the same way.

  Phase 4 — SUMMARY
    Print a comparative table of all agents' metrics to the console.

ATTACKS_HO_FROM_DATASET
-----------------------
  Load scenario.json from the directory configured in env_params.data_traffic_file.
  The file must have been produced by a previous ATTACKS_HO run.
  If scenario.json is missing, a clear error is raised.
"""
import os
import time
import traceback
from colorama import Fore
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from reinforcement_learning.agent_manager import AgentManager
from .network_env_attack_detect_per_host_observable import (
    NetworkEnvAttackDetectPerHostObservable,
)
from .per_host_scan_wrapper import (
    PerHostScanWrapper,
)
from utility.scenario_generator import (
    DEFAULT_ATTACK_LIKELY_EVAL,
    DEFAULT_ATTACK_LIKELY_TRAIN,
    generate_and_save_scenario,
    load_scenario,
)
from utility.evaluation_summary import build_agent_evaluation_summary
from utility.training_summary import build_agent_training_summary
from utility.my_ho_statistics import map_ho_status_id_to_class
from reinforcement_learning.agents.qlearning_agent import QLearningAgent
from reinforcement_learning.agents.sarsa_agent import SARSAAgent
from reinforcement_learning.agents.supervised_agent import SupervisedAgent
from utility.constants import (
    ALGO_SUPERVISED,
    ATTACKS_HO,
    ATTACKS_HO_FROM_DATASET,
    GYM_TYPE,
    SystemLevels,
    SystemModes,
    SystemStatus,
)
from utility.my_files import (
    create_directory_training_execution,
    save_data_to_file,
)
from utility.my_ho_statistics import (
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
    plot_train_types,
)
from utility.utils import ndarray_to_list


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def attack_detect_ho_main(config, am: AgentManager,
                           wrapped_env: PerHostScanWrapper):
    """
    Main entry point.  am.env is already set to wrapped_env by main.py.
    """
    base_env: NetworkEnvAttackDetectPerHostObservable = wrapped_env.unwrapped

    try:
        # ── Phase 1: get or generate scenario ─────────────────────────
        #TODO check if it is from dataset, in this case we don not need scenario.json,
        # but just the statuses_attacks_ho.json, so we can directly load it without the need of the scenario file
        scenario = _get_scenario(config, base_env)
        # Extract the two lists and release the full JSON dict from memory
        scenario_training   = list(scenario["training"])
        scenario_evaluation = list(scenario["evaluation"])
        del scenario

        # ── Stop background update_state thread before sequential replay ───
        # In sequential mode each round drives its own update_state() calls via
        # the wrapper.  The background thread (update_state_thread) would also
        # call update_state() on its own timer, consuming additional df entries
        # and exhausting the scenario dataset prematurely (observed: attacks
        # disappear after ~70 of 100 training episodes).
        if hasattr(base_env, 'stop_update_event') and base_env.stop_update_event:
            base_env.stop_update_event.set()
        if hasattr(base_env, 'update_state_thread_instance'):
            base_env.update_state_thread_instance.join(timeout=5.0)

        # ── Phase 2 + 3: train and evaluate each agent sequentially ───
        agents_metrics  = {}
        agents_summary  = {}
        all_statuses    = []   # accumulates (agent_name, statuses) for root statuses.json

        trainable = [
            agent for agent in am.agents_params
            if not (isinstance(agent.instance, SupervisedAgent)
                    or (agent.skip_learn and not agent.load and not agent.load_dir)  )
        ]

        for agent in trainable:
            if base_env.stop_event.is_set():
                break

            # Clean network state from previous agent training
            if agent != trainable[0]:  # Skip cleanup before first agent
                base_env.clean_network_state()

            _configure_wrapper_for_agent_state(agent, wrapped_env)

            if not agent.skip_learn:
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

                # ── Train ──────────────────────────────────────────────────
                # Reset statuses so each agent's statuses are independent.
                base_env.statuses    = []
                base_env.df          = list(scenario_training)
                base_env.min_accuracy = 2.0   # disable early termination
                train_agent.env       = wrapped_env

                train_agent(agent)
                base_env.min_accuracy = config.env_params.accuracy_min  # restore

                # ── Save per-agent training statuses ───────────────────────
                if base_env.statuses:
                    agent_dir = create_directory_training_execution(
                        config, agent_name=agent.name
                    )
                    agent_statuses = list(base_env.statuses)
                    save_data_to_file(agent_statuses, agent_dir, "train_statuses")
                    all_statuses.extend(agent_statuses)

                if config.env_params.print_training_chart:
                    notify_client(
                        level=SystemLevels.STATUS,
                        status=SystemStatus.RUNNING,
                        mode=SystemModes.PLOTTING,
                        message=f"Plotting training data for {agent.name}...",
                    )
                    data = plot_and_save_data_agent(agent, config)
                    if data is not None:
                        agents_metrics[agent.name] = agent.instance.metrics

            # ── Evaluate ───────────────────────────────────────────────
            if not agent.skip_learn or (agent.load and agent.load_dir):
                notify_client(
                    level=SystemLevels.STATUS,
                    status=SystemStatus.RUNNING,
                    mode=SystemModes.EVALUATION,
                    message=f"Evaluating {agent.name}...",
                )
                _configure_wrapper_for_agent_state(agent, wrapped_env)
                base_env.statuses = []
                base_env.df = list(scenario_evaluation)
                score, gt, pred, mitigation_history = _evaluate_single_agent(
                    agent, wrapped_env, base_env,
                    config.env_params.test_episodes,
                    config.env_params.max_steps,
                )

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
                    plot_ho_test_confusion_matrix(directory_name, gt, pred,
                                                agent.name)
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
                        mitigation_history,
                        directory_name,
                        title=f"{agent.name} Attack Mitigation",
                    )
                except Exception as e:
                    error(Fore.RED + f"Error plotting test for {agent.name}!\n"
                        + f"{e}\n{traceback.format_exc()}\n" + Fore.WHITE)

                save_data_to_file(
                    {"ground_truth": gt, "predicted": {agent.name: pred},
                    "metrics": agents_summary[agent.name],
                    "mitigation_history": mitigation_history},
                    directory_name, "test"
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
                            mitigation_history=mitigation_history,
                            shared_directory=False,
                        ),
                    )
                except Exception as exc:
                    debug(Fore.YELLOW + f"Unable to notify evaluation summary for {agent.name}: {exc}\n" + Fore.WHITE)
                information(
                    f"\n[{agent.name}] "
                    f"Acc={acc*100:.1f}%  "
                    f"P={precision*100:.1f}%  "
                    f"R={recall*100:.1f}%  "
                    f"F1={f1*100:.1f}%\n"
                )
                # Accumulate evaluation statuses for root statuses.json
                if base_env.statuses:
                    all_statuses.extend(list(base_env.statuses))

        # Supervised agents
        for agent in am.agents_params:
            if agent.algorithm.lower() == ALGO_SUPERVISED \
                    and len(all_statuses) > 0:
                if not hasattr(agent.instance, "train"):
                    warning_message = (
                        f"Agent {agent.name} is not supported in {ATTACKS_HO} "
                        f"in this environment (missing train method). Skipping."
                    )
                    information(Fore.YELLOW + warning_message + "\n" + Fore.WHITE)
                    notify_client(
                        level=SystemLevels.STATUS,
                        status=SystemStatus.RUNNING,
                        mode=SystemModes.TRAINING,
                        message=warning_message,
                    )
                    continue

                information(f"Training supervised agent {agent.name}\n")
                agent.instance.train(all_statuses)

        # ── Phase 4: comparative summary ──────────────────────────────
        if len(agents_metrics) > 0:
            plot_comparison_bar_charts(
                config.training_execution_directory, agents_metrics
            )
            plot_radar_chart(
                config.training_execution_directory, agents_metrics
            )

        _print_summary(agents_summary)

        # Send last agent's data to UI (or aggregate)
        if agents_summary:
            try:
                notify_client(
                    level=SystemLevels.DATA,
                    final_data=ndarray_to_list(agents_summary),
                )
            except Exception:
                pass

        # Plot environment statuses (all agents combined)
        base_env.stop()
        # Flush any remaining statuses from the last agent
        if base_env.statuses:
            all_statuses.extend(list(base_env.statuses))
        if len(all_statuses) > 2:
            save_data_to_file(
                all_statuses, config.training_execution_directory, "statuses"
            )
            try:
                plot_ho_enviroment_execution_statutes(
                    all_statuses, config.training_execution_directory, "Statuses"
                )
            except Exception as e:
                error(Fore.RED
                      + f"Error plotting environment statuses!\n{e}\n"
                      + f"{traceback.format_exc()}\n" + Fore.WHITE)

    except Exception as e:
        error(Fore.RED + f"Something went wrong!\n{e}\n"
              f"{traceback.format_exc()}\n" + Fore.WHITE)
        base_env.stop()
        return
    finally:
        information(Fore.WHITE)
        notify_client(
            level=SystemLevels.STATUS,
            status=SystemStatus.FINISHED,
            mode=SystemModes.PLOTTING,
            message="Finished. Ready to start again.",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Scenario helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_scenario(config, base_env) -> dict:
    """
    Return the scenario dict.

    - ATTACKS_HO:              generate a new scenario and save it.
    - ATTACKS_HO_FROM_DATASET: load scenario.json from the configured path.
    """
    gym_type = config.env_params.gym_type

    if gym_type == ATTACKS_HO_FROM_DATASET:
        # Load from the directory specified in data_traffic_file
        # (strip statuses_attacks_ho.json suffix if present, add scenario.json)
        #TODO from dataset probably we don't need the scenario file, 
        # but just the statuses_attacks_ho.json, so we can directly load it without the need of the scenario file    
        base_dir = os.path.dirname(config.env_params.data_traffic_file)
        scenario_path = os.path.join(base_dir, "scenario.json")
        return load_scenario(scenario_path)

    scenario_source = str(getattr(config.env_params, "scenario_source", "generate")).strip().lower()
    configured_scenario = str(getattr(config.env_params, "scenario_file", "") or "").strip()
    if scenario_source == "load" and configured_scenario:
        if not os.path.isabs(configured_scenario):
            configured_scenario = os.path.join(os.getcwd(), configured_scenario)
        return load_scenario(configured_scenario)

    # ATTACKS_HO — generate fresh scenario
    scenario_path = os.path.join(
        config.training_execution_directory, "scenario.json"
    )
    notify_client(
        level=SystemLevels.STATUS,
        status=SystemStatus.RUNNING,
        mode=SystemModes.TRAINING,
        message="Generating traffic scenario...",
    )
    
    # Read attack likelihood values from config, fallback to defaults
    attacks_config = getattr(config.env_params, "attacks", {})
    if isinstance(attacks_config, dict):
        train_likely = attacks_config.get("likely_train", DEFAULT_ATTACK_LIKELY_TRAIN)
        eval_likely = attacks_config.get("likely_eval", DEFAULT_ATTACK_LIKELY_EVAL)
    else:
        train_likely = getattr(attacks_config, "likely_train", DEFAULT_ATTACK_LIKELY_TRAIN)
        eval_likely = getattr(attacks_config, "likely_eval", DEFAULT_ATTACK_LIKELY_EVAL)
    
    return generate_and_save_scenario(
        base_env,
        config,
        scenario_path,
        train_attack_likely=train_likely,
        eval_attack_likely=eval_likely,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_agent(agent):
    """Training loop — custom agents or SB3."""
    start_time = time.time()
    try:
        information(f"Starting training\n", agent.name)
        if agent.is_custom_agent:
            agent.instance.learn(agent.episodes, train_agent.env.stop_event)
        else:
            # Check if this is a SupervisedAgent
            is_supervised = isinstance(agent.instance, SupervisedAgent)

            for episode in range(agent.episodes):
                if train_agent.env.stop_event.is_set():
                    break

                if is_supervised:
                    # Supervised learning: accumulate and train per episode
                    # Get episode statuses from environment
                    episode_statuses = train_agent.env.statuses[-agent.max_steps:] if hasattr(train_agent.env, 'statuses') else []

                    if episode_statuses:
                        agent.instance.accumulate_statuses(episode_statuses)
                        accuracy = agent.instance.train_on_accumulated_per_episode()
                        information(f"Episode {episode+1} - Supervised Training Accuracy: {accuracy * 100:.2f}%\n", agent.name)
                        # Notify metrics
                        try:
                            notify_client(
                                level=SystemLevels.DATA,
                                agent_name=agent.name,
                                metrics={
                                    'episode': episode + 1,
                                    'accuracy': accuracy,
                                    'precision': agent.instance.precision,
                                    'recall': agent.instance.recall,
                                    'f1_score': agent.instance.fscore,
                                }
                            )
                        except Exception as e:
                            debug(f"Error notifying metrics: {e}\n")
                else:
                    # Regular RL agent learning
                    agent.custom_callback.before_episode(episode + 1)
                    agent.instance.learn(
                        total_timesteps=agent.max_steps,
                        callback=agent.custom_callback,
                        progress_bar=agent.progress_bar,
                    )
                    agent.custom_callback.after_episode()
    except Exception as e:
        error(f"Agent {agent.name} training error: {e}\n"
              f"{traceback.format_exc()}")

    agent.elapsed_time = time.time() - start_time
    information(f"Training completed in {agent.elapsed_time:.1f}s\n",
                agent.name)


def _get_deep_agent_state_mode(agent) -> str:
    configured_mode = getattr(agent, "state_input_mode", "normalized")
    configured_mode = str(configured_mode).strip().lower()
    return "raw" if configured_mode == "raw" else "normalized"


def _configure_wrapper_for_agent_state(agent, wrapped_env: PerHostScanWrapper):
    # Tabular agents require raw observations for discretization consistency.
    if isinstance(agent.instance, (QLearningAgent, SARSAAgent, SupervisedAgent)):
        state_mode = "raw"
    else:
        state_mode = _get_deep_agent_state_mode(agent)

    wrapped_env.set_state_mode(state_mode)
    information(f"[{agent.name}] state_input_mode={state_mode}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation — single agent
# ──────────────────────────────────────────────────────────────────────────────

def _evaluate_single_agent(agent, wrapped_env: PerHostScanWrapper,
                            base_env, test_episodes: int,
                            max_steps: int):
    """
    Evaluate one agent on the evaluation sequence already loaded in base_env.df.
    Returns (score, ground_truth_list, predicted_list, mitigation_history).
    """
    score        = 0
    ground_truth = []
    predicted    = []
    model        = agent.instance
    mitigation_history = []

    for episode in range(test_episodes):
        if base_env.stop_event.is_set():
            break
            
        information(f"\n*** Eval episode {episode+1}/{test_episodes} "
                    f"[{agent.name}] ***\n")
        state, _ = wrapped_env.reset(options={"is_real_state": True})
        # Each evaluation episode is exactly 1 round.  By advancing the round
        # counter to max_steps-1 here, the wrapper sees done=True after the
        # single round and skips the trailing update_state() call that would
        # otherwise consume a second df step per episode (halving the usable
        # evaluation scenario).
        wrapped_env._current_round_step = wrapped_env.env.max_steps - 1
        last_infos = None

        for host_idx in range(wrapped_env.num_hosts):
            if base_env.stop_event.is_set():
                break
                
            raw_host_status_id = int(base_env.global_state.status["id"][host_idx])
            host_status_id = map_ho_status_id_to_class(raw_host_status_id)
            ground_truth.append(host_status_id)

            if isinstance(model, SupervisedAgent):
                action = int(model.predict(state))
            elif isinstance(model, (QLearningAgent, SARSAAgent)):
                action = int(model.predict(state))
            else:
                action, _ = model.predict(state, deterministic=True)
                action = int(action)

            predicted.append(action)
            if action == host_status_id:
                score += 1

            color = Fore.GREEN if action == host_status_id else Fore.RED
            information(
                f"  {agent.name} | {base_env.hosts[host_idx].name}"
                + color
                + f" pred={action} gt={host_status_id} sd={[int(i) for i in base_env.get_discretized_state(state)]}"
                + Fore.WHITE + "\n"
            )

            state, _, done, truncated, infos = wrapped_env.step(action)
            last_infos = infos
            # if done or truncated:
            #     break

        if isinstance(last_infos, dict):
            mitigation_history.append({
                "episode": episode + 1,
                "under_attack_count": int(last_infos.get("episode_under_attack_count", 0)),
                "mitigated_under_attack_count": int(last_infos.get("episode_mitigated_under_attack_count", 0)),
                "mitigated_under_attack_ratio": float(last_infos.get("episode_mitigated_under_attack_ratio", 0.0)),
            })
        else:
            mitigation_history.append({
                "episode": episode + 1,
                "under_attack_count": 0,
                "mitigated_under_attack_count": 0,
                "mitigated_under_attack_ratio": 0.0,
            })

    return score, ground_truth, predicted, mitigation_history


# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────

def _print_summary(agents_summary: dict):
    """Print a comparative summary table to the console."""
    if not agents_summary:
        return

    line = "─" * 70
    information(f"\n{line}\n  EXPERIMENT SUMMARY\n{line}\n")
    header = f"{'Agent':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}"
    information(header + "\n")
    information(line + "\n")

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

    # Best agent per metric
    best_acc = max(agents_summary, key=lambda n: agents_summary[n]['accuracy'])
    best_f1  = max(agents_summary, key=lambda n: agents_summary[n]['f1_score'])
    information(f"  Best accuracy : {best_acc}\n")
    information(f"  Best F1-score : {best_f1}\n")
    information(f"{line}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────────────

def plot_and_save_data_agent(agent, config):
    """Collect metrics, plot, save. Returns data object or None."""
    if not agent.is_custom_agent:
        if len(agent.custom_callback.metrics.get('accuracy', [])) == 0:
            error(Fore.RED
                  + f"Agent {agent.name} has no metrics, skipping.\n"
                  + Fore.WHITE)
            return None
        accuracy, precision, recall, f1 = agent.custom_callback.get_metrics()
        agent.instance.metrics    = agent.custom_callback.metrics
        agent.instance.indicators = agent.custom_callback.indicators
    else:
        if not hasattr(agent.instance, 'metrics') or \
                len(agent.instance.metrics.get('accuracy', [])) == 0:
            error(Fore.RED
                  + f"Agent {agent.name} has no metrics, skipping.\n"
                  + Fore.WHITE)
            return None
        accuracy, precision, recall, f1 = agent.instance.get_metrics()

    agent.metrics = {
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1_score": f1,
    }

    data = type('', (), {})()
    data.train_execution_time = agent.elapsed_time
    data.train_metrics        = agent.instance.metrics
    data.train_indicators     = agent.instance.indicators
    if hasattr(agent.instance, 'train_types'):
        data.train_types = agent.instance.train_types

    directory_name = create_directory_training_execution(
        config, agent_name=agent.name
    )
    if agent.save:
        agent.instance.save(directory_name + "/" + agent.name)

    information("Plotting training data\n", agent.name)

    if len(data.train_indicators) > 2:
        for fn, label in [
            (plot_agent_cumulative_rewards,         "cumulative rewards"),
            (plot_ho_agent_execution_confusion_matrix, "confusion matrix"),
            (plot_ho_agent_execution_statuses,      "episode statuses"),
        ]:
            try:
                fn(data.train_indicators, directory_name,
                   *([agent.name] if fn == plot_agent_cumulative_rewards
                     or fn == plot_ho_agent_execution_statuses else []))
            except Exception as e:
                error(Fore.RED
                      + f"Error plotting {label} for {agent.name}!\n"
                      + f"{e}\n{traceback.format_exc()}\n" + Fore.WHITE)

    # ── Q-table coverage (tabular agents) ─────────────────────────────
    if agent.is_custom_agent \
            and hasattr(agent.instance, 'qtable_coverage_history') \
            and len(agent.instance.qtable_coverage_history) > 0:
        try:
            plot_qtable_coverage(
                data.train_indicators,
                agent.instance.qtable_coverage_history,
                directory_name,
                agent.name,
            )
            plot_discrete_feature_bin_coverage(
                data.train_indicators,
                directory_name,
                agent_name=agent.name,
                n_bins=getattr(config.env_params, "n_bins", 4),
            )
        except Exception as e:
            error(Fore.RED
                  + f"Error plotting Q-table coverage for {agent.name}!\n"
                  + f"{e}\n{traceback.format_exc()}\n" + Fore.WHITE)

    # ── Policy exploration metrics (SB3 deep agents) ──────────────────
    if not agent.is_custom_agent \
            and hasattr(agent, 'custom_callback') \
            and hasattr(agent.custom_callback, 'exploration_metric_history') \
            and len(agent.custom_callback.exploration_metric_history) > 0:
        try:
            plot_policy_exploration(
                data.train_indicators,
                agent.custom_callback.exploration_metric_history,
                directory_name,
                agent.name,
            )
        except Exception as e:
            error(Fore.RED
                  + f"Error plotting policy exploration for {agent.name}!\n"
                  + f"{e}\n{traceback.format_exc()}\n" + Fore.WHITE)

    for fn, label in [
        (plot_combined_performance_over_time, "combined performance"),
        (plot_metrics,                        "train metrics"),
    ]:
        try:
            fn(data.train_metrics, directory_name,
               agent.name + (" Combined performance over time"
                             if fn == plot_combined_performance_over_time
                             else " Train metrics"))
        except Exception as e:
            error(Fore.RED
                  + f"Error plotting {label} for {agent.name}!\n"
                  + f"{e}\n{traceback.format_exc()}\n" + Fore.WHITE)

    if (hasattr(data, 'train_types')
            and len(data.train_types.get("explorations", [])) > 0):
        try:
            plot_train_types(
                data.train_types, data.train_execution_time, directory_name
            )
        except Exception as e:
            error(Fore.RED
                  + f"Error plotting train types for {agent.name}!\n"
                  + f"{e}\n{traceback.format_exc()}\n" + Fore.WHITE)

    save_data_to_file(data.__dict__, directory_name)

    # Free the per-step payload from each indicator — plotting and saving are done.
    # The episode-level summary fields (episode, steps, cumulative_reward, …) are kept.
    import gc
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
        debug(Fore.YELLOW + f"Unable to notify training summary for {agent.name}: {exc}\n" + Fore.WHITE)
    information(f"Data saved\n", agent.name)
    return data
