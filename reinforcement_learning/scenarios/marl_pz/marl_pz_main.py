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
from stable_baselines3.common.utils import configure_logger

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
    plot_qtable_coverage_per_host,
    plot_policy_exploration,
)
from utility.my_log import debug, error, information, notify_client
from utility.my_statistics import (
    plot_agent_cumulative_rewards,
    plot_agent_cumulative_rewards_per_host,
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

from .constants import AGENT_ACTIONS, ALERT_COMM_COLUMNS, COORDINATOR, COORDINATOR_ACTIONS, CommStrategy
from .marl_pz_env import MarlPzEnv, SingleAgentView


# ────────────────────────────────────────────────────────────────────────────
# Environment validation (must_check_env=True)
# ────────────────────────────────────────────────────────────────────────────

def _run_env_checks(env: MarlPzEnv, am: AgentManager, scenario_data: list) -> None:
    """Run PettingZoo parallel_api_test + SB3 check_env on every SingleAgentView.

    Must be called AFTER _get_scenario() so that env.df is populated and
    env.reset() can execute successfully.
    """
    from pettingzoo.test import parallel_api_test
    from stable_baselines3.common.env_checker import check_env as sb3_check_env

    # ── 1. PettingZoo parallel_api_test on the full MarlPzEnv ────────────
    information("\n[CHECK ENV] Running PettingZoo parallel_api_test (num_cycles=2)...\n")
    env.df = list(scenario_data)
    env._env_initialized = False
    try:
        parallel_api_test(env, num_cycles=2)
        information("[CHECK ENV] parallel_api_test PASSED ✓\n")
    except Exception as exc:
        information(f"[CHECK ENV] parallel_api_test FAILED ✗ → {exc}\n")
        raise
    finally:
        env.df = list(scenario_data)
        env._env_initialized = False

    # ── 2. SB3 check_env on each SingleAgentView ─────────────────────────
    information("[CHECK ENV] Running SB3 check_env on each SingleAgentView...\n")
    for agent_param in am.agents_params:
        for agent_id, inst_info in (getattr(agent_param, 'instances', None) or {}).items():
            single_view: SingleAgentView = inst_info.get('single_view')
            if single_view is None:
                continue
            env.df = list(scenario_data)
            env._env_initialized = False
            try:
                sb3_check_env(single_view, warn=True)
                information(f"[CHECK ENV]   check_env({agent_id}) PASSED ✓\n")
            except Exception as exc:
                information(f"[CHECK ENV]   check_env({agent_id}) FAILED ✗ → {exc}\n")
                raise

    env.df = list(scenario_data)
    env._env_initialized = False
    information("[CHECK ENV] All environment checks passed.\n\n")


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

        # ── Phase 1b: PZ + SB3 env checks (must_check_env=True only) ──
        if getattr(config.env_params, 'must_check_env', False):
            _run_env_checks(env, am, scenario_training)

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
                    agent_name=agent.name,
                )

                # ── Train ──────────────────────────────────────────────
                env.statuses   = []
                env.df         = list(scenario_training)
                env.min_accuracy = 2.0      # disable early termination
                env._env_initialized = False  # re-init on next reset

                alert_comm_rows, comm_events = _train_marl_agent(agent, env, scenario_training)
                env.min_accuracy = config.env_params.accuracy_min

                if env.statuses:
                    agent_dir = create_directory_training_execution(
                        config, agent_name=agent.name
                    )
                    save_data_to_file(list(env.statuses), agent_dir, "train_statuses")
                    all_statuses.extend(env.statuses)

                    comm_stats = _build_comm_stats_payload(
                        getattr(env, '_comm_strategy', CommStrategy.NONE),
                        alert_comm_rows, comm_events, env.hosts,
                    )
                    if comm_stats is not None:
                        save_data_to_file(comm_stats, agent_dir, "comm_stats")

                if config.env_params.print_training_chart:
                    notify_client(
                        level=SystemLevels.STATUS,
                        status=SystemStatus.RUNNING,
                        mode=SystemModes.PLOTTING,
                        message=f"Plotting training data for {agent.name}...",
                        agent_name=agent.name,
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
                    agent_name=agent.name,
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

def _train_marl_sb3_with_comm(agent, env: MarlPzEnv, scenario_training: list):
    """
    Episode-interleaved SB3 training for S2 / S3 / S4 communication strategies.

    Instead of training each instance for all episodes before moving to the next,
    all instances train one episode at a time. After each round:
      S2 federated_sync   → average policy network weights (FedAvg)
      S3 policy_exchange  → copy best policy to lagging peers
      S4 experience_sharing (DQN only) → share high-|reward| replay buffer entries
    PPO / A2C instances are excluded from S4 (no replay buffer).
    """
    strategy = env._comm_strategy
    comm_cfg  = env._comm_cfg
    comm_events: list = []

    _fed_interval  = 1       # for SB3, sync every episode (not every step)
    _pe_lag        = 10.0
    _es_td_thresh  = 1.0
    _es_share_size = 50

    if strategy == CommStrategy.FEDERATED_SYNC:
        fed_cfg = comm_cfg.get('federated_sync', {}) or {}
        # sync_interval for SB3 is in episodes; default every 1 episode
        _fed_interval = max(1, int(fed_cfg.get('sync_interval_episodes',
                                               fed_cfg.get('sync_interval', 1))))
    elif strategy == CommStrategy.POLICY_EXCHANGE:
        pe_cfg  = comm_cfg.get('policy_exchange', {}) or {}
        _pe_lag = float(pe_cfg.get('lag_threshold', 10.0))
    elif strategy == CommStrategy.EXPERIENCE_SHARING:
        es_cfg         = comm_cfg.get('experience_sharing', {}) or {}
        _es_td_thresh  = float(es_cfg.get('td_error_threshold', 1.0))
        _es_share_size = int(es_cfg.get('buffer_share_size', 50))

    for episode in range(agent.episodes):
        if env.stop_event.is_set():
            break

        episode_rewards: dict = {}

        for agent_id, inst_info in agent.instances.items():
            if env.stop_event.is_set():
                break

            env.df = list(scenario_training)
            env._env_initialized = False

            model    = inst_info['instance']
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
            episode_rewards[agent_id] = _get_sb3_last_episode_reward(inst_info)

        # ── Post-episode communication hooks ─────────────────────────────
        new_events: list = []
        if strategy == CommStrategy.FEDERATED_SYNC and (episode + 1) % _fed_interval == 0:
            new_events = _federated_sync_sb3_weights(agent.instances)

        if strategy == CommStrategy.POLICY_EXCHANGE:
            new_events = _policy_exchange_sb3_end_of_episode(agent.instances, episode_rewards, _pe_lag)

        if strategy == CommStrategy.EXPERIENCE_SHARING:
            new_events = _experience_sharing_dqn(agent.instances, _es_td_thresh, _es_share_size)

        for ev in new_events:
            ev['episode'] = episode + 1
            ev['step'] = None
            comm_events.append(ev)
            notify_client(
                level=SystemLevels.DATA, agent_name=agent.name,
                comm_event=ev, force_immediate=True,
            )

    return [], comm_events


# On-policy algorithms (no replay buffer). DQN (off-policy) now runs every
# communication strategy through the joint per-step loop
# (_train_marl_joint_sb3_naive); these constants only matter for PPO/A2C, which
# have no replay buffer and can't join that loop.
_ON_POLICY_ALGOS = frozenset({'ppo', 'a2c'})
# PPO/A2C + these strategies fall back to the episode-interleaved path
# (_train_marl_sb3_with_comm) since there's no replay buffer to step jointly.
# EXPERIENCE_SHARING (S4) is included for round-robin training UX parity with
# the other strategies, even though the sync itself is a no-op for on-policy
# agents (no replay buffer — see the S4 warning above and
# _experience_sharing_dqn's internal off-policy filter).
_COMM_STRATEGIES_REQUIRING_SB3_INTERLEAVE = frozenset({
    CommStrategy.FEDERATED_SYNC,
    CommStrategy.POLICY_EXCHANGE,
    CommStrategy.EXPERIENCE_SHARING,
})
# PPO/A2C + these strategies have no communication-aware training path at all
# (see the sequential fallback branch below) — kept only for the warning
# message shown when an on-policy agent lands there.
_JOINT_SB3_ELIGIBLE_STRATEGIES = frozenset({
    CommStrategy.NONE, CommStrategy.NAIVE_BROADCAST, CommStrategy.UAQ,
})


def _train_marl_joint_sb3_naive(agent, env: MarlPzEnv, scenario_training: list):
    """
    Joint-step training for off-policy SB3 agents (DQN) — handles ALL implemented
    communication strategies (NONE / NAIVE_BROADCAST / UAQ / FEDERATED_SYNC /
    POLICY_EXCHANGE / EXPERIENCE_SHARING).

    All instances observe and act together every step — one env.step() per round —
    eliminating the N×time multiplication of the sequential/episode-interleaved
    paths. S2/S3/S4 sync hooks are embedded directly in the step loop, mirroring
    _train_marl_joint_tabular exactly: FEDERATED_SYNC and EXPERIENCE_SHARING are
    step-gated (sync_interval / exchange_interval, in environment steps — same
    unit as tabular, NOT episodes), POLICY_EXCHANGE runs once at the end of each
    episode.

    UAQ note: entropy-based confidence gating (see _inject_uaq_uncertainties) only
    has a meaningful signal for tabular agents' Q-values. SB3 agents here always
    inject entropy=1.0 (fully uncertain), so their alerts are filtered out by the
    coordinator exactly as documented in docs/marl_pz_communication.md — this
    matches the tabular joint loop's fallback behaviour for non-tabular instances.

    On-policy models (PPO, A2C) are not supported here — they have no replay
    buffer to step through this way; callers must fall back to
    _train_marl_sb3_with_comm (S2/S3) or the sequential path (NONE/NAIVE_BROADCAST/UAQ).
    """
    instances  = agent.instances
    stop_event = env.stop_event
    strategy   = env._comm_strategy
    comm_cfg   = env._comm_cfg
    alert_comm_rows: list = []
    comm_events: list = []

    # ── Per-strategy config extraction (same units as _train_marl_joint_tabular:
    # sync_interval / exchange_interval are environment STEPS, not episodes) ──
    _fed_interval  = 100
    _pe_lag        = 10.0
    _es_interval   = 50
    _es_td_thresh  = 1.0
    _es_share_size = 50

    if strategy == CommStrategy.FEDERATED_SYNC:
        _fed_cfg      = comm_cfg.get('federated_sync', {}) or {}
        _fed_interval = int(_fed_cfg.get('sync_interval', 100))
        information(f"[S2 FedSync] sync_interval={_fed_interval} steps\n", agent.name)
    elif strategy == CommStrategy.POLICY_EXCHANGE:
        _pe_cfg = comm_cfg.get('policy_exchange', {}) or {}
        _pe_lag = float(_pe_cfg.get('lag_threshold', 10.0))
        information(f"[S3 PolicyEx] lag_threshold={_pe_lag}\n", agent.name)
    elif strategy == CommStrategy.EXPERIENCE_SHARING:
        _es_cfg        = comm_cfg.get('experience_sharing', {}) or {}
        _es_interval   = int(_es_cfg.get('exchange_interval', 50))
        _es_td_thresh  = float(_es_cfg.get('td_error_threshold', 1.0))
        _es_share_size = int(_es_cfg.get('buffer_share_size', 50))
        information(
            f"[S4 ExpShare] exchange_interval={_es_interval} "
            f"td_threshold={_es_td_thresh} share_size={_es_share_size}\n",
            agent.name,
        )

    step_count = 0

    def _emit_comm_events(new_events: list, episode_num: int, step_num) -> None:
        for ev in new_events:
            ev['episode'] = episode_num
            ev['step'] = step_num
            comm_events.append(ev)
            notify_client(
                level=SystemLevels.DATA, agent_name=agent.name,
                comm_event=ev, force_immediate=True,
            )

    env.df = list(scenario_training)
    env._env_initialized = False

    total_steps = agent.episodes * env.max_steps

    # Initialise SB3 bookkeeping so _on_step() and train() work correctly.
    # Calling _on_step()/train() directly (bypassing model.learn(), which
    # normally runs _setup_learn() first) leaves self._logger unset — SB3's
    # logger property then raises AttributeError as soon as _on_step() (DQN)
    # or train() tries to record a metric. Set it up manually here.
    #
    # Dashboard/metric bookkeeping (manage_step_data, evaluate_episode, ...)
    # lives on each instance's CustomCallback, not on the raw SB3 model — SB3
    # only invokes those methods itself via callback.on_step() inside
    # model.learn(), which this joint-step loop never calls. We drive the
    # callback manually below, mirroring what SB3 would do per step.
    for inst_info in instances.values():
        m = inst_info['instance']
        if not hasattr(m, 'replay_buffer'):
            continue
        m.num_timesteps = 0
        m._n_calls = 0
        m._total_timesteps = max(total_steps, 1)
        m._current_progress_remaining = 1.0
        if hasattr(m, 'exploration_initial_eps'):
            m.exploration_rate = m.exploration_initial_eps
        if getattr(m, '_logger', None) is None:
            m.set_logger(configure_logger(verbose=0))
        cb = inst_info.get('custom_callback')
        if cb is not None and hasattr(cb, 'init_callback'):
            cb.init_callback(m)

    for episode in range(agent.episodes):
        if stop_event.is_set():
            break

        information(f"  Episode {episode + 1}/{agent.episodes}\n", agent.name)

        obs_dict, _ = env.reset()

        for inst_info in instances.values():
            cb = inst_info.get('custom_callback')
            if cb is not None and hasattr(cb, 'before_episode'):
                cb.before_episode(episode + 1)

        episode_rewards: dict = {aid: 0.0 for aid in instances}

        done = truncated = False

        while not done and not truncated and not stop_event.is_set():
            actions: dict    = {}
            obs_arrays: dict = {}

            for agent_id, inst_info in instances.items():
                m   = inst_info['instance']
                obs = np.array(obs_dict[agent_id], dtype=np.float32)
                obs_arrays[agent_id] = obs

                if hasattr(m, 'predict'):
                    action, _ = m.predict(obs.reshape(1, -1), deterministic=False)
                    actions[agent_id] = int(action.item() if hasattr(action, 'item') else action[0])
                else:
                    actions[agent_id] = 0

            # ── UAQ: inject per-agent entropy before env step ─────────────
            # No tabular instances exist in this path, so this always sets
            # entropy=1.0 for every agent (see docstring above / disc_states={}
            # is safe here since _inject_uaq_uncertainties only reads it for
            # QLearningAgent/SARSAAgent instances).
            if strategy == CommStrategy.UAQ:
                _inject_uaq_uncertainties(env, instances, {})

            next_obs_dict, rewards, terms, truncs, infos = env.step(actions)
            done      = any(terms.values())
            truncated = any(truncs.values())

            for agent_id, inst_info in instances.items():
                m = inst_info['instance']
                next_obs_np = np.array(next_obs_dict[agent_id], dtype=np.float32).reshape(1, -1)

                if hasattr(m, 'replay_buffer'):
                    obs_np = obs_arrays[agent_id].reshape(1, -1)
                    m.replay_buffer.add(
                        obs_np,
                        next_obs_np,
                        np.array([[actions[agent_id]]]),
                        np.array([rewards[agent_id]]),
                        np.array([done or truncated]),
                        [infos[agent_id]],
                    )
                    m.num_timesteps += 1
                    m._update_current_progress_remaining(m.num_timesteps, m._total_timesteps)

                    if (m.num_timesteps > m.learning_starts
                            and m.replay_buffer.size() >= m.batch_size):
                        m.train(gradient_steps=m.gradient_steps, batch_size=m.batch_size)

                    if hasattr(m, '_on_step'):
                        m._on_step()

                cb = inst_info.get('custom_callback')
                if cb is not None and hasattr(cb, 'on_step'):
                    cb.locals = {
                        'actions': [actions[agent_id]],
                        'rewards': [rewards[agent_id]],
                        'infos':   [infos[agent_id]],
                        'new_obs': next_obs_np,
                    }
                    cb.on_step()

                episode_rewards[agent_id] += float(rewards[agent_id])

            obs_dict = next_obs_dict
            step_count += 1

            # ── S2: federated policy-weight averaging every N steps ───────
            if strategy == CommStrategy.FEDERATED_SYNC and step_count % _fed_interval == 0:
                _emit_comm_events(_federated_sync_sb3_weights(instances), episode + 1, step_count)

            # ── S4: share high-|reward| replay-buffer transitions every N steps ─
            if strategy == CommStrategy.EXPERIENCE_SHARING and step_count % _es_interval == 0:
                _emit_comm_events(
                    _experience_sharing_dqn(instances, _es_td_thresh, _es_share_size),
                    episode + 1, step_count,
                )

        # ── S3: policy exchange at end of episode ─────────────────────────
        if strategy == CommStrategy.POLICY_EXCHANGE:
            _emit_comm_events(
                _policy_exchange_sb3_end_of_episode(instances, episode_rewards, _pe_lag),
                episode + 1, None,
            )

        for inst_info in instances.values():
            cb = inst_info.get('custom_callback')
            if cb is not None and hasattr(cb, 'after_episode'):
                cb.after_episode()

        if strategy in (CommStrategy.NAIVE_BROADCAST, CommStrategy.UAQ):
            for host_idx, h in enumerate(env.hosts):
                hm = env._ep_msgs_per_host.get(h.name, {'total': 0, 'confident': 0, 'uncertain': 0})
                alert_comm_rows.append(np.array(
                    [episode + 1, host_idx, hm['total'], hm['confident'], hm['uncertain']],
                    dtype=np.int32,
                ))

    return alert_comm_rows, comm_events


def _train_marl_agent(agent, env: MarlPzEnv, scenario_training: list):
    """
    Train all instances of one agent.

    For tabular agents (Q-Learning / SARSA): joint-step training where every
    instance observes, acts, and updates its Q-table together each step, so all
    host gauges in the dashboard update simultaneously.

    For SB3 agents (DQN / PPO / A2C):
      - Off-policy (DQN), ANY communication strategy: joint-step training via
        _train_marl_joint_sb3_naive — one env.step() per round, O(E×S) total.
        S2/S3/S4 sync hooks run inline in that loop (same as tabular).
        (UAQ entropy defaults to 1.0 for SB3 agents — see that function's docstring.)
      - On-policy (PPO, A2C) + S2/S3/S4: episode-interleaved training via
        _train_marl_sb3_with_comm (post-episode weight sync / policy exchange;
        S4 sync is a no-op for them — see the warning below) — no replay
        buffer, so the joint per-step loop isn't available to them.
      - On-policy (PPO, A2C) + NONE/NAIVE_BROADCAST/UAQ: sequential per instance
        (on-policy rollout buffers cannot be shared across agents easily).

    S4 warning: PPO and A2C have no replay buffer (on-policy). When S4 is
    selected they are skipped silently during the DQN buffer-sharing step;
    the training itself is unaffected.
    """
    from utility.constants import ALGO_Q_LEARNING, ALGO_SARSA

    start = time.time()
    algo       = getattr(agent, 'algorithm', '').lower()
    is_tabular = algo in (ALGO_Q_LEARNING, ALGO_SARSA)
    strategy   = getattr(env, '_comm_strategy', CommStrategy.NONE)

    # ── S4 incompatibility warning for on-policy agents ──────────────────
    if (strategy == CommStrategy.EXPERIENCE_SHARING
            and not is_tabular
            and algo in _ON_POLICY_ALGOS):
        information(
            f"[WARNING] S4 'experience_sharing' requires a replay buffer. "
            f"{agent.algorithm.upper()} is on-policy (rollout buffer only) — "
            f"experience sharing will be skipped for this agent. "
            f"Training proceeds normally; the coordinator remains active.\n",
            agent.name,
        )

    alert_comm_rows: list = []
    comm_events: list = []

    try:
        if is_tabular:
            information(f"Starting joint training (all agents step together)\n", agent.name)
            alert_comm_rows, comm_events = _train_marl_joint_tabular(agent, env, scenario_training)

        elif algo not in _ON_POLICY_ALGOS:
            # DQN (off-policy): joint per-step loop handles ALL communication
            # strategies now — S2/S3/S4 sync hooks run inline (see that
            # function's docstring), same efficiency as the tabular joint loop.
            information(
                f"Starting joint training (all agents step together, off-policy)\n",
                agent.name,
            )
            alert_comm_rows, comm_events = _train_marl_joint_sb3_naive(agent, env, scenario_training)

        elif strategy in _COMM_STRATEGIES_REQUIRING_SB3_INTERLEAVE:
            # PPO/A2C (on-policy) + S2/S3/S4: no replay buffer, so the joint
            # per-step loop isn't available — fall back to the slower
            # episode-interleaved path. S4 sync is a no-op for these algos
            # (see the warning above) but they still train here for round-robin
            # progress UX parity with the other strategies.
            information(
                f"Starting interleaved training "
                f"(S2/S3/S4 — on-policy, one instance trains a full episode at a time)\n",
                agent.name,
            )
            alert_comm_rows, comm_events = _train_marl_sb3_with_comm(agent, env, scenario_training)

        else:
            if strategy in _JOINT_SB3_ELIGIBLE_STRATEGIES and algo in _ON_POLICY_ALGOS:
                information(
                    f"[WARNING] Joint training requires an off-policy model "
                    f"(replay buffer). {algo.upper()} is on-policy — falling back "
                    f"to sequential per-instance training.\n",
                    agent.name,
                )
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

                    if strategy in (CommStrategy.NAIVE_BROADCAST, CommStrategy.UAQ):
                        for host_idx, h in enumerate(env.hosts):
                            hm = env._ep_msgs_per_host.get(
                                h.name, {'total': 0, 'confident': 0, 'uncertain': 0}
                            )
                            alert_comm_rows.append(np.array(
                                [episode + 1, host_idx, hm['total'],
                                 hm['confident'], hm['uncertain']],
                                dtype=np.int32,
                            ))
    except Exception as exc:
        error(f"[{agent.name}] training error: {exc}\n{traceback.format_exc()}")

    agent.elapsed_time = time.time() - start
    information(f"Training completed in {agent.elapsed_time:.1f}s\n", agent.name)

    return alert_comm_rows, comm_events


# ────────────────────────────────────────────────────────────────────────────
# Communication strategy helpers (S2 / S3 / S4)
# ────────────────────────────────────────────────────────────────────────────

def _host_tabular_agents(instances: dict) -> list:
    """Return [(agent_id, model)] for all tabular HOST agents (coordinator excluded)."""
    return [
        (aid, info['instance'])
        for aid, info in instances.items()
        if aid != COORDINATOR and isinstance(info['instance'], (QLearningAgent, SARSAAgent))
    ]


def _host_sb3_agents(instances: dict) -> list:
    """Return [(agent_id, model)] for all SB3 HOST agents (coordinator excluded)."""
    return [
        (aid, info['instance'])
        for aid, info in instances.items()
        if aid != COORDINATOR
        and not isinstance(info['instance'], (QLearningAgent, SARSAAgent))
        and hasattr(info['instance'], 'policy')
    ]


def _get_sb3_last_episode_reward(inst_info: dict) -> float:
    """Best-effort: read last episode reward from an SB3 agent's custom callback."""
    cb = inst_info.get('custom_callback')
    if cb is None:
        return 0.0
    metrics = getattr(cb, 'metrics', {})
    for key in ('reward', 'total_reward', 'cumulative_reward'):
        vals = metrics.get(key, [])
        if vals:
            return float(vals[-1])
    return 0.0


def _federated_sync_qtables(instances: dict) -> list:
    """S2 — average Q-tables across all tabular host agents in-place.

    Returns a list with one sync event dict (episode/step added by the caller),
    or [] if fewer than 2 tabular host agents are present.
    """
    agents = _host_tabular_agents(instances)
    if len(agents) < 2:
        return []
    tables = np.array([m.q_table for _, m in agents])
    avg = tables.mean(axis=0)
    for _, m in agents:
        np.copyto(m.q_table, avg)
    return [{
        'family': 'policy_coordination',
        'strategy': CommStrategy.FEDERATED_SYNC,
        'eventType': 'fedavg_sync',
        'participants': [aid for aid, _ in agents],
        'detail': {'synced_count': len(agents)},
    }]


def _federated_sync_sb3_weights(instances: dict) -> list:
    """S2 — FedAvg: average policy network weights across all SB3 host agents in-place.

    Returns a list with one sync event dict, or [] if fewer than 2 SB3 host agents.
    """
    import copy
    agents = _host_sb3_agents(instances)
    if len(agents) < 2:
        return []
    state_dicts = [m.policy.state_dict() for _, m in agents]
    avg_sd = copy.deepcopy(state_dicts[0])
    for key in avg_sd:
        avg_sd[key] = (
            sum(sd[key].float() for sd in state_dicts) / len(state_dicts)
        ).to(state_dicts[0][key].dtype)
    for _, m in agents:
        m.policy.load_state_dict(avg_sd)
    return [{
        'family': 'policy_coordination',
        'strategy': CommStrategy.FEDERATED_SYNC,
        'eventType': 'fedavg_sync',
        'participants': [aid for aid, _ in agents],
        'detail': {'synced_count': len(agents)},
    }]


def _policy_exchange_end_of_episode(instances: dict, lag_threshold: float) -> list:
    """S3 tabular — copy best host's Q-table to peers that lag by more than lag_threshold.

    Must be called BEFORE m.rewards is cleared (i.e. before evaluate_episode).
    Returns a list with one sync event dict, or [] if no agent actually lagged.
    """
    agents = _host_tabular_agents(instances)
    if len(agents) < 2:
        return []
    agent_map = dict(agents)
    ep_rewards = {
        aid: float(sum(m.rewards)) if m.rewards else 0.0
        for aid, m in agents
    }
    best_aid = max(ep_rewards, key=ep_rewards.get)
    best_r   = ep_rewards[best_aid]
    best_m   = agent_map[best_aid]
    targets = []
    for aid, m in agents:
        if aid == best_aid:
            continue
        if best_r - ep_rewards[aid] > lag_threshold:
            np.copyto(m.q_table, best_m.q_table)
            targets.append(aid)
    if not targets:
        return []
    from utility.my_log import debug as _dbg
    _dbg(
        f"[PolicyEx] best={best_aid} r={best_r:.1f} → "
        f"copied to {len(targets)} lagging agent(s)\n"
    )
    return [{
        'family': 'policy_coordination',
        'strategy': CommStrategy.POLICY_EXCHANGE,
        'eventType': 'policy_copy',
        'participants': [best_aid] + targets,
        'detail': {'source': best_aid, 'targets': targets,
                   'best_reward': best_r, 'lag_threshold': lag_threshold},
    }]


def _policy_exchange_sb3_end_of_episode(instances: dict,
                                          episode_rewards: dict,
                                          lag_threshold: float) -> list:
    """S3 SB3 — copy best policy weights to SB3 peers that lag by more than lag_threshold.

    Returns a list with one sync event dict, or [] if no agent actually lagged.
    """
    import copy
    agents = _host_sb3_agents(instances)
    if len(agents) < 2:
        return []
    best_aid = max((aid for aid, _ in agents),
                   key=lambda a: episode_rewards.get(a, 0.0))
    best_r  = episode_rewards.get(best_aid, 0.0)
    best_m  = dict(agents)[best_aid]
    best_sd = copy.deepcopy(best_m.policy.state_dict())
    targets = []
    for aid, m in agents:
        if aid == best_aid:
            continue
        if best_r - episode_rewards.get(aid, 0.0) > lag_threshold:
            m.policy.load_state_dict(best_sd)
            targets.append(aid)
    if not targets:
        return []
    from utility.my_log import debug as _dbg
    _dbg(
        f"[PolicyEx SB3] best={best_aid} r={best_r:.1f} → "
        f"copied weights to {len(targets)} lagging agent(s)\n"
    )
    return [{
        'family': 'policy_coordination',
        'strategy': CommStrategy.POLICY_EXCHANGE,
        'eventType': 'policy_copy',
        'participants': [best_aid] + targets,
        'detail': {'source': best_aid, 'targets': targets,
                   'best_reward': best_r, 'lag_threshold': lag_threshold},
    }]


def _experience_sharing_collect(exp_pool: dict, agent_id: str, m,
                                 state: tuple, action: int,
                                 reward: float, next_state: tuple) -> None:
    """S4 — append one (s, a, r, s', td_error) tuple to the agent's pool."""
    try:
        q_sa      = float(m.q_table[state + (action,)])
        q_next    = float(m.q_table[next_state].max())
        td_error  = abs(reward + m.discount_factor * q_next - q_sa)
    except Exception:
        td_error = abs(reward)
    exp_pool.setdefault(agent_id, []).append(
        (state, action, reward, next_state, td_error)
    )


def _experience_sharing_dqn(instances: dict,
                             td_threshold: float, share_size: int) -> list:
    """S4 DQN — share high-|reward| transitions from each DQN agent's replay buffer to peers.

    |reward| is used as a proxy for TD-error: high-magnitude rewards indicate
    surprise (attack detected / false alarm / miss) which are the most informative
    transitions for learning. PPO/A2C agents are silently skipped (no replay buffer).

    Returns a list with one sync event dict per source agent that actually shared
    transitions (empty if none qualified).
    """
    off_policy = [
        (aid, info['instance'])
        for aid, info in instances.items()
        if aid != COORDINATOR and hasattr(info['instance'], 'replay_buffer')
    ]
    if len(off_policy) < 2:
        return []
    events = []
    for src_aid, src_m in off_policy:
        buf = src_m.replay_buffer
        n   = buf.buffer_size if buf.full else buf.pos
        if n == 0:
            continue
        # All available indices, sorted by |reward| descending
        all_idxs = list(range(buf.buffer_size if buf.full else buf.pos))
        qualifying = [
            i for i in all_idxs
            if float(abs(buf.rewards[i])) >= td_threshold
        ]
        if not qualifying:
            continue
        qualifying.sort(key=lambda i: float(abs(buf.rewards[i])), reverse=True)
        to_share = qualifying[:share_size]
        targets = []
        for tgt_aid, tgt_m in off_policy:
            if tgt_aid == src_aid:
                continue
            targets.append(tgt_aid)
            n_envs = buf.observations[0].shape[0]
            for idx in to_share:
                try:
                    tgt_m.replay_buffer.add(
                        buf.observations[idx],
                        buf.next_observations[idx],
                        buf.actions[idx],
                        buf.rewards[idx],
                        buf.dones[idx],
                        [{}] * n_envs,
                    )
                except Exception:
                    pass
        events.append({
            'family': 'policy_coordination',
            'strategy': CommStrategy.EXPERIENCE_SHARING,
            'eventType': 'experience_share',
            'participants': [src_aid] + targets,
            'detail': {'source': src_aid, 'targets': targets,
                       'shared_count': len(to_share), 'td_threshold': td_threshold},
        })
    return events


def _experience_sharing_broadcast(instances: dict, exp_pool: dict,
                                   td_threshold: float, share_size: int) -> list:
    """S4 — broadcast high-TD-error transitions from each host to its peers.

    Applies them as off-policy Q-learning updates and clears the pool.
    Returns a list with one sync event dict per source agent that actually shared
    transitions (empty if none qualified).
    """
    agents = _host_tabular_agents(instances)
    if len(agents) < 2:
        return []
    events = []
    for src_aid, _ in agents:
        pool = exp_pool.get(src_aid, [])
        high_td = [
            (s, a, r, s2)
            for s, a, r, s2, td in pool
            if td >= td_threshold
        ]
        if not high_td:
            exp_pool[src_aid] = []
            continue
        high_td = high_td[-share_size:]
        targets = []
        for tgt_aid, tgt_m in agents:
            if tgt_aid == src_aid:
                continue
            targets.append(tgt_aid)
            for s, a, r, s2 in high_td:
                try:
                    q_next = float(tgt_m.q_table[s2].max())
                    td_err = r + tgt_m.discount_factor * q_next - float(tgt_m.q_table[s + (a,)])
                    tgt_m.q_table[s + (a,)] += tgt_m.learning_rate * td_err
                except Exception:
                    pass
        exp_pool[src_aid] = []
        events.append({
            'family': 'policy_coordination',
            'strategy': CommStrategy.EXPERIENCE_SHARING,
            'eventType': 'experience_share',
            'participants': [src_aid] + targets,
            'detail': {'source': src_aid, 'targets': targets,
                       'shared_count': len(high_td), 'td_threshold': td_threshold},
        })
    return events


def _compute_q_entropy_normalized(q_values: np.ndarray) -> float:
    """Normalized Shannon entropy [0..1] from Q-values via softmax.

    Returns 1.0 (max uncertain) when all Q-values are equal (unexplored state),
    and 0.0 when all probability mass is on one action (fully certain).
    """
    n = len(q_values)
    if n <= 1:
        return 0.0
    q_norm = q_values - q_values.max()
    exp_q = np.exp(q_norm)
    probs = exp_q / (exp_q.sum() + 1e-9)
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    h_max = float(np.log(n))
    return float(np.clip(entropy / h_max, 0.0, 1.0)) if h_max > 0 else 0.0


def _inject_uaq_uncertainties(env: MarlPzEnv, instances: dict,
                               disc_states: dict) -> None:
    """Compute Q-value entropy for each tabular agent and push into env.

    Called each step, only when env._comm_strategy == CommStrategy.UAQ.
    DRL (SB3) agents default to 1.0 (fully uncertain) — their alerts
    are always filtered as 'uncertain' by the coordinator.
    """
    for agent_id, inst_info in instances.items():
        m = inst_info['instance']
        if not isinstance(m, (QLearningAgent, SARSAAgent)):
            env.set_agent_uncertainty(agent_id, 1.0)
            continue
        state = disc_states.get(agent_id)
        if state is None:
            env.set_agent_uncertainty(agent_id, 1.0)
            continue
        try:
            q_values = np.asarray(m.q_table[state], dtype=np.float64)
            entropy = _compute_q_entropy_normalized(q_values)
        except (IndexError, KeyError, TypeError):
            entropy = 1.0
        env.set_agent_uncertainty(agent_id, entropy)


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

    Communication strategy hooks (tabular only):
      S2 federated_sync     — Q-table averaging every sync_interval steps
      S3 policy_exchange    — copy best peer policy at end of each episode
      S4 experience_sharing — broadcast high-TD-error transitions every exchange_interval steps
    """
    from reinforcement_learning.agents.qlearning_agent import QLearningAgent
    from reinforcement_learning.agents.sarsa_agent import SARSAAgent

    instances  = agent.instances
    stop_event = env.stop_event
    strategy   = env._comm_strategy
    comm_cfg   = env._comm_cfg

    # ── Per-strategy config extraction ───────────────────────────────────────
    _fed_interval  = 100
    _pe_lag        = 10.0
    _es_interval   = 50
    _es_td_thresh  = 1.0
    _es_share_size = 50

    if strategy == CommStrategy.FEDERATED_SYNC:
        _fed_cfg      = comm_cfg.get('federated_sync', {}) or {}
        _fed_interval = int(_fed_cfg.get('sync_interval', 100))
        information(
            f"[S2 FedSync] sync_interval={_fed_interval} steps\n", agent.name
        )
    elif strategy == CommStrategy.POLICY_EXCHANGE:
        _pe_cfg = comm_cfg.get('policy_exchange', {}) or {}
        _pe_lag = float(_pe_cfg.get('lag_threshold', 10.0))
        information(
            f"[S3 PolicyEx] lag_threshold={_pe_lag}\n", agent.name
        )
    elif strategy == CommStrategy.EXPERIENCE_SHARING:
        _es_cfg        = comm_cfg.get('experience_sharing', {}) or {}
        _es_interval   = int(_es_cfg.get('exchange_interval', 50))
        _es_td_thresh  = float(_es_cfg.get('td_error_threshold', 1.0))
        _es_share_size = int(_es_cfg.get('buffer_share_size', 50))
        information(
            f"[S4 ExpShare] exchange_interval={_es_interval} "
            f"td_threshold={_es_td_thresh} share_size={_es_share_size}\n",
            agent.name,
        )

    # Load the scenario once; each step() call pops one entry via update_state()
    env.df = list(scenario_training)
    env._env_initialized = False

    step_count = 0                   # global step counter for S2 / S4
    exp_pool: dict = {}              # {agent_id: [transitions]} for S4
    alert_comm_rows: list = []       # per-episode/per-host alert counts (NAIVE_BROADCAST/UAQ)
    comm_events: list = []           # discrete sync events (S2/S3/S4)

    def _emit_comm_events(new_events: list, episode_num: int, step_num) -> None:
        for ev in new_events:
            ev['episode'] = episode_num
            ev['step'] = step_num
            comm_events.append(ev)
            notify_client(
                level=SystemLevels.DATA, agent_name=agent.name,
                comm_event=ev, force_immediate=True,
            )

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

            # ── UAQ: inject per-agent entropy before env step ─────────────
            if strategy == CommStrategy.UAQ:
                _inject_uaq_uncertainties(env, instances, disc_states)

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

                # ── S4: collect transition AFTER Q-update ─────────────────
                if strategy == CommStrategy.EXPERIENCE_SHARING:
                    _experience_sharing_collect(
                        exp_pool, agent_id, m,
                        disc_states[agent_id], actions[agent_id],
                        rewards[agent_id], next_state,
                    )

                m.exploration_rate  *= m.exploration_decay
                disc_states[agent_id] = next_state

                # Each agent gets its own copy — manage_step_data mutates the dict
                agent_status = dict(env.global_state.status)

                # Send step data → updates this agent's gauge in the dashboard
                m.manage_step_data(
                    actions[agent_id], rewards[agent_id], infos[agent_id], agent_status
                )

            obs_dict = next_obs_dict
            step_count += 1

            # ── S2: federated Q-table averaging every N steps ─────────────
            if strategy == CommStrategy.FEDERATED_SYNC and step_count % _fed_interval == 0:
                _emit_comm_events(_federated_sync_qtables(instances), episode + 1, step_count)

            # ── S4: broadcast experience every N steps ────────────────────
            if strategy == CommStrategy.EXPERIENCE_SHARING and step_count % _es_interval == 0:
                _emit_comm_events(
                    _experience_sharing_broadcast(instances, exp_pool, _es_td_thresh, _es_share_size),
                    episode + 1, step_count,
                )

        # ── S3: policy exchange at end of episode (before rewards cleared) ─
        if strategy == CommStrategy.POLICY_EXCHANGE:
            _emit_comm_events(_policy_exchange_end_of_episode(instances, _pe_lag), episode + 1, None)

        # ── Message stats log (sanity check for NAIVE_BROADCAST vs UAQ) ──
        if strategy in (CommStrategy.NAIVE_BROADCAST, CommStrategy.UAQ):
            ms = env._ep_msgs
            if strategy == CommStrategy.UAQ:
                information(
                    f"  [msgs ep={episode+1}] "
                    f"host→coord total={ms['host_total']} "
                    f"confident={ms['confident']} "
                    f"uncertain(filtered)={ms['uncertain']}\n",
                    agent.name,
                )
            else:
                information(
                    f"  [msgs ep={episode+1}] "
                    f"host→coord total={ms['host_total']}\n",
                    agent.name,
                )
            for host_idx, h in enumerate(env.hosts):
                hm = env._ep_msgs_per_host.get(h.name, {'total': 0, 'confident': 0, 'uncertain': 0})
                alert_comm_rows.append(np.array(
                    [episode + 1, host_idx, hm['total'], hm['confident'], hm['uncertain']],
                    dtype=np.int32,
                ))

        # ── End-of-episode metrics for each agent ────────────────────────
        for inst_info in instances.values():
            m = inst_info['instance']
            m.evaluate_episode(
                m.episode, sum(m.rewards),
                m.exploration_count, m.exploitation_count,
            )
            m.print_metrics(sum(m.rewards))

    return alert_comm_rows, comm_events


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

        # Advance scenario state so GT reflects fresh eval data (reset() does not call update_state)
        env.update_state()

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

        # UAQ: inject entropy before the coordinated evaluation step
        if env._comm_strategy == CommStrategy.UAQ:
            for agent_id, inst_info in instances.items():
                m = inst_info['instance']
                if inst_info.get('is_custom_agent', True) and isinstance(
                        m, (QLearningAgent, SARSAAgent)):
                    raw_obs = env._get_obs_raw(agent_id)
                    disc_st = env.get_discretized_state_for_agent(agent_id, raw_obs)
                    try:
                        q_values = np.asarray(m.q_table[disc_st], dtype=np.float64)
                        entropy = _compute_q_entropy_normalized(q_values)
                    except Exception:
                        entropy = 1.0
                    env.set_agent_uncertainty(agent_id, entropy)
                else:
                    env.set_agent_uncertainty(agent_id, 1.0)

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
# Communication stats persistence
# ────────────────────────────────────────────────────────────────────────────

def _build_comm_stats_payload(comm_strategy: str, alert_comm_rows: list,
                               comm_events: list, hosts: list):
    """Build the comm_stats.json payload for one agent's training run.

    Returns None for CommStrategy.NONE/HIERARCHICAL (nothing to persist) or when
    the relevant family produced no data at all (e.g. run stopped before any
    episode completed).
    """
    if comm_strategy in (CommStrategy.NONE, CommStrategy.HIERARCHICAL):
        return None

    if comm_strategy in (CommStrategy.NAIVE_BROADCAST, CommStrategy.UAQ):
        if not alert_comm_rows:
            return None
        # Single conversion point to a compact int32 array — mirrors the
        # custom_callback.py episode_statuses precedent (never keep a running
        # list-of-dicts across the whole training run).
        alert_rows = np.array(alert_comm_rows, dtype=np.int32)
        return {
            'comm_strategy': comm_strategy,
            'family': 'alert',
            'alert_columns': list(ALERT_COMM_COLUMNS),
            'alert_rows': alert_rows,
            'host_names': [h.name for h in hosts],
        }

    # Policy-coordination family (federated_sync / policy_exchange / experience_sharing):
    # event count is bounded by episodes/sync_interval, not by steps — small
    # enough that a plain JSON list-of-dicts is fine, no numpy conversion needed.
    if not comm_events:
        return None
    return {
        'comm_strategy': comm_strategy,
        'family': 'policy_coordination',
        'sync_events': comm_events,
    }


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
    # marl_pz: metrics live on each instance inside agent.instances. Tabular
    # agents track metrics on the instance itself; SB3 agents (DQN/PPO/A2C)
    # track them on the instance's CustomCallback (usually already copied
    # onto the instance by _plot_and_save_agent, but check the callback too
    # in case this is ever called independently).
    if hasattr(agent, 'instances') and agent.instances:
        for inst_info in agent.instances.values():
            m  = inst_info.get('instance')
            cb = inst_info.get('custom_callback')
            if m and hasattr(m, 'metrics') and m.metrics.get('accuracy'):
                return m.metrics
            if cb and hasattr(cb, 'metrics') and cb.metrics.get('accuracy'):
                return cb.metrics
        return {}
    if not getattr(agent, 'is_custom_agent', True):
        return getattr(agent, 'custom_callback', {}).metrics if hasattr(agent, 'custom_callback') else {}
    return getattr(agent, 'instance', type('', (), {'metrics': {}})()).metrics


def _plot_and_save_agent(agent, config):
    """Plot and save training data for one agent. Returns data object or None."""
    import gc

    # Resolve the primary model instance (first non-coordinator for marl_pz),
    # while also collecting EVERY non-coordinator host instance's data for
    # plots that must reflect the whole team, not just one host (confusion
    # matrix / bin coverage aggregate across hosts; reward / qtable coverage
    # get one line per host — see host_indicators_by_name /
    # host_qtable_history_by_name usage below).
    resolved_callback = None
    host_indicators_by_name: dict = {}
    host_qtable_history_by_name: dict = {}
    if hasattr(agent, 'instances') and agent.instances:
        # marl_pz: pick first instance with metrics as the "primary" one used
        # for instance.save() / data.train_metrics / data.train_indicators
        # (the live-dashboard training summary and model persistence keep
        # using this single instance, unchanged). Tabular agents (Q-Learning/
        # SARSA) track metrics directly on the instance and have no callback
        # ({}). SB3 agents (DQN/PPO/A2C) track metrics on their CustomCallback,
        # not on the raw model — mirror the single-agent SB3 branch below by
        # copying the callback's tracked data onto the model instance, so
        # `instance` stays the model (needed for `.save()`) while still
        # exposing `.metrics`/`.indicators`.
        primary = None
        is_custom = True
        for agent_id, inst_info in agent.instances.items():
            if agent_id == COORDINATOR:
                # Binary (0/1) action space — incompatible with the 3-class
                # (Normal/Under Attack/Attacking) host confusion matrix and
                # bin coverage, so it's excluded from the aggregation below.
                continue
            m  = inst_info.get('instance')
            cb = inst_info.get('custom_callback')
            inst_is_custom = inst_info.get('is_custom_agent', True)
            if cb and hasattr(cb, 'metrics') and cb.metrics.get('accuracy'):
                m.metrics    = cb.metrics
                m.indicators = cb.indicators
                if hasattr(cb, 'train_types'):
                    m.train_types = cb.train_types
                inst_is_custom = False
            elif not (m and hasattr(m, 'metrics') and m.metrics.get('accuracy')):
                continue

            host_indicators_by_name[agent_id] = m.indicators
            if inst_is_custom and getattr(m, 'qtable_coverage_history', None):
                host_qtable_history_by_name[agent_id] = m.qtable_coverage_history

            if primary is None:
                primary = m
                resolved_callback = cb if not inst_is_custom else None
                is_custom = inst_is_custom
        if primary is None:
            error(Fore.RED + f"Agent {agent.name} has no metrics, skipping.\n" + Fore.WHITE)
            return None
        instance = primary
    elif not getattr(agent, 'is_custom_agent', True):
        cb = getattr(agent, 'custom_callback', None)
        if cb is None or len(cb.metrics.get('accuracy', [])) == 0:
            error(Fore.RED + f"Agent {agent.name} has no metrics, skipping.\n" + Fore.WHITE)
            return None
        agent.instance.metrics    = cb.metrics
        agent.instance.indicators = cb.indicators
        is_custom = False
        instance  = agent.instance
        resolved_callback = cb
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

    # Concatenate every non-coordinator host's per-episode indicators for the
    # plots that must aggregate across the whole team (confusion matrix, bin
    # coverage) — both functions already flatten `episode_statuses` across
    # all entries regardless of source, so concatenation is all that's
    # needed, no changes to the plotting functions themselves.
    combined_indicators = []
    for host_indicators in host_indicators_by_name.values():
        combined_indicators.extend(host_indicators)
    if not combined_indicators:
        combined_indicators = data.train_indicators

    if len(data.train_indicators) > 2:
        try:
            if len(host_indicators_by_name) > 1:
                plot_agent_cumulative_rewards_per_host(
                    host_indicators_by_name, directory_name, agent.name
                )
            else:
                plot_agent_cumulative_rewards(data.train_indicators, directory_name, agent.name)
        except Exception as exc:
            error(Fore.RED + f"Error plotting cumulative rewards for {agent.name}: {exc}\n" + Fore.WHITE)

        try:
            plot_ho_agent_execution_confusion_matrix(combined_indicators, directory_name)
        except Exception as exc:
            error(Fore.RED + f"Error plotting confusion matrix for {agent.name}: {exc}\n" + Fore.WHITE)

        try:
            plot_ho_agent_execution_statuses(data.train_indicators, directory_name, agent.name)
        except Exception as exc:
            error(Fore.RED + f"Error plotting episode statuses for {agent.name}: {exc}\n" + Fore.WHITE)

    if is_custom \
            and hasattr(instance, 'qtable_coverage_history') \
            and len(instance.qtable_coverage_history) > 0:
        try:
            if len(host_qtable_history_by_name) > 1:
                plot_qtable_coverage_per_host(
                    host_qtable_history_by_name, host_indicators_by_name,
                    directory_name, agent.name,
                )
            else:
                plot_qtable_coverage(
                    data.train_indicators,
                    instance.qtable_coverage_history,
                    directory_name, agent.name,
                )
            plot_discrete_feature_bin_coverage(
                combined_indicators, directory_name,
                agent_name=agent.name,
                n_bins=getattr(config.env_params, "n_bins", 4),
                include_pct_var=getattr(
                    getattr(config.env_params, 'attacks', None),
                    'include_percentage_variations', True,
                ),
            )
        except Exception as exc:
            error(Fore.RED + f"Q-table coverage plot error for {agent.name}: {exc}\n" + Fore.WHITE)

    if not is_custom \
            and resolved_callback is not None \
            and hasattr(resolved_callback, 'exploration_metric_history') \
            and len(resolved_callback.exploration_metric_history) > 0:
        try:
            plot_policy_exploration(
                data.train_indicators,
                resolved_callback.exploration_metric_history,
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
