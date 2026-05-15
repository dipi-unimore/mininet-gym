# custom_callback.py
import time
import numpy as np
from colorama import Fore
from stable_baselines3.common.callbacks import BaseCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from reinforcement_learning.scenarios.marl.constants import COORDINATOR
from utility.constants import ATTACKS_HO, ALGO_DQN, ALGO_PPO, ALGO_A2C, GYM_TYPE, SystemLevels
from utility.my_log import debug, notify_client, information


class CustomCallback(BaseCallback):
    """
    Custom SB3 callback that tracks per-step metrics, indicators,
    training type (exploration vs exploitation), and policy exploration
    metrics across episodes.

    Policy exploration metrics (equivalent to Q-table coverage for deep agents)
    --------------------------------------------------------------------------
    The concept of "how much of the state space has been explored" does not
    directly translate to deep RL agents that use implicit function approximation.
    We instead track two proxies every N steps:

      DQN:
        exploration_rate    — current epsilon (1.0 → exploration_final_eps).
                              Directly measures how often the agent still
                              chooses random actions.
        q_values_std        — standard deviation of Q-values for the current
                              observation.  Low std = policy not yet
                              differentiated; high std = agent has learned
                              strong preferences between actions.

      PPO / A2C:
        policy_entropy      — entropy of the action distribution for the
                              current observation.  High entropy = policy
                              still uncertain (exploring); low entropy =
                              policy confident (exploiting).

    Both metrics are sent to the UI via notify_client and stored in
    indicators at the end of each episode, mirroring what base_agent does
    for qtable_coverage_pct.
    """

    # How often (in steps) to compute and send exploration metrics to the UI.
    EXPLORATION_METRIC_INTERVAL = 10

    def __init__(self, env=None, agent_param=None, name=None,
                 training_start=None, training_end=None,
                 rollout_start=None, rollout_end=None,
                 step_start=None, step_end=None,
                 verbose: int = 0):
        super().__init__(verbose)

        self.env         = env
        self.episodes    = agent_param.episodes
        self.show_action = agent_param.show_action
        self.name        = (
            f"{agent_param.name}_{name}" if name is not None else agent_param.name
        )

        self.train_types = {'explorations': [], 'exploitations': [], 'steps': []}
        self.indicators  = []

        # Metrics tracking
        self.metrics           = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}
        self.rewards           = []
        self.ground_truth      = []
        self.predicted         = []
        self.current_step      = 0
        self.cumulative_reward = 0
        self.episode_rewards   = []
        self.episode           = 0
        self.episode_statuses  = []
        self.correct_predictions = 0

        # Exploration metric history — list of dicts, recorded every N steps
        # Each dict: {global_step, episode, step, <metric_name>: value}
        self.exploration_metric_history: list = []
        self._global_step: int = 0

        self.rollout_start  = rollout_start
        self.step_start     = step_start
        self.step_end       = step_end
        self.rollout_end    = rollout_end
        self.training_start = training_start
        self.training_end   = training_end
        self.rollout        = 0

    # ------------------------------------------------------------------
    # SB3 hooks
    # ------------------------------------------------------------------

    def _on_training_start(self) -> None:
        if self.training_start:
            self.training_start(self)

    def _on_rollout_start(self) -> None:
        """Triggered before collecting new samples (every 4 steps for DQN)."""
        if self.rollout_start:
            self.rollout_start(self)

    def _on_step(self) -> bool:
        """
        Called by SB3 after every env.step().
        Returns False to abort training early.
        """
        if hasattr(self.env, 'stop_event') and self.env.stop_event.is_set():
            return False
        while hasattr(self.env, 'pause_event') and self.env.pause_event.is_set():
            time.sleep(0.1)

        action = self.locals["actions"][0]
        self.count_actions_by_type[action] += 1
        self.current_step += 1
        self._global_step += 1
        reward = self.locals["rewards"][0]
        self.episode_rewards.append(reward)
        self.cumulative_reward += reward

        self.manage_step_data(action, reward, self.locals["infos"][0])

        # Send exploration metrics every N steps
        if self._global_step % self.EXPLORATION_METRIC_INTERVAL == 0:
            self._send_exploration_metric()

        if self.step_end:
            self.step_end(self)

        if self.current_step == self.env.max_steps:
            return False

        return True

    def _on_rollout_end(self) -> None:
        """Triggered before the policy update."""
        if self.rollout_end:
            self.rollout_end(self)

    def _on_training_end(self) -> None:
        """Triggered before exiting learn()."""
        if self.training_end:
            self.training_end(self)

    # ------------------------------------------------------------------
    # Policy exploration metrics
    # ------------------------------------------------------------------

    def _send_exploration_metric(self):
        """
        Compute and send the appropriate exploration metric for the current
        SB3 algorithm.  Records the value in exploration_metric_history.

        DQN   → exploration_rate + Q-values std for current obs
        PPO   → policy entropy for current obs
        A2C   → policy entropy for current obs
        Other → exploration_rate if available, else skip
        """
        metric = {}

        try:
            algo_name = type(self.model).__name__.lower()

            if algo_name == 'dqn':
                # Epsilon (exploration rate): built-in attribute
                eps = float(getattr(self.model, 'exploration_rate', 0.0))
                metric['exploration_rate'] = round(eps, 4)

                # Q-values std for current observation
                obs = self.locals.get('new_obs')
                if obs is not None:
                    import torch
                    with torch.no_grad():
                        obs_t = self.model.policy.obs_to_tensor(obs)[0]
                        q_vals = self.model.q_net(obs_t)
                        metric['q_values_std']  = round(
                            float(q_vals.std()), 4
                        )
                        metric['q_values_mean'] = round(
                            float(q_vals.mean()), 4
                        )
                        metric['q_values_max']  = round(
                            float(q_vals.max()), 4
                        )

            elif algo_name in ('ppo', 'a2c'):
                obs = self.locals.get('obs_tensor') or self.locals.get('new_obs')
                if obs is not None:
                    import torch
                    with torch.no_grad():
                        if not isinstance(obs, torch.Tensor):
                            obs_t = self.model.policy.obs_to_tensor(obs)[0]
                        else:
                            obs_t = obs
                        dist = self.model.policy.get_distribution(obs_t)
                        entropy = dist.entropy()
                        metric['policy_entropy'] = round(
                            float(entropy.mean()), 4
                        )

            else:
                # Generic fallback
                if hasattr(self.model, 'exploration_rate'):
                    metric['exploration_rate'] = round(
                        float(self.model.exploration_rate), 4
                    )

        except Exception as e:
            debug(Fore.RED + f"Error computing exploration metric: {e}\n"
                  + Fore.WHITE)
            return

        if not metric:
            return

        record = {
            'global_step': self._global_step,
            'episode':     self.episode,
            'step':        self.current_step,
            **metric,
        }
        self.exploration_metric_history.append(record)

        try:
            notify_client(
                level=SystemLevels.DATA,
                agent_name=self.name,
                exploration_metric=record,
            )
        except Exception as e:
            debug(Fore.RED + f"Error sending exploration metric: {e}\n"
                  + Fore.WHITE)

    def _get_latest_exploration_metric(self) -> dict:
        """Return the most recent exploration metric record, or {}."""
        if self.exploration_metric_history:
            return self.exploration_metric_history[-1]
        return {}

    # ------------------------------------------------------------------
    # Step data management
    # ------------------------------------------------------------------

    def manage_step_data(self, action, reward, infos):
        """
        Store per-step ground truth, prediction, and status data.

        For ATTACKS_HO: action_correct is an int scalar 0/1/2.
        For other scenarios: action_correct is the raw base env value.
        """
        status = infos["status"]

        if self.env.gym_type == GYM_TYPE[ATTACKS_HO]:
            action_correct           = infos['action_correct']
            status['action_choosen'] = int(action)
            status['traffic_type']   = action_correct
        else:
            status['action_choosen'] = action
            status['traffic_type']   = infos['action_correct']

        self.episode_statuses.append(status)
        self.rewards.append(reward)
        self.ground_truth.append(infos['Ground_truth_step'])
        self.predicted.append(infos['Predicted_step'])

        if infos['is_correct_action']:
            self.correct_predictions += 1

        if self.name.endswith("coordinator"):
            debug(Fore.MAGENTA
                  + f"Coordinator correct predictions so far: {self.correct_predictions}\n"
                  + Fore.WHITE)

        step_data = {
            'episode': self.episode,
            'step':    self.current_step,
            'status': {
                'id':   infos['action_correct'],
                'text': infos["text_action_correct"],
            },
            'action': {
                'choosen':   int(action),
                'isCorrect': bool(infos['is_correct_action']),
            },
            'correctPredictions': self.correct_predictions,
            'reward': round(float(reward), 1),
        }

        if self.env.gym_type == GYM_TYPE[ATTACKS_HO]:
            step_data['host']     = infos.get('host_name', 'unknown')
            step_data['host_idx'] = infos.get('host_idx', -1)
        elif hasattr(self, 'is_team_member') and hasattr(self, 'is_team_coordinator'):
            if self.is_team_member and not self.is_team_coordinator:
                step_data['host'] = self.name.split("_").pop()
            elif self.is_team_coordinator and self.is_team_member:
                step_data['host'] = COORDINATOR
        else:
            step_data['host'] = "single_agent"

        if 'packets' in status:
            step_data.update({
                'packets':                int(status['packets']),
                'bytes':                  int(status['bytes']),
                'packetsPercentageChange': float(status['packets_percentage_change']),
                'bytesPercentageChange':   float(status['bytes_percentage_change']),
            })
        else:
            step_data.update({
                'receivedPackets':
                    int(status.get('received_packets', 0)),
                'receivedPacketsPercentageChange':
                    float(status.get('received_packets_percentage_change', 0)),
                'receivedBytes':
                    int(status.get('received_bytes', 0)),
                'receivedBytesPercentageChange':
                    float(status.get('received_bytes_percentage_change', 0)),
                'transmittedPackets':
                    int(status.get('transmitted_packets', 0)),
                'transmittedPacketsPercentageChange':
                    float(status.get('transmitted_packets_percentage_change', 0)),
                'transmittedBytes':
                    int(status.get('transmitted_bytes', 0)),
                'transmittedBytesPercentageChange':
                    float(status.get('transmitted_bytes_percentage_change', 0)),
            })

        notify_client(level=SystemLevels.DATA, agent_name=self.name,
                      step_data=step_data)

    # ------------------------------------------------------------------
    # Episode evaluation
    # ------------------------------------------------------------------

    def evaluate_episode(self, exploration_count=0, exploitation_count=0):
        """
        Compute and store per-episode metrics.
        Appends exploration metric snapshot to indicators.
        """
        accuracy_episode = accuracy_score(self.ground_truth, self.predicted)
        precision_episode, recall_episode, f1_score_episode, _ = (
            precision_recall_fscore_support(
                self.ground_truth, self.predicted,
                average='weighted', zero_division=0.0
            )
        )
        self.metrics['accuracy'].append(accuracy_episode)
        self.metrics['precision'].append(precision_episode)
        self.metrics['recall'].append(recall_episode)
        self.metrics['f1_score'].append(f1_score_episode)

        if exploration_count > 0 or exploitation_count > 0:
            self.train_types['explorations'].append(exploration_count)
            self.train_types['exploitations'].append(exploitation_count)
            self.train_types['steps'].append(self.current_step)

        # Latest exploration metric snapshot for this episode
        expl_metric = self._get_latest_exploration_metric()

        if self.episode > 0:
            indicator = {
                'episode':            self.episode,
                'steps':              self.current_step,
                'correct_predictions': self.correct_predictions,
                'episode_statuses':   self.episode_statuses,
                'cumulative_reward':  float(self.cumulative_reward),
            }
            # Add exploration metric fields if available
            indicator.update({k: v for k, v in expl_metric.items()
                               if k not in ('global_step', 'episode', 'step')})
            self.indicators.append(indicator)

            try:
                metrics = {
                    'episode':            self.episode,
                    'steps':              self.current_step,
                    'correctPredictions': self.correct_predictions,
                    'accuracy':           accuracy_episode,
                    'precision':          precision_episode,
                    'recall':             recall_episode,
                    'f1Score':            f1_score_episode,
                    'cumulativeReward':   float(self.cumulative_reward),
                }
                metrics.update({k: v for k, v in expl_metric.items()
                                 if k not in ('global_step', 'episode', 'step')})
                notify_client(level=SystemLevels.DATA,
                              agent_name=self.name, metrics=metrics)
            except Exception as e:
                debug(Fore.RED
                      + f"Error notifying client with metrics: {e}\n"
                      + Fore.WHITE)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def print_metrics(self):
        accuracy  = self.metrics['accuracy'][-1]
        precision = self.metrics['precision'][-1]
        recall    = self.metrics['recall'][-1]
        f1_score  = self.metrics['f1_score'][-1]
        expl      = self._get_latest_exploration_metric()
        expl_str  = '  '.join(f"{k}={v}" for k, v in expl.items()
                               if k not in ('global_step', 'episode', 'step'))
        information(
            f"\n\t\tAccuracy  {accuracy  * 100:.2f}%"
            f"\n\t\tPrecision {precision * 100:.2f}%"
            f"\n\t\tRecall    {recall    * 100:.2f}%"
            f"\n\t\tF1-score  {f1_score  * 100:.2f}%"
            f"\n\t\tCumulative reward {self.cumulative_reward}"
            + (f"\n\t\t{expl_str}" if expl_str else "") + "\n",
            self.name
        )

    def episode_reset(self):
        """Reset all per-episode accumulators."""
        self.ground_truth        = []
        self.predicted           = []
        self.rewards             = []
        self.current_step        = 0
        self.correct_predictions = 0
        self.cumulative_reward   = 0
        self.done = self.truncated = False
        self.count_actions_by_type = {i: 0 for i in range(self.env.action_space.n)}
        self.episode_statuses      = []
        self.exploration_count, self.exploitation_count = 0, 0

    def get_metrics(self):
        """Return the last episode's metrics as a 4-tuple."""
        return (
            self.metrics['accuracy'][-1],
            self.metrics['precision'][-1],
            self.metrics['recall'][-1],
            self.metrics['f1_score'][-1],
        )

    def before_episode(self, episode):
        self.episode = episode
        information(f"************* Episode {self.episode} *************\n",
                    self.name)
        self.episode_reset()

    def after_episode(self):
        self.evaluate_episode()
        self.print_metrics()
        information(f"*** Episode {self.episode} finished ***\n", self.name)