# base_agent.py
from abc import ABC, abstractmethod
import time
import numpy as np
import json
from reinforcement_learning.scenarios.marl.constants import COORDINATOR
from reinforcement_learning.network_env import NetworkEnv
from utility.constants import ATTACKS_HO, GYM_TYPE, SystemLevels
from utility.my_log import information, debug, notify_client
from colorama import Fore
from gymnasium import spaces
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class BaseAgent(ABC):
    def __init__(self, env: NetworkEnv, params):
        """
        Base class for RL agents (Q-Learning, SARSA).
        Handles Q-table initialisation, training loop, metrics, save/load.
        """
        self.env = env
        self.learning_rate     = params.learning_rate
        self.discount_factor   = params.discount_factor
        self.exploration_rate  = params.exploration_rate
        self.exploration_decay = params.exploration_decay
        self.is_exploitation   = False

        # How often (in steps) to compute and send Q-table coverage to the UI.
        # Minimum 10. Configurable via agent param qtable_coverage_interval.
        self.qtable_coverage_interval = max(
            10, getattr(params, 'qtable_coverage_interval', 10)
        )

        # Q-table dimensions based on action space
        if isinstance(self.env.action_space, spaces.Discrete):
            action_dims = (self.env.action_space.n,)
        elif isinstance(self.env.action_space, spaces.MultiDiscrete):
            action_dims = tuple(self.env.action_space.nvec)
        else:
            raise ValueError(
                f"Unsupported action space type: {type(self.env.action_space)}"
            )

        self.q_table = np.zeros(
            (self.env.n_bins,) * self.env.observation_space.shape[0] + action_dims,
            dtype=float
        )

        # Total number of cells in the Q-table — fixed, computed once
        self.q_table_total_cells: int = int(self.q_table.size)

        self.train_types = {'explorations': [], 'exploitations': [], 'steps': []}
        self.indicators  = []

        # Metrics tracking
        self.metrics  = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}

        # Q-table coverage tracking:
        #   qtable_coverage_history — list of (global_step, coverage_pct) tuples
        #   recorded every qtable_coverage_interval steps across all episodes.
        self.qtable_coverage_history: list = []

        self.rewards          = []
        self.ground_truth     = []
        self.predicted        = []
        self.current_step     = 0
        self.correct_predictions = 0

        self.name        = params.name
        self.show_action = params.show_action

        # Global step counter across all episodes (used for coverage x-axis)
        self._global_step: int = 0

        information(
            f"[{self.name}] Q-table shape: {self.q_table.shape}  "
            f"total cells: {self.q_table_total_cells}  "
            f"coverage_interval: {self.qtable_coverage_interval} steps\n"
        )

    # ------------------------------------------------------------------
    # Q-table coverage
    # ------------------------------------------------------------------

    def get_qtable_coverage(self) -> float:
        """
        Compute the fraction of Q-table cells that have been updated
        (i.e. are non-zero) as a percentage [0.0, 100.0].

        Non-zero is used as a proxy for 'visited': a cell is updated via
        TD error only when the corresponding (state, action) has been
        experienced at least once with a non-zero reward or non-zero
        initial value propagation.

        Note: cells that receive a TD update of exactly 0.0 are not counted.
        In practice this is rare unless the reward is identically 0 for all
        transitions, which does not occur in this environment.
        """
        non_zero = int(np.count_nonzero(self.q_table))
        return round(100.0 * non_zero / max(1, self.q_table_total_cells), 4)

    def _maybe_send_coverage(self):
        """
        Compute and send Q-table coverage to the UI every
        qtable_coverage_interval global steps.
        Records the value in qtable_coverage_history for final plotting.
        """
        if self._global_step % self.qtable_coverage_interval != 0:
            return

        coverage_pct = self.get_qtable_coverage()

        # Record for final plot
        self.qtable_coverage_history.append({
            'global_step': self._global_step,
            'episode':     self.episode,
            'step':        self.current_step,
            'coverage_pct': coverage_pct,
        })

        # Send to UI in real time
        try:
            notify_client(
                level=SystemLevels.DATA,
                agent_name=self.name,
                qtable_coverage={
                    'global_step':   self._global_step,
                    'episode':       self.episode,
                    'coverage_pct':  coverage_pct,
                    'non_zero_cells': int(np.count_nonzero(self.q_table)),
                    'total_cells':   self.q_table_total_cells,
                    'q_table_shape': list(self.q_table.shape),
                }
            )
        except Exception as e:
            debug(Fore.RED + f"Error sending Q-table coverage: {e}\n" + Fore.WHITE)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            self.exploration_count += 1
            self.is_exploitation = False
            action = self.env.action_space.sample()
        else:
            self.exploitation_count += 1
            self.is_exploitation = True
            action = np.argmax(self.q_table[state]).item()
        debug(
            Fore.GREEN + f"Agent Action {action} in "
            + ("Exploitation" if self.is_exploitation else "Exploration") + "\n"
        )
        return action

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def learn(self, episodes=100, stop_event=None, t=None):
        self.stop_event = stop_event
        for episode in range(episodes):
            self.episode = episode + 1
            information(
                f"************* Episode {self.episode} *************\n", self.name
            )
            cumulative_reward, count_actions_by_type = self.train()
            if self.stop_event.is_set():
                break
            exploration_count  = self.exploration_count
            exploitation_count = self.exploitation_count
            if exploration_count > 0 or exploitation_count > 0:
                information(
                    Fore.YELLOW + f"Exploration used {exploration_count} times\n"
                    + Fore.WHITE, self.name
                )
                information(
                    Fore.BLUE + f"Exploitation used {exploitation_count} times\n"
                    + Fore.WHITE, self.name
                )
            self.evaluate_episode(
                self.episode, cumulative_reward, exploration_count, exploitation_count
            )
            self.print_metrics(cumulative_reward)
            information(f"*** Episode {self.episode} finished ***\n", self.name)
            if t is not None:
                time.sleep(t)

    def get_metrics(self):
        if not self.metrics.get('accuracy'):
            return 0.0, 0.0, 0.0, 0.0
        return (
            self.metrics['accuracy'][-1],
            self.metrics['precision'][-1],
            self.metrics['recall'][-1],
            self.metrics['f1_score'][-1],
        )

    # ------------------------------------------------------------------
    # Step data management
    # ------------------------------------------------------------------

    def manage_step_data(self, action, reward, infos, status):
        """
        Store per-step ground truth, prediction, and status data.
        Also sends Q-table coverage to the UI every qtable_coverage_interval steps.
        """
        if self.env.gym_type == GYM_TYPE[ATTACKS_HO]:
            status['action_choosen'] = int(action)
            status['traffic_type']   = infos['action_correct']
        else:
            status['action_choosen'] = action
            status['traffic_type']   = infos['action_correct']

        self.rewards.append(reward)
        self.ground_truth.append(infos['Ground_truth_step'])
        self.predicted.append(infos['Predicted_step'])
        if infos['is_correct_action']:
            self.correct_predictions += 1

        # Advance global step counter and conditionally send coverage
        self._global_step += 1
        self._maybe_send_coverage()

        if self.name.endswith("coordinator"):
            debug(
                Fore.MAGENTA
                + f"Coordinator correct predictions so far: {self.correct_predictions}\n"
                + Fore.WHITE
            )

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
            'reward': float(reward),
        }

        if hasattr(self, 'is_team_member') and hasattr(self, 'is_team_coordinator'):
            if self.is_team_member and not self.is_team_coordinator:
                step_data['host'] = self.name.split("_").pop()
                step_data.update({
                    'receivedPackets':
                        int(infos['status']['received_packets']),
                    'receivedPacketsPercentageChange':
                        float(infos['status']['received_packets_percentage_change']),
                    'receivedBytes':
                        int(infos['status']['received_bytes']),
                    'receivedBytesPercentageChange':
                        float(infos['status']['received_bytes_percentage_change']),
                    'transmittedPackets':
                        int(infos['status']['transmitted_packets']),
                    'transmittedPacketsPercentageChange':
                        float(infos['status']['transmitted_packets_percentage_change']),
                    'transmittedBytes':
                        int(infos['status']['transmitted_bytes']),
                    'transmittedBytesPercentageChange':
                        float(infos['status']['transmitted_bytes_percentage_change']),
                })
                status.update({
                    'id':     infos['status']['id'],
                    'status': infos['status']['status'],
                    'received_packets':
                        infos['status']['received_packets'],
                    'received_bytes':
                        infos['status']['received_bytes'],
                    'received_packets_percentage_change':
                        infos['status']['received_packets_percentage_change'],
                    'received_bytes_percentage_change':
                        infos['status']['received_bytes_percentage_change'],
                    'transmitted_packets':
                        infos['status']['transmitted_packets'],
                    'transmitted_bytes':
                        infos['status']['transmitted_bytes'],
                    'transmitted_packets_percentage_change':
                        infos['status']['transmitted_packets_percentage_change'],
                    'transmitted_bytes_percentage_change':
                        infos['status']['transmitted_bytes_percentage_change'],
                })
            elif self.is_team_coordinator and self.is_team_member:
                step_data['host'] = COORDINATOR
                step_data.update({
                    'packets':                int(status['packets']),
                    'bytes':                  int(status['bytes']),
                    'packetsPercentageChange': float(status['packets_percentage_change']),
                    'bytesPercentageChange':   float(status['bytes_percentage_change']),
                })
        elif self.env.gym_type == GYM_TYPE[ATTACKS_HO]:
            step_data['host'] = infos.get('host_name', 'unknown')
            step_data.update({
                'receivedPackets':
                    int(infos['status'].get('received_packets', 0)),
                'receivedPacketsPercentageChange':
                    float(infos['status'].get('received_packets_percentage_change', 0)),
                'receivedBytes':
                    int(infos['status'].get('received_bytes', 0)),
                'receivedBytesPercentageChange':
                    float(infos['status'].get('received_bytes_percentage_change', 0)),
                'transmittedPackets':
                    int(infos['status'].get('transmitted_packets', 0)),
                'transmittedPacketsPercentageChange':
                    float(infos['status'].get('transmitted_packets_percentage_change', 0)),
                'transmittedBytes':
                    int(infos['status'].get('transmitted_bytes', 0)),
                'transmittedBytesPercentageChange':
                    float(infos['status'].get('transmitted_bytes_percentage_change', 0)),
            })
        else:
            step_data['host'] = "single_agent"
            step_data.update({
                'packets':                int(status['packets']),
                'bytes':                  int(status['bytes']),
                'packetsPercentageChange': float(status['packets_percentage_change']),
                'bytesPercentageChange':   float(status['bytes_percentage_change']),
            })
            gs = getattr(self.env, 'global_state', None)
            if gs is not None and hasattr(gs, 'received_packets'):
                step_data.update({
                    'receivedPackets':    int(gs.received_packets),
                    'receivedBytes':      int(gs.received_bytes),
                    'transmittedPackets': int(gs.transmitted_packets),
                    'transmittedBytes':   int(gs.transmitted_bytes),
                })

        self.episode_statuses.append(status)
        notify_client(level=SystemLevels.DATA, agent_name=self.name,
                      step_data=step_data)

    # ------------------------------------------------------------------
    # Episode evaluation
    # ------------------------------------------------------------------

    def evaluate_episode(self, episode, cumulative_reward,
                          exploration_count=0, exploitation_count=0):
        """
        Compute and store per-episode metrics.
        Supports int scalars (ATTACKS_HO) and one-hot vectors (other scenarios).
        """
        if (len(self.ground_truth) > 0
                and isinstance(self.ground_truth[0], (int, np.integer))):
            gt   = list(self.ground_truth)
            pred = list(self.predicted)
        else:
            gt   = [next((i for i, v in enumerate(item) if v == 1), -1)
                    for item in self.ground_truth]
            pred = [next((i for i, v in enumerate(item) if v == 1), -1)
                    for item in self.predicted]

        accuracy_episode = accuracy_score(gt, pred)
        precision_episode, recall_episode, f1_score_episode, _ = (
            precision_recall_fscore_support(
                gt, pred, average='weighted', zero_division=0.0
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

        # Final coverage snapshot for this episode
        coverage_pct = self.get_qtable_coverage()

        if episode > 0:
            self.indicators.append({
                'episode':              episode,
                'steps':                self.current_step,
                'correct_predictions':  self.correct_predictions,
                'episode_statuses':     self.episode_statuses,
                'cumulative_reward':    cumulative_reward,
                'qtable_coverage_pct':  coverage_pct,       # end-of-episode snapshot
                'qtable_total_cells':   self.q_table_total_cells,
            })
            try:
                metrics = {
                    'episode':             episode,
                    'steps':               self.current_step,
                    'correctPredictions':  self.correct_predictions,
                    'accuracy':            accuracy_episode,
                    'precision':           precision_episode,
                    'recall':              recall_episode,
                    'f1Score':             f1_score_episode,
                    'cumulativeReward':    float(cumulative_reward),
                    'qtableCoveragePct':   coverage_pct,
                    'qtableTotalCells':    self.q_table_total_cells,
                }
                notify_client(
                    level=SystemLevels.DATA, agent_name=self.name, metrics=metrics
                )
            except Exception as e:
                debug(Fore.RED
                      + f"Error notifying client with metrics: {e}\n"
                      + Fore.WHITE)

    # ------------------------------------------------------------------
    # Episode reset
    # ------------------------------------------------------------------

    def episode_reset(self, is_discretized_state):
        """
        Reset per-episode accumulators and get the initial state.
        The wrapper returns RAW observations; tabular agents discretize here.
        """
        self.ground_truth        = []
        self.predicted           = []
        self.rewards             = []
        self.current_step        = 0
        self.correct_predictions = 0
        cumulative_reward        = 0
        done = truncated         = False
        count_actions_by_type    = {i: 0 for i in range(self.env.action_space.n)}
        self.episode_statuses    = []
        self.exploration_count, self.exploitation_count = 0, 0

        state, _ = self.env.reset(
            options={
                "is_discretized_state": is_discretized_state,
                "is_real_state": False,
            }
        )
        # Skip if already a tuple: classification env returns a discretized tuple when
        # is_discretized_state=True; re-discretizing it collapses all bin indices (0-3)
        # into bin 0, making every state (0,0,0,0) and breaking Q-table learning.
        if is_discretized_state and not isinstance(state, tuple):
            state = self.env.get_discretized_state(state)

        return cumulative_reward, state, done, truncated, count_actions_by_type

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def print_metrics(self, cumulative_reward):
        accuracy  = self.metrics['accuracy'][-1]
        precision = self.metrics['precision'][-1]
        recall    = self.metrics['recall'][-1]
        f1_score  = self.metrics['f1_score'][-1]
        coverage  = self.get_qtable_coverage()
        if cumulative_reward is None:
            information(
                f"Accuracy {accuracy * 100:.2f}%\n"
                f"Precision {precision * 100:.2f}%\n"
                f"Recall {recall * 100:.2f}%\n"
                f"F1-score {f1_score * 100:.2f}%\n"
                f"Q-table coverage {coverage:.2f}% "
                f"({self.q_table_total_cells} cells)\n"
            )
        else:
            information(
                f"\n\t\tAccuracy  {accuracy  * 100:.2f}%"
                f"\n\t\tPrecision {precision * 100:.2f}%"
                f"\n\t\tRecall    {recall    * 100:.2f}%"
                f"\n\t\tF1-score  {f1_score  * 100:.2f}%"
                f"\n\t\tQ-table coverage {coverage:.2f}% "
                f"({self.q_table_total_cells} cells)"
                f"\n\t\tCumulative reward {cumulative_reward}\n",
                self.name,
            )

    @abstractmethod
    def train(self, num_episodes=None):
        pass

    def predict(self, state):
        """Receive raw state, discretize, return best action."""
        discretized_state = self.env.get_discretized_state(state)
        return np.argmax(self.q_table[discretized_state]).item()

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, filename):
        data = {
            'q_table':                self.q_table.tolist(),
            'learning_rate':          self.learning_rate,
            'discount_factor':        self.discount_factor,
            'exploration_rate':       self.exploration_rate,
            'exploration_decay':      self.exploration_decay,
            'qtable_coverage_history': self.qtable_coverage_history,
        }
        with open(filename + ".json", 'w') as file:
            json.dump(data, file)
        information(f"Model saved to {filename}\n", self.name)

    def load(self, filename):
        if not filename.endswith('.json'):
            filename += ".json"
        with open(filename, 'r') as file:
            data = json.load(file)
        self.q_table           = np.array(data['q_table'])
        self.learning_rate     = data['learning_rate']
        self.discount_factor   = data['discount_factor']
        self.exploration_rate  = data['exploration_rate']
        self.exploration_decay = data['exploration_decay']
        self.qtable_coverage_history = data.get('qtable_coverage_history', [])
        information(f"Model loaded from {filename}\n", self.name)

    def discretize_state(self, state):
        discrete_state = []
        for i, val in enumerate(state):
            bin_index = np.digitize(val, self.bins[i]) - 1
            bin_index = max(0, min(bin_index, self.n_bins - 1))
            discrete_state.append(bin_index)
        return tuple(discrete_state)