"""
Classification
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from utility.constants import (
    ATTACKS, ATTACKS_FROM_DATASET,
    CLASSIFICATION_FROM_DATASET, CLASSIFICATION,
    GYM_TYPE, MARL_ATTACKS, MARL_ATTACKS_FROM_DATASET,
    SystemLevels,
)
from utility.my_files import read_data_file
from utility.my_log import information, notify_client


class SupervisedAgent:
    def __init__(self, gym_type, json_file=None, train_test_split_ratio=0.20, name="Supervised"):

        self.name = name
        self.gym_type = gym_type
        self.json_file = json_file
        self.train_test_split_ratio = train_test_split_ratio

        self.is_classification_gym_type = gym_type in [
            GYM_TYPE[CLASSIFICATION_FROM_DATASET], GYM_TYPE[CLASSIFICATION]
        ]

        # Cumulative buffer: each entry is {'id': int, 'features': [f0, f1, f2, f3]}
        self.accumulated_statuses = []

        self.metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}
        self.indicators = []

        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.fscore = 0.0
        self.y_test = []
        self.y_pred = []

        self.clf = DecisionTreeClassifier(max_depth=4)

        # Baseline fit from dataset file — only when the file actually exists.
        # For live gym types (attacks, classification, marl_attacks) the file may
        # not be present; in that case skip the baseline and let learn() build the
        # model from environment data incrementally.
        is_dataset_type = json_file and os.path.isfile(json_file)

        if not is_dataset_type:
            information(f"No dataset file found ({json_file}), starting without baseline fit.\n", self.name)
        else:
            if gym_type in [GYM_TYPE[ATTACKS_FROM_DATASET], GYM_TYPE[ATTACKS]]:
                X, y = self._init_attack_detection_from_file(json_file)
            elif gym_type in [GYM_TYPE[CLASSIFICATION_FROM_DATASET], GYM_TYPE[CLASSIFICATION]]:
                X, y = self._init_traffic_classification_from_file(json_file)
            elif gym_type in [GYM_TYPE[MARL_ATTACKS_FROM_DATASET], GYM_TYPE[MARL_ATTACKS]]:
                X, y = self._init_attack_detection_from_file(json_file)
            else:
                raise ValueError(f"Unsupported gym_type for SupervisedAgent: {gym_type}")

            X_train, X_test, y_train, y_test = train_test_split(
                X.values if hasattr(X, 'values') else X,
                y.values if hasattr(y, 'values') else y,
                test_size=self.train_test_split_ratio, random_state=42
            )
            self.clf.fit(X_train, y_train)
            y_pred = self.clf.predict(X_test)
            self.accuracy = float(accuracy_score(y_test, y_pred))
            self.precision, self.recall, self.fscore, _ = precision_recall_fscore_support(
                y_test, y_pred, average='macro', zero_division=0.0
            )

    # ------------------------------------------------------------------
    # Main training loop — called by train_agent()
    # ------------------------------------------------------------------

    def learn(self, episodes, env, stop_event):
        """
        Run the supervised incremental training loop.

        For each episode:
          1. Step through the environment to collect (state, label) pairs.
          2. Append to the cumulative buffer.
          3. Split accumulated buffer 80/20 (configurable).
          4. Refit the classifier and compute accuracy.
          5. Notify the dashboard with the accuracy point.
        """
        for episode in range(episodes):
            if stop_event.is_set():
                break

            information(f"************* Episode {episode + 1} *************\n", self.name)

            state, _ = env.reset(options={"is_discretized_state": False, "is_real_state": True})
            episode_data = []
            episode_statuses = []
            step = 0
            correct_predictions = 0
            done = truncated = False

            while not done and not truncated and step < env.max_steps:
                if stop_event.is_set():
                    break

                # Use current model to predict; env needs an action but the
                # label comes from the environment truth, not from the action.
                action = self._get_action(state)
                next_state, _reward, done, truncated, info = env.step(
                    action,
                    options={
                        "is_discretized_state": False,
                        "is_real_state": True,
                        "current_step":         step,
                        "correct_predictions":  correct_predictions,
                        "show_action":          False,
                        "name":                 self.name,
                    }

                )

                label = info['action_correct']
                is_correct = (action == label)
                if is_correct:
                    correct_predictions += 1
                episode_data.append({'id': label, 'features': list(state)})

                step_status = {'action_choosen': action, 'action_correct': is_correct}
                if self.is_classification_gym_type:
                    step_status['traffic_type'] = label
                if 'status' in info:
                    step_status.update(info['status'])
                episode_statuses.append(step_status)

                state = next_state
                step += 1

            # Accumulate this episode on top of all previous ones
            self.accumulated_statuses.extend(episode_data)

            accuracy = self._train_and_evaluate(episode + 1, step, correct_predictions, episode_statuses)

            information(
                f"Episode {episode + 1} done — steps: {step} "
                f"| accumulated: {len(self.accumulated_statuses)} "
                f"| accuracy: {accuracy * 100:.2f}%\n",
                self.name,
            )

    # ------------------------------------------------------------------
    # Incremental fit + evaluate on the full accumulated buffer
    # ------------------------------------------------------------------

    def _train_and_evaluate(self, episode, steps, correct_predictions=0, episode_statuses=None):
        """
        Split accumulated buffer, refit, compute metrics, notify dashboard.
        Returns accuracy (float 0-1).
        """
        if len(self.accumulated_statuses) < 2:
            return 0.0

        X = [s['features'] for s in self.accumulated_statuses]
        y = [s['id'] for s in self.accumulated_statuses]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.train_test_split_ratio, random_state=42
        )

        self.clf.fit(X_train, y_train)
        y_pred = self.clf.predict(X_test)

        accuracy = float(accuracy_score(y_test, y_pred))
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            y_test, y_pred, average='macro', zero_division=0.0
        )

        self.accuracy = accuracy
        self.precision = float(precision)
        self.recall = float(recall)
        self.fscore = float(f1_score)
        self.y_test = list(y_test)
        self.y_pred = list(y_pred)

        self.metrics['accuracy'].append(accuracy)
        self.metrics['precision'].append(float(precision))
        self.metrics['recall'].append(float(recall))
        self.metrics['f1_score'].append(float(f1_score))

        self.indicators.append({
            'episode': episode,
            'steps': steps,
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'cumulative_reward': correct_predictions,
            'episode_statuses': episode_statuses or [],
        })

        try:
            notify_client(
                level=SystemLevels.DATA,
                agent_name=self.name,
                metrics={
                    'episode': episode,
                    'accuracy': accuracy,
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1_score),
                    'cumulativeReward': 0,
                },
            )
        except Exception as e:
            from utility.my_log import debug
            debug(f"Error notifying supervised metrics: {e}\n")

        return accuracy

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, state):
        """Predict action from raw state vector."""
        return self.clf.predict([[state[0], state[1], state[2], state[3]]])

    def _get_action(self, state):
        try:
            # DecisionTreeClassifier raises NotFittedError before the first fit
            return int(self.predict(state)[0])
        except Exception:
            return 0  # random-enough default before first fit

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path):
        try:
            joblib.dump(self.clf, path + ".pkl")
            information(f"Model saved to {path}.pkl\n", self.name)
        except Exception as e:
            from utility.my_log import debug
            debug(f"Error saving supervised model: {e}\n")

    def load(self, path):
        self.clf = joblib.load(path + ".pkl")

    def get_metrics(self):
        if self.metrics['accuracy']:
            return (
                self.metrics['accuracy'][-1],
                self.metrics['precision'][-1],
                self.metrics['recall'][-1],
                self.metrics['f1_score'][-1],
            )
        return self.accuracy, self.precision, self.recall, self.fscore

    # ------------------------------------------------------------------
    # Init helpers — read from dataset file (used only for baseline fit)
    # ------------------------------------------------------------------

    def _init_traffic_classification_from_file(self, json_file):
        statuses = read_data_file(json_file)
        rows = []
        for s in statuses:
            rows.append({
                'id':                  s['id'],
                'received_packets':    sum(h['receivedPackets'] for h in s['hostStatusesStructured'].values()),
                'received_bytes':      sum(h['receivedBytes']   for h in s['hostStatusesStructured'].values()),
                'transmitted_packets': sum(h['transmittedPackets'] for h in s['hostStatusesStructured'].values()),
                'transmitted_bytes':   sum(h['transmittedBytes']   for h in s['hostStatusesStructured'].values()),
            })
        df = pd.DataFrame(rows).dropna()
        X = df.drop(columns=['id'])
        y = df['id']
        return X, y

    def _init_attack_detection_from_file(self, json_file):
        statuses = read_data_file(json_file)
        rows = []
        for s in statuses:
            rows.append({
                'is_attack':                  s['id'],
                'packets':                    s['packets'],
                'packets_percentage_change':  s['packetsPercentageChange'],
                'bytes':                      s['bytes'],
                'bytes_percentage_change':    s['bytesPercentageChange'],
            })
        df = pd.DataFrame(rows).dropna()
        X = df.drop(columns=['is_attack'])
        y = df['is_attack']
        return X, y

    # Keep old public names for backward-compat callers
    def init_traffic_classification_env(self, json_file):
        return self._init_traffic_classification_from_file(json_file)

    def init_attack_detection_env(self, json_file):
        return self._init_attack_detection_from_file(json_file)
