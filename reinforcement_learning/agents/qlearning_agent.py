# qlearning_agent.py
import numpy as np, time
from reinforcement_learning.network_env import NetworkEnv
from reinforcement_learning.base_agent import BaseAgent
from utility.my_log import error, set_log_level


class QLearningAgent(BaseAgent):
    """
    Model-free, value-based, off-policy tabular Q-Learning agent.

    Compatible with both the base env and the PerHostScanWrapper.
    When running under the wrapper, env.step() returns a normalised (8,)
    slice which is then discretized here before Q-table lookup.
    """

    def __init__(self, env: NetworkEnv, params):
        super().__init__(env, params)
        self.is_discretized_state = True

    def train_on_batch(self, batch: list):
        """
        Update the Q-table using a batch of experiences.
        This is more efficient than single updates due to reduced Python overhead.
        """
        # For Q-Learning, we can update in any order.
        # For SARSA, the order is important, but here it's fine.
        for state, action, reward, next_state in batch:
            try:
                best_next_action = np.argmax(self.q_table[next_state])
                td_target = (reward
                             + self.discount_factor
                             * self.q_table[next_state + (best_next_action,)])
                td_error = td_target - self.q_table[state + (action,)]
                self.q_table[state + (action,)] += self.learning_rate * td_error
            except (IndexError, TypeError, KeyError):
                # Gestione errore rimane la stessa
                self.q_table[state + (action,)] = 0.0
                self.q_table[next_state + (0,)] = 0.0

    def update_qtable(self, state, action, reward, next_state):
        try:
            best_next_action = np.argmax(self.q_table[next_state])
            td_target = (reward
                         + self.discount_factor
                         * self.q_table[next_state + (best_next_action,)])
            td_error = td_target - self.q_table[state + (action,)]
            self.q_table[state + (action,)] += self.learning_rate * td_error
        except (IndexError, TypeError, KeyError) as e:
            error(
                f"KeyError in update_qtable: {e} -- "
                f"state: {state}, action: {action}, "
                f"reward: {reward}, next_state: {next_state}"
            )
            self.q_table[state + (action,)]    = 0.0
            self.q_table[next_state + (0,)]    = 0.0

    def train(self):
        # episode_reset returns the initial state already discretized
        # (base_agent.episode_reset calls get_discretized_state when
        #  is_discretized_state=True)
        cumulative_reward, state, done, truncated, count_actions_by_type = \
            self.episode_reset(self.is_discretized_state)

        # Determina il numero di host dal wrapper per sapere quando finisce un round
        num_hosts = getattr(self.env, 'num_hosts', 1)

        while not done and not truncated and not self.stop_event.is_set():
            
            # Raccogli un intero round di esperienze
            round_batch = []
            for micro_step in range(num_hosts):
                if done or truncated or self.stop_event.is_set():
                    break

                self.current_step += 1
                status = dict(self.env.global_state.status)

                action = self.choose_action(state)
                count_actions_by_type[action] += 1

                raw_next_state, reward, done, truncated, infos = self.env.step(
                    action,
                    options={
                        "is_discretized_state": self.is_discretized_state,
                        "current_step":         self.current_step,
                        "correct_predictions":  self.correct_predictions,
                        "show_action":          self.show_action,
                        "name":                 self.name,
                    }
                )

                next_state = self.env.get_discretized_state(raw_next_state)

                self.manage_step_data(action, reward, infos, status)
                cumulative_reward += reward

                round_batch.append((state, action, reward, next_state))
                state = next_state

                # Il decay dell'esplorazione va applicato per micro-step
                self.exploration_rate *= self.exploration_decay

            # Aggiorna la Q-table con il batch di esperienze del round
            self.train_on_batch(round_batch)

        return cumulative_reward, count_actions_by_type


if __name__ == '__main__':
    set_log_level('info')
    env = NetworkEnv()
    params = {
        'learning_rate':    0.1,
        'discount_factor':  0.99,
        'exploration_rate': 1,
        'exploration_decay': 0.99995,
    }
    q_learning_agent = QLearningAgent(env=env, params=params)
    q_learning_agent.learn(episodes=1000)
    q_learning_agent.save("QLearning-Main")
    env.close()