import time
from reinforcement_learning.network_env import NetworkEnv
from reinforcement_learning.base_agent import BaseAgent
from utility.my_log import set_log_level

class SARSAAgent(BaseAgent):
    """
    Model free, value based, on policy
    """
    def __init__(self, env: NetworkEnv, params):
        super().__init__(env, params) 
        
    def update_qtable(self, state, action, reward, next_state, next_action):
        # SARSA Update Rule
        td_target = reward + self.discount_factor * self.q_table[next_state + (next_action,)]
        td_error = td_target - self.q_table[state + (action,)]
        self.q_table[state + (action,)] += self.learning_rate * td_error
                 
    def train(self):
        cumulative_reward, state, done, truncated, count_actions_by_type = self.episode_reset(self.is_discretized_state) 
        status = dict(self.env.status)  
        action = self.choose_action(state)
                      
        while not done and not truncated: #episode             
            count_actions_by_type[action] += 1 
            self.current_step+=1
            next_state, reward, done, truncated, infos = self.env.step(action, is_discretized_state = self.is_discretized_state, current_step = self.current_step, correct_predictions= self.correct_predictions )
            
            self.manage_step_data(action,reward,infos,status)
            cumulative_reward += reward            
            
            status = dict(self.env.status) 
            next_action = self.choose_action(next_state)
                      
            self.update_qtable(state, action, reward, next_state, next_action)
            state, action = next_state, next_action
            
            self.exploration_rate *= self.exploration_decay   
            # if self.env.gym_type==4: #attack            
            #     time.sleep(1)            
            # else:
            #     self.env.update_state()    
        self.exploration_count -= 1
        self.exploitation_count -= 1
        return cumulative_reward, count_actions_by_type                  
   
# Initialize the environment and agent
if __name__ == '__main__':
    set_log_level('info')
    env = NetworkEnv()
    params = {
        'learning_rate': 0.1,
        'discount_factor': 0.99,
        'exploration_rate': 1.0,
        'exploration_decay': 0.99995
    }
    sarsa_agent = SARSAAgent(env, params)
    sarsa_agent.learn(episodes=1000)
    sarsa_agent.save("Sarsa-Main")
    env.close()
