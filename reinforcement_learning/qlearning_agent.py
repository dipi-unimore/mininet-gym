import numpy as np, time
from reinforcement_learning.network_env import NetworkEnv
from reinforcement_learning.base_agent import BaseAgent
from utility.my_log import set_log_level

class QLearningAgent(BaseAgent):
    """
    Model free, value based, off policy
    """    
    def __init__(self, env : NetworkEnv, params):         
        super().__init__(env, params)
        

    def update_qtable(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        # max_val = np.max(self.q_table[next_state,:])
        # find_max_val = np.where(self.q_table[next_state, :] == max_val)
        # best_next_action = np.random.choice(find_max_val[0]).item()
        td_target = reward + self.discount_factor * self.q_table[next_state + (best_next_action,)]
        td_error = td_target - self.q_table[state + (action,)]
        self.q_table[state + (action,)] += self.learning_rate * td_error    

    def train(self):   
        cumulative_reward, state, done, truncated, count_actions_by_type = self.episode_reset(self.is_discretized_state) 
        
        while not done and not truncated: #episode
            status = dict(self.env.status)     
            action = self.choose_action(state)         # Step 1: Choose action
            
            count_actions_by_type[action] += 1 
            self.current_step+=1
            
            next_state, reward, done, truncated, infos  = self.env.step(action, is_discretized_state = self.is_discretized_state, current_step = self.current_step, correct_predictions= self.correct_predictions )    # Step 2: Observe result
            
            self.manage_step_data(action,reward,infos,status)
            cumulative_reward += reward

            self.update_qtable(state, action, reward, next_state) #learn each step 
            state = next_state
            
            self.exploration_rate *= self.exploration_decay
            # if self.env.gym_type==4: #attack        
            #     time.sleep(1) 
            # else:
            #     self.env.update_state()           
        return cumulative_reward, count_actions_by_type  
  

if __name__ == '__main__':
    set_log_level('info')
    env = NetworkEnv()
    params = {
                                'learning_rate' : 0.1,
                                'discount_factor' : 0.99,
                                'exploration_rate' : 1, 
                                'exploration_decay' : 0.99995
                            }
    q_learning_agent = QLearningAgent(env=env, params = params)
    q_learning_agent.learn(episodes=1000)
    q_learning_agent.save("QLearning-Main")
    env.close()
