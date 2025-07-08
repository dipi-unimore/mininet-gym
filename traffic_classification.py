# main.py
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from reinforcement_learning.network_env import NetworkEnv
from reinforcement_learning.agent_manager import AgentManager
#from utility.my_statistics import plot_metrics, plot_indicators, plot_train_types, plot_net_metrics
from reinforcement_learning.qlearning_agent import QLearningAgent
from reinforcement_learning.sarsa_agent import SARSAAgent
from supervised_agent import SupervisedAgent
from utility.my_statistics import plot_agent_test_errors, plot_combined_performance_over_time, plot_comparison_bar_charts, plot_metrics, plot_agent_cumulative_rewards, plot_agent_execution_confusion_matrix, plot_agent_execution_traffic_types, plot_enviroment_execution_statutes, plot_radar_chart, plot_train_types, plot_agent_test, plot_test_confusion_matrix
from utility.my_files import save_data_to_file, read_csv_file, create_directory_training_execution
from utility.my_log import information, debug, error
from colorama import Fore, Back, Style
#from utility.my_pdf import create_pdf_from_directory
import time, traceback
import numpy as np



def traffic_classification_main(config, net_env: NetworkEnv):
   
    try:
        am = AgentManager(net_env, config)
         # Step 1: training
        agents_metrics = defaultdict(list)
        for agent in am.agents_params:             
            if isinstance(agent.instance, SupervisedAgent) or agent.skip_learn:
                continue
            if not hasattr(agent, 'episodes'):
                information(Fore.YELLOW+"Param 'total_timesteps' is missing\n")
                continue  
            if config.env_params.gym_type=="classification_from_dataset":
                net_env.df = read_csv_file(net_env.csv_file)  
                agent.episodes = int((len(net_env.df) - config.test_episodes) / (config.env_params.max_steps + 1))
          
            train_agent(agent)   
            
        #Step 2: plotting and saving agent data
        #for agent in am.agents_params:
            # if isinstance(agent.instance, SupervisedAgent) or agent.skip_learn:
            #     continue
            plot_and_save_data_agent(agent, config)  
            agents_metrics[agent.name]=agent.instance.metrics
        if len(agents_metrics)>0:            
            plot_comparison_bar_charts(config.training_execution_directory , agents_metrics)
            plot_radar_chart(config.training_execution_directory , agents_metrics)
        
        #Step 3: starting test
        directory_name = create_directory_training_execution(config, "TEST")
        test_classification_agents(am, directory_name, config)                    
                  
    except Exception as e: 
        #print(traceback.format_exc())
        error(Fore.RED+f"Something went wrong!\n{e}\{traceback.format_exc()}\n"+Fore.WHITE)
    finally:
        # Stop the network when everything has finished
        information(Fore.WHITE)
        if not config.env_params.gym_type=="classification_from_dataset":
            net_env.net.stop()         

def train_agent(agent):
    """
    Function for training a single agent.
    Args:
        agent: Agent to be trained.
    """
    start_time = time.time()
    try:            
        information(f"Starting training\n", agent.name)
        if agent.is_custom_agent:
            #learn for all episodes each one of env.max_steps maximum
            agent.instance.learn(agent.episodes)     
        else:   
            for episode in range(agent.episodes):
                agent.custom_callback.episode = episode+1
                agent.instance.learn(total_timesteps=agent.max_steps, callback=agent.custom_callback, progress_bar=agent.progress_bar)
       
    except Exception as error:
        # handle the exception
        error(f"Agent {agent.name} learn:", error)  
    agent.elapsed_time = time.time() - start_time 
    information(f"Training completed in {agent.elapsed_time}\n",agent.name)

def plot_and_save_data_agent(agent, config):    
    
    # Collect metrics at the end of training
    if agent.is_custom_agent:
        accuracy, precision, recall, f1_score = agent.instance.get_metrics()
    else: 
        accuracy, precision, recall, f1_score = agent.custom_callback.get_metrics()
        agent.instance.metrics = agent.custom_callback.metrics
        agent.instance.indicators = agent.custom_callback.indicators
        
    agent.metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }  
      
    data = type('', (), {})()
    data.train_execution_time = agent.elapsed_time
    data.train_metrics = agent.instance.metrics
    data.train_indicators = agent.instance.indicators
    if "train_types" in agent.instance.__dict__:
        data.train_types = agent.instance.train_types
    #net_env.initialize_storage() #re-initialize for next agent
    
    #create directory to save all files for the agent training excecution
    directory_name = create_directory_training_execution(config, agent_name = agent.name)
    if agent.save:        
        agent.instance.save(directory_name+"/"+agent.name)
        
    #Step 4: plotting training statistics
    information("Plotting training data\n",agent.name)
    if len(data.train_indicators)>2:
        plot_agent_cumulative_rewards(data.train_indicators, directory_name, agent.name)
        #plot_agent_execution_traffic_types(data.train_indicators, directory_name, agent.name)
        plot_agent_execution_confusion_matrix(data.train_indicators, directory_name)
    plot_combined_performance_over_time(data.train_metrics, directory_name, agent.name + " Combined performance over time")
    plot_metrics(data.train_metrics,directory_name,agent.name+" Train metrics")
    if "train_types" in data.__dict__ and len(data.train_types["explorations"]) > 0 and len(data.train_types["exploitations"]) > 0: 
        plot_train_types(data.train_types, data.train_execution_time, directory_name)
    
    #Step 5: saving data
    save_data_to_file(data.__dict__,directory_name)
    information(f"Data saved \n",agent.name)    

def test_classification_agents(am, directory_name, config):
    score, ground_truth, predicted = evaluate_classification_agent(am)
    metrics =  {agent.name: {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0} for agent in config.agents}

    for s, p in zip(score.items(),predicted.items()):
        accuracy = accuracy_score(ground_truth,p[1])
        precision, recall, f1_score, _ = precision_recall_fscore_support(ground_truth, p[1], average='weighted', zero_division=0.0)
        information("\nAgent: "+ Fore.RED +f"{s[0]}"+ Fore.WHITE +f" score: {s[1]}\n")
        metrics[s[0]]['accuracy'] = accuracy
        metrics[s[0]]['precision'] = precision
        metrics[s[0]]['recall'] = recall
        metrics[s[0]]['f1_score'] = f1_score
        #am.env.print_metrics(None, accuracy, precision, recall, f1_score)  
        gt = [int(np.argmax(item)) for item in ground_truth]
        ps = [int(np.argmax(item)) for item in p[1]]
        plot_test_confusion_matrix(directory_name, gt, ps, s[0]) 
         
    data = type('', (), {})()
    data.score = score
    data.ground_truth = ground_truth
    data.predicted = predicted
    data.metrics = metrics
    save_data_to_file(data.__dict__, directory_name,"test")
    # plot test
    plot_agent_test(data.__dict__, directory_name, title='')
    plot_agent_test_errors(data.__dict__, directory_name, title='Agent Evaluation Errors')

def evaluate_classification_agent(am: AgentManager):
    """
    Evaluate for n episodes a classification of traffic types
    None, Ping, UDP, TCP
    """      
    epochs = am.test_episodes
    agents_params = am.agents_params   
    env = am.env
    
    # if env.gym_type == 0:
    #    env.gym_type = 2
    #    #read initial traffic
    #    env.sync_time = env.synchronize_controller()
    #    env.read_time = env.sync_time * 0.6           
            
    information(f"Evaluation started: epochs {epochs}\n")
    score =  {agent.name: 0 for agent in agents_params}
    ground_truth = []
    predicted =  {agent.name: [] for agent in agents_params}

    for episode in range(epochs):
        information(f"\n\n************* Episode {episode+1} *************\n")            
        # self.env.is_state_normalized = True
        state, _ = env.reset() #state continuos
        
        g=np.zeros(env.actions_number)
        g[env.generated_traffic_type]+=1
        ground_truth.append(g)
        real_state = env.real_state #not_normalized
        normalized_state = env.get_normalize_state(real_state) 
        information(f"p_r={real_state[0]}\np_t={real_state[1]}\nb_r={real_state[2]}byte\nb_t={real_state[3]}byte\n")
        
        for agent in agents_params: 
            model = agent.instance
            if model is None:        
                raise("The model can't be None. Create configuration")
            if isinstance(model, SupervisedAgent) or isinstance(model, QLearningAgent) or isinstance(model,SARSAAgent):
                prediction = model.predict(real_state)
            else:
                prediction, _states = model.predict(normalized_state, deterministic=True)
            color = Fore.RED           
            if prediction == env.generated_traffic_type:
                score[agent.name]  += 1 
                color = Fore.GREEN           
                
            p=np.zeros(env.actions_number)
            p[prediction]+=1 
            predicted[agent.name].append(p)    
            information(f"{agent.name}: Action predicted"+color+f" {env.execute_action(prediction)}\n"+Fore.WHITE)

    information(f"Evaluation finished \n")
    return score, ground_truth, predicted
    