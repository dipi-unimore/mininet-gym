# attack_detect_main.py
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from reinforcement_learning.adversarial_agent import  continuous_traffic_generation
from reinforcement_learning.agent_manager import AgentManager
from reinforcement_learning.network_env_attack_detect import NetworkEnvAttackDetect
from reinforcement_learning.qlearning_agent import QLearningAgent
from reinforcement_learning.sarsa_agent import SARSAAgent
from supervised_agent import SupervisedAgent
from utility.network_configurator import stop
from utility.my_statistics import plot_agent_test_errors, plot_combined_performance_over_time, plot_comparison_bar_charts, plot_metrics, plot_agent_cumulative_rewards, plot_agent_execution_confusion_matrix, plot_agent_execution_statuses, plot_enviroment_execution_statutes, plot_radar_chart, plot_train_types, plot_agent_test, plot_test_confusion_matrix
from utility.my_files import save_data_to_file, read_data_file, create_directory_training_execution
from utility.my_log import error, information, debug #,setLogLevel,  notify_client
from colorama import Fore
import time, threading, traceback

def attack_detect_main(config, net_env: NetworkEnvAttackDetect):
    
    try:
        show_normal_traffic = config.env_params.show_normal_traffic
        am = AgentManager(net_env, config)
        
        # Step 1: training
        agents_metrics = defaultdict(list)
        if config.env_params.gym_type=="attacks":
            # Step 1: Start the traffic generation thread
            # Set the desired attack probability (e.g., 0.3 means 30% chance of attack)
            stop_event_traffic, pause_event_traffic, traffic_thread = start_continuous_traffic_generation(net_env, show_normal_traffic)
            
            # Step 2: Start status display thread - moved to env
            #stop_event_status, pause_event_status, status_thread = start_update_status_thread(net_env)
        
            # Step 3: Start training threads for each agent
            training_threads = []
            time.sleep(2)
            for agent in am.agents_params:
                if agent.skip_learn:
                    continue
                if not hasattr(agent, 'episodes'):
                    information(Fore.YELLOW+"Param 'episodes' is missing\n")
                    continue
                training_thread = threading.Thread(
                    target=train_agent,
                    args=(agent,)
                )
                training_threads.append(training_thread)
                training_thread.start()

            # Wait for training threads to finish
            for t in training_threads:
                t.join()
            debug("Train_agent_threads finished")
        else:
            statuses = read_data_file("statuses")
            episodes = int((len(statuses) - config.test_episodes) / (config.env_params.max_steps + 1))
            for agent in am.agents_params:
                net_env.df=list(statuses)
                if "skip_learn" not in agent.__dict__ or agent.skip_learn:
                    continue
                if not hasattr(agent, 'episodes'):
                    information(Fore.YELLOW+"Param 'episodes' is missing\n")
                    continue        
                train_agent(agent)
                agent.episodes = episodes 
                #Step 2: plotting and saving agent data
                plot_and_save_data_agent(agent, config)
                agents_metrics[agent.name]=agent.instance.metrics
            
        
        if config.env_params.gym_type=="attacks":
            pause_event_traffic.set()                 
            #Step 2: plotting and saving all agents data
            for agent in am.agents_params:
                if agent.skip_learn:
                        continue
                plot_and_save_data_agent(agent, config)
                agents_metrics[agent.name]=agent.instance.metrics                
            pause_event_traffic.clear()
            
        if len(agents_metrics)>0:
            plot_comparison_bar_charts(config.training_execution_directory , agents_metrics)
            plot_radar_chart(config.training_execution_directory , agents_metrics)                           
        
        #Step 3: starting test
        directory_name = create_directory_training_execution(config, "TEST")
        test_attack_detect_agents(am, directory_name, config)        
        
        if config.env_params.gym_type=="attacks":
            #if attack version stop the threads
            stop_event_traffic.set()     
            net_env.stop_update_status_event.set()
            traffic_thread.join()
            net_env.update_state_thread_instance.join()            
        
        
        #Step 4: Plotting and saving all network traffic     
        information("Plotting all network traffic\n")
        if len(net_env.statuses)>2:
            statuses = list(net_env.statuses)
            save_data_to_file(statuses, config.training_execution_directory, "statuses")
            plot_enviroment_execution_statutes(statuses, config.training_execution_directory, "Statuses")  
                  
    except Exception as e: 
        #print(traceback.format_exc())
        error(Fore.RED+f"Something went wrong!\n{e}\{traceback.format_exc()}\n"+Fore.WHITE)
    finally:
        # Stop the network when everything has finished
        information(Fore.WHITE)
        if config.env_params.gym_type=="attacks":
            stop(net_env.net) 

def test_attack_detect_agents(am, directory_name, config):
    score, ground_truth, predicted = evaluate_attack_detect_agent(am)
    metrics =  {agent.name: {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0} for agent in config.agents}

    for s, p in zip(score.items(),predicted.items()):
        accuracy = accuracy_score(ground_truth,p[1])
        precision, recall, f1_score, _ = precision_recall_fscore_support(ground_truth, p[1], average='weighted', zero_division=0.0)
        metrics[s[0]]['accuracy'] = accuracy
        metrics[s[0]]['precision'] = precision
        metrics[s[0]]['recall'] = recall
        metrics[s[0]]['f1_score'] = f1_score
        information("Agent: "+ Fore.RED +f"{s[0]}"+ Fore.WHITE +f"\n\tScore: {s[1]}\n\tAccuracy {accuracy * 100 :.2f}%\n\tPrecision {precision * 100 :.2f}%\n\tRecall {recall * 100 :.2f}%\n\tF1-score {f1_score * 100 :.2f}%\n")    
        #transform ground_truth, p[1] -> predicted from multilabel to single
        gt = [1 if item[0] == 0 else 0 for item in ground_truth ]
        ps = [1 if item[0] == 0 else 0 for item in p[1] ]
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

def evaluate_attack_detect_agent(am: AgentManager):
        """
        Evaluate for n episodes a attack detect of traffic types
        Normal, Attack
        """             
        epochs = am.test_episodes
        agents_params = am.agents_params
        env = am.env
                
        information(f"*** Evaluation started: epochs {epochs} ***\n")
        score =  {agent.name: 0 for agent in agents_params}
        ground_truth = []
        predicted =  {agent.name: [] for agent in agents_params}

        for episode in range(epochs):
            if env.gym_type == 4: #attack
                time.sleep(1)
            information(f"\n\n************* Episode {episode+1} *************\n")            
            #self.env.is_state_normalized = True
            state, _ = env.reset(is_real_state= True) #state continuos
            
            g=np.zeros(env.actions_number)
            is_attack = 1 if env.status["id"]>0 else 0
            g[is_attack]+=1
            ground_truth.append(g)
            
            for agent in agents_params: 
                model = agent.instance
                if model is None:        
                    raise("The model can't be None. Create configuration")
                if isinstance(model, SupervisedAgent):
                    prediction = model.predict_attack(state)                    
                elif isinstance(model, QLearningAgent) or isinstance(model,SARSAAgent):
                    #discretized_state = self.env.get_discretized_state(self.env.real_state)
                    prediction = model.predict(state)
                else:
                    normalized_state = env.get_normalize_state(state) 
                    prediction, _states = model.predict(normalized_state, deterministic=True)
                color = Fore.RED           
                if prediction == is_attack:
                    score[agent.name]  += 1 
                    color = Fore.GREEN           
                    
                p=np.zeros(env.actions_number)
                p[prediction]+=1 
                predicted[agent.name].append(p)    
                information(f"{agent.name}: Action predicted"+color+f" {env.execute_action(prediction)}\n"+Fore.WHITE)

        information(f"*** Evaluation finished ***\n")
        return score, ground_truth, predicted
   

def start_continuous_traffic_generation(net_env: NetworkEnvAttackDetect, show_normal_traffic):
    # Create a stop event to manage the thread
    stop_event_traffic = threading.Event()
    pause_event_traffic = threading.Event() #for hosts
    #pause_event_traffic.set()  # Initially, the thread runs        
    traffic_thread = threading.Thread(target=continuous_traffic_generation, args=(net_env, stop_event_traffic, pause_event_traffic, show_normal_traffic))
    traffic_thread.start()
    return stop_event_traffic, pause_event_traffic, traffic_thread  
                
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
        #agent.instance.train_types = agent.custom_callback.train_types
        
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
    if hasattr(agent.instance, 'train_types'):
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
        plot_agent_execution_statuses(data.train_indicators, directory_name, agent.name)
        plot_agent_execution_confusion_matrix(data.train_indicators, directory_name)
    plot_combined_performance_over_time(data.train_metrics, directory_name, agent.name + " Combined performance over time")
    plot_metrics(data.train_metrics,directory_name,agent.name+" Train metrics")
    if hasattr(data, 'train_types') and len(data.train_types["explorations"]) > 0 and len(data.train_types["exploitations"]) > 0: 
        plot_train_types(data.train_types, data.train_execution_time, directory_name)
    
    #Step 5: saving data
    save_data_to_file(data.__dict__,directory_name)
    information(f"Data saved \n",agent.name)     