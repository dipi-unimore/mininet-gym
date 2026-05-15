import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from reinforcement_learning.agent_manager import AgentManager
from .network_env_attack_detect import NetworkEnvAttackDetect
from reinforcement_learning.agents.qlearning_agent import QLearningAgent
from reinforcement_learning.agents.sarsa_agent import SARSAAgent
from reinforcement_learning.agents.supervised_agent import SupervisedAgent
from reinforcement_learning.network_env import get_normalized_state
from utility.constants import ALGO_SUPERVISED, ATTACKS, GYM_TYPE, SystemLevels, SystemModes, SystemStatus
from utility.my_statistics import plot_agent_test_errors, plot_combined_performance_over_time, plot_comparison_bar_charts, plot_metrics, plot_agent_cumulative_rewards, plot_agent_execution_confusion_matrix, plot_agent_execution_statuses, plot_enviroment_execution_statutes, plot_radar_chart, plot_train_types, plot_agent_test, plot_test_confusion_matrix
from utility.my_files import save_data_to_file, read_data_file, create_directory_training_execution
from utility.my_log import error, information, debug, notify_client 
from utility.evaluation_summary import build_agent_evaluation_summary
from utility.training_summary import build_agent_training_summary
from collections import defaultdict
from colorama import Fore
import time, threading, traceback
import numpy as np

from utility.utils import ndarray_to_list

def attack_detect_main(config, am: AgentManager, env: NetworkEnvAttackDetect):
    
    try:
        
        # training
        agents_metrics = defaultdict(list)
        train_agent.env = env
        if config.env_params.gym_type==ATTACKS:
            start_training_agent(am)
        else:
            statuses = read_data_file(config.env_params.data_traffic_file)
            episodes = int((len(statuses) - config.env_params.test_episodes) / (config.env_params.max_steps + 1))
            for agent in am.agents_params:
                env.df=list(statuses)
                agent.episodes = episodes
                if isinstance(agent.instance, SupervisedAgent):
                    train_supervised_from_dataset(agent, statuses, agent.episodes, config)
                    agents_metrics[agent.name] = agent.instance.metrics
                    continue
                if "skip_learn" not in agent.__dict__ or agent.skip_learn:
                    continue
                train_agent(agent)
                #Step 2: plotting and saving agent data
                if config.env_params.print_training_chart :
                    plot_and_save_data_agent(agent, config)
                    agents_metrics[agent.name]=agent.instance.metrics
            
        if not env.stop_event.is_set():
            if config.env_params.print_training_chart:
                plot_training_data(config, env, am, agents_metrics) 
        
        if not env.stop_event.is_set():
            #starting test
            data = start_testing_agents(am, config)   
            data = {
                'score': data.score,
                'groundTruth' : data.ground_truth,
                'predicted' : data.predicted,
                'metrics' : data.metrics
            }   
            try:
                data = ndarray_to_list(data)
                notify_client(level=SystemLevels.DATA, final_data = data)    
            except Exception as e:
                error(Fore.RED+f"Error sending test data to client!\n{e}\n{traceback.format_exc()}\n"+Fore.WHITE)

        env.stop()              

        #Plotting and saving all network traffic     
        information("Plotting all network traffic\n")
        #comunicate with web UI to change status in plotting charts. No buttons  visible
        notify_client(level=SystemLevels.STATUS, status=SystemStatus.RUNNING, mode=SystemModes.PLOTTING, message="Plotting evaluation data...")        
        if len(env.statuses)>2:
            statuses = list(env.statuses)
            save_data_to_file(statuses, config.training_execution_directory, "statuses")
            try:
                plot_enviroment_execution_statutes(statuses, config.training_execution_directory, "Statuses")  
            except Exception as e:
                error(Fore.RED+f"Error plotting environment execution statuses!\n{e}\n{traceback.format_exc()}\n"+Fore.WHITE)
    except Exception as e: 
        #print(traceback.format_exc())
        error(Fore.RED+f"Something went wrong!\n{e}\n{traceback.format_exc()}\n"+Fore.WHITE)
    finally:
        # has finished
        information(Fore.WHITE)
        notify_client(level=SystemLevels.STATUS, status=SystemStatus.FINISHED, mode=SystemModes.TRAINING, message="Finished. Ready to start again...") 


def start_training_agent(am):
    # Start training threads for each agent
    notify_client(level=SystemLevels.STATUS, status=SystemStatus.RUNNING, mode=SystemModes.TRAINING, message="Started training...")
    training_threads = []
    train_agent.env = am.env
    for agent in am.agents_params:
        if not hasattr(agent, 'skip_learn') or agent.skip_learn:
            agent.skip_learn = True
            continue
        if not hasattr(agent, 'episodes'):
            agent.episodes = am.env.episodes  #default
   
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
                    agent.custom_callback.before_episode(episode+1)
                    agent.instance.learn(total_timesteps=agent.max_steps, callback=agent.custom_callback, progress_bar=agent.progress_bar)
                    agent.custom_callback.after_episode()

    except Exception as error:
        # handle the exception
        error(f"Agent {agent.name} learn:", error)
    agent.elapsed_time = time.time() - start_time
    information(f"Training completed in {agent.elapsed_time}\n",agent.name)

def train_supervised_from_dataset(agent, statuses, episodes, config):
    """Train SupervisedAgent standalone from dataset, one episode chunk at a time."""
    start_time = time.time()
    information(f"Starting supervised standalone training\n", agent.name)
    chunk_size = max(1, len(statuses) // episodes)
    for episode_idx in range(episodes):
        start = episode_idx * chunk_size
        end = start + chunk_size if episode_idx < episodes - 1 else len(statuses)
        chunk = statuses[start:end]
        if not chunk:
            continue
        agent.instance.accumulate_statuses(chunk)
        accuracy = agent.instance.train_on_accumulated_per_episode()
        information(f"Episode {episode_idx+1}/{episodes} - Accuracy: {accuracy*100:.2f}%\n", agent.name)
        try:
            notify_client(
                level=SystemLevels.DATA,
                agent_name=agent.name,
                force_immediate=True,
                metrics={
                    'episode': episode_idx + 1,
                    'accuracy': accuracy,
                    'precision': agent.instance.precision,
                    'recall': agent.instance.recall,
                    'f1_score': agent.instance.fscore,
                }
            )
        except Exception as e:
            debug(f"Error notifying metrics: {e}\n")

    agent.elapsed_time = time.time() - start_time
    information(f"Training completed in {agent.elapsed_time:.2f}s\n", agent.name)
    if config.env_params.print_training_chart:
        plot_and_save_data_agent(agent, config)

def plot_training_data(config, net_env, am, agents_metrics):
    if config.env_params.gym_type==ATTACKS:
        net_env.pause_event.set()
        debug("Paused to plot charts")
        #comunicate with web UI to change status in plotting charts. No buttons (start, stop, pause) visible
        notify_client(level=SystemLevels.STATUS, status=SystemStatus.RUNNING, mode=SystemModes.PLOTTING, message="Plotting training data...")
        #plotting and saving all agents data
        for agent in am.agents_params:
            if agent.skip_learn:
                    continue
            plot_and_save_data_agent(agent, config)
            agents_metrics[agent.name]=agent.instance.metrics                
            
    if len(agents_metrics)>0:
        plot_comparison_bar_charts(config.training_execution_directory , agents_metrics)
        plot_radar_chart(config.training_execution_directory , agents_metrics)                           

    net_env.pause_event.clear()
    debug("Resumed after plotted charts")        
    
def plot_and_save_data_agent(agent, config):

    # Collect metrics at the end of training
    if agent.is_custom_agent:
        accuracy, precision, recall, f1_score = agent.instance.get_metrics()
    elif isinstance(agent.instance, SupervisedAgent):
        accuracy = agent.instance.accuracy
        precision = agent.instance.precision
        recall = agent.instance.recall
        f1_score = agent.instance.fscore
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
    if agent.save and hasattr(agent.instance, 'save'):
        agent.instance.save(directory_name+"/"+agent.name)

    #Step 4: plotting training statistics
    information("Plotting training data\n",agent.name)
    if len(data.train_indicators)>2:
        try :
            plot_agent_cumulative_rewards(data.train_indicators, directory_name, agent.name)
            plot_agent_execution_statuses(data.train_indicators, directory_name, agent.name)
            plot_agent_execution_confusion_matrix(data.train_indicators, directory_name)
        except Exception as e:
            error(Fore.RED+f"Error plotting training indicators for {agent.name} !\n{e}\n{traceback.format_exc()}\n"+Fore.WHITE)
    if isinstance(agent.instance, SupervisedAgent) and hasattr(agent.instance, 'y_test') and agent.instance.y_test:
        try:
            plot_test_confusion_matrix(directory_name, agent.instance.y_test, agent.instance.y_pred, agent.name)
        except Exception as e:
            error(Fore.RED+f"Error plotting confusion matrix for {agent.name} !\n{e}\n{traceback.format_exc()}\n"+Fore.WHITE)
    try :
        plot_combined_performance_over_time(data.train_metrics, directory_name, agent.name + " Combined performance over time")
        plot_metrics(data.train_metrics,directory_name,agent.name+" Train metrics")
    except Exception as e:
        error(Fore.RED+f"Error plotting training metrics for {agent.name} !\n{e}\n{traceback.format_exc()}\n"+Fore.WHITE)
    if hasattr(data, 'train_types') and len(data.train_types["explorations"]) > 0 and len(data.train_types["exploitations"]) > 0: 
        try:
            plot_train_types(data.train_types, data.train_execution_time, directory_name)
        except Exception as e:
            error(Fore.RED+f"Error plotting training types for {agent.name} !\n{e}\n{traceback.format_exc()}\n"+Fore.WHITE)
    
    #Step 5: saving data
    save_data_to_file(data.__dict__,directory_name)
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
    information(f"Data saved \n",agent.name)   
    
    return data   

def start_testing_agents(am, config):
    #comunicate with web UI to change status to prepare to Test. Buttons start test visible, again put in pause, waiting start pressed
    notify_client(level=SystemLevels.STATUS, status=SystemStatus.RUNNING, mode=SystemModes.EVALUATION, message="Evaluating started...")
    time.sleep(1)  #wait for web UI to update status
    directory_name = create_directory_training_execution(config, "TEST")
    return test_attack_detect_agents(am, directory_name, config)  

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
        
        try:
            plot_test_confusion_matrix(directory_name, gt, ps, s[0])  
        except Exception as e: 
            error(Fore.RED+f"Error!\n{e}\n{traceback.format_exc()}\n"+Fore.WHITE)

    data = type('', (), {})()
    data.score = score
    data.ground_truth = ground_truth
    data.predicted = predicted
    
    data.metrics = metrics
    save_data_to_file(data.__dict__, directory_name,"test")
    # plot test
    try:
        # for agent_name in metrics.keys():
        #     plot_combined_performance_over_time(metrics[agent_name], directory_name, title='Test Combined performance over time')
        #     plot_metrics(metrics[agent_name], directory_name, title='Test Metrics')
        #     plot_comparison_bar_charts(directory_name , metrics[agent_name], title='Test Comparison Bar Charts')
        #     plot_radar_chart(directory_name , metrics[agent_name], title='Test Radar Chart')
        plot_agent_test(data.__dict__, directory_name, title='')
        plot_agent_test_errors(data.__dict__, directory_name, title='Agent Evaluation Errors')
    except Exception as e:
        error(Fore.RED+f"Error plotting test metrics!\n{e}\n{traceback.format_exc()}\n"+Fore.WHITE)

    for agent_name, agent_metrics in metrics.items():
        try:
            notify_client(
                level=SystemLevels.DATA,
                agent_name=agent_name,
                agent_evaluation_summary=build_agent_evaluation_summary(
                    config=config,
                    agent_name=agent_name,
                    directory_name=directory_name,
                    metrics=agent_metrics,
                    score=score.get(agent_name),
                    test_episodes=len(ground_truth),
                    shared_directory=True,
                ),
            )
        except Exception as exc:
            debug(Fore.YELLOW + f"Unable to notify evaluation summary for {agent_name}: {exc}\n" + Fore.WHITE)
    
    return data

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
            if env.gym_type == GYM_TYPE[ATTACKS]: 
                time.sleep(1)
            information(f"\n\n************* Episode {episode+1} *************\n")            
            #self.env.is_state_normalized = True
            state, _ = env.reset(options={"is_real_state": True}) #state continuos
            
            g=np.zeros(env.actions_number)
            is_attack = 1 if env.global_state.status["id"]>0 else 0
            g[is_attack]+=1
            ground_truth.append(g)
            
            for agent in agents_params: 
                model = agent.instance
                if model is None:        
                    raise("The model can't be None. Create configuration")
                if isinstance(model, SupervisedAgent):
                    prediction = model.predict(state)                    
                elif isinstance(model, QLearningAgent) or isinstance(model,SARSAAgent):
                    #discretized_state = self.env.get_discretized_state(self.env.real_state)
                    prediction = model.predict(state)
                else:
                    normalized_state = get_normalized_state(state, env.low_to_normalize, env.high_to_normalize) 
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
   
     
