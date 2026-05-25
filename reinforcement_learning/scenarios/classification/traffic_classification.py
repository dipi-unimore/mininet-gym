# classification_main.py
import json
import threading
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from reinforcement_learning.network_env import NetworkEnv, get_normalized_state
from reinforcement_learning.agent_manager import AgentManager
#from utility.my_statistics import plot_metrics, plot_indicators, plot_train_types, plot_net_metrics
from reinforcement_learning.agents.qlearning_agent import QLearningAgent
from reinforcement_learning.agents.sarsa_agent import SARSAAgent
from reinforcement_learning.agents.supervised_agent import SupervisedAgent
from utility.constants import CLASSIFICATION_FROM_DATASET, CLASSIFICATION, SystemLevels, SystemModes, SystemStatus
from utility.my_statistics import plot_agent_test_errors, plot_combined_performance_over_time, plot_comparison_bar_charts, plot_metrics, plot_agent_cumulative_rewards, plot_agent_execution_confusion_matrix, plot_agent_execution_traffic_types, plot_enviroment_execution_statutes, plot_radar_chart, plot_train_types, plot_agent_test, plot_test_confusion_matrix
from utility.my_files import read_data_file, save_data_to_file, create_directory_training_execution
from utility.my_log import debug, information,  error, notify_client
from utility.evaluation_summary import build_agent_evaluation_summary
from utility.training_summary import build_agent_training_summary
from colorama import Fore
import copy, time, traceback
from collections import defaultdict
import numpy as np

from utility.utils import ndarray_to_list

def _make_env_copy(base_env, statuses):
    """
    Create a per-agent env copy for parallel dataset-mode training.
    Auto-discovers unpicklable top-level attributes (threading events, gym spaces,
    Mininet objects), stashes them, deep copies the rest, then re-attaches the
    stashed refs so all threads share the same stop/pause signals.
    """
    _SKIP_PROBE = frozenset(('df', 'statuses'))
    stash = {}
    for attr, val in list(vars(base_env).items()):
        if attr in _SKIP_PROBE:
            continue
        try:
            copy.deepcopy(val)
        except Exception:
            stash[attr] = val
    for attr in stash:
        setattr(base_env, attr, None)
    try:
        env_copy = copy.deepcopy(base_env)
    finally:
        for attr, val in stash.items():
            setattr(base_env, attr, val)
    for attr, val in stash.items():
        setattr(env_copy, attr, val)
    env_copy.df = list(statuses)
    env_copy.statuses = []
    env_copy.stop_event = base_env.stop_event
    env_copy.pause_event = base_env.pause_event
    if hasattr(base_env, 'stop_update_event'):
        env_copy.stop_update_event = base_env.stop_update_event
    return env_copy


def traffic_classification_main(config, am: AgentManager, env: NetworkEnv):
   
    try:
         # Step 1: training
        agents_metrics = defaultdict(list)
        train_agent.env = env
        if config.env_params.gym_type==CLASSIFICATION:
            create_and_start_training_agent(am)
        elif config.env_params.gym_type==CLASSIFICATION_FROM_DATASET:
            statuses = read_data_file(config.env_params.data_traffic_file)
            episodes = int((len(statuses) - config.env_params.test_episodes) / (config.env_params.max_steps + 1))
            notify_client(level=SystemLevels.STATUS, status=SystemStatus.RUNNING, mode=SystemModes.TRAINING, message="Started training...")
            threads_data = []
            for agent in am.agents_params:
                agent.episodes = episodes
                if "skip_learn" not in agent.__dict__ or agent.skip_learn:
                    continue
                env_copy = _make_env_copy(env, statuses)
                t = threading.Thread(target=train_agent, args=(agent, env_copy), name=agent.name, daemon=True)
                threads_data.append((t, agent, env_copy))
                t.start()
            for t, _, _ in threads_data:
                t.join()
            for _, agent, env_copy in threads_data:
                env.statuses.extend(env_copy.statuses)
                plot_and_save_data_agent(agent, config)
                agents_metrics[agent.name] = agent.instance.metrics
        
        #env.stop_update_event.set()   
        #Step 2: plotting and saving agent data
        if not env.stop_event.is_set():
            if config.env_params.print_training_chart:
                plot_training_data(config, env, am, agents_metrics)
                
        
        if not env.stop_event.is_set():
            #starting test
            if config.env_params.gym_type == CLASSIFICATION_FROM_DATASET:
                env.df = list(statuses[-config.env_params.test_episodes:])
            data = start_testing_agents(am, config)
            data = {
                'score': data.score,
                'groundTruth' : data.ground_truth,
                'predicted' : data.predicted,
                'metrics' : data.metrics
            }   
            #clean data from ndarrays to be serializable
            try:
                data = ndarray_to_list(data)
                notify_client(level=SystemLevels.DATA, final_data = data)    
            except Exception as e:
                error(Fore.RED+f"Error sending test data to client!\n{e}\n{traceback.format_exc()}\n"+Fore.WHITE)
            
        env.stop()              

        #Plotting and saving all network traffic     
        information("Plotting all network traffic\n")
        if len(env.statuses)>2:
            statuses = list(env.statuses)
            save_data_to_file(statuses, config.training_execution_directory, "statuses")
            try:
                plot_enviroment_execution_statutes(statuses, config.training_execution_directory, "Statuses")  
            except Exception as e:
                error(Fore.RED+f"Error plotting environment execution statuses!\n{e}\n{traceback.format_exc()}\n"+Fore.WHITE)
    except Exception as e: 
        #print(traceback.format_exc())
        error(Fore.RED+f"Something went wrong!\n{e}: {traceback.format_exc()}\n"+Fore.WHITE)
    finally:
        # has finished
        information(Fore.WHITE)
        notify_client(level=SystemLevels.STATUS, status=SystemStatus.FINISHED, mode=SystemModes.TRAINING, message="Finished. Ready to start again...") 

def create_and_start_training_agent(am: AgentManager):   
    # Start training threads for each agent
    notify_client(level=SystemLevels.STATUS, status=SystemStatus.RUNNING, mode=SystemModes.TRAINING, message="Started training...")
    training_threads = []
    train_agent.env = am.env
    for agent in am.agents_params:
        if  not hasattr(agent, 'skip_learn') or agent.skip_learn:
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

def train_agent(agent, env=None):
    """
    Function for training a single agent.
    Args:
        agent: Agent to be trained.
        env: Environment to use. Defaults to train_agent.env (shared, for real-network mode).
    """
    env = env or train_agent.env
    start_time = time.time()
    try:
        information(f"Starting training\n", agent.name)
        if isinstance(agent.instance, SupervisedAgent):
            agent.instance.learn(agent.episodes, env, env.stop_event)
        elif agent.is_custom_agent:
            agent.instance.env = env
            agent.instance.learn(agent.episodes, env.stop_event)
        else:
            try:
                agent.instance.set_env(env)
            except Exception as e:
                debug(Fore.YELLOW + f"Could not swap env for {agent.name}: {e}\n" + Fore.WHITE)
            if hasattr(agent, 'custom_callback') and agent.custom_callback:
                agent.custom_callback.env = env
            for episode in range(agent.episodes):
                if env.stop_event.is_set():
                    break
                agent.custom_callback.before_episode(episode + 1)
                agent.instance.learn(total_timesteps=agent.max_steps, callback=agent.custom_callback, progress_bar=agent.progress_bar)
                agent.custom_callback.after_episode()
    except Exception as err:
        error(Fore.RED + f"Agent {agent.name} learn: {err}\n{traceback.format_exc()}" + Fore.WHITE)

    agent.elapsed_time = time.time() - start_time
    information(f"Training completed in {agent.elapsed_time:.1f}s\n", agent.name)

    try:
        if hasattr(agent.instance, 'rewards') and hasattr(agent.instance, 'metrics'):
            episode_rewards = []
            max_steps = env.max_steps if hasattr(env, 'max_steps') else 1
            for i in range(0, len(agent.instance.rewards), max(1, max_steps)):
                episode_rewards.append(sum(agent.instance.rewards[i:i+max_steps]))
            training_data = {
                'agent': agent.name,
                'episode_rewards': episode_rewards,
                'accuracy': agent.instance.metrics.get('accuracy', []),
                'precision': agent.instance.metrics.get('precision', []),
                'recall': agent.instance.metrics.get('recall', []),
                'f1_score': agent.instance.metrics.get('f1_score', []),
            }
            notify_client(level=SystemLevels.DATA, agent_name=agent.name, training_summary=training_data)
            debug(f"Sent training summary for {agent.name}: {len(episode_rewards)} episodes\n")
    except Exception as e:
        debug(Fore.RED + f"Error sending training summary for {agent.name}: {e}\n" + Fore.WHITE)

def plot_training_data(config, net_env, am, agents_metrics):
    if config.env_params.gym_type==CLASSIFICATION:
        net_env.pause_event.set()
        debug("Paused to plot charts")
        #comunicate with web UI to change status in plotting charts. No buttons (start, stop, pause) visible
        notify_client(level=SystemLevels.STATUS, status=SystemStatus.RUNNING, mode=SystemModes.PLOTTING, message="Plotting training data...")
        #plotting and saving all agents data
        for agent in am.agents_params:
            if agent.skip_learn:
                    continue
            try:    
                plot_and_save_data_agent(agent, config)
            except Exception as e:
                error(Fore.RED+f"Error plotting training data for agent {agent.name}!\n{e}\n{traceback.format_exc()}\n"+Fore.WHITE)
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
    #if hasattr(agent, 'save') and agent.save and hasattr(agent.instance, 'save'):
    agent.instance.save(directory_name+"/"+agent.name)
        
    #Step 4: plotting training statistics
    information("Plotting training data\n",agent.name)
    if len(data.train_indicators)>2 :
        try:
            plot_agent_cumulative_rewards(data.train_indicators, directory_name, agent.name)
            #plot_agent_execution_traffic_types(data.train_indicators, directory_name, agent.name)
            plot_agent_execution_confusion_matrix(data.train_indicators, directory_name)
        except Exception as e:
            error(Fore.RED+f"Error plotting training indicators for {agent.name} !\n{e}\n{traceback.format_exc()}\n"+Fore.WHITE)
    # if isinstance(agent.instance, SupervisedAgent) and hasattr(agent.instance, 'y_test') and agent.instance.y_test:
    #     try:
    #         plot_test_confusion_matrix(directory_name, agent.instance.y_test, agent.instance.y_pred, agent.name)
    #     except Exception as e:
    #         error(Fore.RED+f"Error plotting confusion matrix for {agent.name} !\n{e}\n{traceback.format_exc()}\n"+Fore.WHITE)
    try:
        plot_combined_performance_over_time(data.train_metrics, directory_name, agent.name + " Combined performance over time")
        plot_metrics(data.train_metrics,directory_name,agent.name+" Train metrics")
    except Exception as e:
        error(Fore.RED+f"Error plotting training metrics for {agent.name} !\n{e}\n{traceback.format_exc()}\n"+Fore.WHITE)
    if "train_types" in data.__dict__ and len(data.train_types["explorations"]) > 0 and len(data.train_types["exploitations"]) > 0:
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

def start_testing_agents(am, config):
    #comunicate with web UI to change status to prepare to Test. Buttons start test visible, again put in pause, waiting start pressed    
    notify_client(level=SystemLevels.STATUS, status=SystemStatus.RUNNING, mode=SystemModes.EVALUATION, message="Evaluating started...")
    time.sleep(1)  #wait for web UI to update status
    directory_name = create_directory_training_execution(config, "TEST")
    return test_classification_agents(am, directory_name, config)  

def test_classification_agents(am, directory_name, config):
    score, ground_truth, predicted = evaluate_classification_agent(am)
    metrics =  {agent.name: {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0} for agent in config.agents if agent.enabled}

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
        try:
            plot_test_confusion_matrix(directory_name, gt, ps, s[0], labels=[0,1,2,3], display_labels=["none","ping","udp","tcp"])
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

def evaluate_classification_agent(am: AgentManager):
    """
    Evaluate for n episodes a classification of traffic types
    None, Ping, UDP, TCP
    """      
    epochs = am.test_episodes
    agents_params = am.agents_params   
    env = am.env     
            
    information(f"Evaluation started: epochs {epochs}\n")
    score =  {agent.name: 0 for agent in agents_params}
    ground_truth = []
    predicted =  {agent.name: [] for agent in agents_params}

    for episode in range(epochs):
        information(f"\n\n************* Episode {episode+1} *************\n")            
        real_state, _ = env.reset(options={"is_real_state": True}) #state continuos, not_normalized, not discretized
        
        g=np.zeros(env.actions_number)
        g[env.generated_traffic_type]+=1
        ground_truth.append(g)
        normalized_state = get_normalized_state(real_state, env.low, env.high) 
        #information(f"p_r={real_state[0]}\np_t={real_state[1]}\nb_r={real_state[2]}byte\nb_t={real_state[3]}byte\n")
        
        for agent in agents_params: 
            model = agent.instance
            if model is None:        
                raise("The model can't be None. Create configuration")
            if isinstance(model, SupervisedAgent) :
                prediction = model.predict(real_state)
            elif isinstance(model, QLearningAgent) or isinstance(model,SARSAAgent):
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
    
