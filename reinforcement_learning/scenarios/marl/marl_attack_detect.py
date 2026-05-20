# marl_attack_detect_main.py
from collections import defaultdict
import copy
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from .constants import AGENT_ACTIONS, COORDINATOR, COORDINATOR_ACTIONS, COORDINATOR_LABELS, HOST_LABELS
from .network_env_marl_attack_detect import NetworkEnvMarlAttackDetect
from reinforcement_learning.agent_manager import AgentManager
from reinforcement_learning.agents.qlearning_agent import QLearningAgent
from reinforcement_learning.agents.sarsa_agent import SARSAAgent
from reinforcement_learning.agents.supervised_agent import SupervisedAgent
from reinforcement_learning.network_env import get_normalized_state
from utility.constants import *
from utility.my_marl_statistics import calculate_team_metrics, plot_agent_test_errors, plot_combined_performance_over_time, plot_comparison_bar_charts, plot_metrics, plot_agent_cumulative_rewards, plot_agent_training_confusion_matrix, plot_agent_execution_statuses, plot_enviroment_execution_statutes, plot_radar_chart, plot_train_types, plot_agent_test, plot_test_confusion_matrix
from utility.my_files import save_data_to_file, read_data_file, create_directory_training_execution
from utility.my_log import error, information, debug, notify_client
from utility.evaluation_summary import build_agent_evaluation_summary
from utility.training_summary import build_agent_training_summary
from colorama import Fore
import time, threading, traceback

from utility.utils import ndarray_to_list

def marl_attack_detect_main(config, am: AgentManager, env: NetworkEnvMarlAttackDetect):
    
    try:        
        
        #  training
        agents_metrics = defaultdict(list)
        if config.env_params.gym_type==MARL_ATTACKS:       
            create_and_start_training_agent(am)
        else:
            statuses = read_data_file("statuses")
            episodes = int((len(statuses) - config.test_episodes) / (config.env_params.max_steps + 1))
            for agent in am.agents_params:
                env.df=list(statuses)
                agent.episodes = episodes
                if isinstance(agent.instance, SupervisedAgent) or  "skip_learn" not in agent.__dict__ or agent.skip_learn:
                    continue
                train_agent(agent)  #train the agent TODO: threads
                #Step 2: plotting and saving agent data
                if config.env_params.print_training_chart :
                    data = plot_and_save_data_agent(agent, config)
                    agents_metrics[agent.name] = data.metrics 
        
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

        # Plotting and saving all network traffic     
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
        notify_client(level=SystemLevels.STATUS, status=SystemStatus.FINISHED, mode=SystemModes.PLOTTING, message="Finished. Ready to start again...") 

def create_and_start_training_agent(am: AgentManager):
    # Start training threads for each agent
    notify_client(level=SystemLevels.STATUS, status=SystemStatus.RUNNING, mode=SystemModes.TRAINING, message="Started training...")
    training_threads = []
    train_agent.env = am.env
    for agent in am.agents_params:
        if agent.skip_learn:
            continue
        if not hasattr(agent, 'episodes'):
            agent.episodes = am.env.episodes  #default
        start_time = time.time()                    
        instances = agent.instances   
        new_instances = []
        for host_name, instance_info in instances.items():
            host_agent = copy.copy(agent)
            host_agent.instance = copy.copy(instance_info['instance'])
            host_agent.custom_callback = instance_info['custom_callback']
            host_agent.is_custom_agent = instance_info['is_custom_agent']
            host_agent.max_steps = instance_info['max_steps']
            host_agent.name = host_agent.instance.name
            new_instances.append({'name': host_name, 'agent': host_agent})
            debug(f"Starting training thread for agent {agent.name} on host {host_name}")
            training_thread = threading.Thread(
                target=train_agent,
                args=(host_agent,)
            )
            training_threads.append(training_thread)
            training_thread.start()
        agent.instances = new_instances
        agent.elapsed_time = time.time() - start_time
            
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
            agent.instance.learn(agent.episodes, stop_event = train_agent.env.stop_event)
        else:
            # Check if this is a SupervisedAgent
            is_supervised = isinstance(agent.instance, SupervisedAgent)

            for episode in range(agent.episodes):
                if train_agent.env.stop_event.is_set():
                    break

                while train_agent.env.pause_event.is_set():
                    time.sleep(0.5)

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


def plot_training_data(config, net_env, am, agents_metrics):
    if config.env_params.gym_type==MARL_ATTACKS:
        net_env.pause_event.set()      #only traffic and update state thread remained         
        debug("Paused to plot charts")
        #comunicate with web UI to change status in plotting charts. No buttons (start, stop, pause) visible
        notify_client(level=SystemLevels.STATUS, status=SystemStatus.RUNNING, mode=SystemModes.PLOTTING, message="Plotting training data...")
        #plotting and saving all agents data
        for agent in am.agents_params:
            if agent.skip_learn:
                    continue
            data = plot_and_save_data_agent(agent, config)
            agents_metrics[agent.name] = calculate_team_metrics(data.train_metrics)                
                
    if len(agents_metrics)>0:
        #TODO manipulate metrics for all host as a team
        plot_comparison_bar_charts(config.training_execution_directory , agents_metrics)
        plot_radar_chart(config.training_execution_directory , agents_metrics)        
                    
    net_env.pause_event.clear()
    debug("Resumed after plotted charts")
    
def plot_and_save_data_agent(parent_agent, config):    
    
    data = type('', (), {})()
    data.train_execution_time = parent_agent.elapsed_time
    data.train_metrics =  {}
    data.train_indicators =  {}
    data.train_types = {}
    
    # Collect metrics at the end of training
    for instance in parent_agent.instances:
        agent_name = instance['name']
        agent = instance['agent']

        if not agent.is_custom_agent:
            agent.instance.metrics = agent.custom_callback.metrics
            agent.instance.indicators = agent.custom_callback.indicators

        data.train_metrics[agent_name]=agent.instance.metrics
        data.train_indicators[agent_name]= agent.instance.indicators
        if hasattr(agent.instance, 'train_types'):
            data.train_types[agent_name]= agent.instance.train_types
            
    
    #create directory to save all files for the agent training excecution
    directory_name = create_directory_training_execution(config, agent_name = parent_agent.name)
    if agent.save:  
        #save all instances
        for instance in parent_agent.instances:
            agent_name = f"{parent_agent.name}_{instance['name']}"
            agent = instance['agent']
            agent.instance.save(directory_name+"/"+agent_name)
            information(f"Agent {agent_name} model saved \n",parent_agent.name)       

        
    #Step 4: plotting training statistics
    information("Plotting training data\n",parent_agent.name)
    for host_name, train_indicators in data.train_indicators.items():
        if len(train_indicators)>2:
            try :
                #TODO evaluate to print the rewards as cumulative among all hosts as a team
                plot_agent_cumulative_rewards(train_indicators, directory_name, title=f"{parent_agent.name}_{host_name}", host=host_name)
                plot_agent_execution_statuses(train_indicators, directory_name, title=f"{parent_agent.name}_{host_name}", host=host_name)
                #TODO evaluate to print the confusion matrix as cumulative among all hosts as a team
                labels = COORDINATOR_LABELS if host_name==COORDINATOR else HOST_LABELS
                plot_agent_training_confusion_matrix(train_indicators, directory_name, agent_name=parent_agent.name, host_name=host_name, labels=labels)
            except Exception as e:
                error(Fore.RED+f"Error plotting training indicators for {parent_agent.name}_{host_name} !\n{e}\n{traceback.format_exc()}\n"+Fore.WHITE)
        train_metrics = data.train_metrics[host_name]
        try :
            plot_combined_performance_over_time(train_metrics, directory_name, title=f"{parent_agent.name}_{host_name}  Combined performance over time", host=host_name)
            plot_metrics(train_metrics, directory_name, title=f"{parent_agent.name}_{host_name} Train metrics", host=host_name)
        except Exception as e:
            error(Fore.RED+f"Error plotting training metrics for {parent_agent.name}_{host_name} !\n{e}\n{traceback.format_exc()}\n"+Fore.WHITE)
        if hasattr(data, 'train_types'):
            #TODO verify host_name in data.train_types
            train_types = data.train_types[host_name]
            if 'explorations' in data.train_types.keys() and len(train_types["explorations"]) > 0 and len(train_types["exploitations"]) > 0: 
                try:
                    plot_train_types(train_types, data.train_execution_time, directory_name, title=f"{parent_agent.name}_{host_name} Train types over time", host=host_name)
                except Exception as e:
                    error(Fore.RED+f"Error plotting training types for {parent_agent.name} !\n{e}\n{traceback.format_exc()}\n"+Fore.WHITE)
    
    #Step 5: saving data
    save_data_to_file(data.__dict__, directory_name)
    try:
        notify_client(
            level=SystemLevels.DATA,
            agent_name=parent_agent.name,
            agent_training_summary=build_agent_training_summary(
                config=config,
                agent_name=parent_agent.name,
                directory_name=directory_name,
                train_metrics=data.train_metrics,
                train_indicators=data.train_indicators,
                train_execution_time=data.train_execution_time,
            ),
        )
    except Exception as exc:
        debug(Fore.YELLOW + f"Unable to notify training summary for {parent_agent.name}: {exc}\n" + Fore.WHITE)
    information(f"Data saved \n", parent_agent.name)     
    
    return data    
              
def start_testing_agents(am, config):
    #comunicate with web UI to change status to prepare to Test. Buttons start test visible, again put in pause, waiting start pressed
    notify_client(level=SystemLevels.STATUS, status=SystemStatus.RUNNING, mode=SystemModes.EVALUATION, message="Evaluating started...")
    time.sleep(1)  #wait for web UI to update status
    directory_name = create_directory_training_execution(config, "TEST")
    return test_attack_detect_agents(am, directory_name, config)  
                        
def test_attack_detect_agents(am, directory_name, config):
    score, ground_truth, predicted = evaluate_attack_detect_agent(am)
    notify_client(level=SystemLevels.STATUS, status=SystemStatus.RUNNING, mode=SystemModes.PLOTTING, message="Plotting evaluation data...")
    metrics =  {agent.name: {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []} for agent in config.agents}

    for s, p in zip(score.items(),predicted.items()):
        gs = []
        ps = []
        for host_name in ground_truth.keys():
            gh = ground_truth[host_name]
            ph = [i[host_name] for i in p[1]]
            accuracy = accuracy_score(gh,ph)
            precision, recall, f1_score, _ = precision_recall_fscore_support(gh, ph, average='weighted', zero_division=0.0)
            metrics[s[0]]['accuracy'].append(accuracy)
            metrics[s[0]]['precision'].append(precision)
            metrics[s[0]]['recall'].append(recall)
            metrics[s[0]]['f1_score'].append(f1_score)
            information("Agent: "+ Fore.RED +f"{s[0]} {host_name}"+ Fore.WHITE +f"\n\tScore: {s[1][host_name]}\n\tAccuracy {accuracy * 100 :.2f}%\n\tPrecision {precision * 100 :.2f}%\n\tRecall {recall * 100 :.2f}%\n\tF1-score {f1_score * 100 :.2f}%\n")    
            #transform ground_truth, p[1] -> predicted from multilabel to single
            # if host_name==COORDINATOR:
            #     ghs = [0 if item[0] == 1 else 1 for item in gh ]
            #     phs = [0 if item[0] == 1 else 1 for item in ph ]
            # else:
            #     ghs = [0 if item[0] == 1 else (1 if item[1] == 1 else 2 ) for item in gh ]
            #     phs = [0 if item[0] == 1 else (1 if item[1] == 1 else 2 ) for item in ph ]
            #evaluate only attack detection, discarfing in or out specification for hosts
            try:
                labels = COORDINATOR_LABELS if host_name==COORDINATOR else HOST_LABELS
                plot_test_confusion_matrix(directory_name, gh, ph, agent_name=s[0], host_name=host_name, labels=labels)  
            except Exception as e: 
                error(Fore.RED+f"Error!\n{e}\n{traceback.format_exc()}\n"+Fore.WHITE)
            if not host_name==COORDINATOR:
                gh = [[1,0] if item[0] == 1 else [0,1] for item in gh ]
                ph = [[1,0] if item[0] == 1 else [0,1] for item in ph ]
            gs.extend(gh)
            ps.extend(ph)
        #plot confusion matrix for all hosts as a team
        try:
            plot_test_confusion_matrix(directory_name, gs, ps, agent_name=s[0], host_name="TEAM", labels=COORDINATOR_LABELS)  
        except Exception as e: 
            error(Fore.RED+f"Error!\n{e}\n{traceback.format_exc()}\n"+Fore.WHITE)
        #calculate mean metrics for the agent among all hosts            
        metrics[s[0]]['accuracy']=np.mean(metrics[s[0]]['accuracy'])
        metrics[s[0]]['precision']=np.mean(metrics[s[0]]['precision'])
        metrics[s[0]]['recall']=np.mean(metrics[s[0]]['recall'])
        metrics[s[0]]['f1_score']=np.mean(metrics[s[0]]['f1_score'])
    data = type('', (), {})()
    data.score = score
    data.ground_truth = ground_truth
    data.predicted = predicted    
    data.metrics = metrics
    
    save_data_to_file(data.__dict__, directory_name,"test")
    # plot test
    try:
        # plot_combined_performance_over_time(metrics[s[0]], directory_name, title='Test Combined performance over time')
        # plot_metrics(metrics[s[0]], directory_name, title='Test Metrics')
        # plot_comparison_bar_charts(directory_name , metrics[s[0]], title='Test Comparison Bar Charts')
        # plot_radar_chart(directory_name , metrics[s[0]], title='Test Radar Chart')
        for h in ground_truth.keys():
            plot_agent_test(data.__dict__, directory_name, host=h, title='')
            plot_agent_test_errors(data.__dict__, directory_name,  host=h, title='Agent Evaluation Errors')
    except Exception as e:
        error(Fore.RED+f"Error plotting test metrics!\n{e}\n{traceback.format_exc()}\n"+Fore.WHITE)

    test_episodes = len(next(iter(ground_truth.values()))) if ground_truth else 0
    for agent_name, agent_metrics in metrics.items():
        try:
            agent_score = score.get(agent_name)
            if isinstance(agent_score, dict):
                team_score = sum(int(value) for value in agent_score.values())
            else:
                team_score = agent_score
            notify_client(
                level=SystemLevels.DATA,
                agent_name=agent_name,
                agent_evaluation_summary=build_agent_evaluation_summary(
                    config=config,
                    agent_name=agent_name,
                    directory_name=directory_name,
                    metrics=agent_metrics,
                    score=team_score,
                    test_episodes=test_episodes,
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
        score =  {agent.name: {h["name"]: 0 for h in agents_params[0].instances} for agent in agents_params}
        
        ground_truth = {h["name"]: [] for h in agents_params[0].instances}
        predicted =  {agent.name: [] for agent in agents_params}

        for episode in range(epochs):
            if env.gym_type == GYM_TYPE[MARL_ATTACKS]:
                time.sleep(1)
            information(f"\n\n************* Episode {episode+1} *************\n")            

            states, _ = env.reset(options={"is_real_state": True}) 
            
            #ground_truth prepared for coordinator and every host
            g=np.zeros(COORDINATOR_ACTIONS.NUMBER)
            is_attack = {}
            is_attack[COORDINATOR] = env.global_state.status["id"] #0,1
            g[is_attack[COORDINATOR]]+=1
            ground_truth[COORDINATOR].append(g)
            
            for host in env.hosts:
                g=np.zeros(AGENT_ACTIONS.NUMBER)
                is_attack[host.name] = env.global_state.get_host_status(host.name)["id"] #0,1,2
                g[is_attack[host.name]]+=1
                ground_truth[host.name].append(g)
            
            for agent in agents_params:
                p = {h["name"]: [] for h in agents_params[0].instances}
                #s = {h["name"]: 0 for h in agents_params[0].instances}
                for i in agent.instances:
                    model = i['agent'].instance
                    name = i['name']
                    prediction = predict(env, model, states[name])
                    
                    color = Fore.RED         
                    if prediction == is_attack[name]:
                        score[agent.name][name] += 1 
                        color = Fore.GREEN
                        result = "scored"
                    else:
                        result = "missed"           
                    
                    if name==COORDINATOR:
                        p[name]=np.zeros(COORDINATOR_ACTIONS.NUMBER)
                    else:
                        p[name]=np.zeros(AGENT_ACTIONS.NUMBER)
                    p[name][prediction]+=1 
                    information(f"{agent.name} {name}: Action predicted"+color+f" {prediction} {result}\n"+Fore.WHITE)
                #score[agent.name].append(s) 
                predicted[agent.name].append(p)    

        information(f"*** Evaluation finished ***\n")
        return score, ground_truth, predicted

def predict(env, model, state):
    if model is None:        
        raise("The model can't be None. Create configuration")
    if isinstance(model, SupervisedAgent):
        prediction = model.predict(state)                    
    elif isinstance(model, QLearningAgent) or isinstance(model,SARSAAgent):
        prediction = model.predict(state)
    else:
        normalized_state = get_normalized_state(state, env.low_to_normalize, env.env.high_to_normalize) 
        prediction, _states = model.predict(normalized_state, deterministic=True)
    env.execute_action(prediction)
    return prediction
               
