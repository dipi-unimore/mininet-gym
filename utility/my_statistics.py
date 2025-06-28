# my_statistics.py
import matplotlib
matplotlib.use('Agg')  # Used for a non-GUI backend (Agg is for PNG output)
import matplotlib.pyplot as plt, matplotlib.patches as mpatches, numpy as np
from sklearn import metrics
from utility.my_files import read_data_file, find_latest_execution, read_all_data_from_execution
from utility.params import  Params, read_config_file
from utility.my_log import information, error
import traceback
from colorama import Fore

def plot_all_lines_and_save_png(lines, title, x_label, y_label,legend_lables, colors, markers, file_name):
    """
    generic plot function
    """  
 
    i=1
    x=[]
    y = [[0 for x in range(len(lines))] for y in range(len(lines[0]))] 
    for j in lines:
        x.append(i)
        for w in range(len(j)):
            y[w][i-1]=j[w]
        i+=1

    for w in range(len(lines[0])):
        plt.plot(x, y[w], linestyle='-', linewidth=2.0) # - solid line -- dashed line 
        
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.gca().set_prop_cycle(color=colors,marker=markers)
    plt.title(f'{title}')   
    plt.legend(legend_lables, loc='upper left')
    #plt.semilogy(lines, "b--")
    #plt.show()

    plt.savefig(f"{file_name}.png")
    plt.close()  

def plot_agent_action_accuracy(indicators, dir_name):
    """Plot agent accuracy per action
    
    Args:
        indicators (_type_): _description_
        dir_name (_type_): _description_
    """
    steps = [item['step'] for item in indicators]
    correct_actions = [1 if item['action'] == item['traffic_type'] else 0 for item in indicators]
    accuracy = np.cumsum(correct_actions) / np.arange(1, len(correct_actions) + 1)
    
    plt.plot(steps, accuracy, label='Accuracy', color='green')
    plt.title('Action Accuracy Over Time')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.savefig(f"{dir_name}/accuracy.png")
    plt.close()

def plot_agent_cumulative_rewards(indicators, dir_name, title=''):
    """plot the agent cumulative reward for each episode
        
    Args:
        indicators (_type_): _description_
        dir_name (_type_): _description_
        title (str, optional): _description_. Defaults to ''.
    """
    # Extracting indicators to plot
    episodes = [item['episode'] for item in indicators]
    steps = [item['steps'] for item in indicators]
    correct_predictions = [item['correct_predictions'] for item in indicators]
    reward = [item['cumulative_reward'] for item in indicators]

    x=10+3*int(len(episodes)/70)
    y=10+int(len(episodes)/70)
    plt.figure(figsize=(x, y))
    plt.plot(episodes, reward, label='Rewards', color='purple')
    plt.plot(episodes, steps, label='Steps', color='cyan')
    plt.plot(episodes, correct_predictions, label='Correct predictions', color='green')
    plt.title(f'{title} Cumulative Rewards')
    plt.xlabel('Episodes')
    plt.legend()

    # Save figure
    plt.savefig(f"{dir_name}/rewards.png")
    plt.close()    

def plot_agent_execution_confusion_matrix(indicators, dir_name):
    """print confusion matrix

    Args:
        indicators (_type_): List of dictionaries containing episode statuses.
        dir_name (_type_): Directory name to save the plot.
        title (str, optional): Title of the plot. Defaults to ''.
    """
    # Extracting indicators to plot
    all_steps_status =  []
    for item in indicators:
        for step_status in item['episode_statuses']:
            all_steps_status.append(step_status)

    predicted = [item['action_choosen'] for item in all_steps_status]
    if 'traffic_type' in all_steps_status[0]:
        # If 'traffic_type' exists, use it as ground truth, eg classification
        ground_truth = [item['traffic_type'] for item in all_steps_status]
    else:
        # eg attack the prediction is 0 or 1
        ground_truth = [item['action_choosen'] if item['action_correct'] else 1 - item['action_choosen'] for item in all_steps_status]
    try:
        confusion_matrix = metrics.confusion_matrix(ground_truth, predicted)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
        cm_display.plot()
        plt.savefig(f"{dir_name}/matrix.png")
        plt.close()
    except Exception as e: 
        error(Fore.RED+f"Error!\n{e}\{traceback.format_exc()}\n"+Fore.WHITE)
    
def plot_agent_execution_statuses(indicators, dir_name, title=''):
    """print all registered traffic normal or attack

    Args:
        indicators (_type_): List of dictionaries containing episode statuses.
        dir_name (_type_): Directory name to save the plot.
        title (str, optional): Title of the plot. Defaults to ''.
    """
    # Extracting indicators to plot
    all_steps_status =  []
    n_steps = 0
    for item in indicators:
        for step_status in item['episode_statuses']:
            n_steps += 1
            step_status["step"] = n_steps 
            all_steps_status.append(step_status)

    steps = [item['step'] for item in all_steps_status]
    bytes = [item['bytes'] for item in all_steps_status]
    packets = [item['packets'] for item in all_steps_status]
    predictions = [item['action_correct'] for item in all_steps_status]
    types = [item['id'] for item in all_steps_status]
    
    # Create subplots for different indicators
    x=10+3*int(n_steps/200)
    y=10+int(n_steps/200)
    fig, axs = plt.subplots(3, 1, figsize=(x, y))

    axs[0].set_yscale("log")
    axs[0].plot(steps, bytes, label='Bytes', color='purple')
    axs[0].plot(steps, packets, label='Packets', color='cyan')
    axs[0].set_title(f'{title} Packets-Bytes')
    axs[0].set_xlabel('Steps')
    axs[0].set_ylabel('Log Scale Traffic')    
    axs[0].legend()  

    colors = get_colors_for_types(types)    
    axs[1].scatter(steps, types, label='Types', c=colors, s=3)
    axs[1].set_title(f'{title} - Event Types')
    axs[1].set_xlabel('Steps')
    axs[1].set_ylabel('Type')      
    legend_patches_types = get_legend_labels_for_types()
    axs[1].legend(handles=legend_patches_types, loc='lower left')     
    
    colors = get_colors_for_predictions(predictions) 
    axs[2].scatter(steps, predictions, label='Predictions', c=colors, s=3)
    legend_patches_predictions = get_legend_labels_for_predictions()
    axs[2].legend(handles=legend_patches_predictions, loc='center')
    axs[2].set_title(f'{title} - Correct/Wrong Predictions')
    axs[2].set_xlabel('Steps')
    axs[2].set_ylabel('Prediction')
    
    # Save figure
    plt.savefig(f"{dir_name}/episode_statuses.png")
    plt.close()   
    
def plot_agent_execution_traffic_types(indicators, dir_name, title=''):
    """print all registered traffic types for classification

    Args:
        indicators (_type_): List of dictionaries containing episode statuses.
        dir_name (_type_): Directory name to save the plot.
        title (str, optional): Title of the plot. Defaults to ''.
    """
    # Extracting indicators to plot
    all_steps_status =  []
    n_steps = 0
    for item in indicators:
        for step_status in item['episode_statuses']:
            n_steps += 1
            step_status["step"] = n_steps 
            all_steps_status.append(step_status)

    steps = [item['step'] for item in all_steps_status]
    bytes = [item['bytes_received'] for item in all_steps_status]
    packets = [item['packets_received'] for item in all_steps_status]
    predictions = [item['action_correct'] for item in all_steps_status]
    #types = [item['id'] for item in all_steps_status]
    
    # Create subplots for different indicators
    x=10+3*int(n_steps/200)
    y=10+int(n_steps/200)
    fig, axs = plt.subplots(3, 1, figsize=(x, y))

    axs[0].set_yscale("log")
    axs[0].plot(steps, bytes, label='Bytes', color='purple')
    axs[0].plot(steps, packets, label='Packets', color='cyan')
    axs[0].set_title(f'{title} Packets-Bytes')
    axs[0].set_xlabel('Steps')
    axs[0].set_ylabel('Log Scale Traffic')    
    axs[0].legend()  

    # colors = get_colors_for_types(types)    
    # axs[1].scatter(steps, types, label='Types', c=colors, s=3)
    # axs[1].set_title(f'{title} - Event Types')
    # axs[1].set_xlabel('Steps')
    # axs[1].set_ylabel('Type')      
    # legend_patches_types = get_legend_labels_for_types()
    # axs[1].legend(handles=legend_patches_types, loc='lower left')     
    
    colors = get_colors_for_predictions(predictions) 
    axs[2].scatter(steps, predictions, label='Predictions', c=colors, s=3)
    legend_patches_predictions = get_legend_labels_for_predictions()
    axs[2].legend(handles=legend_patches_predictions, loc='center')
    axs[2].set_title(f'{title} - Correct/Wrong Predictions')
    axs[2].set_xlabel('Steps')
    axs[2].set_ylabel('Prediction')
    
    # Save figure
    plt.savefig(f"{dir_name}/episode_traffic_types.png")
    plt.close()         

def plot_agent_test(test, dir_name, title=''):
    """plot agent test results

    Args:
        test (_type_): _description_
        dir_name (_type_): _description_
        title (str, optional): _description_. Defaults to ''.
    """
    # Extracting indicators to plot
    episodes = range(1,len(test["ground_truth"])+1)
    ground_truth = [item[1] for item in test["ground_truth"]]
    plt.plot(episodes, ground_truth, label='Ground truth', color='purple')
    
    i=1
    for name, items in test["predicted"].items():
        values = [item[1]-i for item in items]
        plt.plot(episodes, values, label=name)
        i+=1
        
    plt.title(f'{title} prediction')
    plt.xlabel('Test Episodes')
    plt.legend()

    # Save figure
    plt.savefig(f"{dir_name}/test_episodes.png")
    plt.close() 
    

def plot_agent_test_errors(test, dir_name, title=''):
    """
    Plot ground truth and quadratic errors of predictions from different agents.

    Args:
        test (dict): Dictionary with 'ground_truth' and 'predicted' keys.
        dir_name (str): Directory to save the plot.
        title (str, optional): Plot title. Defaults to ''.
    """
    ground_truth = np.array([item[1] for item in test["ground_truth"]])
    episodes = np.arange(1, len(ground_truth) + 1)

    num_agents = len(test["predicted"])
    fig, axs = plt.subplots(num_agents, 2, figsize=(14, 4 * num_agents))

    if num_agents == 1:
        axs = np.array([axs])  # Ensure 2D array if only one agent

    for idx, (agent_name, predictions) in enumerate(test["predicted"].items()):
        pred_values = np.array([item[1] for item in predictions])
        errors = (pred_values - ground_truth) ** 2

        # Plot ground truth vs prediction
        axs[idx, 0].plot(episodes, ground_truth, label='Ground Truth', color='black')
        axs[idx, 0].plot(episodes, pred_values, label=f'{agent_name} Prediction', linestyle='--')
        axs[idx, 0].set_title(f'{agent_name} - Predictions vs Ground Truth')
        axs[idx, 0].set_xlabel('Episode')
        axs[idx, 0].set_ylabel('Class')
        axs[idx, 0].legend()
        axs[idx, 0].grid(True)

        # Plot quadratic error
        axs[idx, 1].plot(episodes, errors, label='Quadratic Error', color='red')
        axs[idx, 1].set_title(f'{agent_name} - Squared Error')
        axs[idx, 1].set_xlabel('Episode')
        axs[idx, 1].set_ylabel('Error')
        axs[idx, 1].legend()
        axs[idx, 1].grid(True)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    plt.savefig(f"{dir_name}/test_quadratic_errors.png")
    plt.close()


def plot_enviroment_execution_statutes(statutes, dir_name, title=''):
    """Plot bytes, packets and % variations

    Args:
        statutes (_type_): _description_
        dir_name (_type_): _description_
        title (str, optional): _description_. Defaults to ''.
    """
    # Extracting indicators to plot
    episodes = range(1,len(statutes)+1)
    packets = [item['packets'] for item in statutes]
    packets_varation = [item['variation_packet'] for item in statutes]
    bytes = [item['bytes'] for item in statutes]
    bytes_varation = [item['variation_byte'] for item in statutes]
    ids = [item['id'] for item in statutes] #*100 to view the line

    # Create subplots for different indicators
    x=10+3*int(len(statutes)/200)
    y=15+int(len(statutes)/200)
    fig, axs = plt.subplots(4, 1, figsize=(x, y))
    
    axs[0].set_yscale("log")
    axs[0].plot(episodes, packets, label='Packets Transmitted', color='blue')
    axs[0].set_title(f'{title} Packets Count')
    axs[0].legend()
 
    axs[1].set_yscale("log")
    axs[1].plot(episodes, bytes, label='Bytes Transmitted', color='black')    
    axs[1].set_title(f'{title} Bytes Count')
    axs[1].legend()

    axs[2].set_yscale("log")
    axs[2].plot(episodes, packets_varation, label='% var packet', color='purple')
    axs[2].plot(episodes, bytes_varation, label='% var bytes', color='cyan')
    axs[2].set_title(f'{title} % Variations')
    axs[2].legend()
        
    colors = get_colors_for_types(ids)    
    axs[3].scatter(episodes, ids, label='Type', c=colors, s=3)
    axs[3].set_title(f'{title} Event Types')
    axs[3].set_xlabel('Steps')
    axs[3].set_ylabel('Type')      
    legend_patches_types = get_legend_labels_for_types()
    axs[3].legend(handles=legend_patches_types, loc='upper right')  

    # Save figure
    plt.savefig(f"{dir_name}/statuses.png")
    try:
        plt.show()
    except:
        error("Show is not available")
    plt.close()

def get_legend_labels_for_types():
    return [
        mpatches.Patch(color='purple', label='5 - Trail Attack'),
        mpatches.Patch(color='brown', label='4 - Both Attacks'),
        mpatches.Patch(color='red', label='3 - Long Attack'),
        mpatches.Patch(color='orange', label='2 - Short Attack'),
        mpatches.Patch(color='yellow', label='1 - Init Attack'),
        mpatches.Patch(color='cyan', label='0 - Normal'),
        mpatches.Patch(color='gray', label='-1 - Idle')
    ]  
def get_legend_labels_for_predictions():   
    return [
        mpatches.Patch(color='green', label='Correct'),
        mpatches.Patch(color='red', label='Wrong')
    ]  
   
def get_colors_for_types(items):
    '''
    Assign colors comparing the values of 'types'
    '''
    colors = []
    for value in items:
        if value == 5:
            colors.append('purple')  # purple for 5  trail_attack
        elif value == 4:
            colors.append('brown')  # Brown values greater than 4 both
        elif value == 3:
            colors.append('red')  # Red values greater than 3 long attack
        elif value == 2:
            colors.append('orange')  # Red values greater than 2 short attack
        elif value == 1:
            colors.append('yellow')  # Yellow for 1 init_attack
        elif value == 0:
            colors.append('cyan')  # blue for 0 normal
        else:
            colors.append('gray')  # Gray for idle -1
    return colors

def get_colors_for_predictions(items):
    '''
    Assign colors comparing the values of 'prediction'
    '''
    colors = []
    for value in items:
        if value == 1:
            colors.append('green')  # correct
        else:
            colors.append('red')  # wrong
    return colors
 
def plot_test_confusion_matrix(dir_name,ground_truth, predicted, agent):
    try:
        confusion_matrix = metrics.confusion_matrix(ground_truth, predicted)
        if isinstance(ground_truth[0], int):
            display_labels = None
        else:
            display_labels = [0, 1]
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = display_labels)
        cm_display.plot()
        plt.savefig(f"{dir_name}/{agent}_matrix.png")
        plt.close()  
    except Exception as e: 
        error(Fore.RED+f"Error!\n{e}\{traceback.format_exc()}\n"+Fore.WHITE)

#TODO: remove this function, not usefull, because now we have plot_cumulative_reward and 
# the status traffic is the same for all agents
def plot_indicators(indicators, dir_name, title=''):
    # Extracting indicators to plot
    episodes = [item['episode'] for item in indicators]
    steps = [item['steps'] for item in indicators]
    packets_received = [item['packets_received'] for item in indicators]
    bytes_received = [item['bytes_received'] for item in indicators]
    packets_transmitted = [item['packets_transmitted'] for item in indicators]
    packets_varation = [item['packets_varation'] for item in indicators]
    bytes_transmitted = [item['bytes_transmitted'] for item in indicators]
    bytes_varation = [item['bytes_varation'] for item in indicators]
    reward = [item['cumulative_reward'] for item in indicators]

    # Create subplots for different indicators
    plt.yscale("log")
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    if len(packets_received)>0:
        axs[0].plot(episodes, packets_received, label='Packets Received', color='blue')
    elif len(packets_received)>0:
        axs[0].plot(episodes, packets_varation, label='Packets % Variation', color='blue')
    axs[0].plot(episodes, packets_transmitted, label='Packets Transmitted', color='green')
    axs[0].set_title(f'{title} Packets Count')
    axs[0].legend()

    if len(packets_received)>0:
        axs[1].plot(episodes, bytes_received, label='Bytes Received', color='red')
    elif len(packets_received)>0:
        axs[1].plot(episodes, bytes_varation, label='Bytes % Variation', color='red')    
    axs[1].plot(episodes, bytes_transmitted, label='Bytes Transmitted', color='black')    
    axs[1].set_title(f'{title} Bytes Count')
    axs[1].legend()

    axs[2].plot(episodes, reward, label='Rewards', color='purple')
    axs[2].plot(episodes, steps, label='Steps', color='cyan')
    axs[2].set_title(f'{title} Cumlative Rewards')
    axs[2].legend()

    # Save figure
    plt.savefig(f"{dir_name}/indicators.png")
    plt.close()
    


def plot_metrics(metrics, dir_name, title=''):
    plt.figure(figsize=(12, 6))
    
    # Accuracy
    plt.subplot(2, 2, 1)
    plt.plot(metrics['accuracy'], label='Accuracy', color='blue')
    plt.xlabel('Episodes')
    plt.ylabel('Accuracy')
    plt.title(f'{title} Accuracy per Episode')
    
    # Precision
    plt.subplot(2, 2, 2)
    plt.plot(metrics['precision'], label='Precision', color='green')
    plt.xlabel('Episodes')
    plt.ylabel('Precision')
    plt.title(f'{title} Precision per Episode')
    
    # Recall
    plt.subplot(2, 2, 3)
    plt.plot(metrics['recall'], label='Recall', color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Recall')
    plt.title(f'{title} Recall per Episode')
    
    # F1-Score
    plt.subplot(2, 2, 4)
    plt.plot(metrics['f1_score'], label='F1 Score', color='purple')
    plt.xlabel('Episodes')
    plt.ylabel('F1-Score')
    plt.title(f'{title} F1-Score per Episode')
    
    plt.tight_layout()
    #plt.show()
    # Save figure
    plt.savefig(f"{dir_name}/metrics.png")
    plt.close()
    
def plot_combined_performance_over_time(
    metrics, dir_name,
    title="Model Performance Over Time",
    xlabel="Episodes",
    ylabel="Metric Value"
):
    """
    Plots accuracy, recall, precision, and f-score on a single line chart
    over training episodes or time steps.

    Args:
        episodes (list or np.array): A list or array representing the x-axis (e.g., episode numbers).
        accuracy_scores (list or np.array): List of accuracy scores corresponding to each episode.
        recall_scores (list or np.array): List of recall scores.
        precision_scores (list or np.array): List of precision scores.
        fscore_scores (list or np.array): List of f-score scores.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
   # Extract data for plotting    
    episodes = np.arange(1, len(metrics["accuracy"]) + 1)
    recall_scores = metrics["recall"]
    accuracy_scores = metrics["accuracy"]
    precision_scores = metrics["precision"]
    fscore_scores = metrics["f1_score"]    
    
    plt.figure(figsize=(12, 7))
    plt.plot(episodes, recall_scores, label='Recall', marker='x', linestyle='--')
    plt.plot(episodes, accuracy_scores, label='Accuracy', marker='o', linestyle='-')
    plt.plot(episodes, precision_scores, label='Precision', marker='s', linestyle='-.')
    plt.plot(episodes, fscore_scores, label='F-Score', marker='d', linestyle=':')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.ylim(0, 1.05) # Metrics are usually between 0 and 1
    plt.tight_layout()
    plt.savefig(f"{dir_name}/metrics_combined.png")
    plt.close()   
    
def plot_radar_chart(dir_name,
    scenario_data,
    metrics=['accuracy', 'recall', 'precision', 'f1_score'],
    title="Performance Profile Across Learning Scenarios"
):
    """
    Creates a radar chart to visualize the performance profile of metrics across different scenarios.

    Args:
        scenario_data (dict): A dictionary where keys are scenario names (e.g., 'Low Traffic', 'High DDoS')
                              and values are dictionaries containing metric scores.
                              Example:
                              {
                                  'Scenario A': {'Accuracy': 0.7, 'Recall': 0.8, 'Precision': 0.75, 'F-Score': 0.78},
                                  'Scenario B': {'Accuracy': 0.9, 'Recall': 0.85, 'Precision': 0.92, 'F-Score': 0.88}
                              }
        metrics (list): List of metric names to include in the radar chart.
        title (str): Title of the plot.
    """
    num_metrics = len(metrics)
    # Create angles for each axis in the radar chart
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1] # Complete the circle

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for scenario_name, scores in scenario_data.items():
        # Get scores in the order of 'metrics' list
        values = [np.mean(scores[metric]) for metric in metrics]
        values += values[:1] # Complete the circle

        ax.plot(angles, values, linewidth=1, linestyle='solid', label=scenario_name)
        ax.fill(angles, values, alpha=0.25)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.0) # Metrics are usually between 0 and 1
    ax.set_title(title, va='bottom')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(f"{dir_name}/radar_chart.png")   
    plt.close() 
    
def plot_comparison_bar_charts(dir_name,
    strategies_data,
    metrics=['accuracy', 'recall', 'precision', 'f1_score'],
    title="Comparison of RL Strategies Learning Performance"
):
    """
    Creates grouped bar charts to compare performance metrics across different strategies/models.

    Args:
        strategies_data (dict): A dictionary where keys are strategy names (e.g., 'DQN', 'PPO')
                                and values are dictionaries containing metric scores.
                                Example:
                                {
                                    'DQN': {'Accuracy': 0.85, 'Recall': 0.78, 'Precision': 0.88, 'F-Score': 0.83},
                                    'PPO': {'Accuracy': 0.90, 'Recall': 0.85, 'Precision': 0.92, 'F-Score': 0.88}
                                }
        metrics (list): List of metric names to plot.
        title (str): Title of the plot.
    """
    strategy_names = list(strategies_data.keys())
    num_strategies = len(strategy_names)
    num_metrics = len(metrics)

    bar_width = 0.2
    index = np.arange(num_strategies)

    plt.figure(figsize=(12, 7))

    for i, metric_name in enumerate(metrics):
        # Extract scores for the current metric across all strategies
        scores = [np.mean(strategies_data[strategy][metric_name]) for strategy in strategy_names]
        # Calculate offset for grouped bars
        offset = bar_width * i - (num_metrics - 1) * bar_width / 2
        plt.bar(index + offset, scores, bar_width, label=metric_name)

    plt.title(title)
    plt.xlabel("RL Strategy")
    plt.ylabel("Metric Value")
    plt.xticks(index, strategy_names) # Set x-axis ticks to strategy names
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{dir_name}/metrics_comparison.png")
    plt.close()    
    
#normalized_segments = normalize_individual_segments(segments)     
def normalize_individual_segments(arrays):
    # Normalize each segment individually between 0 and 100
    normalized_segments = []
    for array in arrays:
        if len(array) > 0:
            min_val = np.min(array) 
            max_val = np.max(array)         
            if max_val > min_val:
                normalized_segment = 100 * (array - min_val) / (max_val - min_val)
            else:
                normalized_segment = array  # No normalization if min == max
            normalized_segments.append(normalized_segment)
    # Concatenate normalized segments
    return np.concatenate(normalized_segments)

def filter_labels_segments_empty(pr_normalized, labels):
    filtered_labels = []
    filtered_segments = []
    for segment, label in zip(pr_normalized, labels):
        if isinstance(segment, (np.ndarray, list)) and len(segment) > 0:
            filtered_labels.append(label)
            filtered_segments.append(segment)
        elif isinstance(segment, (np.float64, float, int)):  # Single numeric values
            filtered_labels.append(label)
            filtered_segments.append([segment])  # Treat as a single-element array
    return filtered_labels, filtered_segments

# def indicators_to_net_metrics(indicators):
#     # Define traffic types and initialize data_traffic dictionary
#     traffic_types = ['none', 'ping', 'udp', 'tcp']
#     data_traffic = {key: {'p_r': [], 'p_t': [], 'b_r': [], 'b_t': []} for key in traffic_types}

#     # Iterate through the data list and populate data_traffic
#     for entry in indicators:
#         traffic_type_index = entry["traffic_type"]
#         traffic_key = traffic_types[traffic_type_index]

#         data_traffic[traffic_key]['p_r'].append(entry["packets_received"])
#         data_traffic[traffic_key]['p_t'].append(entry["packets_transmitted"])
#         data_traffic[traffic_key]['b_r'].append(entry["bytes_received"])
#         data_traffic[traffic_key]['b_t'].append(entry["bytes_transmitted"])
#     return data_traffic

def plot_net_metrics(data, dir_name, title=''):
    plt.figure(figsize=(12, 6))

    # Colors and labels for each segment
    colors = ['blue', 'red', 'purple', 'green']
    labels = ['none', 'ping', 'udp', 'tcp']

    # Plotting each metric without normalization
    for i, metric in enumerate(['p_r', 'p_t', 'b_r', 'b_t'], 1):
        plt.subplot(2, 2, i)
        
        # Retrieve data for each label
        segments = [data[label][metric] for label in labels]

        # Plot each non-empty segment sequentially
        offset = 0  # Initialize offset for sequential plotting
        for idx, (segment, label) in enumerate(zip(segments, labels)):
            if len(segment) > 0:  # Only plot non-empty segments
                plt.plot(range(offset, offset + len(segment)), segment, color=colors[idx], label=f'{label} {metric}')
                offset += len(segment)  # Update offset for the next segment

        plt.xlabel('Episodes')
        plt.yscale("log")
        plt.ylabel(metric)
        plt.title(f'{title} {metric.capitalize()} per Episode')

    plt.tight_layout()
    plt.legend(loc='lower right')

    # Save figure
    plt.savefig(f"{dir_name}/net_metrics.png")
    plt.close()
    
def plot_train_types(train_types, training_execution_time, dir_name):
    plt.figure(figsize=(12, 6))
    plt.plot(train_types['explorations'], label='Exploration', color='blue')
    plt.plot(train_types['exploitations'], label='Exploitation', color='green')
    plt.plot(train_types['steps'], label='Step', color='red')
    plt.title(f'Train Explorations, Exploitations and Steps per Episode:  time {training_execution_time:.2f} s') 
    plt.tight_layout()
    plt.legend()
    #plt.show()
    # Save figure
    plt.savefig(f"{dir_name}/train_types.png")
    plt.close()    
    
if __name__ == '__main__':

    # Assuming `net` is Mininet object
    net = None  # Replace with actual Mininet network object
    # stats = collect_mystatistics(net)
    # plot_mystatistics(stats)
    # lines=[(3,9,3),(8,6,4),(8,8,2),(5,3,1),(3,9,5),(8,6,7),(8,8.8),(5,3,9)]
    # plot_all_lines_and_save_png(lines, "title", ['a', 'b','c'],"prova")
    
    config,config_dict = read_config_file('config.yaml')
    #test plot traffic from indicators previously recorded
    #indicators = read_data_file("../traffic")
    last_dir = find_latest_execution(config.training_directory)
    store = read_all_data_from_execution(last_dir)
    
    #plot_net_metrics(store, '..', '')
    name = store[0]["folder"]
    #data = js.loads(js.dumps(store[1]["data"]), object_hook=Params)
    data = store[0]["data"]
    
    plot_agent_test(data,config.training_directory)
    information("plotting training statistics\n",name)
    #if len(data["train_indicators"])>2:
        #plot_cumulative_rewards(data.train_indicators, config.training_directory, "")
        #plot_episode_statuses(data["train_indicators"], config.training_directory, "Episode statuses")
    # plot_metrics(data.train_metrics,config.training_directory,"Train metrics")
    # if len(data["train_types"]["explorations"]) > 0 and len(data["train_types"]["exploitations"]) > 0: 
    #     plot_train_types(data["train_types"], data["train_excution_time"], config.training_directory)
    information("plotting training metrics\n",name)
    
    
    
    #test plot traffic from indicators previously recorded
    # indicators = read_data_file("../traffic")
    # store = indicators_to_net_metrics(indicators)
    # plot_net_metrics(store, '..', '')