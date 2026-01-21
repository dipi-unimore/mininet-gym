# my_statistics.py
import matplotlib

from utility.constants import SystemModes
matplotlib.use('Agg')  # Used for a non-GUI backend (Agg is for PNG output)
import matplotlib.pyplot as plt, matplotlib.patches as mpatches, numpy as np, seaborn as sns,  pandas as pd
from sklearn import metrics

#TODO add new functions from colab adapeted to marl

#TODO evaluate to print the rewards as cumulative among all hosts as a team
def plot_agent_cumulative_rewards(indicators, dir_name, title='', host=''):
    """plot the agent cumulative reward for each episode
        
    Args:
        indicators (_type_): data to plot
        dir_name (_type_): directory to save the plot
        title (str, optional): chart title. Defaults to ''.
        host (str, optional): host name. Defaults to ''.
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
    plt.savefig(f"{dir_name}/rewards_{host}.png")
    plt.close()    



    
def plot_agent_execution_statuses(indicators, dir_name, title='', host=''):
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
    if host=='coordinator':
        packets = [item['packets'] for item in all_steps_status]
        bytes = [item['bytes'] for item in all_steps_status]        
    else:
        received_bytes = [item['received_bytes'].item() for item in all_steps_status]
        received_packets = [item['received_packets'].item() for item in all_steps_status]
        transmitted_bytes = [item['transmitted_bytes'].item() for item in all_steps_status]
        transmitted_packets = [item['transmitted_packets'].item() for item in all_steps_status]
    #message = [item['message'] for item in all_steps_status]
    predictions = [item['action_choosen'] for item in all_steps_status]
    types = [item['id'] for item in all_steps_status]
    
    # Create subplots for different indicators
    x=10+5*int(n_steps/200)
    y=10+int(n_steps/200)
    fig, axs = plt.subplots(2, 2, figsize=(x, y))

    if host=='coordinator':
        axs[0][0].set_yscale("log")
        axs[0][0].plot(steps, packets, label='Packets', color='purple')
        axs[0][0].set_title(f'{title} Packets')
        axs[0][0].set_xlabel('Steps')
        axs[0][0].set_ylabel('Log Scale Packet Traffic')    
        axs[0][0].legend()  
        
        axs[0][1].set_yscale("log")
        axs[0][1].plot(steps, bytes, label='Bytes', color='royalblue')
        axs[0][1].set_title(f'{title} Bytes')
        axs[0][1].set_xlabel('Steps')
        axs[0][1].set_ylabel('Log Scale Bytes Traffic')    
        axs[0][1].legend()        
    else:
        axs[0][0].set_yscale("log")
        axs[0][0].plot(steps, transmitted_packets, label='TX packets', color='purple')
        axs[0][0].plot(steps, received_packets, label='RX packets', color='cyan')
        axs[0][0].set_title(f'{title} TX-RX Packets')
        axs[0][0].set_xlabel('Steps')
        axs[0][0].set_ylabel('Log Scale Packet Traffic')    
        axs[0][0].legend()  
        
        axs[0][1].set_yscale("log")
        axs[0][1].plot(steps, transmitted_bytes, label='TX bytes', color='royalblue')
        axs[0][1].plot(steps, received_bytes, label='RX bytes', color='green')
        axs[0][1].set_title(f'{title} TX-RX Bytes')
        axs[0][1].set_xlabel('Steps')
        axs[0][1].set_ylabel('Log Scale Bytes Traffic')    
        axs[0][1].legend()   

    colors = get_colors_for_types(types)    
    legend_patches_types = get_legend_labels_for_types()
    axs[1][0].scatter(steps, types, label='Types', c=colors, s=3)
    axs[1][0].set_title(f'{title} - Event Types')
    axs[1][0].set_xlabel('Steps')
    axs[1][0].set_ylabel('Type')      
    axs[1][0].legend(handles=legend_patches_types, loc='lower left')     
    
    colors = get_colors_for_predictions(predictions) 
    legend_patches_predictions = get_legend_labels_for_predictions()
    axs[1][1].scatter(steps, predictions, label='Predictions', c=colors, s=3)
    axs[1][1].legend(handles=legend_patches_predictions, loc='center')
    axs[1][1].set_title(f'{title} - Correct/Wrong Predictions')
    axs[1][1].set_xlabel('Steps')
    axs[1][1].set_ylabel('Prediction')
    
    # Save figure
    plt.savefig(f"{dir_name}/episode_statuses_{host}.png")
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

def plot_agent_test(test, dir_name, host='', title=''):
    """plot agent test results

    Args:
        test (_type_): _description_
        dir_name (_type_): _description_
        title (str, optional): _description_. Defaults to ''.
    """
    # Extracting indicators to plot
    ground_truth = [item for item in test["ground_truth"][host]]
    episodes = range(1,len(ground_truth)+1)
    plt.plot(episodes, ground_truth, label='Ground truth', color='purple')
    
    i=1
    for name, items in test["predicted"].items():
        values = [item[host]-i for item in items]
        plt.plot(episodes, values, label=name)
        i+=1
        
    plt.title(f'{title} prediction')
    plt.xlabel(f'Test Episodes {host}')
    plt.legend()

    # Save figure
    plt.savefig(f"{dir_name}/test_episodes_{host}.png")
    plt.close() 
    

def plot_agent_test_errors(test, dir_name, host='', title=''):
    """
    Plot ground truth and quadratic errors of predictions from different agents.

    Args:
        test (dict): Dictionary with 'ground_truth' and 'predicted' keys.
        dir_name (str): Directory to save the plot.
        title (str, optional): Plot title. Defaults to ''.
    """
    ground_truth = np.array([item for item in test["ground_truth"][host]])
    episodes = np.arange(1, len(ground_truth) + 1)

    num_agents = len(test["predicted"])
    fig, axs = plt.subplots(num_agents, 2, figsize=(14, 4 * num_agents))

    if num_agents == 1:
        axs = np.array([axs])  # Ensure 2D array if only one agent

    for idx, (agent_name, predictions) in enumerate(test["predicted"].items()):
        pred_values = np.array([item[host] for item in predictions])
        errors = (pred_values - ground_truth) ** 2

        # Plot ground truth vs prediction
        axs[idx, 0].plot(episodes, ground_truth, label='Ground Truth', color='black')
        axs[idx, 0].plot(episodes, pred_values, label=f'{agent_name} {host} Prediction', linestyle='--')
        axs[idx, 0].set_title(f'{agent_name} - Predictions vs Ground Truth')
        axs[idx, 0].set_xlabel('Episode')
        axs[idx, 0].set_ylabel('Class')
        axs[idx, 0].legend()
        axs[idx, 0].grid(True)

        # Plot quadratic error
        axs[idx, 1].plot(episodes, errors, label='Quadratic Error', color='red')
        axs[idx, 1].set_title(f'{agent_name} {host} - Squared Error')
        axs[idx, 1].set_xlabel('Episode')
        axs[idx, 1].set_ylabel('Error')
        axs[idx, 1].legend()
        axs[idx, 1].grid(True)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    plt.savefig(f"{dir_name}/test_quadratic_errors_{host}.png")
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
    packets_varation = [item['packetsPercentageChange'] for item in statutes]
    bytes = [item['bytes'] for item in statutes]
    bytes_varation = [item['bytesPercentageChange'] for item in statutes]
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
    # try:
    #     plt.show()
    # except:
    #     error("Show is not available")
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

def plot_agent_training_confusion_matrix(indicators, dir_name, agent_name, must_print = True, host_name='', labels = ['Normal', 'Dos Attack']):
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
        # eg attack the prediction is 0 or 1 for Coordinator and 0.1.2 for hosts
        ground_truth = [item['id'] for item in all_steps_status]
        #ground_truth = [item['action_choosen'] if item['action_correct'] else 1 - item['action_choosen'] for item in all_steps_status]
    
    confusion_matrix = metrics.confusion_matrix(ground_truth, predicted, labels=range(len(labels)))
    if must_print:
        mode = SystemModes.TRAINING
        plot_confusion_matrix(dir_name, confusion_matrix, mode, agent_name, host_name, labels)
    return confusion_matrix
 
def plot_test_confusion_matrix(dir_name, ground_truth, predicted, agent_name, must_print = True, host_name='', labels = ['Normal', 'Dos Attack']):
    """
    plot confusion matrix for test results
    
    Args:
        dir_name (_type_): _description_
        ground_truth (_type_): _description_
        predicted (_type_): _description_
        agent_name (_type_): _description_
        host_name (str, optional): _description_. Defaults to ''.
    """

    y_true = np.argmax(np.array(ground_truth), axis=1)
    y_pred = np.argmax(np.array(predicted), axis=1)
    # Define the labels for the 2x2 matrix

    confusion_matrix = metrics.confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    if must_print:
        mode = SystemModes.EVALUATION
        plot_confusion_matrix(dir_name, confusion_matrix, mode, agent_name, host_name, labels)
    return confusion_matrix
    
def plot_confusion_matrix(dir_name, confusion_matrix, mode, agent_name, host_name='', labels = ['Normal', 'Dos Attack']):
    # Calculate percentages for each row (actual class)
    title_cm_font_size = 26
    label_cm_font_size = 26
    tick_cm_font_size = 26
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    percentage_matrix = (confusion_matrix / row_sums) * 100

    

    # Create the confusion matrix DataFrame
    confusion_matrix_df = pd.DataFrame(percentage_matrix, index=labels, columns=labels)

    # Create an annotations DataFrame with the '%' symbol
    annotations_df = confusion_matrix_df.map(lambda x: f'{x:.1f}%')

    # Create the plot
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(confusion_matrix_df, annot=annotations_df, fmt='', cmap='viridis', annot_kws={'fontsize': label_cm_font_size * 2},
                     linewidths=1, cbar=True)

    # Replace underscores with spaces in the agent name for a cleaner title
    #ag = agent_name.split(' ')[0] if 'score' not in agent_name else agent_name.replace(' in Evaluation', '').replace(' ', '_')
    cleaned_agent_name = agent_name.replace('_', ' ')

    plt.ylabel('Truth', fontsize=label_cm_font_size)
    plt.xlabel('Predicted', fontsize=label_cm_font_size)

    # Set the font size for the tick labels
    ax.tick_params(axis='x', labelsize=tick_cm_font_size)
    ax.tick_params(axis='y', labelsize=tick_cm_font_size)

    # Set the font size of the color bar labels
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=tick_cm_font_size)

    plt.tight_layout()

    plt.title(f'Confusion Matrix - {mode} - {cleaned_agent_name} {host_name}', fontsize=title_cm_font_size)
    plt.savefig(f"{dir_name}/{agent_name}_matrix_{host_name}.png")
    plt.close()  


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
    


def plot_metrics(metrics, dir_name, title='', host=''):
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
    plt.savefig(f"{dir_name}/metrics_{host}.png")
    plt.close()
    
def plot_combined_performance_over_time(
    metrics, dir_name,
    title="Model Performance Over Time",
    xlabel="Episodes",
    ylabel="Metric Value",
    host=''
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
   # TODO: verify that all metrics have the same length
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
    plt.savefig(f"{dir_name}/metrics_combined_{host}.png")
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
        dir_name (str): The directory where the chart image will be saved.
        strategies_data (dict): A dictionary where keys are strategy names (e.g., 'DQN', 'PPO')
                                and values are dictionaries containing metric scores.
                                Scores for each metric can be a single number or a list/array of numbers.
                                Example:
                                {
                                    'DQN': {'accuracy': [0.85, 0.86, ...], 'recall': [0.78, ...], ...},
                                    'PPO': {'accuracy': [0.90, 0.91, ...], 'recall': [0.85, ...], ...}
                                }
        metrics (list): List of metric names to plot.
        title (str): Title of the plot.
    """
    strategy_names = list(strategies_data.keys())
    num_strategies = len(strategy_names)
    num_metrics = len(metrics)

    # --- Calculate dynamic y-axis limits and ensure scores are scalar ---
    all_scalar_scores = [] # This will store the single aggregated score for each metric/strategy
    
    # First, process the data to ensure we have scalar values for plotting
    # and to collect all scores for dynamic y-axis calculation
    processed_strategies_data = {}
    for strategy in strategy_names:
        processed_strategies_data[strategy] = {}
        for metric_name in metrics:
            raw_score = strategies_data[strategy].get(metric_name)
            
            if raw_score is None:
                print(f"Warning: Metric '{metric_name}' not found for strategy '{strategy}'. Setting to 0 or NaN.")
                aggregated_score = 0 # Or np.nan if you prefer to exclude it from calculations
            elif isinstance(raw_score, (list, np.ndarray)):
                # If it's a list/array, take the mean (or median, max, etc.)
                if len(raw_score) > 0:
                    aggregated_score = np.mean(raw_score)
                else:
                    print(f"Warning: Empty list of scores for metric '{metric_name}' in strategy '{strategy}'. Setting to 0.")
                    aggregated_score = 0
            else:
                # If it's already a scalar, use it directly
                aggregated_score = raw_score
            
            processed_strategies_data[strategy][metric_name] = aggregated_score
            all_scalar_scores.append(aggregated_score)

    if not all_scalar_scores:
        print("Error: No valid scores found to plot after aggregation. Cannot create chart.")
        return

    min_score = np.min(all_scalar_scores)
    
    # Calculate the lower limit: minimum score minus 10% of the minimum score
    # Ensure it doesn't go below 0 if min_score is very small or zero
    y_lower_limit = max(0, min_score - (min_score * 0.10))
    y_upper_limit = 1.05 # Common upper limit for these metrics

    # --- Plotting logic ---
    bar_width = 0.2
    index = np.arange(num_strategies)

    plt.figure(figsize=(12, 7))

    for i, metric_name in enumerate(metrics):
        # Extract the already processed (scalar) scores for the current metric across all strategies
        scores = [processed_strategies_data[strategy][metric_name] for strategy in strategy_names]
        
        # Calculate offset for grouped bars
        offset = bar_width * i - (num_metrics - 1) * bar_width / 2
        plt.bar(index + offset, scores, bar_width, label=metric_name)

    plt.title(title)
    plt.xlabel("RL Strategy")
    plt.ylabel("Metric Value")
    plt.xticks(index + bar_width * (num_metrics - 1) / 2, strategy_names) # Adjust x-tick positions for better centering
    plt.ylim(y_lower_limit, y_upper_limit) # Apply the dynamic y-axis limits
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Ensure the directory exists before saving
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
    
def plot_train_types(train_types, training_execution_time, dir_name, title='', host=''):
    plt.figure(figsize=(12, 6))
    plt.plot(train_types['explorations'], label='Exploration', color='blue')
    plt.plot(train_types['exploitations'], label='Exploitation', color='green')
    plt.plot(train_types['steps'], label='Step', color='red')
    plt.title(f'{title}:  time {training_execution_time:.2f} s') 
    plt.tight_layout()
    plt.legend()
    #plt.show()
    # Save figure
    plt.savefig(f"{dir_name}/train_types_{host}.png")
    plt.close()   
    
    
from typing import Dict, List, Union

# Tipo di dati per le metriche di un singolo agente
AgentMetrics = Dict[str, List[float]]
# Tipo di dati per l'intero dataset (Agent Name -> Metrics)
Dataset = Dict[str, AgentMetrics]

def calculate_team_metrics(data: Dataset) -> AgentMetrics:
    """
    Evaluates and aggregates the performance metrics of a team of agents.
    Args:
        data: A dictionary where keys are agent names (e.g., 'h1', 'coordinator') 
              and values are their metrics (e.g., {'accuracy': [0.5, 0.5, 0.0], ...}).
    Returns:
        A dictionary containing the aggregated team metrics, where each metric's 
        values are averaged across all agents for each time step.
    """
    
    if not data:
        return {}

    # Extract a metric and its list to determine the number of observation points (time steps)
    first_agent_name = next(iter(data))
    first_metrics = data[first_agent_name]
    
    # Extract the names of the metrics (e.g., 'accuracy', 'precision')
    metric_names = list(first_metrics.keys())
    
    if not metric_names:
        return {}
        
    num_time_steps = len(first_metrics[metric_names[0]])

    summed_metrics: AgentMetrics = {metric: [0.0] * num_time_steps for metric in metric_names}
    
    num_agents = 0
    
    for agent_name, agent_data in data.items():
        if all(metric in agent_data for metric in metric_names):
            num_agents += 1
            
            for metric_name in metric_names:
                metric_values = agent_data[metric_name]
                
                for i in range(num_time_steps):
                    summed_metrics[metric_name][i] += metric_values[i]

    team_metrics: AgentMetrics = {}
    if num_agents > 0:
        for metric_name in metric_names:
            # Calculate the average for each time step
            team_metrics[metric_name] = [
                total_sum / num_agents
                for total_sum in summed_metrics[metric_name]
            ]
    
    return team_metrics
     
  

