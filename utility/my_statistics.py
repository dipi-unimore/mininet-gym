# my_statistics.py
import matplotlib
matplotlib.use('Agg')  # Used for a non-GUI backend (Agg is for PNG output)
import matplotlib.pyplot as plt, matplotlib.patches as mpatches, numpy as np, seaborn as sns,  pandas as pd
from sklearn import metrics


#TODO copy new from colab
def plot_bar_chart_scores(scores, dir_name, title=''):
    """
    Plots a vertical bar chart of agent scores.
    
    Args:
        scores (dict): A dictionary where keys are agent names and values are their scores.
    """
    df_scores = pd.DataFrame(scores.items(), columns=['Agent', 'Score'])
    df_scores = df_scores.sort_values(by='Score', ascending=False)
    
    plt.figure(figsize=(15, 8))
    ax = sns.barplot(x='Agent', y='Score', data=df_scores, palette='viridis', hue='Agent', legend=False)
    
    # Add labels on top of the bars
    ax.bar_label(ax.containers[0], fmt='%.0f', padding=3)
    
    ax.set_xlabel('Agent', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    plt.title('Agent Performance Scores', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{dir_name}/scores.png")
    plt.close()
    
def plot_horizontal_bar_chart_test_data(scores, dir_name, title=''):
    """
    Plots a horizontal bar chart of agent scores from test data.
    
    Args:
        scores (dict): A dictionary where keys are agent names and values are their scores.
    """
    df_scores = pd.DataFrame(scores.items(), columns=['Agent', 'Score'])
    df_scores = df_scores.sort_values(by='Score', ascending=True)
    
    plt.figure(figsize=(15, 8))
    ax = sns.barplot(x='Score', y='Agent', data=df_scores, palette='viridis', hue='Agent', legend=False)
    
    # Add labels to the bars
    for index, row in df_scores.iterrows():
        ax.text(row.Score + 1, index, f'{row.Score:.0f}', va='center')
    # for container in ax.containers:
    #     ax.bar_label(container)
        
    plt.title('Agent Performance Scores (Horizontal)', fontsize=16)
    plt.xlabel('Score', fontsize=12)
    plt.ylabel('Agent', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{dir_name}/scores_horizontal.png")
    plt.close()  

# Parse the agent names to get the algorithm and variant
def parse_agent_name(name):
    parts = name.split('_')
    algorithm = parts[0]
    speed = '_'.join(parts[1:]) if len(parts) > 1 else 'N/A'
    return algorithm, speed

def plot_grouped_by_algo_score_bar_chart(scores, dir_name, title=''):
    """
    Plots a bar chart of scores for grouped by algorithm.
    
    Args:
        scores (dict): A dictionary of agent scores.
    """
    df_scores = pd.DataFrame(scores.items(), columns=['Agent', 'Score'])
    df_scores = df_scores.sort_values(by='Score', ascending=True)
    
    df_scores[['Algorithm', 'Speed']] = df_scores['Agent'].apply(
        lambda x: pd.Series(parse_agent_name(x))
    )

    # Plot the grouped bar chart
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(
        x='Algorithm',
        y='Score',
        hue='Speed',
        data=df_scores,
        palette='pastel'
    )

    # Add labels on top of the bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', padding=3)

    ax.set_title('Agent Scores Grouped by Algorithm', fontsize=16)
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(215, 300) # Set a lower y-limit for better visibility
    plt.tight_layout()
    plt.savefig(f"{dir_name}/scores_grouped_by_algo.png")
    plt.close()  
    
def plot_total_score_bar_chart(scores, dir_name, title=''):
    """
    Plots a bar chart of the total score.
    
    Args:
        scores (dict): A dictionary of agent scores.
    """
    total_score = sum(scores.values())
    
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=['Total Score'], y=[total_score], palette='viridis', hue=['Total Score'], legend=False)
    
    # Add labels on the bar
    ax.bar_label(ax.containers[0], fmt='%.0f', padding=3)

    plt.title('Total Score Across All Agents', fontsize=16)
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('')
    plt.tight_layout()
    plt.savefig(f"{dir_name}/scores_total.png")
    plt.close()    

def plot_score_metrics_bar_chart(scores, metrics, dir_name, title=''):
    """
    Plots a bar chart of scores for grouped by algorithm.
    
    Args:
        scores (dict): A dictionary of agent scores.
        metrics (dict): A dictionary of agent metrics.
    """
    df_scores = pd.DataFrame(scores.items(), columns=['Agent', 'Score'])
    df_scores = df_scores.sort_values(by='Score', ascending=True)
    # Get the ordered list of agents for the y-axis labels
    agent_order = df_scores['Agent'].tolist()
    df_metrics = pd.DataFrame(metrics.items(), columns=['Agent', 'Metrics'])

    # Convert the dictionary to a DataFrame suitable for plotting
    # The data is in a "wide" format, so we need to "melt" it to a "long" format for seaborn
    df_metrics = pd.DataFrame(metrics).T.reset_index()
    df_metrics = df_metrics.rename(columns={'index': 'Agent'})
    df_long = pd.melt(df_metrics, id_vars=['Agent'], var_name='Metric', value_name='Value')

    # Merge the metrics data with the scores data
    df_merged = pd.merge(df_long, df_scores, on='Agent')

    # Sort the merged DataFrame by the scores in descending order
    df_merged = df_merged.sort_values(by='Score', ascending=False)

    # Create a list of formatted labels for the y-axis
    y_labels = []
    for agent in agent_order:
        score = df_scores[df_scores['Agent'] == agent]['Score'].iloc[0]

        # Format the agent name (e.g., A2C_quick -> A2C quick)
        formatted_name = agent.replace('_', ' ')

        # Create the two-line label string
        y_labels.append(f'{formatted_name}\nscore: {score}')

        # Create the grouped bar chart, ordering agents by score
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x='Value', y='Agent', hue='Metric', data=df_merged, order=agent_order, palette='Paired')

    # Set the x-axis limits
    ax.set_xlim(0.65, 1.0)

    # Apply the custom y-axis labels
    ax.set_yticklabels(y_labels)

    # Add percentage labels at the end of each bar
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=5, labels=[f'{val:.1%}' for val in container.datavalues], fontsize=8)

    # Customize the plot
    plt.title('Agent Performance Metrics (Ordered by Score)')
    plt.xlabel('Value')
    plt.ylabel('Agent & Score')
    ax.legend(title='Metric', loc='lower right')
    plt.tight_layout()
    plt.savefig(f"{dir_name}/metrics_and_scores.png")
    plt.close()  


def plot_predictions_over_time(data, agents_to_plot, dir_name, title=''):
    """
    Plots agent predictions against ground truth over time.
    
    Args:
        data (dict): The complete evaluation data.
        agents_to_plot (list): A list of agent names to include in the plot.
    """
    df_predictions = pd.DataFrame(data['predicted']).melt(
        var_name='Agent', value_name='Prediction'
    )
    df_predictions['Time Step'] = df_predictions.index
    
    df_ground_truth = pd.DataFrame({'Time Step': np.arange(len(data['ground_truth'])),
                                    'Ground Truth': data['ground_truth']})

    plt.figure(figsize=(15, 8))
    
    # Create the prediction line plot for the selected agents
    for agent in agents_to_plot:
        sns.lineplot(
            x='Time Step',
            y='Prediction',
            data=df_predictions[df_predictions['Agent'] == agent],
            label=f'{agent} Prediction',
            linestyle='--'
        )
    
    # Create the ground truth line plot
    sns.lineplot(
        x='Time Step',
        y='Ground Truth',
        data=df_ground_truth,
        color='black',
        linewidth=2,
        label='Ground Truth'
    )

    # Find where the ground truth changes and add vertical lines
    ground_truth_array = np.array(data['ground_truth'])
    change_points = np.where(ground_truth_array[:-1] != ground_truth_array[1:])[0]
    
    for cp in change_points:
        plt.axvline(x=cp + 1, color='red', linestyle='--', linewidth=1)

    plt.title('Agent Predictions vs. Ground Truth Over Time', fontsize=16)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Prediction (0=Normal, 1=Attack)', fontsize=12)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_fn_or_fp_linear_chart(data, is_fn=True):
    """
    Plots a linear chart for False Negatives or False Positives for the top 5 agents.
    
    Args:
        data (dict): The complete evaluation data with 'predicted' and 'ground_truth'.
        is_fn (bool): If True, plots False Negatives. If False, plots False Positives.
    """
    metric_type = 'FN' if is_fn else 'FP'
    
    # Calculate FP/FN for each agent
    agent_metrics = {}
    ground_truth = np.array(data['ground_truth'])
    
    for agent_name, predictions in data['predicted'].items():
        predictions = np.array(predictions)
        
        # Calculate True Positive (TP), False Positive (FP), False Negative (FN), and True Negative (TN)
        tp = np.sum((predictions == 1) & (ground_truth == 1))
        fp = np.sum((predictions == 1) & (ground_truth == 0))
        fn = np.sum((predictions == 0) & (ground_truth == 1))
        tn = np.sum((predictions == 0) & (ground_truth == 0))
        
        if is_fn:
            agent_metrics[agent_name] = fn
        else:
            agent_metrics[agent_name] = fp

    # Sort agents and select the top 5
    sorted_agents = sorted(agent_metrics.items(), key=lambda item: item[1], reverse=True)
    top_5_agents = {agent[0]: agent[1] for agent in sorted_agents[:5]}

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    
    # Define colors for the agents
    agent_colors = plt.cm.get_cmap('viridis', len(top_5_agents))
    
    # Plot bars for the top 5 agents
    sns.barplot(x=list(top_5_agents.keys()), y=list(top_5_agents.values()), palette=agent_colors, ax=ax)
    
    # Add labels on the bars
    for i, (agent, metric_value) in enumerate(top_5_agents.items()):
        ax.text(i, metric_value, f'{metric_value}', ha='center', va='bottom', fontsize=10)
    
    plt.title(f'Top 5 Agents with Most {metric_type}s', fontsize=16)
    plt.xlabel('Agent', fontsize=12)
    plt.ylabel(f'Number of {metric_type}s', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()



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

#END new from colab

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

def plot_agent_execution_confusion_matrix(indicators, dir_name, must_print = True, title=''):
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
        if must_print:
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
            cm_display.plot()
            plt.savefig(f"{dir_name}/matrix.png")
            plt.close()
        return confusion_matrix
    except Exception as e: 
        error(Fore.RED+f"Error!\n{e}: {traceback.format_exc()}\n"+Fore.WHITE)
    
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
    predictions = [item['action_choosen'] for item in all_steps_status]
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
    #TODO adapt to global_state
    packets_varation = [item['packetsPercentageChange'] for item in statutes]
    bytes = [item['bytes'] for item in statutes]
    bytes_varation = [item['packetsPercentageChange'] for item in statutes]
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
    """
    print confusion matrix
    Args:
        dir_name (_type_): _description_
        ground_truth (_type_): _description_
        predicted (_type_): _description_
        agent (_type_): _description_
    """
    confusion_matrix = metrics.confusion_matrix(ground_truth, predicted)
    if isinstance(ground_truth[0], int):
        display_labels = None
    else:
        display_labels = [0, 1]
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = display_labels)
    cm_display.plot()
    plt.savefig(f"{dir_name}/{agent}_matrix.png")
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
  

from utility.my_files import find_latest_execution, read_all_data_from_execution
from utility.params import  read_config_file
from utility.my_log import information, error
import traceback
from colorama import Fore
    
if __name__ == '__main__':

    # Assuming `net` is Mininet object
    net = None  # Replace with actual Mininet network object
    # stats = collect_mystatistics(net)
    # plot_mystatistics(stats)
    # lines=[(3,9,3),(8,6,4),(8,8,2),(5,3,1),(3,9,5),(8,6,7),(8,8.8),(5,3,9)]
    
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