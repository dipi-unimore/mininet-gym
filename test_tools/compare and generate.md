# Comparative Graphs and Tables Generation (Paper 8)

This documentation describes the workflow for generating the comparative graphs and ranking tables used in the analysis of experiments for "Paper 8" (working draft on MARL).

The scripts analyze the results of experiment groups, defined in YAML configuration files, to produce standardized outputs (images and Markdown tables) that facilitate comparison between different configurations.

## Analysis Scripts

### `generate_confusion_plots.py`
This script reads a YAML configuration file and generates a grid image containing the confusion matrices for each listed experiment.

**Purpose:** It allows for a visual comparison of the agents' classification performance. It is particularly useful for observing how specific errors (e.g., confusion between `ATTACK_IN` and `ATTACK_OUT`) change with different configurations (e.g., with or without `reward_scaling`).

### `generate_ranking_table.py`
This script, starting from the same YAML file, generates a ranking table in Markdown format.

**Purpose:** It provides a quantitative and synthetic evaluation of performance. The table ranks the different experimental configurations based on a composite "Winner Score" and displays the main metrics (Accuracy, Precision, Recall, F1), allowing for the quick identification of the best configuration for a given test group.

## Workflow and Commands

The process for generating the analysis is as follows:

1.  **Virtual Environment Setup:** Create a virtual Python environment and install the necessary dependencies for data analysis and visualization.
2.  **Script Execution:** Run the two generation scripts for each YAML configuration file that defines a group of experiments.

The commands below execute the entire process, generating all the necessary figures and tables.

```bash
# 1. Setup the virtual environment (to be run only once)
python3 -m venv .venv-confusion
source .venv-confusion/bin/activate
python -m pip install --upgrade pip
pip install numpy pandas pyyaml matplotlib seaborn scikit-learn

# 2. Generation of graphs and tables for each group of experiments
# Group 1: Ablation study on reward scaling
python /home/finistrellas/mininet-gym/test_tools/generate_confusion_plots.py /home/finistrellas/mininet-gym/test_tools/compare_config_reward_scaling.yaml
python /home/finistrellas/mininet-gym/test_tools/generate_ranking_table.py /home/finistrellas/mininet-gym/test_tools/compare_config_reward_scaling.yaml

# Group 2: Ablation study on variation features (deltas)
python /home/finistrellas/mininet-gym/test_tools/generate_confusion_plots.py /home/finistrellas/mininet-gym/test_tools/compare_config_deltas.yaml
python /home/finistrellas/mininet-gym/test_tools/generate_ranking_table.py /home/finistrellas/mininet-gym/test_tools/compare_config_deltas.yaml

# Group 3: General comparison with all flags enabled/disabled
python /home/finistrellas/mininet-gym/test_tools/generate_confusion_plots.py /home/finistrellas/mininet-gym/test_tools/compare_config_all_flags.yaml
python /home/finistrellas/mininet-gym/test_tools/generate_ranking_table.py /home/finistrellas/mininet-gym/test_tools/compare_config_all_flags.yaml

# Group 4: Comparison on the impact of resources (net_arch, n_bins)
python /home/finistrellas/mininet-gym/test_tools/generate_confusion_plots.py /home/finistrellas/mininet-gym/test_tools/compare_config_group3_resources.yaml
python /home/finistrellas/mininet-gym/test_tools/generate_ranking_table.py /home/finistrellas/mininet-gym/test_tools/compare_config_group3_resources.yaml
```