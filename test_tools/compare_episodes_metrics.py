#!/usr/bin/env python3
"""
Compare metrics between experiments (e.g. 100 vs 300 episodes) and plot a grouped
bar chart with four metrics per algorithm: accuracy, precision (macro), recall (macro), F1 (macro).

Configuration is provided via a YAML file specifying the experiment folders to
compare and labels.

Example config (see test_tools/compare_config.yaml):
experiments:
  - label: "100_episodes"
    path: "_training/attacks_ho/20260806-213847_1_5_5"
  - label: "300_episodes"
    path: "_training/attacks_ho/20260802-235108_1_5_5"

The script will look for TEST*/test.json inside each experiment folder and use
its `ground_truth` and `predicted` arrays to compute metrics. If multiple
TEST_* folders exist, all agents inside will be included and matched by
agent name where possible.
"""

import argparse
import os
from pathlib import Path
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# --- font sizes (bumped for print legibility in the two-column paper) ---
FS_VALUE_LABEL = 17   # numeric value above each bar
FS_EXP_LABEL = 16     # per-experiment cluster label ("100 ep." / "300 ep.")
FS_TICK = 18          # x tick labels (algorithm names)
FS_YTICK = 15         # y tick labels
FS_AXIS_LABEL = 17    # x/y axis labels
FS_TITLE = 19         # plot title
FS_LEGEND = 14        # legend entries
FS_LEGEND_TITLE = 14  # legend section title (multi-row legend)


def _short_label(label: str) -> str:
    """'100_episodes' -> '100 ep.' for compact in-figure captions."""
    return label.replace('_', ' ').replace('episodes', 'ep.').replace('episode', 'ep.')


# --- helper metric functions (no sklearn dependency) ---

def accuracy_score(gt, pred):
    gt = np.array(gt, dtype=int)
    pred = np.array(pred, dtype=int)
    return float((pred == gt).sum()) / max(1, gt.size)


def precision_recall_f1_macro(gt, pred, n_classes=None):
    gt = np.array(gt, dtype=int)
    pred = np.array(pred, dtype=int)

    if n_classes is None:
        n_classes = int(max(gt.max() if gt.size>0 else 0, pred.max() if pred.size>0 else 0)) + 1

    precisions = []
    recalls = []
    f1s = []

    for c in range(n_classes):
        tp = int(((pred == c) & (gt == c)).sum())
        fp = int(((pred == c) & (gt != c)).sum())
        fn = int(((pred != c) & (gt == c)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s))


# --- load test results ---

def find_test_results(experiment_path):
    """Return a dict agent_name -> test_data loaded from TEST*/test.json"""
    results = {}
    path = Path(experiment_path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment path does not exist: {experiment_path}")

    for test_dir in path.iterdir():
        if not test_dir.is_dir():
            continue
        if not test_dir.name.startswith("TEST"):
            continue
        test_json = test_dir / "test.json"
        if not test_json.exists():
            continue
        try:
            with open(test_json, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: could not load {test_json}: {e}")
            continue

        # if TEST folder is plain TEST, try to infer agent name from data
        if test_dir.name == "TEST":
            # if predicted is a dict keyed by agent names, merge those
            if isinstance(data.get('predicted'), dict):
                for agent, preds in data['predicted'].items():
                    results[agent] = {
                        'ground_truth': data.get('ground_truth', []),
                        'predicted': preds
                    }
            else:
                # use a synthetic agent name if only one prediction vector exists
                results["TEST"] = data
        else:
            # TEST_<agentname>
            agent_name = test_dir.name[5:]
            results[agent_name] = data

    return results


# --- aggregate metrics per algorithm/agent ---

def get_episodes(experiment_path):
    """Read the training episode count from <experiment_path>/config.yaml, or None
    if unavailable. Looks for a top-level `episodes` key first (matches marl_pz's
    schema), falling back to a shallow one-level-nested search."""
    config_path = Path(experiment_path) / 'config.yaml'
    if not config_path.exists():
        return None
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
    except Exception:
        return None
    if not isinstance(cfg, dict):
        return None
    if 'episodes' in cfg:
        return cfg['episodes']
    for v in cfg.values():
        if isinstance(v, dict) and 'episodes' in v:
            return v['episodes']
    return None


def format_episode_title(experiments_info):
    """Build a parametric episode-count fragment for plot titles: '100 episodes' if
    every experiment shares the same count, '100 vs 300 episodes' (in file order) if
    they differ, or '' if no count could be read for any experiment."""
    counts = [get_episodes(exp['path']) for exp in experiments_info]
    counts = [c for c in counts if c is not None]
    if not counts:
        return ''
    distinct = sorted(set(counts))
    if len(distinct) == 1:
        return f'{distinct[0]} episodes'
    return ' vs '.join(str(c) for c in dict.fromkeys(counts)) + ' episodes'


def compute_metrics_for_experiment(experiment_path):
    """Read training summaries from <algorithm>/data.json when available.

    Each agent folder contains a data.json with arrays such as
    train_metrics.accuracy, train_metrics.precision, train_metrics.recall,
    train_metrics.f1_score. Those are episode-wise training metrics; we aggregate
    them by mean to compare experiments at algorithm level.
    """
    path = Path(experiment_path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment path does not exist: {experiment_path}")

    metrics = {}
    for child in sorted(path.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith('TEST'):
            continue

        data_json = child / 'data.json'
        if not data_json.exists():
            continue

        try:
            with open(data_json, 'r') as f:
                data = json.load(f)
        except Exception as exc:
            print(f"Warning: could not load {data_json}: {exc}")
            continue

        train_metrics = data.get('train_metrics', {})
        if not train_metrics:
            # Fallback to legacy flat keys if present.
            train_metrics = {
                'accuracy': data.get('accuracy', []),
                'precision': data.get('precision', []),
                'recall': data.get('recall', []),
                'f1_score': data.get('f1_score', []),
            }

        # Use the training summary arrays rather than TEST*/test.json classification arrays.
        # The arrays are already per-episode metrics; average them for the experiment-level summary.
        acc = np.mean(train_metrics.get('accuracy', [])) if train_metrics.get('accuracy') else 0.0
        prec = np.mean(train_metrics.get('precision', [])) if train_metrics.get('precision') else 0.0
        rec = np.mean(train_metrics.get('recall', [])) if train_metrics.get('recall') else 0.0
        f1 = np.mean(train_metrics.get('f1_score', [])) if train_metrics.get('f1_score') else 0.0

        agent_name = child.name
        normalized_name = agent_name.split('_', 1)[0]
        metrics.setdefault(normalized_name, []).append({
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
        })

    merged_metrics = {}
    for algo_name, entries in metrics.items():
        merged_metrics[algo_name] = {
            'accuracy': float(np.mean([m['accuracy'] for m in entries])),
            'precision': float(np.mean([m['precision'] for m in entries])),
            'recall': float(np.mean([m['recall'] for m in entries])),
            'f1_score': float(np.mean([m['f1_score'] for m in entries])),
        }

    return merged_metrics


# --- plotting ---

def plot_comparison(experiments_info, output_dir, variant_name='comparison', excluded_algorithms=None,
                     fig_height=7.0, fig_width_scale=0.8):
    """experiments_info: list of {'label':str, 'path':str}

    Produces a single grouped bar chart containing all algorithms present across
    the experiments.
    X-axis: algorithms.
    For each algorithm: one block for each experiment label, and inside each
    experiment block 4 adjacent metric bars (accuracy, precision, recall, f1).

    fig_height: figure height in inches (default 7.0, matches the original fixed
    value). Lower it for a flatter chart when there are few algorithm groups (e.g.
    a 2-algorithm comparison looks cramped at the default height/width ratio).
    fig_width_scale: multiplier applied to the algorithm-group width when computing
    the auto figure width (default 0.8, matches the original hardcoded formula).
    """
    os.makedirs(output_dir, exist_ok=True)

    from matplotlib.patches import Patch

    excluded_set = {name.strip() for name in (excluded_algorithms or []) if str(name).strip()}

    # collect per-experiment metrics
    all_metrics = {}
    algorithms = set()
    for exp in experiments_info:
        label = exp['label']
        path = exp['path']
        try:
            m = compute_metrics_for_experiment(path)
        except FileNotFoundError as e:
            print(e)
            m = {}
        if excluded_set:
            m = {algo: vals for algo, vals in m.items() if algo not in excluded_set}
            print(f"Info: excluding algorithms for '{label}': {sorted(excluded_set)}")
        all_metrics[label] = m
        algorithms.update(m.keys())

    algorithms = sorted(list(algorithms))
    labels = list(all_metrics.keys())

    if not algorithms:
        print("No algorithms found in the provided experiments. Nothing to plot.")
        return

    metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_labels = {
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1_score': 'F1'
    }
    metric_palette = {
        'accuracy': ('#dfeef7', '#78b7d8'),
        'precision': ('#dff3e3', '#7fc89c'),
        'recall': ('#f7efc8', '#e6c56c'),
        'f1_score': ('#f8dfe3', '#d98ca5'),
    }

    n_algorithms = len(algorithms)
    n_metrics = len(metric_names)
    n_experiments = len(labels)

    # --- Bar and group spacing logic (thicker bars, tighter gaps) ---
    bar_width = 0.42
    # Space between bars within a metric group (e.g., between acc and prec)
    metric_step = bar_width * 1.04
    # Total width of one experiment's 4-metric block
    exp_block_width = n_metrics * metric_step
    # Space between experiment blocks (e.g., between 100ep and 300ep)
    exp_gap = bar_width * 0.5
    # Total width for one algorithm's group (all experiments)
    algo_group_width = (n_experiments * exp_block_width) + ((n_experiments - 1) * exp_gap)
    # Space between different algorithm groups (e.g., between A2C and DQN)
    algo_gap = exp_block_width * 0.6

    fig_width = max(12, n_algorithms * (algo_group_width + algo_gap) * fig_width_scale)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    algo_centers = []
    legend_patches = {}

    for algo_idx, algo in enumerate(algorithms):
        algo_start_pos = algo_idx * (algo_group_width + algo_gap)

        for exp_idx, label in enumerate(labels):
            exp_start_pos = algo_start_pos + exp_idx * (exp_block_width + exp_gap)

            for metric_idx, metric_name in enumerate(metric_names):
                x_pos = exp_start_pos + metric_idx * metric_step
                algo_metrics = all_metrics.get(label, {}).get(algo, {})
                score = algo_metrics.get(metric_name, 0.0) * 100.0

                light_color, dark_color = metric_palette[metric_name]
                # Use light/dark colors for different experiments
                color = light_color if exp_idx % 2 == 0 else dark_color

                ax.bar(x_pos, score, width=bar_width, color=color, edgecolor='black', linewidth=0.6, alpha=0.95)

                # Numeric value label directly above each bar
                ax.text(
                    x_pos,
                    score + 1.2,
                    f'{score:.1f}',
                    ha='center',
                    va='bottom',
                    fontsize=FS_VALUE_LABEL,
                    rotation=90,
                    color='black'
                )

                # Create legend entry if it doesn't exist
                legend_key = f"{label} - {metric_labels[metric_name]}"
                if legend_key not in legend_patches:
                    legend_patches[legend_key] = Patch(facecolor=color, edgecolor='black', label=legend_key)

            # Experiment label above its own block of bars, at a fixed height well clear
            # of the numeric value labels, in bold with underscores rendered as spaces
            cluster_center = exp_start_pos + (exp_block_width - metric_step) / 2.0
            ax.text(
                cluster_center,
                120.0,
                _short_label(label),
                ha='center',
                va='bottom',
                fontsize=FS_EXP_LABEL,
                fontweight='bold',
                color='dimgray'
            )

        # Center of the entire group for this algorithm
        algo_center = algo_start_pos + (algo_group_width / 2)
        algo_centers.append(algo_center)

    ax.set_xticks(algo_centers)
    ax.set_xticklabels(algorithms, ha='center', fontweight='bold', fontsize=FS_TICK)
    ax.set_ylim(0, 132.0)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.tick_params(axis='y', labelsize=FS_YTICK)
    ax.set_ylabel('Weighted Score (%) — training set', fontsize=FS_AXIS_LABEL)
    episode_title = format_episode_title(experiments_info)
    title_suffix = f'{episode_title} - ' if episode_title else ''
    ax.set_title(f'Metrics comparison - {title_suffix}{variant_name.replace("_", " ")}', fontsize=FS_TITLE)
    ax.grid(axis='y', linestyle='--', alpha=0.35)

    # Legend: a single row, grouped by experiment --
    #   "100 ep."  [Accuracy] [Precision] [Recall] [F1]   "300 ep."  [Accuracy] ...
    # The experiment names are inline text pseudo-entries (invisible handle) acting
    # as group separators.
    spacer = Patch(facecolor='none', edgecolor='none')
    handles, labels_row = [], []
    for label in labels:
        handles.append(spacer)
        labels_row.append(_short_label(label))
        for m in metric_names:
            handles.append(legend_patches[f"{label} - {metric_labels[m]}"])
            labels_row.append(metric_labels[m])
    leg = ax.legend(
        handles=handles,
        labels=labels_row,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.10),
        ncol=len(handles),
        frameon=False,
        fontsize=FS_LEGEND,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    legend_artists = [leg]

    out_path = os.path.join(output_dir, 'metrics_comparison_all_algorithms.png')
    plt.tight_layout()
    # Crop right after the last rendered artist (the legend) instead of
    # reserving a fixed fraction of the figure for it, to avoid leftover
    # blank space below the legend.
    plt.savefig(out_path, dpi=180, bbox_inches='tight', bbox_extra_artists=legend_artists, pad_inches=0.1)
    plt.close(fig)
    print(f"Saved {out_path}")


# --- CLI ---

def main():
    parser = argparse.ArgumentParser(description="Compare experiment metrics (100 vs 300 episodes)")
    parser.add_argument('--config', type=str, default='test_tools/compare_config.yaml', help='YAML config pointing to experiments to compare')
    parser.add_argument('--output_dir', type=str, default='_training/_debug/compare', help='Output directory for charts')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    experiments = cfg.get('experiments', [])
    if len(experiments) < 2:
        raise SystemExit('Config must list at least two experiments to compare')

    output_prefix = cfg.get('output_prefix', 'comparison')
    excluded_algorithms = cfg.get('excluded_algorithms', [])
    fig_height = cfg.get('fig_height', 7.0)
    fig_width_scale = cfg.get('fig_width_scale', 0.8)
    plot_comparison(experiments, args.output_dir, output_prefix, excluded_algorithms=excluded_algorithms,
                     fig_height=fig_height, fig_width_scale=fig_width_scale)


if __name__ == '__main__':
    main()
