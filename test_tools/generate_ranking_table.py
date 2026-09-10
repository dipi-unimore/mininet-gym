#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

DEFAULT_OUTPUT_DIR = Path("_training/_debug/generate")
ALGORITHM_ORDER = ["Q-Learning", "SARSA", "DQN", "PPO", "A2C"]


def normalize_algorithm_name(raw_name: str) -> str:
    base = raw_name.split("_", 1)[0]
    mapping = {
        "Q-learning": "Q-Learning",
        "Sarsa": "SARSA",
        "SARSA": "SARSA",
        "DQN": "DQN",
        "PPO": "PPO",
        "A2C": "A2C",
    }
    return mapping.get(base, base)


def parse_int_label(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def extract_prediction_vector(predicted_block, agent_name_hint):
    if isinstance(predicted_block, list):
        return predicted_block
    if isinstance(predicted_block, dict):
        if agent_name_hint in predicted_block and isinstance(predicted_block[agent_name_hint], list):
            return predicted_block[agent_name_hint]
        if len(predicted_block) == 1:
            first_value = next(iter(predicted_block.values()))
            if isinstance(first_value, list):
                return first_value
    raise ValueError("Unsupported predicted payload format in test.json")


def map_ho_status_id_to_class(value):
    """Mirror utility/my_ho_statistics.py:map_ho_status_id_to_class.

    Raw ground-truth status ids can include two mitigation-aware states beyond the
    3 confusion-matrix classes (0=NORMAL, 1=UNDER_ATTACK, 2=ATTACKING), produced only
    when attacks.apply_drop_rules=true: 3 (incoming_blocked_attack) belongs to NORMAL
    for classification purposes (the incoming attack has been neutralized), and
    4 (out_attack_blocked) belongs to ATTACKING (the host's own role hasn't changed).
    Predicted labels are always in {0,1,2} (Discrete(3) action space) and never need
    remapping.
    """
    if value == 3:
        return 0
    if value == 4:
        return 2
    return value


def get_num_hosts(experiment_path: Path):
    """Read num_hosts + num_iots from the experiment's config.yaml, or None if unavailable."""
    config_path = experiment_path / "config.yaml"
    if not config_path.exists():
        return None
    try:
        with open(config_path, "r") as handle:
            cfg = yaml.safe_load(handle)
        net_params = cfg.get("env_params", {}).get("net_params", {})
        num_hosts = net_params.get("num_hosts")
        num_iots = net_params.get("num_iots", 0)
        if num_hosts is None:
            return None
        return int(num_hosts) + int(num_iots)
    except Exception:
        return None


def remap_ground_truth_sequence(gt_labels, n_hosts):
    """Remap raw ground-truth status ids to the 3 confusion-matrix classes, accounting for
    the same-round observation lag documented in per_host_scan_wrapper.py's
    _is_incoming_attack_mitigated(): a victim's status can flip to 3
    (incoming_blocked_attack) one round before its own (round-frozen) observation
    actually reflects the block, if it is processed after its attacker within that round.

    Only the *first* round a given host reads 3 (i.e. the previous round for that same
    host was not 3) is kept as class 1 (UNDER_ATTACK); every subsequent round the same
    host stays at 3 is mapped to class 0 (NORMAL). Status 4 (out_attack_blocked) always
    maps to 2 (ATTACKING). Falls back to the simple element-wise remap if n_hosts doesn't
    evenly divide the sequence length.
    """
    n = len(gt_labels)
    if not n_hosts or n % n_hosts != 0:
        return [map_ho_status_id_to_class(v) for v in gt_labels]

    n_rounds = n // n_hosts
    remapped = [0] * n
    for host in range(n_hosts):
        prev_raw = None
        for round_idx in range(n_rounds):
            i = round_idx * n_hosts + host
            raw = gt_labels[i]
            if raw == 3:
                remapped[i] = 1 if prev_raw != 3 else 0
            elif raw == 4:
                remapped[i] = 2
            else:
                remapped[i] = raw
            prev_raw = raw
    return remapped


def read_test_payload(test_json: Path, agent_name_hint: str, n_hosts=None):
    with open(test_json, "r") as handle:
        data = json.load(handle)

    ground_truth = data.get("ground_truth", [])
    predicted = extract_prediction_vector(data.get("predicted"), agent_name_hint)
    if not isinstance(ground_truth, list) or not isinstance(predicted, list):
        raise ValueError("ground_truth/predicted must be lists")
    if len(ground_truth) != len(predicted):
        raise ValueError("ground_truth and predicted have different lengths")

    raw_gt = []
    filtered_pred = []
    dropped = 0
    for gt, pred in zip(ground_truth, predicted):
        gt_label = parse_int_label(gt)
        pred_label = parse_int_label(pred)
        if gt_label is None or pred_label is None:
            dropped += 1
            continue
        raw_gt.append(gt_label)
        filtered_pred.append(pred_label)

    if dropped == 0:
        filtered_gt = remap_ground_truth_sequence(raw_gt, n_hosts)
    else:
        filtered_gt = [map_ho_status_id_to_class(v) for v in raw_gt]

    return np.array(filtered_gt, dtype=int), np.array(filtered_pred, dtype=int)


def precision_recall_f1_macro(gt, pred):
    if gt.size == 0:
        return 0.0, 0.0, 0.0

    labels = sorted(set(gt.tolist()) | set(pred.tolist()))
    precisions = []
    recalls = []
    f1_scores = []

    for label in labels:
        tp = int(((pred == label) & (gt == label)).sum())
        fp = int(((pred == label) & (gt != label)).sum())
        fn = int(((pred != label) & (gt == label)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1_scores))


def compute_metrics_for_experiment(experiment_path: str):
    exp_path = Path(experiment_path)
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment path not found: {exp_path}")

    n_hosts = get_num_hosts(exp_path)

    per_algorithm_data = {}
    for test_dir in sorted(exp_path.iterdir()):
        if not test_dir.is_dir() or not test_dir.name.startswith("TEST"):
            continue
        test_json = test_dir / "test.json"
        if not test_json.exists():
            continue

        agent_name = test_dir.name[5:] if test_dir.name.startswith("TEST_") else test_dir.name
        algorithm = normalize_algorithm_name(agent_name)

        try:
            gt, pred = read_test_payload(test_json, agent_name, n_hosts)
        except Exception as exc:
            print(f"Warning: skipping {test_json}: {exc}")
            continue

        if gt.size == 0:
            continue
        per_algorithm_data.setdefault(algorithm, {"gt": [], "pred": []})
        per_algorithm_data[algorithm]["gt"].append(gt)
        per_algorithm_data[algorithm]["pred"].append(pred)

    metrics = {}
    for algorithm, vectors in per_algorithm_data.items():
        gt = np.concatenate(vectors["gt"])
        pred = np.concatenate(vectors["pred"])

        precision, recall, f1 = precision_recall_f1_macro(gt, pred)
        accuracy = float((pred == gt).sum()) / float(gt.size)

        victims = gt == 1
        victim_total = int(victims.sum())
        victim_as_attacker = int(((gt == 1) & (pred == 2)).sum())
        confusion_rate = victim_as_attacker / victim_total if victim_total > 0 else 0.0

        metrics[algorithm] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "victim_as_attacker_rate": confusion_rate,
        }

    return metrics


def algorithm_sort_key(name):
    return ALGORITHM_ORDER.index(name) if name in ALGORITHM_ORDER else len(ALGORITHM_ORDER)


def rank_values(values_by_algo, higher_is_better):
    valid = [(algo, val) for algo, val in values_by_algo.items() if not pd.isna(val)]
    valid = sorted(valid, key=lambda item: item[1], reverse=higher_is_better)

    ranks = {algo: "-" for algo in values_by_algo}
    if not valid:
        return ranks

    current_rank = 1
    prev_val = None
    for idx, (algo, val) in enumerate(valid):
        if prev_val is not None and val != prev_val:
            current_rank = idx + 1
        ranks[algo] = current_rank
        prev_val = val
    return ranks


def generate_latex_table(results_by_label, metric_name, caption, label_name, higher_is_better=True, as_percentage=False):
    experiment_labels = list(results_by_label.keys())
    all_algorithms = sorted(
        {algo for exp_metrics in results_by_label.values() for algo in exp_metrics.keys()},
        key=algorithm_sort_key,
    )

    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append(f"\\caption{{{caption}}}")
    latex.append(f"\\label{{{label_name}}}")
    latex.append("\\renewcommand{\\arraystretch}{1.2}")
    latex.append("\\begin{tabular}{l" + ("c" * len(experiment_labels)) + "}")
    latex.append("\\toprule")
    latex.append("Algorithm & " + " & ".join(experiment_labels) + " \\\\")
    latex.append("\\midrule")

    column_ranks = {}
    column_bests = {}
    for exp_label in experiment_labels:
        values = {algo: results_by_label[exp_label].get(algo, {}).get(metric_name, float("nan")) for algo in all_algorithms}
        column_ranks[exp_label] = rank_values(values, higher_is_better=higher_is_better)

        valid = [v for v in values.values() if not pd.isna(v)]
        if not valid:
            column_bests[exp_label] = None
        else:
            column_bests[exp_label] = max(valid) if higher_is_better else min(valid)

    for algo in all_algorithms:
        row = [algo]
        for exp_label in experiment_labels:
            value = results_by_label[exp_label].get(algo, {}).get(metric_name, float("nan"))
            rank = column_ranks[exp_label][algo]
            best = column_bests[exp_label]

            if pd.isna(value):
                row.append("-")
                continue

            shown_value = f"{value * 100:.2f}\\%" if as_percentage else f"{value:.3f}"
            cell = f"{shown_value} ({rank})"
            if best is not None and value == best:
                row.append(f"\\textbf{{{cell}}}")
            else:
                row.append(cell)

        latex.append(" & ".join(row) + " \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    return "\n".join(latex)


def main():
    parser = argparse.ArgumentParser(
        description="Generate test-set ranking tables from TEST*/test.json using compare-style YAML configs."
    )
    parser.add_argument("config_file", type=str, help="Path to YAML config (same schema as compare_episodes_metrics).")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory for .tex tables.")
    args = parser.parse_args()

    with open(args.config_file, "r") as handle:
        config = yaml.safe_load(handle) or {}

    experiments = config.get("experiments", [])
    if not experiments:
        raise SystemExit("Config must define at least one experiment in 'experiments'.")

    output_prefix = config.get("output_prefix", "test_ranking")
    excluded_algorithms = config.get("excluded_algorithms", [])
    if excluded_algorithms is None:
        excluded_algorithms = []
    if not isinstance(excluded_algorithms, list):
        raise SystemExit("Config key 'excluded_algorithms' must be a list (e.g. []).")
    excluded_set = {name.strip() for name in excluded_algorithms if str(name).strip()}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_by_label = {}
    for exp in experiments:
        label = exp.get("label", "experiment")
        path = exp.get("path")
        if not path:
            print(f"Warning: missing path for experiment '{label}', skipping.")
            continue

        try:
            metrics = compute_metrics_for_experiment(path)
        except FileNotFoundError as exc:
            print(f"Warning: {exc}")
            continue
        if excluded_set:
            metrics = {algo: vals for algo, vals in metrics.items() if algo not in excluded_set}
            print(f"Info: excluding algorithms for '{label}': {sorted(excluded_set)}")
        results_by_label[label] = metrics

    if not results_by_label:
        raise SystemExit("No valid experiment data loaded.")

    f1_table = generate_latex_table(
        results_by_label=results_by_label,
        metric_name="f1_score",
        caption="Test-set F1 macro ranking across selected experiments",
        label_name=f"tab:test_f1_ranking_{output_prefix}",
        higher_is_better=True,
        as_percentage=False,
    )
    confusion_table = generate_latex_table(
        results_by_label=results_by_label,
        metric_name="victim_as_attacker_rate",
        caption="Victim-as-attacker confusion rate (lower is better)",
        label_name=f"tab:victim_as_attacker_{output_prefix}",
        higher_is_better=False,
        as_percentage=True,
    )

    f1_path = output_dir / f"{output_prefix}_f1_ranking_table.tex"
    confusion_path = output_dir / f"{output_prefix}_victim_as_attacker_table.tex"
    f1_path.write_text(f1_table + "\n")
    confusion_path.write_text(confusion_table + "\n")

    print(f"Saved: {f1_path}")
    print(f"Saved: {confusion_path}")
    print("\n--- F1 ranking table ---")
    print(f1_table)
    print("\n--- Victim-as-attacker table ---")
    print(confusion_table)


if __name__ == "__main__":
    main()
