#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import confusion_matrix

DEFAULT_OUTPUT_DIR = Path("_training/_debug/generate")
ALGORITHM_ORDER = ["Q-Learning", "SARSA", "DQN", "PPO", "A2C"]

# --- font sizes (bumped for print legibility in the two-column paper) ---
FS_CM_ANNOT = 26        # confusion-matrix cell annotations
FS_CM_TITLE = 19        # per-panel title (algorithm name)
FS_CM_AXIS = 17         # "True Label" / "Predicted Label"
FS_CM_TICK = 16         # class-name ticks
FS_SUPTITLE = 22        # figure-level title
FS_BAR_VALUE = 11       # numeric value above each bar
FS_BAR_TICK = 14        # x tick labels (algorithm names)
FS_BAR_AXIS = 14        # y axis label
FS_BAR_TITLE = 15       # plot title
FS_BAR_LEGEND = 11      # legend entries
# role-confusion charts render single-column, so they are drawn near their final
# print size with smaller absolute fonts
RC_VALUE = 8
RC_TICK = 10
RC_AXIS = 10
RC_TITLE = 11
RC_LEGEND = 9
LABEL_MAPPING = {
    0: "NORMAL",
    1: "VICTIM",
    2: "ATTACKER",
    3: "CLASS_3",
}


def slugify(value: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in value).strip("_")


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


def get_episodes(experiment_path: Path):
    """Read the training episode count from the experiment's config.yaml, or None."""
    config_path = experiment_path / "config.yaml"
    if not config_path.exists():
        return None
    try:
        with open(config_path, "r") as handle:
            cfg = yaml.safe_load(handle)
    except Exception:
        return None
    if not isinstance(cfg, dict):
        return None
    if "episodes" in cfg:
        return cfg["episodes"]
    for v in cfg.values():
        if isinstance(v, dict) and "episodes" in v:
            return v["episodes"]
    return None


def format_episode_title(episode_counts):
    """Build a parametric episode-count fragment: '100 episodes' if every count is the
    same, '100 vs 300 episodes' (in order) if they differ, '' if none were readable."""
    counts = [c for c in episode_counts if c is not None]
    if not counts:
        return ""
    distinct = sorted(set(counts))
    if len(distinct) == 1:
        return f"{distinct[0]} episodes"
    return " vs ".join(str(c) for c in dict.fromkeys(counts)) + " episodes"


def remap_ground_truth_sequence(gt_labels, n_hosts):
    """Remap raw ground-truth status ids to the 3 confusion-matrix classes, accounting for
    the same-round observation lag documented in per_host_scan_wrapper.py's
    _is_incoming_attack_mitigated(): a victim's status can flip to 3
    (incoming_blocked_attack) one round before its own (round-frozen) observation
    actually reflects the block, if it is processed after its attacker within that round.

    Only the *first* round a given host reads 3 (i.e. the previous round for that same
    host was not 3) is kept as class 1 (UNDER_ATTACK) -- the observation for that specific
    round is still consistent with an active attack. Every subsequent round the same host
    stays at 3 is mapped to class 0 (NORMAL) as usual, since by then the observation has
    had at least one round to catch up. Status 4 (out_attack_blocked) always maps to 2
    (ATTACKING), unaffected by this transition logic.

    Requires reconstructing the per-host, per-round structure of the flat gt sequence
    (n_rounds x n_hosts); if n_hosts doesn't evenly divide the sequence length, falls back
    to the simple element-wise remap (3->0, 4->2).
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

    # The transition-aware remap needs the original per-round ordering, so only apply it
    # when nothing was dropped (dropped entries would break the round reconstruction);
    # otherwise fall back to the simple element-wise remap.
    if dropped == 0:
        filtered_gt = remap_ground_truth_sequence(raw_gt, n_hosts)
    else:
        filtered_gt = [map_ho_status_id_to_class(v) for v in raw_gt]

    if dropped:
        print(f"Warning: dropped {dropped} non-integer labels from {test_json}")

    return pd.DataFrame({"true_label": filtered_gt, "predicted_label": filtered_pred})


def load_experiment_predictions(experiment_path: str):
    exp_path = Path(experiment_path)
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment path not found: {exp_path}")

    n_hosts = get_num_hosts(exp_path)

    all_predictions = {}
    for test_dir in sorted(exp_path.iterdir()):
        if not test_dir.is_dir() or not test_dir.name.startswith("TEST"):
            continue
        test_json = test_dir / "test.json"
        if not test_json.exists():
            continue

        agent_name = test_dir.name[5:] if test_dir.name.startswith("TEST_") else test_dir.name
        algorithm = normalize_algorithm_name(agent_name)

        try:
            df = read_test_payload(test_json, agent_name, n_hosts)
        except Exception as exc:
            print(f"Warning: skipping {test_json}: {exc}")
            continue

        if df.empty:
            continue
        all_predictions.setdefault(algorithm, []).append(df)

    return {algo: pd.concat(parts, ignore_index=True) for algo, parts in all_predictions.items()}


def get_class_ids(predictions_by_algo):
    class_ids = sorted(
        {
            int(label)
            for df in predictions_by_algo.values()
            for label in pd.concat([df["true_label"], df["predicted_label"]]).unique()
        }
    )
    return class_ids


def class_names(class_ids):
    return [LABEL_MAPPING.get(class_id, f"CLASS_{class_id}") for class_id in class_ids]


def compute_victim_as_attacker_stats(df: pd.DataFrame):
    victims = df[df["true_label"] == 1]
    if victims.empty:
        return 0.0, 0, 0
    misclassified = int((victims["predicted_label"] == 2).sum())
    total = int(len(victims))
    return float(misclassified) / float(total), misclassified, total


def compute_attacker_as_victim_stats(df: pd.DataFrame):
    attackers = df[df["true_label"] == 2]
    if attackers.empty:
        return 0.0, 0, 0
    misclassified = int((attackers["predicted_label"] == 1).sum())
    total = int(len(attackers))
    return float(misclassified) / float(total), misclassified, total


def render_confusion_matrix(ax, df, title, class_ids, class_labels, show_y=True):
    """Render one confusion-matrix panel. When ``show_y`` is False the y tick labels
    and "True Label" axis label are hidden -- used for every panel except the
    leftmost one, so a row of panels does not repeat the same class ticks N times.
    Returns the QuadMesh mappable, so a caller can attach a single shared colorbar."""
    cm_norm = confusion_matrix(df["true_label"], df["predicted_label"], labels=class_ids, normalize="true")
    cm_counts = confusion_matrix(df["true_label"], df["predicted_label"], labels=class_ids)
    annot_labels = np.array(
        [
            [f"{cm_norm[i, j]:.1%}\n{cm_counts[i, j]}" for j in range(cm_norm.shape[1])]
            for i in range(cm_norm.shape[0])
        ]
    )
    sns.heatmap(
        cm_norm,
        annot=annot_labels,
        fmt="",
        cmap="Blues",
        ax=ax,
        xticklabels=class_labels,
        yticklabels=(class_labels if show_y else False),
        vmin=0,
        vmax=1,
        square=True,
        annot_kws={"size": FS_CM_ANNOT},
        cbar=False,
    )
    ax.set_title(title, fontsize=FS_CM_TITLE, fontweight="bold")
    ax.set_ylabel("True Label" if show_y else "", fontsize=FS_CM_AXIS)
    ax.set_xlabel("Predicted Label", fontsize=FS_CM_AXIS)
    ax.tick_params(axis="both", labelsize=FS_CM_TICK)
    return ax.collections[0]


def select_algorithms(predictions_by_algo, excluded_algorithms):
    excluded_set = {name.strip() for name in excluded_algorithms if str(name).strip()}
    selected = {algo: df for algo, df in predictions_by_algo.items() if algo not in excluded_set}
    return selected, excluded_set


def algorithm_sort_key(name):
    return ALGORITHM_ORDER.index(name) if name in ALGORITHM_ORDER else len(ALGORITHM_ORDER)


def format_episode_variant_title(experiment_label: str, prefix: str) -> str:
    """Build a 'confusion matrices by algorithm (<n_episodes> episode <variant flag>)'
    style title. Falls back gracefully when experiment_label isn't a '<N>_episodes' string
    (e.g. merged 'all_experiments' views or Group-3-style variant labels)."""
    match = re.match(r"^(\d+)_episodes?$", experiment_label)
    episode_part = f"{match.group(1)} episode" if match else experiment_label.replace("_", " ")
    variant_part = prefix.replace("_", " ")
    return f"Confusion Matrices by Algorithm ({episode_part} {variant_part})"


def plot_confusion_matrices(experiment_label, predictions_by_algo, output_dir: Path, prefix: str):
    if not predictions_by_algo:
        print(f"Warning: no algorithms available for '{experiment_label}', skipping confusion plot.")
        return

    class_ids = get_class_ids(predictions_by_algo)
    labels = class_names(class_ids)

    ordered_algorithms = sorted(predictions_by_algo.keys(), key=algorithm_sort_key)
    n_algorithms = len(ordered_algorithms)
    # square=True (in render_confusion_matrix) keeps the cells square; the panel
    # boxes are sized close to square so little horizontal space is wasted, with
    # a small gap between them and the shared colorbar tucked in on the right.
    fig, axes = plt.subplots(
        1, n_algorithms, figsize=(5.2 * n_algorithms + 1.0, 6.2),
    )
    if n_algorithms == 1:
        axes = [axes]
    fig.subplots_adjust(left=0.04, right=0.945, top=0.88, bottom=0.10, wspace=0.13)

    fig.suptitle(format_episode_variant_title(experiment_label, prefix), fontsize=FS_SUPTITLE, fontweight="bold", y=0.97)
    mappable = None
    for idx, (ax, algo) in enumerate(zip(axes, ordered_algorithms)):
        mappable = render_confusion_matrix(
            ax, predictions_by_algo[algo], algo, class_ids, labels, show_y=(idx == 0)
        )

    # one shared colorbar on the far right instead of one per panel
    if mappable is not None:
        cax = fig.add_axes([0.952, 0.17, 0.012, 0.58])
        cb = fig.colorbar(mappable, cax=cax)
        cb.ax.tick_params(labelsize=FS_CM_TICK - 2)
    output_path = output_dir / f"{prefix}_{slugify(experiment_label)}_confusion_matrices.png"
    plt.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_confusion_matrices_grid(loaded: dict, output_dir: Path, prefix: str, layout: str = "grid"):
    """One figure per config, NOT pooled: each (experiment label, algorithm) pair gets
    its own confusion matrix computed only from that label's own data. This is what a
    side-by-side Tier A/B comparison needs -- unlike
    plot_confusion_matrices("all_experiments", merge_predictions(loaded), ...), which
    merges every label's data into one matrix per algorithm and loses the
    per-condition comparison entirely.

    layout: "grid" (default) lays out rows = experiment labels, columns = algorithms
    (e.g. 2 labels x 2 algorithms = a 2x2 grid). "inline" instead puts every
    (label, algorithm) panel in a single row, in label-then-algorithm order.
    """
    labels = list(loaded.keys())
    if not labels:
        print(f"Warning: no experiments to plot for '{prefix}' confusion matrix grid, skipping.")
        return

    all_algos = set()
    for preds in loaded.values():
        all_algos.update(preds.keys())
    ordered_algorithms = sorted(all_algos, key=algorithm_sort_key)
    if not ordered_algorithms:
        print(f"Warning: no algorithms to plot for '{prefix}' confusion matrix grid, skipping.")
        return

    class_ids = get_class_ids({algo: df for preds in loaded.values() for algo, df in preds.items()})
    class_labels = class_names(class_ids)

    if layout == "inline":
        panels = [(label, algo) for label in labels for algo in ordered_algorithms]
        n_rows, n_cols = 1, len(panels)
        fig, axes_flat = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 6.2 * n_rows), squeeze=False)
        axes = [[axes_flat[0][i] for i in range(n_cols)]]
        panel_grid = [panels]
    elif layout == "grid":
        n_rows, n_cols = len(labels), len(ordered_algorithms)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 6.2 * n_rows), squeeze=False)
        panel_grid = [[(label, algo) for algo in ordered_algorithms] for label in labels]
    else:
        raise ValueError(f"Unknown confusion_matrix_layout '{layout}', expected 'grid' or 'inline'.")

    variant_display = VARIANT_DISPLAY_NAMES.get(prefix, prefix.replace("_", " "))
    fig.suptitle(f"Confusion Matrices: {' vs. '.join(labels)} ({variant_display})", fontsize=FS_SUPTITLE, fontweight="bold")

    for row_idx, row in enumerate(panel_grid):
        for col_idx, (label, algo) in enumerate(row):
            ax = axes[row_idx][col_idx]
            preds = loaded[label]
            if algo in preds:
                render_confusion_matrix(ax, preds[algo], f"{label} — {algo}", class_ids, class_labels)
            else:
                ax.axis("off")

    plt.tight_layout(rect=(0, 0.02, 1, 0.95))
    output_path = output_dir / f"{prefix}_grid_confusion_matrices.png"
    plt.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {output_path}")


VARIANT_DISPLAY_NAMES = {
    "reward_scaling": "Reward Scaling Only",
    "deltas": "Deltas",
    "all_flags": "All Flags",
}


def plot_role_confusion_bars_combined(loaded: dict, output_dir: Path, prefix: str, episode_title: str = ""):
    """Combined 100-vs-300-episodes role-confusion chart: one figure per variant instead
    of two, using light shades for the shorter run and full/dark shades for the longer
    one (same light/dark-per-run convention as compare_episodes_metrics.py)."""
    labels = list(loaded.keys())
    if len(labels) != 2:
        print(f"Warning: combine_role_confusion requires exactly 2 experiment labels for '{prefix}', skipping.")
        return

    algorithms = sorted(
        {algo for label in labels for algo in loaded[label].keys()},
        key=algorithm_sort_key,
    )
    if not algorithms:
        return

    stats = {
        label: {
            "victim": [
                compute_victim_as_attacker_stats(loaded[label][a]) if a in loaded[label] else (0.0, 0, 0)
                for a in algorithms
            ],
            "attacker": [
                compute_attacker_as_victim_stats(loaded[label][a]) if a in loaded[label] else (0.0, 0, 0)
                for a in algorithms
            ],
        }
        for label in labels
    }

    x = np.arange(len(algorithms), dtype=float)
    bar_width = 0.19
    offsets = {
        (labels[0], "victim"): -1.5 * bar_width,
        (labels[0], "attacker"): -0.5 * bar_width,
        (labels[1], "victim"): 0.5 * bar_width,
        (labels[1], "attacker"): 1.5 * bar_width,
    }
    palette = {
        "victim": ("#ffd9c7", "coral"),
        "attacker": ("#c9dff0", "steelblue"),
    }
    shade_index = {labels[0]: 0, labels[1]: 1}

    # single-column figure: 4:3 landscape, compact fonts, % labels only,
    # legend inside the top band
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    bars_by_key = {}
    for label in labels:
        for kind in ("victim", "attacker"):
            rates = [item[0] for item in stats[label][kind]]
            color = palette[kind][shade_index[label]]
            bars_by_key[(label, kind)] = ax.bar(
                x + offsets[(label, kind)], rates, width=bar_width,
                color=color, edgecolor="black", linewidth=0.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=RC_TICK)
    ax.tick_params(axis="y", labelsize=RC_TICK - 1)
    ax.set_ylabel("Conditional error rate", fontsize=RC_AXIS)
    variant_display = VARIANT_DISPLAY_NAMES.get(prefix, prefix.replace("_", " "))
    episode_prefix = f"{episode_title}: " if episode_title else ""
    ax.set_title(f'{episode_prefix}Role confusion ("{variant_display}")', fontsize=RC_TITLE)

    all_rates = [item[0] for label in labels for kind in ("victim", "attacker") for item in stats[label][kind]]
    max_rate = max(all_rates) if all_rates else 0.0
    ax.set_ylim(0, max_rate + 0.36)   # headroom for the in-plot legend
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_yticks(np.arange(0, min(1.0, max_rate + 0.05) + 0.001, 0.2))
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for label in labels:
        for kind in ("victim", "attacker"):
            for bar, (rate, _m, _t) in zip(bars_by_key[(label, kind)], stats[label][kind]):
                ax.text(bar.get_x() + bar.get_width() / 2.0, rate + 0.005,
                        f"{rate:.0%}", ha="center", va="bottom",
                        fontsize=RC_VALUE, rotation=90)

    handles = [
        Patch(facecolor=palette["victim"][1], edgecolor="black", label=r"Victim$\to$Attacker"),
        Patch(facecolor=palette["attacker"][1], edgecolor="black", label=r"Attacker$\to$Victim"),
        Patch(facecolor="#cccccc", edgecolor="black", label=f"{labels[0].replace('_',' ')} (light)"),
        Patch(facecolor="#555555", edgecolor="black", label=f"{labels[1].replace('_',' ')} (dark)"),
    ]
    ax.legend(handles=handles, loc="upper left", ncol=2, frameon=False,
              fontsize=RC_LEGEND, handlelength=1.2, columnspacing=1.0)

    output_path = output_dir / f"{prefix}_role_confusion_rates.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_role_confusion_bars(experiment_label, predictions_by_algo, output_dir: Path, prefix: str):
    if not predictions_by_algo:
        return

    ordered_algorithms = sorted(predictions_by_algo.keys(), key=algorithm_sort_key)
    victim_as_attacker_stats = [compute_victim_as_attacker_stats(predictions_by_algo[algo]) for algo in ordered_algorithms]
    attacker_as_victim_stats = [compute_attacker_as_victim_stats(predictions_by_algo[algo]) for algo in ordered_algorithms]

    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    x = np.arange(len(ordered_algorithms), dtype=float)
    bar_width = 0.38

    victim_as_attacker_rates = [item[0] for item in victim_as_attacker_stats]
    attacker_as_victim_rates = [item[0] for item in attacker_as_victim_stats]
    bars_victim = ax.bar(x - bar_width / 2.0, victim_as_attacker_rates, width=bar_width,
                         color="coral", edgecolor="black", linewidth=0.5,
                         label=r"Victim$\to$Attacker")
    bars_attacker = ax.bar(x + bar_width / 2.0, attacker_as_victim_rates, width=bar_width,
                           color="steelblue", edgecolor="black", linewidth=0.5,
                           label=r"Attacker$\to$Victim")

    ax.set_xticks(x)
    ax.set_xticklabels(ordered_algorithms, fontsize=RC_TICK)
    ax.tick_params(axis="y", labelsize=RC_TICK - 1)
    ax.set_ylabel("Conditional error rate", fontsize=RC_AXIS)
    ax.set_title(f"Role confusion ({experiment_label})", fontsize=RC_TITLE)
    max_rate = max(victim_as_attacker_rates + attacker_as_victim_rates) if ordered_algorithms else 0.0
    ax.set_ylim(0, max_rate + 0.30)   # headroom for the in-plot legend
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_yticks(np.arange(0, min(1.0, max_rate + 0.05) + 0.001, 0.2))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=RC_LEGEND,
              handlelength=1.2, columnspacing=1.0)

    for bar, (rate, _m, _t) in zip(bars_victim, victim_as_attacker_stats):
        ax.text(bar.get_x() + bar.get_width() / 2.0, rate + 0.005, f"{rate:.0%}",
                ha="center", va="bottom", fontsize=RC_VALUE, rotation=90)
    for bar, (rate, _m, _t) in zip(bars_attacker, attacker_as_victim_stats):
        ax.text(bar.get_x() + bar.get_width() / 2.0, rate + 0.005, f"{rate:.0%}",
                ha="center", va="bottom", fontsize=RC_VALUE, rotation=90)

    output_path = output_dir / f"{prefix}_{slugify(experiment_label)}_role_confusion_rates.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {output_path}")


def merge_predictions(experiments_predictions):
    merged = {}
    for predictions in experiments_predictions.values():
        for algo, df in predictions.items():
            merged.setdefault(algo, []).append(df)
    return {algo: pd.concat(parts, ignore_index=True) for algo, parts in merged.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Generate confusion analysis plots from TEST*/test.json using compare-style YAML configs."
    )
    parser.add_argument("config_file", type=str, help="Path to YAML config (same schema as compare_episodes_metrics).")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory for images.")
    args = parser.parse_args()

    with open(args.config_file, "r") as handle:
        config = yaml.safe_load(handle) or {}

    experiments = config.get("experiments", [])
    if not experiments:
        raise SystemExit("Config must define at least one experiment in 'experiments'.")

    output_prefix = config.get("output_prefix", "confusion_analysis")
    combine_role_confusion = bool(config.get("combine_role_confusion", False))
    confusion_matrix_grid = bool(config.get("confusion_matrix_grid", False))
    confusion_matrix_layout = config.get("confusion_matrix_layout", "grid")
    if confusion_matrix_layout not in ("grid", "inline"):
        raise SystemExit("Config key 'confusion_matrix_layout' must be 'grid' or 'inline'.")
    excluded_algorithms = config.get("excluded_algorithms", [])
    if excluded_algorithms is None:
        excluded_algorithms = []
    if not isinstance(excluded_algorithms, list):
        raise SystemExit("Config key 'excluded_algorithms' must be a list (e.g. []).")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = {}
    episode_counts = []
    for exp in experiments:
        label = exp.get("label", "experiment")
        path = exp.get("path")
        if not path:
            print(f"Warning: missing path for experiment '{label}', skipping.")
            continue
        episode_counts.append(get_episodes(Path(path)))
        try:
            predictions = load_experiment_predictions(path)
        except FileNotFoundError as exc:
            print(f"Warning: {exc}")
            continue

        if not predictions:
            print(f"Warning: no TEST*/test.json data loaded for experiment '{label}'.")
            continue
        selected_predictions, excluded_set = select_algorithms(predictions, excluded_algorithms)
        if excluded_set:
            print(f"Info: excluding algorithms for '{label}': {sorted(excluded_set)}")
        if not selected_predictions:
            print(f"Warning: all algorithms excluded or missing for '{label}'.")
            continue
        loaded[label] = selected_predictions

    if not loaded:
        raise SystemExit("No valid experiment data loaded.")

    for label, predictions in loaded.items():
        if not confusion_matrix_grid:
            plot_confusion_matrices(label, predictions, output_dir, output_prefix)
        if not combine_role_confusion:
            plot_role_confusion_bars(label, predictions, output_dir, output_prefix)

    if combine_role_confusion:
        plot_role_confusion_bars_combined(loaded, output_dir, output_prefix, format_episode_title(episode_counts))

    if confusion_matrix_grid and len(loaded) > 1:
        plot_confusion_matrices_grid(loaded, output_dir, output_prefix, layout=confusion_matrix_layout)
    elif len(loaded) > 1:
        merged = merge_predictions(loaded)
        plot_confusion_matrices("all_experiments", merged, output_dir, output_prefix)
        if not combine_role_confusion:
            plot_role_confusion_bars("all_experiments", merged, output_dir, output_prefix)

    print("Confusion analysis completed.")


if __name__ == "__main__":
    main()
