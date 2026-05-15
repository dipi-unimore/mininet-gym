import orjson
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reinforcement_learning.scenarios.attack_detect_host_observable.network_env_attack_detect_per_host_observable import (
    discretize_attack_detect_ho_state,
)
from utility.params import read_config_file


def find_latest_statuses_file(root_dir="_training", preferred_keywords=None):
    """Return the most recent statuses.json under root_dir."""
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Training directory not found: {root_dir}")

    candidates = list(root.rglob("statuses.json"))
    if not candidates:
        raise FileNotFoundError(f"No statuses.json files found under {root_dir}")

    if preferred_keywords:
        lowered_keywords = [keyword.lower() for keyword in preferred_keywords]
        preferred = [
            path for path in candidates
            if any(keyword in str(path).lower() for keyword in lowered_keywords)
        ]
        if preferred:
            candidates = preferred

    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_statuses_json(statuses_path=None, preferred_keywords=None):
    """Load a statuses.json file and return (path, statuses_list)."""
    if statuses_path is None:
        statuses_path = find_latest_statuses_file(
            preferred_keywords=preferred_keywords
        )
    statuses_path = Path(statuses_path)
    with open(statuses_path, "rb") as file:
        data = orjson.loads(file.read())

    if isinstance(data, dict):
        if "statuses" in data and isinstance(data["statuses"], list):
            data = data["statuses"]
        else:
            raise ValueError(
                f"Unsupported statuses.json format in {statuses_path}: expected a list"
            )

    if not isinstance(data, list):
        raise ValueError(f"Unsupported statuses.json format in {statuses_path}")

    return statuses_path, data


def _extract_statuses_list(statuses_data):
    if isinstance(statuses_data, dict):
        if isinstance(statuses_data.get("statuses"), list):
            return statuses_data.get("statuses", [])
        if isinstance(statuses_data.get("data"), list):
            return statuses_data.get("data", [])
    if isinstance(statuses_data, list):
        return statuses_data
    return []


def _normalize_action_label(host_status):
    status_text = str(host_status.get("status", "")).strip().lower()
    task_text = str(host_status.get("taskType", "")).strip().lower()
    raw_id = host_status.get("id", None)

    if status_text in ("normal",):
        return "normal"
    if status_text in ("under_attack", "attack_in", "incoming_attack"):
        return "attack_in"
    if status_text in ("attacking", "attack_out", "outgoing_attack"):
        return "attack_out"

    if isinstance(raw_id, (int, np.integer)):
        id_map = {0: "normal", 1: "attack_in", 2: "attack_out"}
        mapped = id_map.get(int(raw_id))
        if mapped:
            return mapped

    if task_text:
        return task_text
    if status_text:
        return status_text
    return "unknown"


def _pretty_feature_name(name):
    mapping = {
        "receivedPackets": "rx_pkt",
        "transmittedPackets": "tx_pkt",
        "receivedBytes": "rx_bytes",
        "transmittedBytes": "tx_bytes",
        "receivedPacketsPercentageChange": "rx_pkt_%diff",
        "transmittedPacketsPercentageChange": "tx_pkt_%diff",
        "receivedBytesPercentageChange": "rx_bytes_%diff",
        "transmittedBytesPercentageChange": "tx_bytes_%diff",
    }
    return mapping.get(name, name)


def plot_qtable_coverage_dynamic_actions_from_statuses(
    statuses_data,
    output_path,
    low,
    high,
    n_bins,
    features=None,
):
    """Plot one subplot per action using env discretization and save a PNG."""
    statuses_list = _extract_statuses_list(statuses_data)
    if not statuses_list:
        raise ValueError("No statuses data available")

    if features is None:
        features = [
            "receivedPackets",
            "transmittedPackets",
            "receivedBytes",
            "transmittedBytes",
            "receivedPacketsPercentageChange",
            "transmittedPacketsPercentageChange",
            "receivedBytesPercentageChange",
            "transmittedBytesPercentageChange",
        ]

    state_feature_order = [
        "receivedPackets",
        "receivedPacketsPercentageChange",
        "receivedBytes",
        "receivedBytesPercentageChange",
        "transmittedPackets",
        "transmittedPacketsPercentageChange",
        "transmittedBytes",
        "transmittedBytesPercentageChange",
    ]

    n_bins = max(2, int(n_bins))
    rows = []
    for step_index, status in enumerate(statuses_list):
        hosts = status.get("hostStatusesStructured", {})
        if not isinstance(hosts, dict):
            continue

        for host_name, host_status in hosts.items():
            if not isinstance(host_status, dict):
                continue

            action = _normalize_action_label(host_status)
            try:
                raw_state = [float(host_status.get(feature, 0) or 0) for feature in state_feature_order]
            except (TypeError, ValueError):
                continue

            discrete_state = discretize_attack_detect_ho_state(raw_state, low, high, n_bins)
            for feature_idx, feature in enumerate(state_feature_order):
                rows.append({
                    "step": step_index,
                    "host": host_name,
                    "feature": feature,
                    "value": float(raw_state[feature_idx]),
                    "bin": int(discrete_state[feature_idx]),
                    "action": str(action),
                })

    if not rows:
        raise ValueError("No feature rows could be extracted from statuses data")

    df = pd.DataFrame(rows)
    counts = (
        df.groupby(["action", "feature", "bin"], sort=False)
        .size()
        .reset_index(name="count")
        .sort_values(["action", "feature", "bin"], ascending=[True, True, True])
    )

    total_traffic_records = int(df[["step", "host"]].drop_duplicates().shape[0])
    preferred_actions = ["normal", "attack_in", "attack_out"]
    available_actions = [str(action) for action in pd.unique(df["action"]).tolist()]
    action_order = [action for action in preferred_actions if action in available_actions]
    if not action_order:
        action_order = sorted(available_actions)

    block_gap = 1
    block_width = n_bins + block_gap
    x_rows = []
    for action in action_order:
        for feature_idx, feature in enumerate(features):
            for bin_idx in range(n_bins):
                x_rows.append({
                    "action": action,
                    "feature": feature,
                    "bin": bin_idx,
                    "x": feature_idx * block_width + bin_idx,
                })

    full_grid = pd.DataFrame(x_rows)
    counts = full_grid.merge(counts, on=["action", "feature", "bin"], how="left")
    counts["count"] = counts["count"].fillna(0).astype(int)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_actions = max(1, len(action_order))
    fig, axes = plt.subplots(
        n_actions,
        1,
        figsize=(24, max(6, 4.8 * n_actions)),
        constrained_layout=True,
        squeeze=False,
    )
    cmap = plt.get_cmap("tab10", n_bins)
    bin_colors = [cmap(i) for i in range(n_bins)]

    block_centers = [feature_idx * block_width + (n_bins - 1) / 2 for feature_idx in range(len(features))]
    feature_labels = [_pretty_feature_name(feature) for feature in features]

    for axis, action in zip(axes.flatten(), action_order):
        action_counts = counts[counts["action"] == action].sort_values("x")
        max_count = max(1, int(action_counts["count"].max()))
        zero_label_y = max_count * 0.03
        positive_label_pad = max_count * 0.01

        non_zero = action_counts[action_counts["count"] > 0]
        bar_colors = [bin_colors[int(bin_idx)] for bin_idx in non_zero["bin"].tolist()]
        axis.bar(non_zero["x"], non_zero["count"], color=bar_colors, width=0.85)

        for _, row in action_counts.iterrows():
            count_value = int(row["count"])
            is_zero = count_value == 0
            axis.text(
                row["x"],
                zero_label_y if is_zero else count_value + positive_label_pad,
                str(count_value),
                color="red" if is_zero else "black",
                ha="center",
                va="bottom",
                fontsize=6,
            )

        for feature_idx in range(1, len(features)):
            separator_x = feature_idx * block_width - 0.5
            axis.axvline(separator_x, color="#BBBBBB", linestyle="--", linewidth=0.6)

        axis.set_title(f"Q-table coverage (discretized) - action: {action}")
        axis.set_ylabel("Occurrences")
        axis.set_xticks(block_centers)
        axis.set_xticklabels(feature_labels, rotation=25, ha="right")
        axis.set_ylim(0, max_count * 1.08)
        axis.grid(axis="y", alpha=0.25)

    axes[-1][0].set_xlabel(f"Feature blocks with {n_bins} discretized bins each")
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=bin_colors[bin_idx], label=f"bin {bin_idx}")
        for bin_idx in range(n_bins)
    ]
    axes[0][0].legend(
        handles=legend_handles,
        loc="upper right",
        ncol=min(n_bins, 8),
        fontsize=8,
        title="Bin colors",
    )

    plt.suptitle(
        f"Q-table coverage (discretized bins) from statuses.json | total traffics considered: {total_traffic_records}"
    )
    plt.savefig(output_path, dpi=160)
    plt.close()

    return {
        "output_path": str(output_path),
        "rows": int(len(df)),
        "counts": counts,
        "total_traffic_records": total_traffic_records,
    }


def plot_qtable_coverage_from_statuses(
    statuses_path=None,
    output_path=None,
    preferred_keywords=("qlearning", "sarsa", "tabular"),
    features=None,
    max_bins=8,
    n_bins=None,
):
    """Plot discretized Q-table coverage blocks from statuses.json."""
    statuses_path, statuses = load_statuses_json(
        statuses_path=statuses_path,
        preferred_keywords=preferred_keywords,
    )

    if features is None:
        features = [
            "receivedPackets",
            "receivedPacketsPercentageChange",
            "receivedBytes",
            "receivedBytesPercentageChange",
            "transmittedPackets",
            "transmittedPacketsPercentageChange",
            "transmittedBytes",
            "transmittedBytesPercentageChange",
        ]

    if n_bins is None:
        try:
            cfg, _ = read_config_file("config/default.yaml")
            n_bins = int(getattr(cfg.env_params, "n_bins", max_bins))
        except Exception:
            n_bins = max_bins
    n_bins = max(2, int(n_bins))

    packet_byte_features = {
        "receivedPackets",
        "receivedBytes",
        "transmittedPackets",
        "transmittedBytes",
    }

    def normalize_action_label(host_status):
        status_text = str(host_status.get("status", "")).strip().lower()
        task_text = str(host_status.get("taskType", "")).strip().lower()
        raw_id = host_status.get("id", None)

        if status_text in ("normal",):
            return "normal"
        if status_text in ("under_attack", "attack_in", "incoming_attack"):
            return "attack_in"
        if status_text in ("attacking", "attack_out", "outgoing_attack"):
            return "attack_out"

        if isinstance(raw_id, (int, np.integer)):
            id_map = {0: "normal", 1: "attack_in", 2: "attack_out"}
            if int(raw_id) in id_map:
                return id_map[int(raw_id)]

        if task_text:
            return task_text
        if status_text:
            return status_text
        return "unknown"

    def pretty_feature_name(name):
        out = []
        for i, ch in enumerate(name):
            if ch.isupper() and i > 0:
                out.append("_")
            out.append(ch.lower())
        return "".join(out)

    rows = []
    for step_index, status in enumerate(statuses):
        hosts = status.get("hostStatusesStructured", {})
        if not isinstance(hosts, dict):
            continue

        for host_name, host_status in hosts.items():
            action_label = normalize_action_label(host_status)
            for feature in features:
                if feature not in host_status:
                    continue
                value = host_status.get(feature)
                if value is None:
                    continue
                rows.append(
                    {
                        "step": step_index,
                        "host": host_name,
                        "feature": feature,
                        "value": float(value),
                        "action": str(action_label),
                    }
                )

    if not rows:
        raise ValueError(f"No feature rows could be extracted from {statuses_path}")

    df = pd.DataFrame(rows)
    df["bin"] = -1
    for feature in features:
        feature_mask = df["feature"] == feature
        feature_values = df.loc[feature_mask, "value"]
        if feature_values.empty:
            continue

        if feature in packet_byte_features:
            zero_mask = feature_values <= 0
            df.loc[feature_mask & zero_mask, "bin"] = 0

            positive_values = feature_values[~zero_mask]
            if positive_values.empty:
                continue

            positive_bins = max(1, n_bins - 1)
            transformed = np.log1p(positive_values)

            if transformed.nunique() > 1:
                cap = transformed.quantile(0.995)
                transformed = transformed.clip(upper=cap)

            try:
                q_labels = pd.qcut(
                    transformed,
                    q=positive_bins,
                    labels=False,
                    duplicates="drop",
                )
                df.loc[q_labels.index, "bin"] = q_labels.astype(int) + 1
            except ValueError:
                tmin = float(transformed.min())
                tmax = float(transformed.max())
                if tmax == tmin:
                    df.loc[transformed.index, "bin"] = 1
                else:
                    scaled = (transformed - tmin) / (tmax - tmin)
                    bins = np.floor(scaled * positive_bins).astype(int)
                    df.loc[transformed.index, "bin"] = bins.clip(0, positive_bins - 1) + 1
            continue

        if feature_values.nunique() <= 1:
            df.loc[feature_mask, "bin"] = 0
            continue

        vmin = float(feature_values.min())
        vmax = float(feature_values.max())
        if vmax == vmin:
            df.loc[feature_mask, "bin"] = 0
        else:
            normalized = (feature_values - vmin) / (vmax - vmin)
            bins = np.floor(normalized * n_bins).astype(int)
            df.loc[feature_mask, "bin"] = bins.clip(0, n_bins - 1)

    counts = (
        df.groupby(["action", "feature", "bin"], sort=False)
        .size()
        .reset_index(name="count")
        .sort_values(["action", "feature", "bin"], ascending=[True, True, True])
    )

    total_traffic_records = int(df[["step", "host"]].drop_duplicates().shape[0])

    preferred_actions = ["normal", "attack_in", "attack_out"]
    available_actions = [str(action) for action in pd.unique(df["action"]).tolist()]
    action_order = [action for action in preferred_actions if action in available_actions]
    for action in available_actions:
        if action not in action_order:
            action_order.append(action)

    block_gap = 1
    block_width = n_bins + block_gap
    x_rows = []
    for action in action_order:
        for feature_idx, feature in enumerate(features):
            for bin_idx in range(n_bins):
                x_rows.append(
                    {
                        "action": action,
                        "feature": feature,
                        "bin": bin_idx,
                        "x": feature_idx * block_width + bin_idx,
                    }
                )
    full_grid = pd.DataFrame(x_rows)
    counts = full_grid.merge(counts, on=["action", "feature", "bin"], how="left")
    counts["count"] = counts["count"].fillna(0).astype(int)

    if output_path is None:
        output_path = statuses_path.with_name("qtable_coverage_preview.png")
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_actions = max(1, len(action_order))
    fig, axes = plt.subplots(
        n_actions,
        1,
        figsize=(24, max(6, 4.8 * n_actions)),
        constrained_layout=True,
        squeeze=False,
    )
    cmap = plt.get_cmap("tab10", n_bins)
    bin_colors = [cmap(i) for i in range(n_bins)]

    block_centers = [feature_idx * block_width + (n_bins - 1) / 2 for feature_idx in range(len(features))]
    feature_labels = [pretty_feature_name(feature) for feature in features]

    for axis, action in zip(axes.flatten(), action_order):
        action_counts = counts[counts["action"] == action].sort_values("x")
        max_count = max(1, int(action_counts["count"].max()))
        zero_label_y = max_count * 0.03
        positive_label_pad = max_count * 0.01

        non_zero = action_counts[action_counts["count"] > 0]
        bar_colors = [bin_colors[int(bin_idx)] for bin_idx in non_zero["bin"].tolist()]
        axis.bar(non_zero["x"], non_zero["count"], color=bar_colors, width=0.85)

        for _, row in action_counts.iterrows():
            count_value = int(row["count"])
            is_zero = count_value == 0
            axis.text(
                row["x"],
                zero_label_y if is_zero else count_value + positive_label_pad,
                str(count_value),
                color="red" if is_zero else "black",
                ha="center",
                va="bottom",
                fontsize=6,
            )

        for feature_idx in range(1, len(features)):
            separator_x = feature_idx * block_width - 0.5
            axis.axvline(separator_x, color="#BBBBBB", linestyle="--", linewidth=0.6)

        axis.set_title(f"Q-table coverage (discretized) - action: {action}")
        axis.set_ylabel("Occurrences")
        axis.set_xticks(block_centers)
        axis.set_xticklabels(feature_labels, rotation=25, ha="right")
        axis.set_ylim(0, max_count * 1.08)
        axis.grid(axis="y", alpha=0.25)

    axes[-1][0].set_xlabel(f"Feature blocks with {n_bins} discretized bins each")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=bin_colors[bin_idx], label=f"bin {bin_idx}")
        for bin_idx in range(n_bins)
    ]
    axes[0][0].legend(
        handles=legend_handles,
        loc="upper right",
        ncol=min(n_bins, 8),
        fontsize=8,
        title="Bin colors",
    )

    plt.suptitle(
        f"Q-table coverage (discretized bins) from {statuses_path.name} | total traffics considered: {total_traffic_records}"
    )
    plt.savefig(output_path, dpi=160)
    plt.close()

    preview_path = statuses_path.with_name("qtable_coverage_preview.png")
    if preview_path.resolve() != output_path.resolve():
        try:
            shutil.copy2(output_path, preview_path)
        except Exception:
            pass

    print(f"Saved qtable coverage preview to {output_path}")
    return {
        "statuses_path": str(statuses_path),
        "output_path": str(output_path),
        "preview_path": str(preview_path),
        "rows": int(len(df)),
        "counts": counts,
    }
