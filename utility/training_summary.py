import os


CORE_METRIC_KEYS = ("accuracy", "precision", "recall", "f1_score")
INDICATOR_METRIC_KEYS = (
    "cumulative_reward",
    "qtable_coverage_pct",
    "exploration_rate",
    "q_values_std",
    "q_values_mean",
    "q_values_max",
    "policy_entropy",
)


def _to_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _last_numeric(values):
    if not isinstance(values, list):
        return _to_float(values)
    for value in reversed(values):
        numeric_value = _to_float(value)
        if numeric_value is not None:
            return numeric_value
    return None


def _last_indicator(indicators):
    if isinstance(indicators, list) and indicators:
        last_item = indicators[-1]
        if isinstance(last_item, dict):
            return last_item
    return {}


def _list_chart_files(directory_name):
    if not os.path.isdir(directory_name):
        return []
    return sorted(
        file_name
        for file_name in os.listdir(directory_name)
        if str(file_name).lower().endswith(".png")
    )


def _relative_training_dir(training_directory, directory_name):
    base_path = os.path.abspath(str(training_directory or ""))
    target_path = os.path.abspath(str(directory_name or ""))
    relative_dir = os.path.relpath(target_path, base_path)
    return relative_dir.replace(os.sep, "/")


def _collect_single_agent_metrics(train_metrics, train_indicators):
    latest_indicator = _last_indicator(train_indicators)
    latest_metrics = {}

    for metric_key in CORE_METRIC_KEYS:
        metric_value = _last_numeric(train_metrics.get(metric_key, []))
        if metric_value is not None:
            latest_metrics[metric_key] = metric_value

    for metric_key in INDICATOR_METRIC_KEYS:
        metric_value = _to_float(latest_indicator.get(metric_key))
        if metric_value is not None:
            latest_metrics[metric_key] = metric_value

    episodes_completed = _to_int(
        latest_indicator.get("episode"),
        default=len(train_metrics.get("accuracy", [])) if isinstance(train_metrics, dict) else 0,
    )
    steps_last_episode = _to_int(latest_indicator.get("steps"), default=0)

    return {
        "latest_metrics": latest_metrics,
        "episodes_completed": episodes_completed,
        "steps_last_episode": steps_last_episode,
    }


def _mean_metric(metric_maps, metric_key):
    values = []
    for metric_map in metric_maps:
        metric_value = _to_float(metric_map.get(metric_key))
        if metric_value is not None:
            values.append(metric_value)
    if not values:
        return None
    return float(sum(values) / len(values))


def build_agent_training_summary(
    config,
    agent_name,
    directory_name,
    train_metrics=None,
    train_indicators=None,
    train_execution_time=None,
):
    train_metrics = train_metrics if isinstance(train_metrics, dict) else {}
    train_indicators = train_indicators if isinstance(train_indicators, (dict, list)) else {}

    summary = {
        "agent_name": agent_name,
        "relative_dir": _relative_training_dir(config.training_directory, directory_name),
        "chart_files": _list_chart_files(directory_name),
        "train_execution_time": _to_float(train_execution_time),
        "latest_metrics": {},
        "episodes_completed": 0,
        "steps_last_episode": 0,
        "per_host_metrics": [],
    }
    summary["chart_count"] = len(summary["chart_files"])

    is_multi_host = bool(train_metrics) and all(isinstance(value, dict) for value in train_metrics.values())

    if is_multi_host:
        per_host_metrics = []
        for host_name, host_metrics in train_metrics.items():
            host_indicators = train_indicators.get(host_name, []) if isinstance(train_indicators, dict) else []
            host_summary = _collect_single_agent_metrics(host_metrics, host_indicators)
            host_summary["host"] = host_name
            per_host_metrics.append(host_summary)

        summary["per_host_metrics"] = per_host_metrics
        summary["episodes_completed"] = max(
            (item.get("episodes_completed", 0) for item in per_host_metrics),
            default=0,
        )
        summary["steps_last_episode"] = max(
            (item.get("steps_last_episode", 0) for item in per_host_metrics),
            default=0,
        )

        aggregated_metrics = {}
        for metric_key in CORE_METRIC_KEYS + INDICATOR_METRIC_KEYS:
            mean_value = _mean_metric(
                [item.get("latest_metrics", {}) for item in per_host_metrics],
                metric_key,
            )
            if mean_value is not None:
                aggregated_metrics[metric_key] = mean_value
        summary["latest_metrics"] = aggregated_metrics
        return summary

    single_summary = _collect_single_agent_metrics(train_metrics, train_indicators)
    summary["latest_metrics"] = single_summary["latest_metrics"]
    summary["episodes_completed"] = single_summary["episodes_completed"]
    summary["steps_last_episode"] = single_summary["steps_last_episode"]
    return summary
