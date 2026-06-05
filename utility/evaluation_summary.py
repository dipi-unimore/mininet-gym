import os


CORE_METRIC_KEYS = ("accuracy", "precision", "recall", "f1_score")
SHARED_OVERVIEW_PREFIXES = (
    "test_episodes",
    "test_quadratic_errors",
    "scores",
    "metrics_and_scores",
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


def _relative_training_dir(training_directory, directory_name):
    base_path = os.path.abspath(str(training_directory or ""))
    target_path = os.path.abspath(str(directory_name or ""))
    return os.path.relpath(target_path, base_path).replace(os.sep, "/")


def _list_chart_files(directory_name):
    if not os.path.isdir(directory_name):
        return []
    return sorted(
        file_name
        for file_name in os.listdir(directory_name)
        if str(file_name).lower().endswith(".png")
    )


def _is_shared_overview_chart(file_name):
    lower_name = str(file_name or "").lower()
    return any(lower_name.startswith(prefix) for prefix in SHARED_OVERVIEW_PREFIXES)


def _filter_chart_files_for_agent(chart_files, agent_name, shared_directory):
    if not shared_directory:
        return sorted(chart_files)

    filtered = []
    agent_token = str(agent_name or "").lower()
    for file_name in chart_files:
        lower_name = str(file_name or "").lower()
        if agent_token and agent_token in lower_name:
            filtered.append(file_name)
            continue
        if _is_shared_overview_chart(lower_name):
            filtered.append(file_name)
    return sorted(set(filtered))


def _build_latest_metrics(metrics):
    latest_metrics = {}
    if not isinstance(metrics, dict):
        return latest_metrics
    for metric_key in CORE_METRIC_KEYS:
        metric_value = _to_float(metrics.get(metric_key))
        if metric_value is not None:
            latest_metrics[metric_key] = metric_value
    return latest_metrics


def _build_mitigation_summary(mitigation_history):
    if not isinstance(mitigation_history, list) or not mitigation_history:
        return None

    total_under_attack = 0
    ratios = []
    for item in mitigation_history:
        if not isinstance(item, dict):
            continue
        total_under_attack += _to_int(item.get("under_attack_count"), default=0)
        ratio_value = item.get("mitigated_under_attack_ratio", None)
        if ratio_value is None:
            under_attack = _to_int(item.get("under_attack_count"), default=0)
            mitigated = _to_int(item.get("mitigated_under_attack_count"), default=0)
            ratio_value = float(min(mitigated, under_attack) / under_attack) if under_attack > 0 else 0.0
        ratios.append(_to_float(ratio_value) or 0.0)

    ratio = float(sum(ratios) / len(ratios)) if ratios else 0.0
    return {
        "episodes_with_mitigation_data": len(mitigation_history),
        "attack_episodes": total_under_attack,
        "mitigation_ratio": ratio,
        "total_under_attack_count": total_under_attack,
        "mitigated_under_attack_ratio": ratio,
    }


def build_agent_evaluation_summary(
    config,
    agent_name,
    directory_name,
    metrics=None,
    score=None,
    test_episodes=None,
    mitigation_history=None,
    shared_directory=False,
):
    all_chart_files = _list_chart_files(directory_name)
    chart_files = _filter_chart_files_for_agent(
        all_chart_files,
        agent_name=agent_name,
        shared_directory=shared_directory,
    )

    summary = {
        "agent_name": agent_name,
        "relative_dir": _relative_training_dir(config.training_directory, directory_name),
        "chart_files": chart_files,
        "chart_count": len(chart_files),
        "latest_metrics": _build_latest_metrics(metrics),
        "score": score,
        "test_episodes": _to_int(test_episodes, default=0),
        "shared_directory": bool(shared_directory),
    }

    mitigation_summary = _build_mitigation_summary(mitigation_history)
    if mitigation_summary:
        summary["mitigation_summary"] = mitigation_summary

    return summary
