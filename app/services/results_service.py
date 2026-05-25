import datetime
import io
import os
import shutil
from collections import Counter

from itsdangerous import exc
import numpy as np
import yaml
from fpdf import FPDF
from PIL import Image as PILImage
from flask import send_file

from reinforcement_learning.scenarios.marl.constants import COORDINATOR
from utility.constants import ALGO_A2C, ALGO_DQN, ALGO_PPO, ALGO_Q_LEARNING, ALGO_SARSA, ALGO_SUPERVISED, ATTACKS, ATTACKS_FROM_DATASET, ATTACKS_HO, CLASSIFICATION, CLASSIFICATION_FROM_DATASET, FROM_DATASET, MARL_ATTACKS, MARL_ATTACKS_FROM_DATASET
from utility.my_files import read_data_file
from utility.network_configurator import get_host_agents_by_network_config
from utility.params import read_config_file
from utility.my_log import information, debug, error
from utility.scenario_generator import (
    DEFAULT_ATTACK_LIKELY_EVAL,
    DEFAULT_ATTACK_LIKELY_TRAIN,
    preview_scenario_statistics_from_config,
)
from test_tools.qtable_coverage import plot_qtable_coverage_dynamic_actions_from_statuses


APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_net_param(net_params_dict, key_new, key_old, default=''):
    """
    Helper function to handle backward compatibility for net_params.
    Tries the new key first, then falls back to the old key.
    
    Args:
        net_params_dict: Dictionary of net parameters
        key_new: New parameter key name (e.g., 'num_iots')
        key_old: Old parameter key name (e.g., 'num_iot') for backward compatibility
        default: Default value if neither key is found
    
    Returns:
        The parameter value or default
    """
    return net_params_dict.get(key_new) or net_params_dict.get(key_old, default)


def _add_incomplete(incomplete_list, str_date, relative_path, reason):
    incomplete_list.append({
        "datetime": str_date,
        "path": relative_path,
        "reason": reason,
    })


def _get_candidate_test_dirs(dir_path, test_dir):
    candidate_test_dirs = []
    default_test_dir_path = os.path.join(dir_path, test_dir)
    if os.path.isdir(default_test_dir_path):
        candidate_test_dirs.append(default_test_dir_path)
    for test_candidate in os.listdir(dir_path):
        test_candidate_path = os.path.join(dir_path, test_candidate)
        if test_candidate.startswith(f"{test_dir}_") and os.path.isdir(test_candidate_path):
            candidate_test_dirs.append(test_candidate_path)
    return candidate_test_dirs


def _collect_test_results(candidate_test_dirs, dir_path, test_dir, gym_type_in_config, agent_names):
    merged_test_scores = {}
    test_charts = []
    detected_test_episodes = []
    mitigation_entries = []

    for test_dir_path in candidate_test_dirs:
        test_file_path = os.path.join(test_dir_path, "test.json")
        if not os.path.isfile(test_file_path):
            continue

        test_data = read_data_file(test_file_path)
        if "ground_truth" in test_data:
            gt = test_data.get("ground_truth")
            if gym_type_in_config.startswith(MARL_ATTACKS):
                if isinstance(gt, dict):
                    detected_test_episodes.append(len(gt.get(COORDINATOR, [])))
            elif isinstance(gt, (list, dict)):
                detected_test_episodes.append(len(gt))

        score_dict = test_data.get("score", None)
        if isinstance(score_dict, dict):
            merged_test_scores.update(score_dict)
        else:
            metrics = test_data.get("metrics", {}) if isinstance(test_data.get("metrics", {}), dict) else {}
            score_value = metrics.get("score", score_dict)
            if score_value is not None:
                test_agent_name = ""
                predicted = test_data.get("predicted", {})
                if isinstance(predicted, dict) and predicted:
                    test_agent_name = next(iter(predicted.keys()))
                if not test_agent_name:
                    test_folder_name = os.path.basename(test_dir_path)
                    if test_folder_name.startswith(f"{test_dir}_"):
                        test_agent_name = test_folder_name[len(f"{test_dir}_"):]
                if not test_agent_name and len(agent_names) == 1:
                    test_agent_name = agent_names[0]
                if test_agent_name:
                    merged_test_scores[test_agent_name] = score_value

        history = test_data.get("mitigation_history", [])
        if isinstance(history, list):
            for entry in history:
                if isinstance(entry, dict):
                    mitigation_entries.append(entry)

        test_folder_rel = os.path.relpath(test_dir_path, dir_path)
        for file_name in os.listdir(test_dir_path):
            if file_name.endswith(".png"):
                if test_folder_rel == test_dir:
                    test_charts.append(file_name)
                else:
                    test_charts.append(f"{test_folder_rel}/{file_name}")

    mitigation_summary = _summarize_mitigation_entries(mitigation_entries)
    return merged_test_scores, test_charts, detected_test_episodes, mitigation_summary


def _summarize_mitigation_entries(mitigation_entries):
    total_under_attack = 0
    total_mitigated = 0

    for entry in mitigation_entries:
        try:
            under_attack = int(entry.get("under_attack_count", 0))
        except (TypeError, ValueError):
            under_attack = 0
        try:
            mitigated = int(entry.get("mitigated_under_attack_count", 0))
        except (TypeError, ValueError):
            mitigated = 0

        if under_attack < 0:
            under_attack = 0
        if mitigated < 0:
            mitigated = 0

        total_under_attack += under_attack
        total_mitigated += min(mitigated, under_attack)

    ratio = float(total_mitigated / total_under_attack) if total_under_attack > 0 else 0.0
    return {
        "episodes_with_mitigation_data": len(mitigation_entries),
        "total_under_attack_count": int(total_under_attack),
        "total_mitigated_under_attack_count": int(total_mitigated),
        "mitigated_under_attack_ratio": ratio,
    }


def _compute_test_score_stats(test_scores, gym_type_in_config, test_episodes):
    min_score = test_episodes
    name_min_score = ""
    max_score = 0
    name_max_score = ""
    mean_scores = []

    for agent_name, agent_score in test_scores.items():
        if gym_type_in_config.startswith(MARL_ATTACKS) and isinstance(agent_score, dict):
            host_scores = list(agent_score.values())
            mean_agent_score = float(np.mean(host_scores)) if host_scores else 0
        else:
            try:
                mean_agent_score = float(agent_score)
            except (TypeError, ValueError):
                continue
        if mean_agent_score < min_score:
            min_score = mean_agent_score
            name_min_score = agent_name
        if mean_agent_score > max_score:
            max_score = mean_agent_score
            name_max_score = agent_name
        mean_scores.append(mean_agent_score)

    mean_score = float(np.mean(mean_scores)) if mean_scores else 0
    return min_score, name_min_score, max_score, name_max_score, mean_score


def _find_missing_expected_file(dir_path, expected_files):
    for expected_file in expected_files:
        if not os.path.isfile(os.path.join(dir_path, expected_file)):
            return expected_file
    return None


def _collect_agents_data(
    dir_path,
    agent_names,
    data_file_in_agent_folder,
    print_training_chart,
    expected_files_in_agent_folder_print_training_chart_enabled,
    gym_type_in_config,
    training_episodes,
    max_steps,
):
    agents_data = []
    all_accuracies = []
    min_accuracy = 0
    name_min_accuracy = ""
    max_accuracy = 0
    name_max_accuracy = ""

    for agent_name in agent_names:
        agent_path = os.path.join(dir_path, agent_name)
        if not os.path.isdir(agent_path):
            return False, f"Missing agent folder: {agent_name}", None

        if print_training_chart and not gym_type_in_config.startswith(MARL_ATTACKS):
            for expected_chart in expected_files_in_agent_folder_print_training_chart_enabled:
                if not os.path.isfile(os.path.join(agent_path, expected_chart)):
                    return False, f"Missing training chart file in {agent_name}: {expected_chart}", None

        if not os.path.isfile(os.path.join(agent_path, data_file_in_agent_folder)):
            return False, f"Missing data.json in {agent_name}", None

        data = read_data_file(os.path.join(agent_path, data_file_in_agent_folder))
        if gym_type_in_config.startswith(MARL_ATTACKS):
            accuracy_values = []
            for _, metrics in data.get("train_metrics", {}).items():
                accuracy_values.append(float(np.mean(metrics.get("accuracy", 0))))
            accuracy = float(np.mean(accuracy_values)) if accuracy_values else 0
            training_episodes = len(data.get("ground_truth", {}).get(COORDINATOR, [])) if "ground_truth" in data else training_episodes
            max_steps = data["train_indicators"][COORDINATOR][0]["steps"]
        else:
            max_steps = data["train_indicators"][0]["steps"]
            training_episodes = len(data.get("ground_truth", {})) if "ground_truth" in data else training_episodes
            accuracy = float(np.mean(data.get("train_metrics", {}).get("accuracy", 0))) if data else 0

        if accuracy == 0:
            return False, f"Accuracy is zero or missing for {agent_name}", None

        charts = []
        if print_training_chart:
            for file_name in os.listdir(agent_path):
                if file_name.endswith(".png"):
                    charts.append(file_name)

        agents_data.append({
            "agent_name": agent_name,
            "accuracy": accuracy,
            "charts": charts,
            "print_training_chart": print_training_chart,
        })

        if accuracy > min_accuracy:
            min_accuracy = accuracy
            name_min_accuracy = agent_name
        if accuracy < max_accuracy:
            max_accuracy = accuracy
            name_max_accuracy = agent_name
        all_accuracies.append(accuracy)

    return True, "", {
        "agents_data": agents_data,
        "mean_accuracy": float(np.mean(all_accuracies)) if all_accuracies else 0,
        "min_accuracy": min_accuracy,
        "name_min_accuracy": name_min_accuracy,
        "max_accuracy": max_accuracy,
        "name_max_accuracy": name_max_accuracy,
        "training_episodes": training_episodes,
        "max_steps": max_steps,
    }


def _safe_text(value):
    return str(value).encode("latin-1", "replace").decode("latin-1")


def _training_base_abs_path(current_config):
    training_directory = current_config.get("training_directory", "_training")
    if os.path.isabs(training_directory):
        return os.path.abspath(training_directory)
    return os.path.abspath(os.path.join(APP_ROOT, "..", training_directory))


def _resolve_result_abs_path(current_config, relative_result_path):
    base_path = _training_base_abs_path(current_config)
    normalized = os.path.normpath(str(relative_result_path or "")).lstrip("/")
    result_abs_path = os.path.abspath(os.path.join(base_path, normalized))
    if not result_abs_path.startswith(base_path + os.sep) and result_abs_path != base_path:
        raise ValueError("Invalid result path")
    return result_abs_path


def delete_result_dir(current_config, relative_result_path):
    if not relative_result_path:
        return {"status": "error", "message": "Missing result path"}, 400

    try:
        result_abs_path = _resolve_result_abs_path(current_config, relative_result_path)
        base_path = _training_base_abs_path(current_config)
    except ValueError:
        return {"status": "error", "message": "Invalid result path"}, 400

    # Never allow deleting the whole training root.
    if os.path.abspath(result_abs_path) == os.path.abspath(base_path):
        return {"status": "error", "message": "Refusing to delete training root"}, 400

    if not os.path.exists(result_abs_path):
        return {"status": "error", "message": "Result path not found"}, 404

    if not os.path.isdir(result_abs_path):
        return {"status": "error", "message": "Result path is not a directory"}, 400

    try:
        shutil.rmtree(result_abs_path)
    except Exception as exc:
        return {"status": "error", "message": f"Unable to delete result path: {exc}"}, 500

    return {"status": "success", "message": "Result directory deleted"}, 200


def _pdf_usable_width(pdf):
    return max(1, pdf.w - pdf.l_margin - pdf.r_margin)


def _pdf_fit_image_size(image_path, max_w, max_h):
    try:
        with PILImage.open(image_path) as img:
            img_w, img_h = img.size
        if img_w <= 0 or img_h <= 0:
            return max_w, max_h
        scale = min(max_w / img_w, max_h / img_h)
        return img_w * scale, img_h * scale
    except Exception:
        return max_w, max_h


def _pdf_section_title(pdf, title):
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_fill_color(34, 97, 149)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(_pdf_usable_width(pdf), 8, _safe_text(title), ln=True, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)


def _pdf_summary_cards(pdf, cards):
    usable_width = _pdf_usable_width(pdf)
    gap = 4
    col_width = (usable_width - gap) / 2
    card_height = 18
    x_left = pdf.l_margin
    x_right = pdf.l_margin + col_width + gap

    row_y = pdf.get_y()
    for index, card in enumerate(cards):
        if len(card) >= 3:
            label, value, value_color = card[0], card[1], card[2]
        else:
            label, value = card[0], card[1]
            value_color = (0, 0, 0)

        if index % 2 == 0:
            row_y = pdf.get_y()
            if row_y + card_height > pdf.page_break_trigger:
                pdf.add_page()
                row_y = pdf.get_y()

        x = x_left if index % 2 == 0 else x_right
        pdf.set_fill_color(245, 248, 252)
        pdf.set_draw_color(206, 214, 224)
        pdf.rect(x, row_y, col_width, card_height)
        pdf.set_xy(x + 2, row_y + 2)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(col_width - 4, 4, _safe_text(label), ln=True)
        pdf.set_x(x + 2)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*value_color)
        pdf.multi_cell(col_width - 4, 4, _safe_text(value))
        pdf.set_text_color(0, 0, 0)

        if index % 2 == 1 or index == len(cards) - 1:
            pdf.set_y(row_y + card_height + 3)


def _mitigation_ratio_color(ratio):
    if ratio >= 0.8:
        return (22, 163, 74)
    if ratio >= 0.5:
        return (202, 138, 4)
    return (220, 38, 38)


def _pdf_image_card(pdf, image_path, caption, x, y, width, height):
    pdf.set_draw_color(210, 215, 224)
    pdf.set_fill_color(255, 255, 255)
    pdf.rect(x, y, width, height, style="D")

    inner_margin = 2
    pdf.set_xy(x + inner_margin, y + 1)
    pdf.set_font("Helvetica", "B", 8)
    pdf.multi_cell(width - inner_margin * 2, 4, _safe_text(caption))

    image_top = pdf.get_y() + 1
    image_max_w = width - inner_margin * 2
    image_max_h = max(8, height - (image_top - y) - 2)

    if not os.path.isfile(image_path):
        pdf.set_xy(x + inner_margin, image_top)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(160, 50, 50)
        pdf.multi_cell(image_max_w, 4, _safe_text("Missing image"))
        pdf.set_text_color(0, 0, 0)
        return

    img_w, img_h = _pdf_fit_image_size(image_path, image_max_w, image_max_h)
    img_x = x + (width - img_w) / 2
    img_y = image_top + max(0, (image_max_h - img_h) / 2)
    try:
        pdf.image(image_path, x=img_x, y=img_y, w=img_w, h=img_h)
    except Exception:
        pdf.set_xy(x + inner_margin, image_top)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(160, 50, 50)
        pdf.multi_cell(image_max_w, 4, _safe_text("Unable to render image"))
        pdf.set_text_color(0, 0, 0)


def _pdf_image_grid(pdf, items, title):
    if not items:
        return

    _pdf_section_title(pdf, title)
    usable_width = _pdf_usable_width(pdf)
    gap = 5
    col_width = (usable_width - gap) / 2
    row_height = 72
    x_left = pdf.l_margin
    x_right = pdf.l_margin + col_width + gap

    index = 0
    while index < len(items):
        if pdf.get_y() + row_height > pdf.page_break_trigger:
            pdf.add_page()
            _pdf_section_title(pdf, title)

        row_y = pdf.get_y()
        _pdf_image_card(pdf, items[index][0], items[index][1], x_left, row_y, col_width, row_height)
        index += 1

        if index < len(items):
            _pdf_image_card(pdf, items[index][0], items[index][1], x_right, row_y, col_width, row_height)
            index += 1

        pdf.set_y(row_y + row_height + 4)


def build_result_pdf_response(current_config, gym_type, result_path, data):
    if not result_path:
        return {"status": "error", "message": "Missing result path"}, 400

    try:
        result_abs_path = _resolve_result_abs_path(current_config, result_path)
    except ValueError:
        return {"status": "error", "message": "Invalid result path"}, 400

    if not os.path.isdir(result_abs_path):
        return {"status": "error", "message": "Result directory not found"}, 404

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.set_margins(12, 12, 12)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(34, 97, 149)
    pdf.cell(0, 10, _safe_text("Result Report"), ln=True, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, _safe_text(f"Scenario: {gym_type}"), ln=True)
    pdf.cell(0, 6, _safe_text(f"Path: {result_path}"), ln=True)
    pdf.cell(0, 6, _safe_text(f"Date: {data.get('datetime', '-') }"), ln=True)
    pdf.ln(2)

    _pdf_section_title(pdf, "Summary")
    mitigation_summary = data.get("mitigation_summary", {}) if isinstance(data.get("mitigation_summary", {}), dict) else {}
    mitigation_ratio = float(mitigation_summary.get("mitigated_under_attack_ratio", 0) or 0)
    summary_cards = [
        ("Network config", data.get("network_config", "-")),
        ("Training episodes", data.get("training_episodes", "-")),
        ("Max steps", data.get("max_steps", "-")),
        ("Test episodes", data.get("test_episodes", "-")),
        ("Mean accuracy", f"{round(float(data.get('mean_accuracy', 0)) * 100, 2)}%"),
        ("Mean score", data.get("mean_score", "-")),
        ("Agents", len(data.get("agents_data", []) if isinstance(data.get("agents_data", []), list) else [])),
        ("Test charts", len(data.get("test_charts", []) if isinstance(data.get("test_charts", []), list) else [])),
    ]
    if mitigation_summary:
        mitigation_ratio_color = _mitigation_ratio_color(mitigation_ratio)
        summary_cards.extend([
            ("Mitigation episodes", mitigation_summary.get("episodes_with_mitigation_data", 0)),
            ("Under attack total", mitigation_summary.get("total_under_attack_count", 0)),
            ("Mitigated total", mitigation_summary.get("total_mitigated_under_attack_count", 0)),
            ("Mitigation ratio", f"{round(mitigation_ratio * 100, 2)}%", mitigation_ratio_color),
        ])
    _pdf_summary_cards(pdf, summary_cards)

    agents_data = data.get("agents_data", []) if isinstance(data.get("agents_data", []), list) else []
    test_scores = data.get("test_scores", {}) if isinstance(data.get("test_scores", {}), dict) else {}
    if agents_data:
        agent_cards = []
        for agent in agents_data:
            name = agent.get("agent_name", "unknown")
            charts = agent.get("charts", []) if isinstance(agent.get("charts", []), list) else []
            accuracy = round(float(agent.get("accuracy", 0)) * 100, 2)
            score = test_scores.get(name, "-")
            for chart in charts:
                agent_cards.append((
                    os.path.join(result_abs_path, name, chart),
                    f"{name} | acc {accuracy}% | score {score} | {chart}",
                ))

        if agent_cards:
            pdf.add_page()
            _pdf_image_grid(pdf, agent_cards, "Agents")

    general_files = data.get("files", []) if isinstance(data.get("files", []), list) else []
    general_images = []
    for image_name in general_files:
        if str(image_name).lower().endswith(".png"):
            general_images.append((os.path.join(result_abs_path, image_name), f"General chart: {image_name}"))
    if general_images:
        pdf.add_page()
        _pdf_image_grid(pdf, general_images, "General Charts")

    test_charts = data.get("test_charts", []) if isinstance(data.get("test_charts", []), list) else []
    if test_charts:
        test_images = []
        for test_img in test_charts:
            test_img = str(test_img)
            image_path = os.path.join(result_abs_path, test_img) if "/" in test_img else os.path.join(result_abs_path, "TEST", test_img)
            test_images.append((image_path, f"Test chart: {test_img}"))
        pdf.add_page()
        _pdf_image_grid(pdf, test_images, "Test Charts")

    output_raw = pdf.output(dest="S")
    output_bytes = output_raw.encode("latin-1", "ignore") if isinstance(output_raw, str) else bytes(output_raw)

    safe_name = str(result_path).replace("/", "_").replace(" ", "_")
    filename = f"result_report_{safe_name}.pdf"
    return send_file(
        io.BytesIO(output_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename,
    )


RESULT_GYM_TYPES = [MARL_ATTACKS, ATTACKS, ATTACKS_HO, CLASSIFICATION, MARL_ATTACKS_FROM_DATASET, ATTACKS_FROM_DATASET, CLASSIFICATION_FROM_DATASET]


def build_results_dir_list(current_config):
    from utility.my_log import information, debug, error as log_error
    
    results_dir_list = []
    expected_files = ["log.txt", "metrics_comparison.png", "radar_chart.png", "statuses.json"]
    data_file_in_agent_folder = "data.json"
    test_dir = "TEST"
    expected_files_in_agent_folder_print_training_chart_enabled = ["matrix.png", "metrics.png", "metrics_combined.png", "rewards.png"]
    training_directory = current_config.get("training_directory", None)
    
    debug(f"[BUILD_RESULTS] Starting build_results_dir_list with training_directory: {training_directory}")

    if training_directory and os.path.isdir(training_directory):
        for gym_type in RESULT_GYM_TYPES:
            path = os.path.join(training_directory, gym_type)
            if not os.path.isdir(path):
                continue

            datas_gym_type = []
            incomplete_datas_gym_type = []

            for d in os.listdir(path):
                dir_path = os.path.join(path, d)
                if not os.path.isdir(dir_path):
                    continue

                str_date = d.split("_")[0]
                relative_path = dir_path.replace(f"{training_directory}/", "")

                try:
                    datetime.datetime.strptime(str_date, "%Y%m%d-%H%M%S")
                except ValueError:
                    _add_incomplete(incomplete_datas_gym_type, str_date, relative_path, "Invalid datetime format in directory name")
                    continue
                try:
                    _, config_yaml = read_config_file(os.path.join(dir_path, "config.yaml"))
                except FileNotFoundError:
                    _add_incomplete(incomplete_datas_gym_type, str_date, relative_path, "Missing config.yaml")
                    continue    
                env_params = config_yaml.get("env_params", None)
                if env_params is None:
                    _add_incomplete(incomplete_datas_gym_type, str_date, relative_path, "Missing env_params in config.yaml")
                    continue

                gym_type_in_config = env_params.get("gym_type", "")
                if gym_type_in_config != gym_type:
                    _add_incomplete(incomplete_datas_gym_type, str_date, relative_path, f"Gym type mismatch: expected {gym_type}, found {gym_type_in_config}")
                    continue

                print_training_chart = env_params.get("print_training_chart", False)
                net_params = env_params.get("net_params", None)
                if net_params is None:
                    _add_incomplete(incomplete_datas_gym_type, str_date, relative_path, "Missing net_params in config.yaml")
                    continue

                training_episodes = env_params.get("episodes", 0)
                max_steps = env_params.get("max_steps", 0)
                test_episodes = env_params.get("test_episodes", 0)
                # Convert Params object to dict to access net_params values
                net_params_dict = net_params.__dict__ if hasattr(net_params, '__dict__') else net_params
                # Use helper for backward compatibility: try num_iots first, then num_iot                
                network_config = f"{_get_net_param(net_params_dict, 'num_switches', 0)}_{_get_net_param(net_params_dict, 'num_hosts', 0)}_{_get_net_param(net_params_dict, 'num_iots',  'num_iot',0)}"
                agents = config_yaml.get("agents", [])
                agent_names = [
                    agent.get("name", "")
                    for agent in agents
                    if agent.get("enabled", False)
                    and not agent.get("skip_learn", True)
                    and not agent.get("name", "").startswith(ALGO_SUPERVISED)
                ]
                if not agent_names:
                    _add_incomplete(incomplete_datas_gym_type, str_date, relative_path, "No enabled learning agents found in config")
                    continue

                data_gym_type = {
                    "network_config": network_config,
                    "datetime": str_date,
                    "training_episodes": training_episodes,
                    "max_steps": max_steps,
                    "test_episodes": test_episodes,
                }

                agents_ok, agents_reason, agents_summary = _collect_agents_data(
                    dir_path=dir_path,
                    agent_names=agent_names,
                    data_file_in_agent_folder=data_file_in_agent_folder,
                    print_training_chart=print_training_chart,
                    expected_files_in_agent_folder_print_training_chart_enabled=expected_files_in_agent_folder_print_training_chart_enabled,
                    gym_type_in_config=gym_type_in_config,
                    training_episodes=training_episodes,
                    max_steps=max_steps,
                )
                if not agents_ok:
                    _add_incomplete(incomplete_datas_gym_type, str_date, relative_path, agents_reason or "Agent artifacts are incomplete")
                    continue

                data_gym_type["training_episodes"] = agents_summary["training_episodes"]
                data_gym_type["max_steps"] = agents_summary["max_steps"]
                data_gym_type["mean_accuracy"] = agents_summary["mean_accuracy"]
                data_gym_type["min_accuracy"] = agents_summary["min_accuracy"]
                data_gym_type["name_min_accuracy"] = agents_summary["name_min_accuracy"]
                data_gym_type["max_accuracy"] = agents_summary["max_accuracy"]
                data_gym_type["name_max_accuracy"] = agents_summary["name_max_accuracy"]
                data_gym_type["agents_data"] = agents_summary["agents_data"]

                candidate_test_dirs = _get_candidate_test_dirs(dir_path, test_dir)
                if not candidate_test_dirs:
                    _add_incomplete(incomplete_datas_gym_type, str_date, relative_path, "Missing test directory: expected TEST or TEST_<algorithm>")
                    continue

                merged_test_scores, test_charts, detected_test_episodes, mitigation_summary = _collect_test_results(
                    candidate_test_dirs=candidate_test_dirs,
                    dir_path=dir_path,
                    test_dir=test_dir,
                    gym_type_in_config=gym_type_in_config,
                    agent_names=agent_names,
                )
                if not merged_test_scores:
                    _add_incomplete(incomplete_datas_gym_type, str_date, relative_path, "No valid test scores found in test.json files")
                    continue

                min_score, name_min_score, max_score, name_max_score, mean_score = _compute_test_score_stats(
                    test_scores=merged_test_scores,
                    gym_type_in_config=gym_type_in_config,
                    test_episodes=test_episodes,
                )
                if detected_test_episodes:
                    test_episodes = max(detected_test_episodes)

                data_gym_type["test_scores"] = merged_test_scores
                data_gym_type["test_episodes"] = test_episodes
                data_gym_type["mean_score"] = mean_score
                data_gym_type["min_score"] = min_score
                data_gym_type["name_min_score"] = name_min_score
                data_gym_type["name_max_score"] = name_max_score
                data_gym_type["max_score"] = max_score
                data_gym_type["test_charts"] = test_charts
                data_gym_type["mitigation_summary"] = mitigation_summary

                missing_expected_file = _find_missing_expected_file(dir_path, expected_files)
                if missing_expected_file is not None:
                    _add_incomplete(incomplete_datas_gym_type, str_date, relative_path, f"Missing experiment file: {missing_expected_file}")
                    continue

                # _ensure_qtable_coverage_dynamic_actions_preview(current_config, dir_path)

                # Collect all root-level PNG files (preserving expected_files order first)
                all_root_pngs = [f for f in expected_files if f.lower().endswith(".png")]
                for fname in sorted(os.listdir(dir_path)):
                    if fname.lower().endswith(".png") and fname not in all_root_pngs:
                        all_root_pngs.append(fname)
                data_gym_type["files"] = all_root_pngs
                data_gym_type["path"] = relative_path
                datas_gym_type.append(data_gym_type)

            results_dir_list.append({"gym_type": gym_type, "data": datas_gym_type, "incomplete_data": incomplete_datas_gym_type})

    information(f"[BUILD_RESULTS] Found {len(results_dir_list)} gym types with {sum(len(item['data']) for item in results_dir_list)} complete results")
    return results_dir_list


def build_load_dir_list(current_config, gym_type, network_config, agent_name):
    if agent_name.lower().startswith(ALGO_DQN) or agent_name.lower().startswith(ALGO_A2C) or agent_name.lower().startswith(ALGO_PPO):
        extension = "zip"
    elif agent_name.lower().startswith(ALGO_Q_LEARNING) or agent_name.lower().startswith(ALGO_SARSA):
        extension = "json"
    else:
        extension = "zip"

    load_dir_list = []
    gym_type_training_directories = []
    training_directory = current_config.get("training_directory", None)
    gym_type_training_directory = training_directory + f"/{gym_type}"
    gym_type_training_directories.append(gym_type_training_directory)
    if gym_type.endswith(FROM_DATASET):
        gym_type_training_directories.append(gym_type_training_directory.replace(f"_{FROM_DATASET}", ""))
    else:
        gym_type_training_directories.append(gym_type_training_directory + f"_{FROM_DATASET}")

    dir_list = []
    for directory in gym_type_training_directories:
        if os.path.isdir(directory):
            dir_list.extend([
                os.path.join(directory, d)
                for d in os.listdir(directory)
                if network_config in d and os.path.isdir(os.path.join(directory, d))
            ])

    for d in dir_list:
        for da in os.listdir(d):
            path = os.path.join(d, da)
            if da == agent_name and os.path.isdir(path):
                if gym_type.startswith(MARL_ATTACKS):
                    host_agents = get_host_agents_by_network_config(network_config)
                    host_agents.append(COORDINATOR)
                    is_ok = True
                    for host_agent in host_agents:
                        if not os.path.isfile(f"{path}/{agent_name}_{host_agent}.{extension}"):
                            is_ok = False
                            break
                    if not is_ok or not os.path.isfile(f"{path}/data.json"):
                        continue
                    data = read_data_file(f"{path}/data.json")
                    complete_path = path.replace(f"{training_directory}/", "")
                    accuracy = []
                    for _, metrics in data.get("train_metrics", {}).items():
                        accuracy.append(float(np.mean(metrics.get("accuracy", 0))))

                    load_dir_list.append({
                        "accuracy": float(np.mean(accuracy)) if accuracy else 0,
                        "datetime": path.split('/')[2].split('_')[0],
                        "path": complete_path,
                    })
                else:
                    if not os.path.isfile(f"{path}/{agent_name}.{extension}") or not os.path.isfile(f"{path}/data.json"):
                        continue
                    try:
                        data = read_data_file(f"{path}/data.json")
                        complete_path = path.replace(f"{training_directory}/", "") + f"/{agent_name}.{extension}"
                        raw_accuracy = data.get("train_metrics", {}).get("accuracy", []) if data else []
                        acc_val = float(np.mean(raw_accuracy)) if raw_accuracy else 0.0
                        if np.isnan(acc_val) or np.isinf(acc_val):
                            acc_val = 0.0
                        load_dir_list.append({
                            "accuracy": acc_val,
                            "datetime": path.split('/')[2].split('_')[0],
                            "path": complete_path,
                        })
                    except Exception as exc:
                        error(f"Error reading data.json for {path}: {exc}")
                        continue

    return load_dir_list


def _candidate_gym_type_directories(training_directory, gym_type):
    candidates = []
    base = os.path.join(training_directory, gym_type)
    candidates.append(base)
    if gym_type.endswith(FROM_DATASET):
        candidates.append(base.replace(f"_{FROM_DATASET}", ""))
    else:
        candidates.append(base + f"_{FROM_DATASET}")

    seen = set()
    out = []
    for path in candidates:
        path = os.path.abspath(path)
        if path in seen:
            continue
        seen.add(path)
        if os.path.isdir(path):
            out.append(path)
    return out


def _iter_scenario_run_dirs(training_directory, gym_type, network_config):
    for root in _candidate_gym_type_directories(training_directory, gym_type):
        for entry in os.listdir(root):
            run_dir = os.path.join(root, entry)
            if not os.path.isdir(run_dir):
                continue
            if network_config and network_config not in entry:
                continue
            yield run_dir


def _safe_load_json_file(path):
    try:
        return read_data_file(path)
    except Exception:
        return None


def _resolve_statuses_json_path(result_abs_path):
    if not os.path.isdir(result_abs_path):
        return None

    preferred = os.path.join(result_abs_path, "statuses.json")
    if os.path.isfile(preferred):
        return preferred

    for file_name in sorted(os.listdir(result_abs_path)):
        if file_name.startswith("statuses") and file_name.endswith(".json"):
            candidate = os.path.join(result_abs_path, file_name)
            if os.path.isfile(candidate):
                return candidate

    return None


def _ensure_qtable_coverage_dynamic_actions_preview(current_config, result_abs_path, statuses_data=None):
    output_path = os.path.join(result_abs_path, "qtable_coverage_dynamic_actions.png")
    if os.path.isfile(output_path):
        return True

    if statuses_data is None:
        statuses_path = _resolve_statuses_json_path(result_abs_path)
        if not statuses_path:
            return False
        statuses_data = _safe_load_json_file(statuses_path)
        if statuses_data is None:
            return False

    try:
        low, high, n_bins = _resolve_attack_detect_ho_plot_params(current_config)
        plot_qtable_coverage_dynamic_actions_from_statuses(
            statuses_data=statuses_data,
            output_path=output_path,
            low=low,
            high=high,
            n_bins=n_bins,
        )
        return True
    except Exception:
        return False


def _dataset_summary(dataset_data):
    if not isinstance(dataset_data, list) or not dataset_data:
        return {
            "entries": 0,
            "hosts": 0,
            "status_kinds": 0,
            "attack_like_entries": 0,
            "mean_packets": 0,
        }

    first = dataset_data[0] if isinstance(dataset_data[0], dict) else {}
    statuses = []
    packets = []
    attack_like = 0

    for row in dataset_data:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status", ""))
        statuses.append(status)
        pkt = row.get("packets", 0)
        try:
            packets.append(float(pkt))
        except (TypeError, ValueError):
            pass

        status_id = row.get("id", None)
        if status in {"under_attack", "attacking", "attack"} or status_id in {1, 2}:
            attack_like += 1

    return {
        "entries": len(dataset_data),
        "hosts": len(first.get("hostStatusesStructured", {})) if isinstance(first, dict) else 0,
        "status_kinds": len(set(statuses)),
        "attack_like_entries": int(attack_like),
        "mean_packets": round(float(np.mean(packets)), 2) if packets else 0,
    }


def _resolve_preview_bins(current_config, fallback=8):
    raw_n_bins = fallback

    if isinstance(current_config, dict):
        env_params = current_config.get("env_params", {})
        if isinstance(env_params, dict):
            raw_n_bins = env_params.get("n_bins", fallback)
        else:
            raw_n_bins = getattr(env_params, "n_bins", fallback)
    else:
        env_params = getattr(current_config, "env_params", None)
        raw_n_bins = getattr(env_params, "n_bins", fallback)

    try:
        return max(2, int(raw_n_bins))
    except (TypeError, ValueError):
        return max(2, int(fallback))


def _config_value(root, *keys, default=None):
    current = root
    for key in keys:
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            current = getattr(current, key, default)
    return current if current is not None else default


def _resolve_attack_detect_ho_plot_params(current_config):
    env_params = _config_value(current_config, "env_params", default={})
    net_params = _config_value(env_params, "net_params", default={})
    attacks_cfg = _config_value(env_params, "attacks", default={})
    thresholds = _config_value(attacks_cfg, "thresholds", default={})

    n_bins = _resolve_preview_bins(current_config, fallback=4)
    num_hosts = max(1, int(_config_value(net_params, "num_hosts", default=1) or 1))
    threshold_packets = float(_config_value(thresholds, "packets", default=22000) or 22000)
    threshold_var_packets = float(_config_value(thresholds, "var_packets", default=50) or 50)
    threshold_bytes = float(_config_value(thresholds, "bytes", default=423000000) or 423000000)
    threshold_var_bytes = float(_config_value(thresholds, "var_bytes", default=30) or 30)

    low = np.array([
        0,
        -threshold_var_packets,
        0,
        -threshold_var_bytes,
        0,
        -threshold_var_packets,
        0,
        -threshold_var_bytes,
    ], dtype=np.float32)
    high = np.array([
        threshold_packets * num_hosts,
        threshold_var_packets,
        threshold_bytes * num_hosts,
        threshold_var_bytes,
        threshold_packets * num_hosts,
        threshold_var_packets,
        threshold_bytes * num_hosts,
        threshold_var_bytes,
    ], dtype=np.float32)
    return low, high, n_bins


def _compute_traffic_distribution_data(statuses_list, n_bins=8):
    """Compute qtable-like discretized coverage grouped by action -> feature -> bin."""
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

    feature_labels = {
        "receivedPackets": "rx_pkt",
        "transmittedPackets": "tx_pkt",
        "receivedBytes": "rx_bytes",
        "transmittedBytes": "tx_bytes",
        "receivedPacketsPercentageChange": "rx_pkt_%diff",
        "transmittedPacketsPercentageChange": "tx_pkt_%diff",
        "receivedBytesPercentageChange": "rx_bytes_%diff",
        "transmittedBytesPercentageChange": "tx_bytes_%diff",
    }

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
            mapped = id_map.get(int(raw_id))
            if mapped:
                return mapped

        if task_text:
            return task_text
        if status_text:
            return status_text
        return "unknown"

    rows = []
    for status in statuses_list:
        if not isinstance(status, dict):
            continue
        hosts = status.get("hostStatusesStructured", {})
        if not isinstance(hosts, dict):
            continue

        for host_status in hosts.values():
            if not isinstance(host_status, dict):
                continue
            action = normalize_action_label(host_status)
            for feature in features:
                if feature not in host_status:
                    continue
                try:
                    value = float(host_status.get(feature))
                except (TypeError, ValueError):
                    continue
                rows.append({
                    "action": str(action),
                    "feature": feature,
                    "value": value,
                })

    if not rows:
        return {
            "n_bins": int(n_bins),
            "features": features,
            "feature_labels": feature_labels,
            "action_order": [],
            "actions": [],
        }

    # Compute bins per feature using all actions together, exactly like qtable coverage logic.
    n_bins = max(2, int(n_bins))
    feature_values = {feature: [] for feature in features}
    for row in rows:
        feature_values[row["feature"]].append(row["value"])

    feature_bin_edges = {}
    for feature in features:
        values = feature_values[feature]
        if not values:
            feature_bin_edges[feature] = None
            continue

        arr = np.asarray(values, dtype=float)
        if feature in packet_byte_features:
            positive_values = arr[arr > 0]
            if positive_values.size == 0:
                feature_bin_edges[feature] = {"type": "packet_byte", "positive_edges": None}
                continue

            transformed = np.log1p(positive_values)
            if transformed.size > 1:
                cap = float(np.quantile(transformed, 0.995))
                transformed = np.clip(transformed, a_min=None, a_max=cap)

            positive_bins = max(1, n_bins - 1)
            if np.unique(transformed).size <= 1:
                feature_bin_edges[feature] = {
                    "type": "packet_byte",
                    "positive_edges": "single_value",
                }
                continue

            quantiles = np.linspace(0, 1, positive_bins + 1)
            edges = np.quantile(transformed, quantiles)
            if np.unique(edges).size <= 1:
                feature_bin_edges[feature] = {
                    "type": "packet_byte",
                    "positive_edges": "single_value",
                }
            else:
                feature_bin_edges[feature] = {
                    "type": "packet_byte",
                    "positive_edges": edges.tolist(),
                }
            continue

        vmin = float(arr.min())
        vmax = float(arr.max())
        if vmax == vmin:
            feature_bin_edges[feature] = {"type": "minmax", "vmin": vmin, "vmax": vmax, "single_value": True}
        else:
            feature_bin_edges[feature] = {"type": "minmax", "vmin": vmin, "vmax": vmax, "single_value": False}

    def assign_bin(feature, value):
        rule = feature_bin_edges.get(feature)
        if not rule:
            return 0

        if rule["type"] == "packet_byte":
            if value <= 0:
                return 0
            positive_edges = rule.get("positive_edges")
            if positive_edges == "single_value" or positive_edges is None:
                return 1

            transformed = float(np.log1p(value))
            edges = np.asarray(positive_edges, dtype=float)
            idx = int(np.searchsorted(edges, transformed, side="right") - 1)
            positive_bins = max(1, n_bins - 1)
            idx = max(0, min(idx, positive_bins - 1))
            return idx + 1

        vmin = rule["vmin"]
        vmax = rule["vmax"]
        if rule.get("single_value") or vmax == vmin:
            return 0
        scaled = (value - vmin) / (vmax - vmin)
        b = int(np.floor(scaled * n_bins))
        return max(0, min(b, n_bins - 1))

    preferred_actions = ["normal", "attack_in", "attack_out"]
    available_actions = list({row["action"] for row in rows})
    action_order = [a for a in preferred_actions if a in available_actions]
    if not action_order:
        action_order = sorted(available_actions)

    counts = {
        action: {feature: [0 for _ in range(n_bins)] for feature in features}
        for action in action_order
    }
    for row in rows:
        action = row["action"]
        if action not in counts:
            continue
        feature = row["feature"]
        bin_idx = assign_bin(feature, row["value"])
        counts[action][feature][bin_idx] += 1

    actions_payload = []
    for action in action_order:
        feature_bin_counts = counts[action]
        action_total = int(sum(sum(bin_counts) for bin_counts in feature_bin_counts.values()))
        actions_payload.append({
            "name": action,
            "feature_bin_counts": feature_bin_counts,
            "total": action_total,
        })

    return {
        "n_bins": int(n_bins),
        "features": features,
        "feature_labels": feature_labels,
        "action_order": action_order,
        "actions": actions_payload,
    }


def _summarize_statuses_json(statuses_data):
    statuses_list = []
    if isinstance(statuses_data, dict):
        if isinstance(statuses_data.get("statuses"), list):
            statuses_list = statuses_data.get("statuses", [])
        elif isinstance(statuses_data.get("data"), list):
            statuses_list = statuses_data.get("data", [])
    elif isinstance(statuses_data, list):
        statuses_list = statuses_data

    if not isinstance(statuses_list, list):
        statuses_list = []

    total_entries = len(statuses_list)
    first_entry = statuses_list[0] if total_entries > 0 and isinstance(statuses_list[0], dict) else {}
    sample_size = min(20, total_entries)
    sample_entries = statuses_list[:sample_size]

    host_names = []
    if isinstance(first_entry, dict) and isinstance(first_entry.get("hostStatusesStructured", {}), dict):
        host_names = list(first_entry.get("hostStatusesStructured", {}).keys())

    status_kinds = set()
    attack_like_entries = 0
    total_packets = []
    total_bytes = []

    def _is_attack_like_text(value):
        text = str(value or "").strip().lower()
        return text in {"under_attack", "attacking", "attack", "attack_in", "attack_out", "incoming_attack", "outgoing_attack"}

    for row in statuses_list:
        if not isinstance(row, dict):
            continue

        if row.get("status") is not None:
            status_value = row.get("status")
            if isinstance(status_value, list):
                for item in status_value:
                    status_kinds.add(str(item))
                    if _is_attack_like_text(item):
                        attack_like_entries += 1
            else:
                status_kinds.add(str(status_value))
                if _is_attack_like_text(status_value):
                    attack_like_entries += 1

        if isinstance(row.get("hostStatusesStructured"), dict):
            for host_status in row.get("hostStatusesStructured", {}).values():
                if not isinstance(host_status, dict):
                    continue
                status_kinds.add(str(host_status.get("status", "")))
                if _is_attack_like_text(host_status.get("status")):
                    attack_like_entries += 1
                if host_status.get("id") in {1, 2}:
                    attack_like_entries += 1

        for key, container in (("packets", total_packets), ("bytes", total_bytes)):
            try:
                container.append(float(row.get(key, 0) or 0))
            except (TypeError, ValueError):
                pass

    sample_payload = {
        "sample_size": sample_size,
        "sample": sample_entries,
    }

    return {
        "total_entries": total_entries,
        "sample_size": sample_size,
        "hosts": len(host_names),
        "host_names": host_names,
        "status_kinds": len({kind for kind in status_kinds if kind}),
        "attack_like_entries": int(attack_like_entries),
        "mean_packets": round(float(np.mean(total_packets)), 2) if total_packets else 0,
        "mean_bytes": round(float(np.mean(total_bytes)), 2) if total_bytes else 0,
        "sample_payload": sample_payload,
    }


def build_result_statuses_preview(current_config, result_path, sample_size=20):
    if not result_path:
        return {"status": "error", "message": "Missing result path"}, 400

    try:
        result_abs_path = _resolve_result_abs_path(current_config, result_path)
    except ValueError:
        return {"status": "error", "message": "Invalid result path"}, 400

    statuses_path = _resolve_statuses_json_path(result_abs_path)
    if not statuses_path:
        return {"status": "error", "message": "statuses.json not found in result directory"}, 404

    statuses_data = _safe_load_json_file(statuses_path)
    if statuses_data is None:
        return {"status": "error", "message": "Unable to read statuses.json"}, 500

    # Extract statuses list
    statuses_list = []
    if isinstance(statuses_data, dict):
        if isinstance(statuses_data.get("statuses"), list):
            statuses_list = statuses_data.get("statuses", [])
        elif isinstance(statuses_data.get("data"), list):
            statuses_list = statuses_data.get("data", [])
    elif isinstance(statuses_data, list):
        statuses_list = statuses_data

    preview = _summarize_statuses_json(statuses_data)
    preview["file_name"] = os.path.basename(statuses_path)
    preview["file_path"] = os.path.relpath(statuses_path, _training_base_abs_path(current_config)).replace("\\", "/")
    preview["sample_size"] = min(int(sample_size or 20), preview.get("sample_size", 0) or 0)
    sample_payload = preview.get("sample_payload", {"sample_size": 0, "sample": []})
    sample_payload["sample_size"] = preview["sample_size"]
    sample_payload["sample"] = sample_payload.get("sample", [])[:preview["sample_size"]]
    preview["sample_payload"] = sample_payload
    
    try:
        _ensure_qtable_coverage_dynamic_actions_preview(current_config, result_abs_path, statuses_data=statuses_data)
        output_path = os.path.join(result_abs_path, "qtable_coverage_dynamic_actions.png")
        preview["traffic_distribution_image"] = os.path.relpath(
            output_path,
            _training_base_abs_path(current_config),
        ).replace("\\", "/")
    except Exception as exc:
        preview["traffic_distribution_image_error"] = str(exc)
    
    return preview, 200


def build_dataset_list(current_config, gym_type, network_config):
    training_directory = current_config.get("training_directory", None)
    if not training_directory or not os.path.isdir(training_directory):
        return []

    dataset_rows = []
    training_base_abs = _training_base_abs_path(current_config)
    for run_dir in _iter_scenario_run_dirs(training_directory, gym_type, network_config):
        run_name = os.path.basename(run_dir)
        run_dt = run_name.split("_")[0]

        for file_name in os.listdir(run_dir):
            if not (file_name.startswith("statuses") and file_name.endswith(".json")):
                continue
            dataset_path = os.path.join(run_dir, file_name)
            dataset_data = _safe_load_json_file(dataset_path)
            summary = _dataset_summary(dataset_data)
            dataset_abs_path = os.path.abspath(dataset_path)

            dataset_rows.append({
                "datetime": run_dt,
                "run": run_name,
                "file": file_name,
                "path": dataset_abs_path,
                "relative_path": os.path.relpath(dataset_abs_path, training_base_abs).replace("\\", "/"),
                "summary": summary,
            })

    dataset_rows.sort(key=lambda row: row.get("datetime", ""), reverse=True)
    return dataset_rows


def _extract_scenario_summary(data):
    if not isinstance(data, dict):
        return {
            "train_steps": 0,
            "train_episodes": 0,
            "train_max_steps": 0,
            "eval_steps": 0,
            "eval_episodes": 0,
            "attack_likely_used": None,
        }

    statistics = data.get("statistics", {}) if isinstance(data.get("statistics", {}), dict) else {}
    train_stats = statistics.get("training", {}) if isinstance(statistics.get("training", {}), dict) else {}
    eval_stats = statistics.get("evaluation", {}) if isinstance(statistics.get("evaluation", {}), dict) else {}

    train_steps = train_stats.get("total_steps")
    if train_steps is None:
        train_steps = len(data.get("training", [])) if isinstance(data.get("training", []), list) else 0

    eval_steps = eval_stats.get("total_steps")
    if eval_steps is None:
        eval_steps = len(data.get("evaluation", [])) if isinstance(data.get("evaluation", []), list) else 0

    return {
        "train_steps": int(train_steps or 0),
        "train_episodes": int(train_stats.get("episodes", 0) or 0),
        "train_max_steps": int(train_stats.get("max_steps", 0) or 0),
        "eval_steps": int(eval_steps or 0),
        "eval_episodes": int(eval_stats.get("episodes", 0) or 0),
        "attack_likely_used": train_stats.get("attack_likely_used", None),
    }


def _extract_scenario_statistics_without_hosts(data):
    if not isinstance(data, dict):
        return {}

    statistics = data.get("statistics", {})
    if not isinstance(statistics, dict):
        return {}

    filtered = {}
    for section_name, section_value in statistics.items():
        if not isinstance(section_value, dict):
            filtered[section_name] = section_value
            continue

        filtered[section_name] = {
            key: value
            for key, value in section_value.items()
            if key != "hosts"
        }

    return filtered


def build_scenario_list(current_config, gym_type, network_config):
    configured_training_directory = current_config.get("training_directory", "_training")
    training_directory = _training_base_abs_path(current_config)
    if not os.path.isdir(training_directory):
        return []

    scenario_rows = []
    for run_dir in _iter_scenario_run_dirs(training_directory, gym_type, network_config):
        scenario_file = os.path.join(run_dir, "scenario.json")
        if not os.path.isfile(scenario_file):
            continue

        run_name = os.path.basename(run_dir)
        run_dt = run_name.split("_")[0]
        scenario_data = _safe_load_json_file(scenario_file)
        scenario_rel_path = os.path.relpath(scenario_file, training_directory)
        scenario_rows.append({
            "datetime": run_dt,
            "run": run_name,
            "path": os.path.join(configured_training_directory, scenario_rel_path).replace("\\", "/"),
            "summary": _extract_scenario_summary(scenario_data),
            "statistics": _extract_scenario_statistics_without_hosts(scenario_data),
        })

    scenario_rows.sort(key=lambda row: row.get("datetime", ""), reverse=True)
    return scenario_rows


def build_test_scenario_preview(current_config):
    # Read attack likelihood values from config, fallback to defaults
    attacks_config = current_config.get("env_params", {}).get("attacks", {})
    if isinstance(attacks_config, dict):
        train_likely = attacks_config.get("likely_train", DEFAULT_ATTACK_LIKELY_TRAIN)
        eval_likely = attacks_config.get("likely_eval", DEFAULT_ATTACK_LIKELY_EVAL)
    else:
        train_likely = getattr(attacks_config, "likely_train", DEFAULT_ATTACK_LIKELY_TRAIN)
        eval_likely = getattr(attacks_config, "likely_eval", DEFAULT_ATTACK_LIKELY_EVAL)
    
    statistics = preview_scenario_statistics_from_config(
        current_config,
        train_attack_likely=train_likely,
        eval_attack_likely=eval_likely,
    )
    scenario_like = {"statistics": statistics}

    return {
        "summary": _extract_scenario_summary(scenario_like),
        "statistics": _extract_scenario_statistics_without_hosts(scenario_like),
        "storage": "memory",
    }


def build_saved_configs_list(app_root):
    config_root = os.path.abspath(os.path.join(app_root, "..", "config"))
    if not os.path.isdir(config_root):
        return []

    rows = []
    for root, _, files in os.walk(config_root):
        for file_name in files:
            if not (file_name.endswith(".yaml") or file_name.endswith(".yml")):
                continue

            abs_path = os.path.join(root, file_name)
            try:
                with open(abs_path, "r", encoding="utf-8") as handle:
                    cfg = yaml.safe_load(handle) or {}
            except Exception:
                continue

            env_params_raw = cfg.get("env_params", {})
            env_params = env_params_raw if isinstance(env_params_raw, dict) else (env_params_raw.__dict__ if hasattr(env_params_raw, '__dict__') else {})
            
            net_params_raw = env_params.get("net_params", {})
            net_params_dict = net_params_raw if isinstance(net_params_raw, dict) else (net_params_raw.__dict__ if hasattr(net_params_raw, '__dict__') else {})
            
            agents_raw = cfg.get("agents", [])
            agents = agents_raw if isinstance(agents_raw, list) else (agents_raw.__dict__ if hasattr(agents_raw, '__dict__') else [])
            enabled_agents = sum(1 for agent in agents if isinstance(agent, dict) and agent.get("enabled", False))

            rows.append({
                "path": abs_path.replace(f"{os.path.abspath(os.path.join(app_root, '..'))}/", ""),
                "gym_type": env_params.get("gym_type", ""),
                "episodes": env_params.get("episodes", 0),
                "test_episodes": env_params.get("test_episodes", 0),
                "network_config": f"{_get_net_param(net_params_dict, 'num_switches', 'num_switches', '')}_{_get_net_param(net_params_dict, 'num_hosts', 'num_hosts', '')}_{_get_net_param(net_params_dict, 'num_iots', 'num_iot', '')}",
                "enabled_agents": enabled_agents,
                "modified_ts": int(os.path.getmtime(abs_path)),
                "modified": datetime.datetime.fromtimestamp(os.path.getmtime(abs_path)).strftime("%Y-%m-%d %H:%M:%S"),
            })

    rows.sort(key=lambda row: row.get("modified_ts", 0), reverse=True)
    for row in rows:
        row.pop("modified_ts", None)
    return rows


def load_saved_config_by_relative_path(app_root, relative_path):
    workspace_root = os.path.abspath(os.path.join(app_root, ".."))
    rel_norm = os.path.normpath(str(relative_path or "")).lstrip("/")
    abs_path = os.path.abspath(os.path.join(workspace_root, rel_norm))
    if not abs_path.startswith(workspace_root + os.sep) and abs_path != workspace_root:
        raise ValueError("Invalid config path")
    if not os.path.isfile(abs_path):
        raise FileNotFoundError("Config file not found")

    with open(abs_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def reprint_result_charts(current_config, gym_type, relative_result_path):
    """
    Regenerate all metric charts for a result directory using the current plot functions.

    Reprints per-agent charts (metrics.png, metrics_combined.png) and root-level
    charts (metrics_comparison.png, radar_chart.png).  For MARL scenarios the
    per-host breakdown is also regenerated.
    """
    try:
        result_abs_path = _resolve_result_abs_path(current_config, relative_result_path)
    except ValueError:
        return {"status": "error", "message": "Invalid result path"}, 400

    if not os.path.isdir(result_abs_path):
        return {"status": "error", "message": "Result directory not found"}, 404

    config_path = os.path.join(result_abs_path, "config.yaml")
    if not os.path.isfile(config_path):
        return {"status": "error", "message": "Missing config.yaml in result folder"}, 404

    try:
        _, config_yaml = read_config_file(config_path)
    except Exception as exc:
        return {"status": "error", "message": f"Error reading config.yaml: {exc}"}, 500

    env_params = config_yaml.get("env_params", {}) or {}
    detected_gym_type = env_params.get("gym_type", gym_type or "")
    agents_cfg = config_yaml.get("agents", []) or []
    agent_names = [
        a.get("name", "")
        for a in agents_cfg
        if a.get("enabled", False)
        and not a.get("skip_learn", True)
        and not str(a.get("name", "")).startswith(ALGO_SUPERVISED)
    ]

    is_marl = detected_gym_type.startswith(MARL_ATTACKS)

    if is_marl:
        from utility.my_marl_statistics import (
            calculate_team_metrics,
            plot_metrics as _plot_metrics,
            plot_combined_performance_over_time as _plot_combined,
            plot_comparison_bar_charts as _plot_comparison,
            plot_radar_chart as _plot_radar,
            plot_metrics_kfold as _plot_kfold,
            plot_metrics_violin as _plot_violin,
        )
    else:
        from utility.my_statistics import (
            plot_metrics as _plot_metrics,
            plot_combined_performance_over_time as _plot_combined,
            plot_comparison_bar_charts as _plot_comparison,
            plot_radar_chart as _plot_radar,
            plot_metrics_kfold as _plot_kfold,
            plot_metrics_violin as _plot_violin,
        )

    reprinted = []
    errors = []
    agents_metrics = {}

    for agent_name in agent_names:
        agent_dir = os.path.join(result_abs_path, agent_name)
        data_file = os.path.join(agent_dir, "data.json")
        if not os.path.isfile(data_file):
            errors.append(f"Missing data.json for agent '{agent_name}'")
            continue

        try:
            data = read_data_file(data_file)
        except Exception as exc:
            errors.append(f"Cannot read data.json for '{agent_name}': {exc}")
            continue

        raw_metrics = data.get("train_metrics", {}) or {}

        if is_marl:
            host_metrics = {h: m for h, m in raw_metrics.items() if isinstance(m, dict)}
            team_metrics = calculate_team_metrics(host_metrics)
            agent_plot_metrics = team_metrics

            for host_name, host_m in host_metrics.items():
                try:
                    _plot_metrics(host_m, agent_dir,
                                  title=f"{agent_name}_{host_name} Train metrics",
                                  host=host_name)
                    _plot_combined(host_m, agent_dir,
                                   title=f"{agent_name}_{host_name} Combined performance over time",
                                   host=host_name)
                    reprinted.append(f"{agent_name}/{host_name}")
                except Exception as exc:
                    errors.append(f"Error reprinting {agent_name}/{host_name}: {exc}")
        else:
            agent_plot_metrics = raw_metrics if isinstance(raw_metrics, dict) else {}

        if not agent_plot_metrics:
            errors.append(f"No usable train_metrics for agent '{agent_name}'")
            continue

        try:
            _plot_metrics(agent_plot_metrics, agent_dir,
                          title=f"{agent_name} Train metrics")
            _plot_combined(agent_plot_metrics, agent_dir,
                           title=f"{agent_name} Combined performance over time")
            reprinted.append(agent_name)
        except Exception as exc:
            errors.append(f"Error reprinting {agent_name}: {exc}")
            continue

        agents_metrics[agent_name] = agent_plot_metrics

    if agents_metrics:
        try:
            _plot_comparison(result_abs_path, agents_metrics)
            _plot_radar(result_abs_path, agents_metrics)
            reprinted.append("metrics_comparison + radar_chart")
        except Exception as exc:
            errors.append(f"Error reprinting root charts: {exc}")

        # K-Fold chart: treat each agent as one fold
        try:
            metric_keys = ["accuracy", "precision", "recall", "f1_score"]
            kfold_data = {}
            for k in metric_keys:
                folds = [m[k] for m in agents_metrics.values()
                         if k in m and isinstance(m.get(k), list) and m[k]]
                if folds:
                    kfold_data[k] = folds

            if kfold_data:
                n_folds = len(agents_metrics)
                fold_label = f"{n_folds} agent{'s' if n_folds != 1 else ''} as fold{'s' if n_folds != 1 else ''}"
                _plot_kfold(
                    kfold_data,
                    result_abs_path,
                    title=f"Cross-Agent K-Fold Performance ({fold_label})",
                    xlabel="Episodes",
                )
                reprinted.append("metrics_kfold")
                try:
                    _plot_violin(
                        kfold_data,
                        result_abs_path,
                        title=f"Metric Distribution at Key Training Steps ({fold_label})",
                    )
                    reprinted.append("metrics_violin")
                except Exception as ve:
                    errors.append(f"Error generating violin chart: {ve}")
        except Exception as exc:
            errors.append(f"Error generating K-Fold chart: {exc}")

    status = "success" if not errors else ("partial" if reprinted else "error")
    return {
        "status": status,
        "reprinted": reprinted,
        "errors": errors,
    }, 200
