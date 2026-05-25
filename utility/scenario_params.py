import os
import yaml

SCENARIOS_BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'reinforcement_learning', 'scenarios'
)
SCENARIO_PARAM_FILE = 'scenario_env_param.yaml'


def gym_type_to_scenario_folder(gym_type: str) -> str | None:
    if not gym_type:
        return None
    if gym_type.startswith('attacks_ho'):
        return 'attack_detect_host_observable'
    if gym_type.startswith('marl_attacks'):
        return 'marl'
    if gym_type.startswith('attacks'):
        return 'attack_detect'
    if gym_type.startswith('classification'):
        return 'classification'
    return None


def load_scenario_env_params(gym_type: str) -> dict:
    folder = gym_type_to_scenario_folder(gym_type)
    if folder is None:
        return {}
    param_file = os.path.join(SCENARIOS_BASE_DIR, folder, SCENARIO_PARAM_FILE)
    if not os.path.exists(param_file):
        return {}
    with open(param_file, 'r') as f:
        return yaml.safe_load(f) or {}


_SCENARIO_SECTION_KEYS = ('attacks', 'classification')


def reorder_env_params_for_display(env_params: dict) -> dict:
    """Return env_params with scenario section (attacks/classification) placed right after gym_type."""
    scenario_keys = [k for k in _SCENARIO_SECTION_KEYS if k in env_params]
    if not scenario_keys:
        return env_params
    ordered: dict = {}
    for key, val in env_params.items():
        if key in _SCENARIO_SECTION_KEYS:
            continue
        ordered[key] = val
        if key == 'gym_type':
            for sk in scenario_keys:
                ordered[sk] = env_params[sk]
    return ordered
