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
    if gym_type.startswith('marl_pz'):
        return 'marl_pz'
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


def _deep_merge_defaults(base: dict, defaults: dict) -> dict:
    """Merge defaults into base, only filling in missing keys (base has priority)."""
    for key, default_val in defaults.items():
        if key not in base:
            base[key] = default_val
        elif isinstance(base[key], dict) and isinstance(default_val, dict):
            _deep_merge_defaults(base[key], default_val)
    return base


def merge_scenario_defaults(config: dict) -> dict:
    """Fill in env_params keys missing from config (e.g. 'attacks') using the
    selected scenario's scenario_env_param.yaml, and reorder for display.

    Any config loaded from disk without going through this (e.g. a raw saved
    config.yaml) can be missing scenario-specific sections that scenario code
    assumes are always present. Mutates and returns config.
    """
    gym_type = config.get('env_params', {}).get('gym_type', '')
    if gym_type:
        scenario_params = load_scenario_env_params(gym_type)
        if scenario_params:
            _deep_merge_defaults(config.setdefault('env_params', {}), scenario_params)
        config['env_params'] = reorder_env_params_for_display(config['env_params'])
    return config


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
