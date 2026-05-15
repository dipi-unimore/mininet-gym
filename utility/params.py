import json as js
import yaml
from utility.scenario_params import load_scenario_env_params, reorder_env_params_for_display

class Params:
    # constructor
    def __init__(self, dict1):
        self.__dict__.update(dict1)

def read_config_file(file_name='config/default.yaml', config_cleaner=None):
    with open(file_name, 'r') as file:
        config = yaml.safe_load(file)

    if callable(config_cleaner):
        cleaned_config = config_cleaner(config)
        if cleaned_config is not None:
            config = cleaned_config

    gym_type = config.get('env_params', {}).get('gym_type', '')
    if gym_type:
        scenario_params = load_scenario_env_params(gym_type)
        if scenario_params:
            config.setdefault('env_params', {}).update(scenario_params)
        config['env_params'] = reorder_env_params_for_display(config['env_params'])

    return js.loads(js.dumps(config), object_hook=Params), config