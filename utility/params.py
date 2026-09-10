import json as js
import yaml
from utility.scenario_params import merge_scenario_defaults

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

    config = merge_scenario_defaults(config)

    return js.loads(js.dumps(config), object_hook=Params), config