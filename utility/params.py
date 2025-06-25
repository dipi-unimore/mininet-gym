import json as js
import yaml

class Params:     
    # constructor
    def __init__(self, dict1):
        self.__dict__.update(dict1)
        
def read_config_file( file_name='config.yaml'):
    with open(file_name, 'r') as file:
        config = yaml.safe_load(file)
    return js.loads(js.dumps(config), object_hook=Params), config