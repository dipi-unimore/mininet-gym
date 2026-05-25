# my_files.py
import shutil
import os, time, yaml, json as js, orjson, numpy as np, pwd
from utility.my_log import set_log_level, information, error   
from utility.params import read_config_file

def drop_privileges(username: str):
    """Change the effective user ID of the process to the specified username."""
    try:
        user_info = pwd.getpwnam(username)
        os.setegid(user_info.pw_gid)  # Change the group
        os.seteuid(user_info.pw_uid)  # Change the user
    except KeyError:
        raise ValueError(f"User {username} does not exist")
    except PermissionError:
        raise PermissionError("You must run the script as root to change user")

def regain_root():
    """Return the process to root privileges."""
    try:
        os.seteuid(0)  # Restore root privileges
    except:
        error("no root instance to recover")

def _build_experiment_config(config_dict: dict) -> dict:
    """Return a filtered config with only experiment-relevant keys."""
    experiment = {}
    if 'random_seed' in config_dict:
        experiment['random_seed'] = config_dict['random_seed']
    if 'env_params' in config_dict:
        experiment['env_params'] = config_dict['env_params']
    if 'agents' in config_dict:
        experiment['agents'] = [a for a in config_dict['agents'] if a.get('enabled', False)]
    return experiment


def copy_config_file_to_training_dir(training_dir: str, config_dict: dict = None) -> bool:
    """
    Saves the experiment config to a specified training subdirectory.
    Only random_seed, env_params and enabled agents are persisted.
    If config_dict is None, falls back to copying config/default.yaml.

    Args:
        training_dir (str): The path to the destination directory.
        config_dict (dict): The actual configuration dictionary used for training.

    Returns:
        bool: True if the save was successful, False otherwise.
    """
    execution_dir = os.getcwd()
    destination_full_dir = os.path.join(execution_dir, training_dir)
    destination_path = os.path.join(destination_full_dir, "config.yaml")

    try:
        if config_dict is not None:
            experiment_config = _build_experiment_config(config_dict)
            with open(destination_path, 'w') as f:
                yaml.dump(experiment_config, f, default_flow_style=False)
            print(f"Successfully saved actual config to '{destination_path}'.")
            return True
        else:
            source_file_name = "config/default.yaml"
            source_path = os.path.join(execution_dir, source_file_name)

            if not os.path.exists(source_path):
                print(f"Error: Source file '{source_file_name}' not found.")
                return False
            if not os.path.isfile(source_path):
                print(f"Error: '{source_path}' is not a file.")
                return False

            if not os.path.exists(destination_full_dir):
                try:
                    os.makedirs(destination_full_dir)
                except OSError as e:
                    print(f"Error creating destination directory: {e}")
                    return False

            shutil.copy2(source_path, destination_path)
            print(f"Successfully copied default config to '{destination_path}'.")
            return True

    except PermissionError:
        print(f"Error: Permission denied. Unable to save config to '{destination_path}'.")
        return False
    except Exception as e:
        print(f"An error occurred while saving config: {e}")
        return False

def save_data_to_file(data, dir_name, file_name="data"):
    def convert_np(obj):
        # Converts np.float32 or np.float64 to regular float
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.ndarray)):
            return list(obj)
        if isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    
   
    with open(f"{dir_name}/{file_name}.json", 'w') as f:
        js.dump(data, f, default=convert_np) 
    

    
def read_csv_file(csv_file):
    """
    Reads a CSV file and returns its content as a list of dictionaries.
    
    :param csv_file: Path to the CSV file.
    :return: List of dictionaries representing the CSV data.
    """
    import pandas as pd
    df = pd.read_csv(csv_file)
    return df.to_dict(orient='records')

def read_data_file(file_name:str = '../data' , extension:str = None):
    if extension is not None:
        file_name = f"{file_name}.{extension}"
    file_name = resolve_data_file_path(file_name)
    with open(file_name, 'r') as file:
        data = orjson.loads(file.read()) # faster than this -> data = yaml.safe_load(file)
    return data #js.loads(js.dumps(config), object_hook=Params), config


def resolve_data_file_path(file_name: str) -> str:
    """
    Resolve dataset paths that may have been saved as relative paths,
    absolute paths, or malformed concatenations such as:
    '_training//home/.../classification/...'.
    """
    if os.path.exists(file_name):
        return file_name

    normalized = os.path.normpath(file_name)
    if os.path.exists(normalized):
        return normalized

    cwd = os.getcwd()

    # If the current working directory appears inside the path, rebuild it as
    # a path under the local _training directory.
    cwd_marker = cwd + os.sep
    if cwd_marker in file_name:
        suffix = file_name.split(cwd_marker, 1)[1]
        candidate = os.path.join(cwd, "_training", suffix)
        if os.path.exists(candidate):
            return candidate

    # Common UI case: relative path without the _training prefix.
    if not os.path.isabs(file_name):
        candidate = os.path.join(cwd, "_training", file_name.lstrip(os.sep))
        if os.path.exists(candidate):
            return candidate

    return file_name

def read_all_data_from_execution(dir_execution: str):
    store = []
    for folder in os.listdir(dir_execution):
        if not os.path.isdir(os.path.join(dir_execution, folder)):
            continue
        data = {}
        data["folder"] = folder
        file = "data" if folder != "TEST" else "test"
        file_path = os.path.join(dir_execution,folder, file)
        data["data"]  = read_data_file(file_path)
        store.append(data )
    return store

def create_directory_training_execution(config, agent_name = None):

    #create training base path if not exists
    if os.path.exists(config.training_directory) == False:
        try:
            os.mkdir(config.training_directory)
            information(f"Directory '{config.training_directory}' created successfully.\n")
        except FileExistsError:
            error(f"Directory '{config.training_directory}' already exists.\n")
        except PermissionError:
            error(f"Permission denied: Unable to create '{config.training_directory}'.\n")
        except Exception as e:
            error(f"An error occurred: {e}")
            
    #create training gym_type path if not exists
    training_gym_type_directory = f"{config.training_directory}/{config.env_params.gym_type}"
    if os.path.exists(training_gym_type_directory) == False:  
        try:
            os.mkdir(training_gym_type_directory)
            information(f"Directory '{training_gym_type_directory}' created successfully.\n")
        except FileExistsError:
            error(f"Directory '{training_gym_type_directory}' already exists.\n")
        except PermissionError:
            error(f"Permission denied: Unable to create '{training_gym_type_directory}'.\n")
        except Exception as e:
            error(f"An error occurred: {e}")
     
    #create execution directory 
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if agent_name is None:
        directory_name= f"{training_gym_type_directory}/{timestr}_{config.net_config_filter}"
    else:    
        directory_name= f"{config.training_execution_directory}/{agent_name}"
        
    try:
        os.mkdir(directory_name)
        information(f"Directory '{directory_name}' created successfully.\n")
    except FileExistsError:
        error(f"Directory '{directory_name}' already exists.\n")
    except PermissionError:
        error(f"Permission denied: Unable to create '{directory_name}'.\n")
    except Exception as e:
        error(f"An error occurred: {e}")

    return directory_name

def find_latest_execution(base_path: str) -> str | None:
    
    folders = sorted(
        [folder for folder in os.listdir(base_path) if folder[:8].isdigit()],
        reverse=True
    )  
       
    return os.path.join(base_path, folders[0])


def find_all_file_by_name(base_path: str, file_name: str) -> list | None:
    
    folders = sorted(
        [folder for folder in os.listdir(base_path) if folder[:8].isdigit()],
        reverse=True
    )  
    
    statutes_files = []
    
    for folder in folders:
        path = os.path.join(base_path, folder)

        if os.path.isdir(path):  # Assicura che sia una directory
            file_path = os.path.join(path, file_name)
            if os.path.exists(file_path):  # Controlla se il file esiste
                statutes_files.append(file_path)

    return statutes_files if statutes_files else None

def find_latest_file(base_path: str, file_key: str, extension: str, net_config_filter: str) -> str | None:
    """
    Trova il file ZIP più recente all'interno della cartella più recente.
    
    :param base_path: Percorso della cartella principale (X).
    :param file_key: Nome del file da cercare ('a', 'b' o 'c').
    :return: Percorso completo del file ZIP più recente, o None se non trovato.
    """
    
    folders = sorted(
        [folder for folder in os.listdir(base_path) if folder[:8].isdigit()],
        reverse=True
    )
    
    # Trova la cartella più recente  
    for folder in folders:
        if folder[16:] != net_config_filter:
            continue
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path) and folder[:8].isdigit() and folder[9:15].isdigit():
            try:
                # Controlla la presenza del file JSON richiesto
                target_folder = os.path.join(folder_path, file_key)
                target_file = os.path.join(target_folder, f"{file_key}.{extension}")
                if os.path.exists(target_file):
                    return target_file
            except ValueError:
                continue
    
    return f"{file_key}.{extension}"


# Esempio di utilizzo:
# base_path = "X"  # Sostituisci con il percorso reale della cartella
# print(find_latest_json(base_path, 'a'))

if __name__ == '__main__':
    set_log_level('info')
    config,config_dict = read_config_file('config/default.yaml')
    #config.net_config_filter = f"{config.env_params.net_params.num_switches}_{config.env_params.net_params.num_hosts}_{config.env_params.net_params.num_iots}"
    #training_execution_directory = create_directory_training_execution(config)
    #set_log_file(f"{training_execution_directory}/log.txt")
    # file = find_latest_file(config.training_directory, 'Q-learning', 'json', '1_10_1')
    # information(file)
    # last_execution = find_latest_execution(config.training_directory)
    # information(last_execution)
    
    
    # to write statuses.json from train indicators of one agent in a execution without it
    # dir = f"{config.training_directory}/20250301-144317_1_10_1"
    # file = f"{dir}/Q-learning/data"
    # data = read_data_file(file)
    # print(len(data["train_indicators"]))
    # statuses = []
    # for episode in data["train_indicators"]:
    #     for status in episode["episode_statuses"]:
    #         del status['action_choosen']
    #         del status['action_correct']
    #         del status['step']
    #         statuses.append(status)
    # save_data_to_file(statuses, dir, "statuses")    
    
    statuses = []
    # read a big file   
    with open(config.training_directory+"/statuses.json", 'r') as file:
        statuses = orjson.loads(file.read())
    
    
    
    #trovare tutti i file di un certo nome
    # statuses_files = find_all_file_by_name(config.training_directory, "statuses.json")
    # for file_name in statuses_files:
    #     with open(file_name, 'r') as file:
    #         data = orjson.loads(file.read())
    #     # for d in data:
    #     #     statuses.append(d)
    #     statuses.extend(data)
    # save_data_to_file(statuses, config.training_directory, "statuses")