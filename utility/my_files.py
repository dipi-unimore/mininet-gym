# my_files.py
import os, time, yaml, json as js, orjson, numpy as np, pwd
from utility.my_log import set_log_level,set_log_file, information, debug, error, notify_client   
from utility.params import read_config_file

def drop_privileges(username: str):
    """Cambia l'utente effettivo del processo al dato username."""
    try:
        user_info = pwd.getpwnam(username)
        os.setegid(user_info.pw_gid)  # Cambia il gruppo
        os.seteuid(user_info.pw_uid)  # Cambia l'utente
    except KeyError:
        raise ValueError(f"L'utente {username} non esiste")
    except PermissionError:
        raise PermissionError("Devi eseguire lo script come root per cambiare utente")

def regain_root():
    """Ritorna a root (solo se lo script è stato avviato come root)."""
    try:
        os.seteuid(0)  # Ripristina i privilegi root
    except:
        error("no root instance to recover")


def save_data_to_file(data, dir_name, file_name="data"):
    def convert_np(obj):
        # Converts np.float32 or np.float64 to regular float
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.ndarray)):
            return list(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    
    #drop_privileges("salvo")
    
    with open(f"{dir_name}/{file_name}.json", 'w') as f:
        js.dump(data, f, default=convert_np) 
    
    #regain_root()
    
def read_csv_file(csv_file):
    """
    Reads a CSV file and returns its content as a list of dictionaries.
    
    :param csv_file: Path to the CSV file.
    :return: List of dictionaries representing the CSV data.
    """
    import pandas as pd
    df = pd.read_csv(csv_file)
    return df.to_dict(orient='records')

def read_data_file(file_name:str = '../data' , add_extension_file:bool = True, extension:str = "json"):
    if add_extension_file:
        file_name = f"{file_name}.{extension}"
    with open(file_name, 'r') as file:
        data = orjson.loads(file.read()) # faster than this -> data = yaml.safe_load(file)
    return data #js.loads(js.dumps(config), object_hook=Params), config

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
    config.training_directory += f"/{config.env_params.gym_type}"
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
     
    #create execution directory 
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if agent_name is None:
        directory_name= f"{config.training_directory}/{timestr}_{config.net_config_filter}"
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
    config,config_dict = read_config_file('config.yaml')
    #config.net_config_filter = f"{config.env_params.net_params.num_switches}_{config.env_params.net_params.num_hosts}_{config.env_params.net_params.num_iot}"
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