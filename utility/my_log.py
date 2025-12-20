from mininet.log import setLogLevel as mininet_setLogLevel, info as mininet_info, debug as mininet_debug, error as mininet_error
from app.socket_handler import send_live_data
import datetime, logging, re
from colorama import Fore

from utility.constants import SYSTEM, SystemLevels
from utility.utils import convert_ansi_to_html

logger = None

def set_log_file(log_path: str = "log.txt"):
    global logger
    logging.basicConfig(
        filename=log_path,
        filemode='a',
        format='%(asctime)s,%(msecs)03d %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.DEBUG
    )
    logger = logging.getLogger('app')

def set_log_level(level):
    mininet_setLogLevel(level)

def add_timestamp_to_message(message):
    now = datetime.datetime.now()
    return f"{Fore.LIGHTYELLOW_EX}{now.strftime('%H:%M:%S')}{Fore.WHITE} {message}"

def add_author_to_message(author, message):
    return f"{Fore.YELLOW}{author}{Fore.WHITE} {message}"

def remove_colors(message):
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')  # Regex for ANSI color codes
    return ansi_escape.sub('', message)

def log_message(level, message, author=None):
    if not message.endswith("\n") and  not message.endswith("\n"+Fore.WHITE) :
        message += "\n"
        
    if author:
        message = add_author_to_message(author, message)
    
    #logging message on file
    if logger is not None:
        log_func = getattr(logging, level)
        log_func(remove_colors(message).replace("\n", " "))

    message = add_timestamp_to_message(message)
    
    #sending message throught web socket
    notify_client(level=level,agent_name=author,message=message)

    #show message on screen
    mininet_func = {
        "info": mininet_info,
        "debug": mininet_debug,
        "error": mininet_error
    }.get(level, mininet_info)

    mininet_func(message)

def information(message, author=None):
    log_message("info", message, author)

def debug(message):
    log_message("debug", message)

def error(message):
    log_message("error", message)

_client_data_notifier_func = None 
_client_status_notifier_func = None
_is_from_dataset = False

def set_is_from_dataset(value: bool):
    """
    Imposta se i dati provengono da un dataset.
    """
    global _is_from_dataset
    _is_from_dataset = value

def initialize_client_notifier(func_send_data, func_training_status):
    """
    Usata da app/app_api.py per iniettare la funzione send_live_data e send_training_status 
    dal modulo socket_handler.
    """
    global _client_data_notifier_func
    _client_data_notifier_func = func_send_data
    global _client_status_notifier_func
    _client_status_notifier_func = func_training_status
    start_buffer_flush()
    # NOTA: Questa funzione viene chiamata una volta, all'avvio del server.

# def notify_client(level=None, agent_name = None, message = None, 
#                   config = None, traffic_data = None, final_data = None, host_tasks = None,
#                   global_state = None, metrics = None, final_metrics = None, step_data = None,
#                   status=None, mode=None):
#     """
#     Function to notify the web client about various events or data. 
#     Checks if the sending function has been initialized before calling it.
#     """
#     if message is None and config is None and \
#         traffic_data is None and global_state is None \
#             and metrics is None and status is None and step_data is None and \
#             final_data is None and final_metrics is None and host_tasks is None and \
#                 mode is None:
#         return
    
#     global _client_status_notifier_func
#     global _client_data_notifier_func
    
#     if level == SystemLevels.STATUS and mode is not None and _client_status_notifier_func: 
#         try:
#             _client_status_notifier_func(status=status, mode = mode, message = message)            
#         except Exception as e:
#             log_message("error",f"Error sending to client: {e}", author="notify_client") 
#         return
    
#     # Questo è il controllo che verifica se la funzione è "unbound" (cioè, None)
#     if  _client_data_notifier_func: 
#         try:
#             if level is None and message is None:
#                 level = SystemLevels.DATA
#             if agent_name is None:
#                 agent_name = SYSTEM
#             _client_data_notifier_func(level=level, agent_name = agent_name, message = message,
#                                   config = config, traffic_data = traffic_data,  final_data = final_data, host_tasks = host_tasks, 
#                                   global_state = global_state, metrics = metrics, final_metrics = final_metrics, step_data = step_data)
#         except Exception as e:
#             # Utile per debug, nel caso la funzione iniettata fallisca
#             log_message("error",f"Error sending to client: {e}", author="notify_client") 
            
import threading
import time
from collections import deque

# Variabili globali per il buffer
_message_buffer = deque()
_buffer_lock = threading.Lock()
_flush_thread = None
_flush_interval = 2.0  # secondi
_should_stop = False

def _flush_buffer():
    """Thread che svuota il buffer ogni N secondi"""
    global _message_buffer, _buffer_lock, _should_stop, _client_data_notifier_func
    
    while not _should_stop:
        time.sleep(_flush_interval)
        
        with _buffer_lock:
            if len(_message_buffer) > 0 and _client_data_notifier_func:
                # Crea una copia del buffer e lo svuota
                messages_to_send = list(_message_buffer)
                _message_buffer.clear()
                
                try:
                    # Invia l'array di messaggi
                    _client_data_notifier_func(messages=messages_to_send)
                except Exception as e:
                    log_message("error", f"Error flushing buffer to client: {e}", author="flush_buffer")

def start_buffer_flush():
    """Inizializza il thread di flush del buffer"""
    global _flush_thread, _should_stop
    
    if _flush_thread is None or not _flush_thread.is_alive():
        _should_stop = False
        _flush_thread = threading.Thread(target=_flush_buffer, daemon=True)
        _flush_thread.start()
        log_message("info", "Buffer flush thread started", author="notify_client")

def stop_buffer_flush():
    """Ferma il thread di flush del buffer"""
    global _should_stop, _flush_thread
    
    _should_stop = True
    if _flush_thread:
        _flush_thread.join(timeout=5)
        log_message("info", "Buffer flush thread stopped", author="notify_client")

def notify_client(level=None, agent_name=None, message=None, 
                  config=None, traffic_data=None, final_data=None, host_tasks=None,
                  global_state=None, metrics=None, final_metrics=None, step_data=None,
                  status=None, mode=None, force_immediate=False):
    """
    Function to notify the web client about various events or data.
    Messages are buffered and sent every 2 seconds unless force_immediate=True.
    """
    if message is None and config is None and \
        traffic_data is None and global_state is None \
            and metrics is None and status is None and step_data is None and \
            final_data is None and final_metrics is None and host_tasks is None and \
                mode is None:
        return
    
    global _client_status_notifier_func
    global _client_data_notifier_func
    global _message_buffer, _buffer_lock
    
    # I messaggi di STATUS vengono sempre inviati immediatamente
    if level == SystemLevels.STATUS and mode is not None and _client_status_notifier_func: 
        try:
            _client_status_notifier_func(status=status, mode=mode, message=message)            
        except Exception as e:
            log_message("error", f"Error sending status to client: {e}", author="notify_client") 
        return
    
    # Prepara il messaggio da bufferizzare
    if _client_data_notifier_func:
        if level is None and message is None:
            level = SystemLevels.DATA
        if agent_name is None:
            agent_name = SYSTEM
        if message is not None:
            message = convert_ansi_to_html(message)
            message_data = {
                'agent': agent_name,
                'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
                'level': level, 
                'message': message
            }
        elif config is not None:
            message_data = {
                'level': 'config',
                'config': config
            }                
        else:
            message_data = {
                'agent': agent_name,
                'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
                'level': level, 
                'config': config,
                'trafficData': traffic_data,
                'finalData': final_data,
                'hostTasks': host_tasks,
                'globalState': global_state,
                'metrics': metrics,
                'finalMetrics': final_metrics,
                'stepData': step_data                   
            }                
            
        
        # Se richiesto invio immediato o per messaggi critici
        if not _is_from_dataset or config is not None:
            try:
                _client_data_notifier_func(messages=[message_data])
            except Exception as e:
                log_message("error", f"Error sending immediate message to client: {e}", author="notify_client")
        else:
            if message_data.get('level') == SystemLevels.DEBUG or (message_data.get('level') == SystemLevels.INFO and message_data.get('agent') == SYSTEM):
                return
            # Aggiungi al buffer
            with _buffer_lock:
                _message_buffer.append(message_data)
            time.sleep(0.01)  # Piccola pausa per permettere al thread di flush di lavorare
