from mininet.log import setLogLevel as mininet_setLogLevel, info as mininet_info, debug as mininet_debug, error as mininet_error
from app.app_socket import send_message
import datetime, logging, re
from colorama import Fore

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
    notify_client(message)

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

def notify_client(message):
    # try:
        send_message(message)
    # except Exception as e:
        #logging.error(f"Failed to send WebSocket message: {e}")
