import datetime, re
from flask import request
from flask_socketio import SocketIO
from utility.constants import SystemStatus, SystemModes


socketio = None # initialized in app_api.py
flask_app = None 
app_state = None # Application state dict (training thread, pause/stop events)

def init_socketio(app):
    """
    Initialize SocketIO associating to Flask app
    """
    global socketio, flask_app
    flask_app = app
    # logger=True and engineio_logger=True useful for debug
    socketio = SocketIO(app, cors_allowed_origins="*", logger=False, engineio_logger=False, transports=['websocket', 'polling'], async_mode="threading") 
    
    # Register handler SocketIO
    #register_handlers(socketio)
    return socketio

def get_socketio_instance():
    """Return the SocketIO instance."""
    global socketio
    return socketio

def set_app_state(state):
    """Set the application state dict for use in handlers."""
    global app_state
    app_state = state

def _get_training_status():
    """Get current training status."""
    if not app_state:
        return SystemStatus.IDLE, 'IDLE', 'No active training'
    
    is_training = app_state.get('training_thread') is not None and app_state['training_thread'].is_alive()
    pause_set = app_state.get('pause_event') and app_state['pause_event'].is_set()
    stop_set = app_state.get('stop_event') and app_state['stop_event'].is_set()
    
    if is_training:
        if pause_set:
            return SystemStatus.PAUSED, 'PAUSED', 'Training is paused'
        elif stop_set:
            return SystemStatus.STOPPED, 'STOPPED', 'Training is stopping'
        else:
            return SystemStatus.RUNNING, 'RUNNING', 'Training is running'
    else:
        return SystemStatus.IDLE, 'IDLE', 'No active training'

def register_handlers(socketio_instance):
    """Register handler SocketIO like connect, disconnect, message, ecc."""
    
    @socketio_instance.on('connect')
    def handle_connect():
        print(f'Client connected {request.sid}')
        # Get current training status and send it immediately
        status, mode, message = _get_training_status()
        send_status(status=status, mode=mode, message=f'Client Connected. {message}')
        # Resend cfg on reconnect so the client can reinitialize charts for any gym type
        if status not in (SystemStatus.IDLE, SystemStatus.DISCONNECTED) \
                and hasattr(socketio_instance, 'cfg'):
            send_live_data([{'level': 'config', 'config': socketio_instance.cfg}])
        
    @socketio_instance.on('disconnect')
    def handle_disconnect(param = None):
        print('Client disconnected')
        send_status(status=SystemStatus.DISCONNECTED,  mode='', message='Client Disconnected!')

    # Handler di errore
    @socketio_instance.on_error()
    def error_handler(e):
        print(f'SocketIO Error: {e}')
        pass

def _emit_live_data(messages):
    """Helper to send in context."""
    global flask_app
    if flask_app:
        with flask_app.app_context():    
            try:
                socketio.emit('live_update', messages)
                # force flush buffer I/O.
                socketio.sleep(0) 
            except Exception as e:
                print(f"Error emitting live data in background task: {e}")
    else:
        print("flask_app non è disponibile per l'emissione in _emit_live_data.")
        
def send_live_data(messages):
    """
    
    # old version
    # agent_name = None, level = None, message = None, 
    #                config = None, traffic_data = None, final_data = None, host_tasks = None,
    #                global_state = None, metrics = None, final_metrics = None, step_data = None):

    Send training data real time to web client.
    """
    if socketio is not None and flask_app is not None:
        try:
            _emit_live_data(messages)
        except Exception as e:
            print(f"Error sending socket data: {e}")
    else:
        print("SocketIO not initialized.")
 
def _emit_status(status_data):
    """Helper"""
    global flask_app
    if flask_app:
        with flask_app.app_context():    
            try:
                socketio.emit('status_update', status_data)
            except Exception as e:
                print(f"Error emitting training status in background task: {e}")
               
def send_status(status: str, mode: str, message: str = None):
    """
    Send to client the training status update (event 'status_update').

    :param status: Training status (es. 'RUNNING', 'PAUSED', 'COMPLETED', 'ERROR').
    :param details: addon details.
    """
    if socketio is not None and flask_app is not None:
        status_data = {
            'status': status, 
            'message': message,
            'mode': mode
        }
        try:
            _emit_status(status_data)
            
        except Exception as e:
            print(f"Error sending socket data (status): {e}")
    else:
        print("SocketIO not initialized.")
