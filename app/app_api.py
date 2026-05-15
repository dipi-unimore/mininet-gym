import os
from threading import Event

from flask import Flask, render_template, send_from_directory

from .routes import create_config_blueprint, create_results_blueprint, create_training_blueprint
from .socket_handler import init_socketio, set_app_state


CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'default.yaml')
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    static_folder=os.path.join(APP_ROOT, 'static'),
    template_folder=os.path.join(APP_ROOT, 'templates')
)

state = {
    "current_config": {},
    "main_ref": None,
    "socketio_instance": None,
    "training_thread": None,
    "pause_event": Event(),
    "stop_event": Event(),
    "app_root": APP_ROOT,
}

app.register_blueprint(create_config_blueprint(state, CONFIG_PATH, APP_ROOT))
app.register_blueprint(create_results_blueprint(state))
app.register_blueprint(create_training_blueprint(state))

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')  

def start_api(main_training_func, loaded_config, host='0.0.0.0', port=5000):
    """Start server Flask + SocketIO in a separate thread."""
    state["main_ref"] = main_training_func
    state["current_config"] = loaded_config
    state["socketio_instance"] = init_socketio(app)
    # Pass state to socket handler so it can check training status on connect
    set_app_state(state)
    print("Server Web API + SocketIO started...")
    state["socketio_instance"].run(app, host=host, port=port, debug=False)
