import os
from threading import Event
import ssl
from flask_httpauth import HTTPBasicAuth, request
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, Response, request, send_from_directory

from utility.my_log import information

from .routes import (
    create_batch_blueprint,
    create_calibration_blueprint,
    create_config_blueprint,
    create_results_blueprint,
    create_training_blueprint,
)
from .socket_handler import init_socketio, set_app_state


CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'default.yaml')
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    static_folder=os.path.join(APP_ROOT, 'static'),
    template_folder=os.path.join(APP_ROOT, 'templates')
)
auth = HTTPBasicAuth()


@auth.verify_password
def verify_password(username, password):
    """Verify the username and password against the database."""
    return check_auth(username, password)

def check_auth(username, password):
    """Check if a username/password combination is valid."""
    import json
    with open(os.path.join('config', 'users_hashed.json'), 'r') as f:
        users = json.load(f)
    user_hash = users.get(username)
    if username in users and check_password_hash(user_hash, password):
        information(f"User {username}")
        return username
    return None

def authenticate():
    """Send a 401 response to force the browser popup."""
    return Response(
        'Access denied. Missing or incorrect credentials.\n', 401,
        {'WWW-Authenticate': 'Basic realm="Development Area"'}
    )

@app.before_request
def require_auth():
    """Intercept ALL incoming requests."""
    auth = request.authorization
    if not auth or not check_auth(auth.username, auth.password):
        return authenticate()

state = {
    "current_config": {},
    "main_ref": None,
    "socketio_instance": None,
    "training_thread": None,
    "pause_event": Event(),
    "stop_event": Event(),
    "app_root": APP_ROOT,
    "batch": {"active": False},
}

app.register_blueprint(create_config_blueprint(state, CONFIG_PATH, APP_ROOT))
app.register_blueprint(create_results_blueprint(state))
app.register_blueprint(create_training_blueprint(state))
app.register_blueprint(create_calibration_blueprint(state))
app.register_blueprint(create_batch_blueprint(state))

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
    # # 1. create an SSL context for the server
    # context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    
    # # 2. Load the server's certificate and private key
    # context.load_cert_chain(certfile='../server_cert.pem', keyfile='../server_key.pem')
    
    # # 3. Ask the server to require a client certificate for authentication
    # context.verify_mode = ssl.CERT_REQUIRED
    
    # # 4. Load the CA certificate that will be used to verify the client's certificate
    # context.load_verify_locations(cafile='../ca_client_cert.pem')    
    
    
    #state["socketio_instance"].run(app, host=host, port=port, ssl_context='adhoc', debug=False)
    state["socketio_instance"].run(app, host=host, port=port, allow_unsafe_werkzeug=True, debug=False)
