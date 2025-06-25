from flask import Flask, request
from flask_socketio import SocketIO, send, emit
#from flask_cors import CORS  # Import CORS
import time

app = Flask(__name__)
# app.config['SECRET_KEY'] = 'secret!'
# app.config['DEBUG'] = True
#socketio = SocketIO(app, logger=True, engineio_logger=True, cors_allowed_origins="*")
socketio = SocketIO(app, cors_allowed_origins="*")

# WebSocket handler
@socketio.on('connect')
def handle_connect():
    sid = request.sid
    print(f'Client connected {sid}') 
    socketio.emit('message', {'message': 'Connected'})
    # time.sleep(10)
    # socketio.emit('message', {'message': 'Message after 5 seconds'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

#to receive from client  
#message
@socketio.on('message')
def handle_message(message):
    print(f"Received message: {message}")
    #socketio.emit('response', {'message': f"Server received: {message}"})
#json   
@socketio.on('json')
def handle_json(json):
    print('received json: ' + str(json))
#custom
@socketio.on('custom_event')
def handle_custom_event(data):
    print(f"Received data: {data}")
    socketio.emit('response', {'message': f"Server processed your event: {data}"})
#custom multi params
@socketio.on('custom_event_multi_params')
def handle_custom_event_multi_params(arg1, arg2, arg3):
    print('received args: ' + arg1 + arg2 + arg3)    
#or general
# @socketio.event
# def my_custom_event(arg1, arg2, arg3):
#     print('received args: ' + arg1 + arg2 + arg3)

#to send to client @socketio.on('to_client')
def send_message(message):    
    #socketio.send(message)
    # socketio.emit('to_client', {'message': message}, callback=ack)   
    socketio.emit('response', {'message': message}, callback=ack) 
    # if (sid is not None):
    # socketio.emit('prova', {'message': message}, callback=ack, to=sid)

def ack(): 
    print('message was received!')
    
@socketio.on_error()        # Handles the default namespace
def error_handler(e):
    print(f'message was received! {e}')
    print(request.event["message"]) # "my error event"
    print(request.event["args"])    # (data,)
    pass

# Function to run the app and set the initial config
def start_socket(static_config):
    global config    
    config = static_config  # Set the global config to the passed config
    socketio.run(app, host='0.0.0.0', port=8001)



if __name__ == '__main__':
    config = {
        "param1": 10,
        "param2": "default_value",
        "episodes": 123
    }
    socketio.run(app, host='0.0.0.0', port=8001)
