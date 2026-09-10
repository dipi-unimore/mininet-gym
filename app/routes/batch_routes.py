from flask import Blueprint, jsonify


def create_batch_blueprint(state):
    """Read-only view of the batch runner's progress.

    utility/run_batch.py runs in this same process (it owns the main thread and
    drives start_experiment in a loop) and writes its progress into
    state['batch']. This endpoint just exposes that dict so the web UI can render
    a "3/20 - <config> - running" banner while the batch is going. There is no
    control here: pause/stop of the *current* experiment still goes through
    /pause_training and /stop_training.
    """
    bp = Blueprint("batch_routes", __name__)

    state.setdefault('batch', {'active': False})

    @bp.route('/batch_status', methods=['GET'])
    def batch_status():
        return jsonify(state.get('batch', {'active': False})), 200

    return bp
