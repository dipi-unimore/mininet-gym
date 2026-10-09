import os
import threading

from flask import Blueprint, jsonify, request

_ROUTES_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_ROUTES_DIR))  # app/routes/.. /..


def create_batch_blueprint(state):
    """Batch runner: progress view + launching a batch from the web UI.

    utility/run_batch.py's core loop (run_batch()) runs in this same process --
    either the CLI bootstrap's main thread, or (here) a background thread started
    by /start_batch -- and writes its progress into state['batch']. /batch_status
    just exposes that dict so the web UI can render a "3/20 - <config> - running"
    banner while the batch is going. Pause/stop of the *current* experiment still
    goes through /pause_training and /stop_training (Stop also halts the whole
    batch -- see training_routes.py).
    """
    bp = Blueprint("batch_routes", __name__)

    state.setdefault('batch', {'active': False})

    @bp.route('/batch_status', methods=['GET'])
    def batch_status():
        return jsonify(state.get('batch', {'active': False})), 200

    @bp.route('/list_batch_dirs', methods=['GET'])
    def list_batch_dirs():
        """Every directory under config/ that directly holds staged experiment
        configs (config/marl_pz and config/marl_pz/tier1 both show up separately),
        with to-run/done/stuck counts for the picker. The top-level config/ folder
        itself is deliberately excluded -- it also holds config/default.yaml and
        one-off saved snapshots, which utility/run_batch.py must never rename."""
        from utility.run_batch import _plan

        config_root = os.path.join(REPO_ROOT, 'config')
        candidate_dirs = []
        for root, _dirnames, files in os.walk(config_root):
            if root == config_root:
                continue
            if any(f.endswith(('.yaml', '.yml')) and not f.startswith('.') for f in files):
                candidate_dirs.append(root)
        candidate_dirs.sort()

        rows = []
        for abs_dir in candidate_dirs:
            todo, skipped = _plan(abs_dir, None, force=False, retry=False)
            done = sum(1 for _f, reason in skipped if reason.startswith('already completed'))
            stuck = sum(1 for _f, reason in skipped if reason.startswith('marked _'))
            rows.append({
                "path": os.path.relpath(abs_dir, REPO_ROOT),
                "to_run": len(todo),
                "done": done,
                "running_or_failed": stuck,
                "total": len(todo) + len(skipped),
            })
        return jsonify({"dirs": rows}), 200

    @bp.route('/start_batch', methods=['POST'])
    def start_batch():
        if state.get('training_thread') is not None and state['training_thread'].is_alive():
            return jsonify({"status": "error",
                            "message": "Cannot start a batch while training is running."}), 409
        if state.get('calibration_thread') is not None and state['calibration_thread'].is_alive():
            return jsonify({"status": "error",
                            "message": "Cannot start a batch while threshold calibration is running."}), 409
        if isinstance(state.get('batch'), dict) and state['batch'].get('active'):
            return jsonify({"status": "error", "message": "A batch is already running."}), 409

        payload = request.get_json(silent=True) or {}
        rel_dir = str(payload.get('config_dir') or '').strip()
        if not rel_dir:
            return jsonify({"status": "error", "message": "config_dir is required"}), 400

        abs_dir = os.path.abspath(os.path.join(REPO_ROOT, rel_dir))
        if abs_dir != REPO_ROOT and not abs_dir.startswith(REPO_ROOT + os.sep):
            return jsonify({"status": "error", "message": "Invalid config_dir"}), 403
        if os.path.abspath(os.path.join(REPO_ROOT, 'config')) == abs_dir:
            return jsonify({"status": "error",
                            "message": "Refusing to batch the top-level config/ folder (it holds default.yaml)."}), 400
        if not os.path.isdir(abs_dir):
            return jsonify({"status": "error", "message": f"Not a directory: {rel_dir}"}), 404

        from utility.run_batch import run_batch

        kwargs = dict(
            config_dir=abs_dir,
            gym_type=(str(payload.get('gym_type') or '').strip() or None),
            force=bool(payload.get('force')),
            retry=bool(payload.get('retry')),
            sync_after=bool(payload.get('sync_after')),
        )
        thread = threading.Thread(target=run_batch, kwargs=kwargs, daemon=True, name="ui-batch")
        thread.start()
        return jsonify({"status": "starting", "message": f"Batch starting on {rel_dir}"}), 200

    return bp
