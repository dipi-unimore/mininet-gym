#!/usr/bin/env python3
"""Run a folder of experiment configs one after another, unattended.

This is the launching half of ``utility/manage_configs.py``: that script tags a
config file with the status of its *existing* run; this one actually *runs* the
configs it finds and tags them as it goes --

  ``<name>.yaml``               -> ``<name>_running.yaml``   while it is training
  (on success)                  -> ``<name>_<ddMMyyyyHHmm>.yaml``  (completion stamp,
                                   identical to what manage_configs would apply)
  (on failure)                  -> ``<name>_failed.yaml``    (and the batch continues)
  (user stop / Ctrl-C)          -> back to ``<name>.yaml``   (batch halts)

"Success" is not re-implemented here: after each run the just-created run folder is
checked against ``app.services.results_service.build_results_dir_list`` -- the same
authority the /results panel and manage_configs use.

The whole batch runs in ONE process: the web app (Flask + SocketIO) is started in a
background thread exactly as ``main.py`` does, and each experiment is launched the same
way the *Start Training* button does (``main.start_experiment`` in a thread, tracked in
the shared ``app.app_api.state``). So while the batch runs you can open the web UI and
watch the current experiment live -- charts, log, status -- plus a batch-progress banner
fed by ``GET /batch_status``. Once the batch finishes, the process does NOT exit: it
falls back to the same idle, ready-for-interactive-use state ``main.py`` leaves you in
(pass ``--exit-after`` to get the old exit-on-completion behavior, e.g. for cron).

The same core (``run_batch()``) is also what the web UI's Batch Runner panel calls --
picking a folder there and pressing *Start Batch* runs this exact code in a background
thread inside the already-running app, no terminal needed.

Must run as root (Mininet), like ``main.py``:

    sudo python3 utility/run_batch.py config/marl_pz/tier1
    sudo python3 utility/run_batch.py config/marl_pz --gym-type marl_pz --dry-run
    sudo python3 utility/run_batch.py config/marl_pz/tier2 --retry --sync-after
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import signal
import sys
import threading
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.app_api import state, start_api  # noqa: E402
from app.services.results_service import build_results_dir_list  # noqa: E402
from main import start_experiment  # noqa: E402
from utility import manage_configs as mc  # noqa: E402
from utility.config_sanitizer import clean_load_dir  # noqa: E402
from utility.constants import GYM_TYPE  # noqa: E402
from utility.params import read_config_file  # noqa: E402

_aborting = threading.Event()


def _abort_requested():
    """True when the batch (not just the current experiment) should stop:
    our own SIGINT/SIGTERM handler, or the web-UI Stop button while a batch is
    active (training_routes.py sets state['batch']['abort'])."""
    return _aborting.is_set() or bool(state.get("batch", {}).get("abort"))


# --------------------------------------------------------------------------- #
# batch status (exposed to the web UI via GET /batch_status)
# --------------------------------------------------------------------------- #
def _now():
    return _dt.datetime.now().isoformat(timespec="seconds")


def _set_batch(**changes):
    batch = state.setdefault("batch", {})
    batch.update(changes)
    batch["updated_at"] = _now()


def _record_result(fname, status, run=None, error=None):
    entry = {"file": fname, "status": status}
    if run:
        entry["run"] = run
    if error:
        entry["error"] = str(error)[:500]
    state["batch"].setdefault("results", []).append(entry)
    _set_batch()


# --------------------------------------------------------------------------- #
# filename markers (reuse manage_configs conventions)
# --------------------------------------------------------------------------- #
def _marked_path(path, marker):
    """``/x/foo_running.yaml`` -> ``/x/foo_<marker>.yaml`` (or bare when marker is None)."""
    directory, fname = os.path.split(path)
    stem, ext = os.path.splitext(fname)
    base = mc._strip_marker(stem)
    new_stem = base if not marker else f"{base}_{marker}"
    return os.path.join(directory, f"{new_stem}{ext}")


def _rename(src, dst):
    if os.path.abspath(src) == os.path.abspath(dst):
        return src
    os.replace(src, dst)  # our own markers -- last writer wins
    return dst


# --------------------------------------------------------------------------- #
# completion check -- delegate to the /results authority
# --------------------------------------------------------------------------- #
def _run_is_complete(training_dir, gym_type, run_basename):
    try:
        results = build_results_dir_list({"training_directory": training_dir})
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  ! could not evaluate run completeness: {exc}")
        return False
    for entry in results:
        if entry.get("gym_type") != gym_type:
            continue
        for data in entry.get("data", []) or []:
            rel = (data.get("path") or "").rstrip("/")
            if rel and os.path.basename(rel) == run_basename:
                return True
    return False


# --------------------------------------------------------------------------- #
# discovery / planning
# --------------------------------------------------------------------------- #
def _plan(config_dir, gym_type_filter, force, retry):
    """Return (todo, skipped) where todo is [(path, cfg_dict)] in run order."""
    todo, skipped = [], []
    for path in mc.iter_config_files(config_dir):
        fname = os.path.basename(path)
        if fname == "default.yaml":
            # never touch the live app config, even if someone points a batch at
            # the top-level config/ folder that holds it (the /start_batch route
            # already refuses that folder outright; this is the CLI's backstop).
            skipped.append((fname, "refusing to treat config/default.yaml as a batch config"))
            continue
        raw = mc._load_yaml(path)
        if not isinstance(raw, dict) or "env_params" not in raw:
            skipped.append((fname, "not an experiment config"))
            continue

        gym_type = (raw.get("env_params") or {}).get("gym_type")
        if gym_type_filter and gym_type != gym_type_filter:
            skipped.append((fname, f"gym_type {gym_type} != {gym_type_filter}"))
            continue

        stem = os.path.splitext(fname)[0]
        kind, _ = mc._current_marker(stem)
        if kind == "stamp" and not force:
            skipped.append((fname, "already completed (use --force)"))
            continue
        if kind in ("running", "failed") and not retry:
            skipped.append((fname, f"marked _{kind} (use --retry)"))
            continue

        try:
            _cfg_params, cfg_dict = read_config_file(
                path, config_cleaner=lambda cfg: clean_load_dir(cfg, GYM_TYPE.keys())
            )
        except Exception as exc:
            skipped.append((fname, f"cannot load: {exc}"))
            continue
        cfg_dict["enable_web_interface"] = True
        cfg_dict.pop("training_execution_directory", None)
        todo.append((path, cfg_dict))
    return todo, skipped


# --------------------------------------------------------------------------- #
# one experiment
# --------------------------------------------------------------------------- #
def _run_one(path, cfg_dict, training_dir_override):
    """Run a single config to completion. Returns ('done'|'failed'|'stopped', detail)."""
    from threading import Event

    running_path = _rename(path, _marked_path(path, "running"))
    fname = os.path.basename(running_path)
    _set_batch(current_file=fname, current_status="running")

    err = {}

    def _target():
        try:
            start_experiment(cfg_dict, state["pause_event"], state["stop_event"])
        except BaseException as exc:  # start_experiment usually swallows its own
            err["error"] = repr(exc)

    state["pause_event"] = Event()
    state["stop_event"] = Event()
    state["current_config"] = cfg_dict
    start_experiment.env = {"param": 123}

    worker = threading.Thread(target=_target, name="batch-experiment")
    state["training_thread"] = worker
    worker.start()

    signalled = False
    while worker.is_alive():
        worker.join(timeout=1.0)
        if _abort_requested() and not signalled:
            print("  .. abort requested -- signalling the current experiment to stop")
            state["stop_event"].set()
            signalled = True

    run_dir = cfg_dict.get("training_execution_directory")
    run_basename = os.path.basename(run_dir.rstrip("/")) if run_dir else None
    training_dir = training_dir_override or cfg_dict.get("training_directory", "_training")

    # NOTE: do NOT read state['stop_event'].is_set() as "user aborted". Every
    # normal experiment ends by calling env.stop() (network_env.py), which *sets*
    # the stop_event as part of teardown -- so it is set after every run whether
    # it succeeded or crashed. The only signals that mean "stop the batch" are our
    # SIGINT/SIGTERM handler and the web-UI Stop button (_abort_requested()).
    # done-vs-failed is decided by the /results completeness check below.
    if _abort_requested():
        _rename(running_path, _marked_path(running_path, None))
        return "stopped", "Ctrl-C / web-UI Stop"

    if run_basename and _run_is_complete(training_dir, cfg_dict["env_params"]["gym_type"], run_basename):
        stamp_dt = mc._run_datetime(run_basename)
        stamp = stamp_dt.strftime(mc.STAMP_FMT) if stamp_dt else _dt.datetime.now().strftime(mc.STAMP_FMT)
        final = _rename(running_path, _marked_path(running_path, stamp))
        return "done", f"{os.path.basename(final)}  (run {run_basename})"

    _rename(running_path, _marked_path(running_path, "failed"))
    detail = err.get("error") or f"run folder {run_basename} not complete"
    _mininet_cleanup()
    return "failed", detail


def _mininet_cleanup():
    """Best-effort teardown after a hard failure (normal runs self-clean on next build)."""
    try:
        import mininet.clean

        mininet.clean.cleanup()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  ! mininet cleanup skipped: {exc}")


def _print_plan(print_fn, rel_dir, todo, skipped):
    print_fn(f"\n=== batch: {rel_dir} ===")
    for fname, reason in skipped:
        print_fn(f"  [skip] {fname}  -- {reason}")
    for path, _cfg in todo:
        print_fn(f"  [run ] {os.path.basename(path)}")
    print_fn(f"\n{len(todo)} to run, {len(skipped)} skipped")


# --------------------------------------------------------------------------- #
# the reusable core -- called by main() below (CLI) and by /start_batch (web UI)
# --------------------------------------------------------------------------- #
def run_batch(config_dir, gym_type=None, force=False, retry=False,
              training_dir=None, sync_after=False, print_fn=print):
    """Plan and run every eligible config under ``config_dir``, in the CALLING
    thread, to completion or abort. Does not touch the web server -- the caller
    is expected to already have one running (main.py's, or this module's own
    CLI bootstrap via _start_web_server). Safe to call from a background thread
    (the /start_batch route does exactly that).

    Returns a summary dict; also visible live via state['batch'] / GET /batch_status.
    """
    config_dir = config_dir if os.path.isabs(config_dir) else os.path.join(REPO_ROOT, config_dir)
    if not os.path.isdir(config_dir):
        print_fn(f"not a directory: {config_dir}")
        return {"error": f"not a directory: {config_dir}"}

    todo, skipped = _plan(config_dir, gym_type, force, retry)
    rel_dir = os.path.relpath(config_dir, REPO_ROOT)
    _print_plan(print_fn, rel_dir, todo, skipped)

    _set_batch(
        active=True, abort=False, config_dir=rel_dir, total=len(todo), index=0,
        current_file=None, current_status="starting",
        started_at=_now(), results=[],
        skipped=[{"file": f, "reason": r} for f, r in skipped],
    )

    counts = {"done": 0, "failed": 0, "stopped": 0}
    for idx, (path, cfg_dict) in enumerate(todo, start=1):
        if _abort_requested():
            print_fn("aborted before starting the next experiment")
            break
        print_fn(f"\n--- [{idx}/{len(todo)}] {os.path.basename(path)} ---")
        _set_batch(index=idx, current_file=os.path.basename(path), current_status="running")
        status, detail = _run_one(path, cfg_dict, training_dir)
        counts[status] = counts.get(status, 0) + 1
        _record_result(os.path.basename(_marked_path(path, None)), status, error=detail if status == "failed" else None)
        print_fn(f"    -> {status}: {detail}")
        if status == "stopped":
            break

    final_status = "stopped" if (_abort_requested() or counts["stopped"]) else "finished"
    _set_batch(active=False, current_status=final_status, current_file=None)
    print_fn(f"\n=== batch {final_status}: {counts['done']} done, {counts['failed']} failed, "
             f"{counts['stopped']} stopped ===")

    if sync_after and not _abort_requested():
        print_fn("\nreconciling markers with _training/ (manage_configs) ...")
        gym_args = ["--gym-type", gym_type] if gym_type else []
        mc.main(gym_args)

    return {"config_dir": rel_dir, "total": len(todo), "skipped": len(skipped),
            "status": final_status, **counts}


# --------------------------------------------------------------------------- #
# main (CLI only)
# --------------------------------------------------------------------------- #
def _start_web_server(port):
    """Boot the Flask+SocketIO app in a daemon thread, CLI-only (a UI-launched
    batch already has one running -- this is main.py's job then)."""
    _defaults, default_dict = read_config_file(
        os.path.join(REPO_ROOT, "config", "default.yaml"),
        config_cleaner=lambda cfg: clean_load_dir(cfg, GYM_TYPE.keys()),
    )
    web_port = port or default_dict.get("web_server_port", 5000)
    start_experiment.env = {"param": 123}
    thread = threading.Thread(
        target=start_api,
        args=(start_experiment, default_dict, "127.0.0.1", web_port),
        daemon=True,
        name="batch-web-api",
    )
    thread.start()
    for _ in range(100):
        if state.get("socketio_instance") is not None:
            break
        time.sleep(0.1)
    print(f"web UI: http://127.0.0.1:{web_port}  (batch progress banner on the Training page)")
    return web_port, thread, default_dict


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config_dir", help="folder of config .yaml files to run (searched recursively)")
    parser.add_argument("--gym-type", default=None, help="only run configs of this gym type")
    parser.add_argument("--port", type=int, default=None, help="web UI port (default: config web_server_port)")
    parser.add_argument("--force", action="store_true", help="also re-run configs already stamped complete")
    parser.add_argument("--retry", action="store_true", help="also (re-)run configs marked _running / _failed")
    parser.add_argument("--training-dir", default=None, help="override training results root for the completeness check")
    parser.add_argument("--dry-run", action="store_true", help="list what would run, touch nothing")
    parser.add_argument("--sync-after", action="store_true",
                        help="run manage_configs after the batch to reconcile every marker with _training/")
    parser.add_argument("--exit-after", action="store_true",
                        help="exit once the batch finishes instead of staying up as a normal interactive app "
                             "(use for unattended/cron launches; default is to keep serving the web UI)")
    args = parser.parse_args(argv)

    config_dir = args.config_dir if os.path.isabs(args.config_dir) else os.path.join(REPO_ROOT, args.config_dir)
    if not os.path.isdir(config_dir):
        print(f"not a directory: {config_dir}")
        return 2

    if args.dry_run:
        todo, skipped = _plan(config_dir, args.gym_type, args.force, args.retry)
        _print_plan(print, os.path.relpath(config_dir, REPO_ROOT), todo, skipped)
        print("(dry run -- nothing launched)")
        return 0

    def _on_signal(signum, _frame):
        print(f"\nreceived signal {signum} -- finishing the current experiment then stopping")
        _aborting.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    web_port, web_thread, default_dict = _start_web_server(args.port)

    run_batch(config_dir, gym_type=args.gym_type, force=args.force, retry=args.retry,
              training_dir=args.training_dir, sync_after=args.sync_after)

    # Back to a plain, on-disk-default config -- exactly what a fresh
    # `sudo python3 main.py` would show, not whichever config ran last.
    state["current_config"] = default_dict

    if args.exit_after:
        return 0

    # The batch's signal handler only sets a flag; restore normal Ctrl-C/kill
    # behavior now that there is no batch left to finish cleanly.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    print(f"\nbatch finished -- app still running at http://127.0.0.1:{web_port} (Ctrl-C to quit)")
    web_thread.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
