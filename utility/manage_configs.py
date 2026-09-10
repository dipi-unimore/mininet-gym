#!/usr/bin/env python3
"""Tag staged experiment configs with the status of their training run.

For every gym type the results panel understands, this looks for a staged-config
directory ``config/<gym_type>/`` (e.g. ``config/marl_pz/``). If that directory
does not exist the gym type is skipped silently -- so on a checkout that only has
``config/marl_pz/`` only that one is touched.

For each config file found under such a directory (recursively -- ``tier1/``,
``tier2/`` ...), the matching run(s) in ``_training/<gym_type>/`` are looked up
and the file is renamed to record the outcome:

  ``<name>_<ddMMyyyyHHmm>.yaml``  -- a *complete* run exists (per the exact rules
                                    the /results panel uses); the stamp is the
                                    datetime of the most recent such run.
  ``<name>_incomplete.yaml``      -- a matching run folder exists but no run of
                                    it is complete yet.
  ``<name>.yaml``                 -- no matching run folder at all.

"Complete" is not re-implemented here: it is decided by
``app.services.results_service.build_results_dir_list`` -- the very function the
/results panel calls -- so this stays in sync with the panel automatically.

A config is matched to a run by a scenario-identity signature -- gym type,
network shape, ``random_seed``, episode/step/bin counts and (for ``marl_pz``)
the coordination strategy / strategies / UAQ ``tau`` -- plus the requirement that
every learning algorithm the config enables was also trained in the run. Attack
thresholds / timing and agent hyper-parameters are deliberately NOT compared:
they drift as the protocol is refined without changing which experiment cell a
run represents. Diff ``config.yaml`` by hand before treating an old run as a
drop-in for the current protocol.

The script is idempotent. Running it again never stacks a second marker: it
strips the marker it finds, recomputes the status from scratch and re-applies.
So across runs a file only moves when the underlying facts move --

  * a newer complete run appears        -> the stamp is bumped to the newer one
  * an ``_incomplete`` run completes    -> ``_incomplete`` becomes the stamp
  * the last complete run disappears    -> the stamp falls back to ``_incomplete``
  * no run matches any more             -> the marker is dropped

Usage:
    python3 utility/manage_configs.py                 # apply
    python3 utility/manage_configs.py --dry-run       # preview only
    python3 utility/manage_configs.py --gym-type marl_pz
    python3 utility/manage_configs.py --config-root config --training-dir _training
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.services.results_service import RESULT_GYM_TYPES, build_results_dir_list  # noqa: E402

RUN_DIR_RE = re.compile(r"^(\d{8}-\d{6})")
STAMP_FMT = "%d%m%Y%H%M"          # ddMMyyyyHHmm
_STAMP_RE = re.compile(r"_(\d{12})$")
_WORD_MARKER_RE = re.compile(r"_(?:incomplete|done|running|failed)$")


# --------------------------------------------------------------------------- #
# run datetime / markers
# --------------------------------------------------------------------------- #
def _run_datetime(run_dir_name):
    m = RUN_DIR_RE.match(run_dir_name)
    if not m:
        return None
    try:
        return _dt.datetime.strptime(m.group(1), "%Y%m%d-%H%M%S")
    except ValueError:
        return None


def _looks_like_stamp(digits):
    try:
        _dt.datetime.strptime(digits, STAMP_FMT)
        return True
    except ValueError:
        return False


def _strip_marker(stem):
    """Remove a trailing status marker this script may have added, idempotently.

    A ``_<12 digits>`` tail is only treated as a marker when it parses as a real
    ddMMyyyyHHmm value, so a config legitimately named ``..._202601011200`` (an
    ISO-ish stamp -- month 26 -> not a valid ddMMyyyyHHmm) is left alone.
    """
    changed = True
    while changed:
        changed = False
        new = _WORD_MARKER_RE.sub("", stem)
        if new != stem:
            stem, changed = new, True
            continue
        m = _STAMP_RE.search(stem)
        if m and _looks_like_stamp(m.group(1)):
            stem, changed = stem[: m.start()], True
    return stem


def _current_marker(stem):
    """What marker is on this filename right now: ('none'|'incomplete'|'running'|
    'failed'|'done'|'stamp', value).

    ``running`` / ``failed`` are written by ``utility/run_batch.py`` while it drives a
    batch of configs; this script treats them like any other word marker -- strip and
    recompute the status from ``_training/`` -- so a batch run and a later
    ``manage_configs`` pass converge on the same answer.
    """
    if _WORD_MARKER_RE.search(stem):
        for word in ("incomplete", "running", "failed", "done"):
            if stem.endswith(f"_{word}"):
                return word, None
        return "done", None
    m = _STAMP_RE.search(stem)
    if m and _looks_like_stamp(m.group(1)):
        return "stamp", m.group(1)
    return "none", None


# --------------------------------------------------------------------------- #
# matching: scenario-identity signature + learning-algorithm subset
# --------------------------------------------------------------------------- #
def _norm_strategy(value):
    if not value or str(value).lower() == "none":
        return "none"
    return str(value)


def _identity_key(raw_cfg):
    """Stable string identifying the *experiment cell* a config / run describes.

    Scenario shape + seed + episode budget + (marl_pz) the coordination knobs.
    Deliberately ignores attack thresholds / timing and agent hyper-parameters --
    those drift with protocol tweaks without changing the cell.
    """
    if not isinstance(raw_cfg, dict):
        return None
    env = raw_cfg.get("env_params", {}) or {}
    net = env.get("net_params", {}) or {}
    comm = env.get("communication", {}) or {}
    uaq = comm.get("uaq", {}) or {}
    strategies = comm.get("strategies") or []
    payload = {
        "gym_type": env.get("gym_type"),
        "net": [net.get("num_switches"), net.get("num_hosts"),
                net.get("num_iots", net.get("num_iot"))],
        "random_seed": raw_cfg.get("random_seed"),
        "episodes": env.get("episodes"),
        "max_steps": env.get("max_steps"),
        "test_episodes": env.get("test_episodes"),
        "n_bins": env.get("n_bins"),
        "comm_strategy": _norm_strategy(comm.get("strategy")),
        "comm_strategies": sorted(str(s) for s in strategies),
        "uaq_tau": uaq.get("tau"),
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _learning_algos(raw_cfg):
    """Lower-cased algorithms of the enabled, learning, non-supervised agents."""
    algos = set()
    for a in (raw_cfg.get("agents") or []):
        if not isinstance(a, dict) or not a.get("enabled"):
            continue
        if a.get("skip_learn", True):
            continue
        name = str(a.get("name", "")).lower()
        if name.startswith("supervised"):
            continue
        algo = str(a.get("algorithm", "")).lower()
        if algo:
            algos.add(algo)
    return algos


def _load_yaml(path):
    try:
        with open(path) as fh:
            return yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError):
        return None


def _index_runs(results_entry, training_dir):
    """Return a list of {key, algos, dt, complete} for every run of one gym type."""
    runs = []
    for is_complete, key_name in ((True, "data"), (False, "incomplete_data")):
        for entry in results_entry.get(key_name, []) or []:
            rel_path = entry.get("path")
            if not rel_path:
                continue
            dt = _run_datetime(os.path.basename(rel_path))
            if dt is None:
                continue
            raw = _load_yaml(os.path.join(training_dir, rel_path, "config.yaml"))
            key = _identity_key(raw)
            if key is None:
                continue
            runs.append({
                "key": key,
                "algos": _learning_algos(raw),
                "dt": dt,
                "complete": is_complete,
            })
    return runs


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #
def _same_file_content(path_a, path_b):
    try:
        with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
            return fa.read() == fb.read()
    except OSError:
        return False


def iter_config_files(config_dir):
    for dirpath, _dirs, files in os.walk(config_dir):
        for fname in sorted(files):
            if fname.endswith((".yaml", ".yml")) and not fname.startswith("."):
                yield os.path.join(dirpath, fname)


def plan_gym_type(config_dir, runs):
    """Return [(old_path, new_path, status, detail)] for one gym type's config dir."""
    plan = []
    for path in iter_config_files(config_dir):
        raw = _load_yaml(path)
        if raw is None:
            plan.append((path, path, "error", "cannot parse YAML"))
            continue
        if not isinstance(raw, dict) or "env_params" not in raw:
            continue

        key = _identity_key(raw)
        algos = _learning_algos(raw)
        matches = [r for r in runs if r["key"] == key and algos <= r["algos"]]
        complete = [r for r in matches if r["complete"]]
        incomplete = [r for r in matches if not r["complete"]]

        directory, fname = os.path.split(path)
        stem, ext = os.path.splitext(fname)
        old_kind, old_val = _current_marker(stem)
        base_stem = _strip_marker(stem)

        if complete:
            latest = max(r["dt"] for r in complete)
            stamp = latest.strftime(STAMP_FMT)
            new_stem = f"{base_stem}_{stamp}"
            status = "done"
            if old_kind == "stamp" and old_val == stamp:
                detail = f"unchanged - complete run {latest:%Y-%m-%d %H:%M:%S}"
            elif old_kind == "stamp":
                try:
                    prev = _dt.datetime.strptime(old_val, STAMP_FMT)
                    rel = "newer" if latest > prev else "older"
                except ValueError:
                    rel = "different"
                detail = f"{rel} complete run -> {latest:%Y-%m-%d %H:%M:%S} (stamp was {old_val})"
            elif old_kind == "incomplete":
                detail = f"completed -> stamped {latest:%Y-%m-%d %H:%M:%S}"
            else:
                detail = f"{len(complete)} complete run(s); latest {latest:%Y-%m-%d %H:%M:%S}"
        elif incomplete:
            latest = max(r["dt"] for r in incomplete)
            new_stem = f"{base_stem}_incomplete"
            status = "incomplete"
            if old_kind == "incomplete":
                detail = f"unchanged - run folder(s) present, none complete (latest {latest:%Y-%m-%d %H:%M:%S})"
            elif old_kind == "stamp":
                detail = f"no complete run any more -> back to incomplete (latest {latest:%Y-%m-%d %H:%M:%S})"
            else:
                detail = f"{len(incomplete)} run folder(s), none complete; latest {latest:%Y-%m-%d %H:%M:%S}"
        else:
            new_stem = base_stem
            status = "pending"
            detail = "no matching run folder" if old_kind == "none" else "no matching run any more - marker removed"

        new_path = os.path.join(directory, f"{new_stem}{ext}")
        plan.append((path, new_path, status, detail))
    return plan


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def _apply(plan, dry_run):
    counts = {"done": 0, "incomplete": 0, "pending": 0, "error": 0}
    width = max((len(os.path.basename(o)) for o, *_ in plan), default=0)
    renamed = 0
    for old_path, new_path, status, detail in plan:
        counts[status] = counts.get(status, 0) + 1
        arrow = "" if old_path == new_path else f"  ->  {os.path.basename(new_path)}"
        print(f"  [{status:10}] {os.path.basename(old_path):<{width}}{arrow}")
        print(f"               {detail}")

        if old_path == new_path or status == "error":
            continue
        if os.path.exists(new_path):
            if _same_file_content(old_path, new_path):
                print(f"               de-duplicating identical '{os.path.basename(new_path)}'")
                if not dry_run:
                    os.replace(old_path, new_path)
                    renamed += 1
            else:
                print(f"               CONFLICT: '{os.path.basename(new_path)}' exists "
                      f"with different content - resolve by hand")
            continue
        if not dry_run:
            os.rename(old_path, new_path)
            renamed += 1
    return counts, renamed


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-root", default="config",
                        help="root holding the per-gym-type config dirs (default: config)")
    parser.add_argument("--training-dir", default="_training",
                        help="training results root (default: _training)")
    parser.add_argument("--gym-type", action="append", default=None,
                        help="limit to this gym type (repeatable); default: all present")
    parser.add_argument("--dry-run", action="store_true", help="print the plan without renaming")
    args = parser.parse_args(argv)

    config_root = args.config_root if os.path.isabs(args.config_root) else os.path.join(REPO_ROOT, args.config_root)
    training_dir = args.training_dir if os.path.isabs(args.training_dir) else os.path.join(REPO_ROOT, args.training_dir)

    gym_types = [g for g in RESULT_GYM_TYPES if not args.gym_type or g in args.gym_type]
    present = [g for g in gym_types if os.path.isdir(os.path.join(config_root, g))]
    if not present:
        print(f"no config/<gym_type>/ directory found under {config_root} - nothing to do")
        return 0

    print(f"scanning runs in {training_dir} ...")
    results = build_results_dir_list({"training_directory": training_dir})
    results_by_gym = {r["gym_type"]: r for r in results}

    grand = {"done": 0, "incomplete": 0, "pending": 0, "error": 0}
    grand_renamed = 0
    for gym_type in present:
        entry = results_by_gym.get(gym_type, {"data": [], "incomplete_data": []})
        runs = _index_runs(entry, training_dir)
        config_dir = os.path.join(config_root, gym_type)
        print(f"\n=== {gym_type} ===  ({len(runs)} run folder(s) in {args.training_dir}/{gym_type})")
        plan = plan_gym_type(config_dir, runs)
        if not plan:
            print("  (no config files)")
            continue
        counts, renamed = _apply(plan, args.dry_run)
        for k, v in counts.items():
            grand[k] += v
        grand_renamed += renamed

    print()
    print(f"total: done={grand['done']}  incomplete={grand['incomplete']}  "
          f"pending={grand['pending']}  error={grand['error']}")
    print("(dry run - nothing renamed)" if args.dry_run else f"renamed {grand_renamed} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
