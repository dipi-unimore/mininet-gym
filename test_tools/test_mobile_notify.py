#!/usr/bin/env python3
"""Send a test mobile notification without running an experiment.

Reads config_notify_mobile.yaml exactly like a real run does and pushes a
message through the configured provider, so you can confirm the phone actually
receives it.

Usage (from the repo root):

    python test_tools/test_mobile_notify.py              # plain "hello" push
    python test_tools/test_mobile_notify.py --finish     # simulate end-of-run
    python test_tools/test_mobile_notify.py --error      # simulate a failed run
    python test_tools/test_mobile_notify.py --dry-run    # don't send, just report

The --finish / --error paths go through utility.my_log, i.e. the same code an
experiment triggers.
"""

import argparse
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utility import mobile_notifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finish", action="store_true",
                        help="simulate the end-of-run push (via my_log hook)")
    parser.add_argument("--error", action="store_true",
                        help="simulate the run-failed push (via my_log hook)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report config only, send nothing")
    parser.add_argument("--message", default="Test notification from mininet-gym.",
                        help="body text for the plain push")
    args = parser.parse_args()

    mobile_notifier.reload_config()
    config = mobile_notifier._load_config()

    if not config:
        print("Feature DISABLED: config_notify_mobile.yaml is missing, or "
              "'notify_mobile' is not set to true.")
        print("  -> cp config_notify_mobile.yaml.example config_notify_mobile.yaml")
        return 1

    provider = (config.get("provider") or "ntfy").lower()
    print(f"Feature ENABLED  provider={provider}  "
          f"on_finish={config.get('on_finish', True)}  "
          f"on_error={config.get('on_error', True)}")
    if provider == "ntfy":
        p = config.get("ntfy") or {}
        print(f"  ntfy: {(p.get('server') or 'https://ntfy.sh').rstrip('/')}/{p.get('topic')}")

    if args.dry_run:
        print("--dry-run: nothing sent.")
        return 0

    if args.finish:
        print("Sending simulated end-of-run notification (via my_log hook) ...")
        from utility.my_log import _push_mobile_run_finished
        _push_mobile_run_finished("Finished. Ready to start again... (simulated)")
        _join_pending()
        ok, detail = True, "dispatched via _push_mobile_run_finished (see above)"
    elif args.error:
        print("Sending simulated run-failed notification (via my_log hook) ...")
        from utility.my_log import notify_mobile_error
        notify_mobile_error("simulated failure: RuntimeError('boom')")
        _join_pending()
        ok, detail = True, "dispatched via notify_mobile_error (see above)"
    else:
        print("Sending plain test notification (blocking) ...")
        ok, detail = mobile_notifier.notify(
            "mininet-gym: test", args.message, event="finish", blocking=True
        )

    print(f"\nResult: {'OK' if ok else 'FAILED'}  -  {detail}")
    if ok:
        print("Check your device.")
    return 0 if ok else 2


def _join_pending():
    """Wait for the non-daemon send thread started by the my_log hooks."""
    for thread in threading.enumerate():
        if thread is not threading.current_thread() and thread.is_alive():
            thread.join(timeout=12)


if __name__ == "__main__":
    raise SystemExit(main())
