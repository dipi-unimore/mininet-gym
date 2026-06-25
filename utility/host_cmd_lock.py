import threading

_host_cmd_locks: dict = {}
_host_cmd_locks_lock = threading.Lock()


def get_host_cmd_lock(host) -> threading.RLock:
    """Return (creating if needed) a per-host RLock.

    RLock is used so that a call chain where an outer function already holds
    the lock (e.g. launch_udp_flood → prepare_attacker_for_dos) can re-enter
    without deadlocking.
    """
    host_name = getattr(host, 'name', str(id(host)))
    with _host_cmd_locks_lock:
        lock = _host_cmd_locks.get(host_name)
        if lock is None:
            lock = threading.RLock()
            _host_cmd_locks[host_name] = lock
        return lock
