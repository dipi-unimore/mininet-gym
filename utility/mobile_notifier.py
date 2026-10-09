"""Optional mobile push notifications for experiment lifecycle events.

The whole feature is driven by a single optional file at the repository root:
``config_notify_mobile.yaml``. Behaviour:

* file missing                        -> feature disabled, nothing is ever sent
* file present, ``notify_mobile`` absent / false  -> same as missing
* file present, ``notify_mobile: true``           -> feature enabled

So the config file can simply not exist; when it does exist it decides *whether*
and *what* to notify to the phone.

Supported providers: ``ntfy`` (default), ``telegram``, ``pushover``.

Every send is best-effort: it runs on a daemon thread with a short timeout and
never raises into the caller. A failure is only logged.
"""

import os
import threading

import yaml

_CONFIG_FILENAME = "config_notify_mobile.yaml"

_config = None          # parsed dict when enabled, {} when disabled
_config_loaded = False   # _load_config() memoises after the first call


def _repo_root():
    # utility/mobile_notifier.py -> repo root
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _log_error(message):
    # Imported lazily to avoid an import cycle (my_log imports this module).
    try:
        from utility.my_log import error
        error(f"[mobile_notifier] {message}\n")
    except Exception:
        pass


def _log_info(message):
    try:
        from utility.my_log import information
        information(f"[mobile_notifier] {message}\n")
    except Exception:
        pass


def _load_config():
    """Return the config dict when the feature is enabled, otherwise ``{}``."""
    global _config, _config_loaded
    if _config_loaded:
        return _config

    _config_loaded = True
    _config = {}

    path = os.path.join(_repo_root(), _CONFIG_FILENAME)
    if not os.path.isfile(path):
        return _config

    try:
        with open(path, "r") as handle:
            raw = yaml.safe_load(handle) or {}
    except Exception as exc:
        _log_error(f"could not read {_CONFIG_FILENAME}: {exc}")
        return _config

    if not isinstance(raw, dict) or raw.get("notify_mobile") is not True:
        # File is there but the master switch is off / missing: act as if absent.
        return _config

    _config = raw
    _log_info(f"enabled (provider: {(raw.get('provider') or 'ntfy')})")
    return _config


def is_enabled():
    return bool(_load_config())


def reload_config():
    """Forget the cached config so the next call re-reads the file (tests / REPL)."""
    global _config, _config_loaded
    _config = None
    _config_loaded = False


def notify(title, body, event="finish", blocking=False):
    """Send a push notification for ``event`` ("finish" or "error").

    No-op when the feature is disabled or the specific event is switched off in
    the config.

    By default the HTTP call runs on a background thread. That thread is
    *non-daemon* on purpose: a short-lived process (e.g. the very end of a run,
    or a one-liner) then waits for the send to finish instead of exiting and
    killing it mid-request. ``blocking=True`` sends inline and returns
    ``(ok, detail)`` — used by the test helper.
    """
    config = _load_config()
    if not config:
        return (False, "feature disabled") if blocking else None

    if event == "finish" and not config.get("on_finish", True):
        return (False, "on_finish is false") if blocking else None
    if event == "error" and not config.get("on_error", True):
        return (False, "on_error is false") if blocking else None

    if blocking:
        return _send(config, str(title), str(body))

    thread = threading.Thread(
        target=_send, args=(config, str(title), str(body)), daemon=False
    )
    thread.start()
    return thread


def _send(config, title, body):
    """Perform the HTTP request. Returns ``(ok: bool, detail: str)``; never raises."""
    provider = (config.get("provider") or "ntfy").lower()
    try:
        import requests

        if provider == "ntfy":
            params = config.get("ntfy") or {}
            server = (params.get("server") or "https://ntfy.sh").rstrip("/")
            topic = params.get("topic")
            if not topic:
                return _fail("ntfy.topic is not set")
            headers = {"Title": _header_safe(title)}
            if params.get("priority"):
                headers["Priority"] = str(params["priority"])
            if params.get("tags"):
                headers["Tags"] = str(params["tags"])
            response = requests.post(
                f"{server}/{topic}",
                data=body.encode("utf-8"),
                headers=headers,
                timeout=10,
            )

        elif provider == "telegram":
            params = config.get("telegram") or {}
            token = params.get("bot_token")
            chat_id = params.get("chat_id")
            if not token or not chat_id:
                return _fail("telegram.bot_token / telegram.chat_id are not set")
            response = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": f"*{title}*\n{body}",
                    "parse_mode": "Markdown",
                },
                timeout=10,
            )

        elif provider == "pushover":
            params = config.get("pushover") or {}
            token = params.get("token")
            user = params.get("user")
            if not token or not user:
                return _fail("pushover.token / pushover.user are not set")
            response = requests.post(
                "https://api.pushover.net/1/messages.json",
                data={
                    "token": token,
                    "user": user,
                    "title": title,
                    "message": body,
                },
                timeout=10,
            )

        else:
            return _fail(f"unknown provider: {provider}")

        if response.status_code >= 300:
            snippet = (response.text or "").strip().replace("\n", " ")[:300]
            return _fail(f"{provider} HTTP {response.status_code}: {snippet}")
        return (True, f"{provider} HTTP {response.status_code}")

    except Exception as exc:
        return _fail(f"send failed ({provider}): {exc!r}")


def _fail(message):
    _log_error(message)
    return (False, message)


def _header_safe(text):
    # HTTP header values must be latin-1 / ASCII; ntfy reads the title from a
    # header, so strip anything that would make requests raise.
    return text.encode("ascii", "replace").decode("ascii")


def build_run_params_summary(config):
    """Compact one-block digest of the main experiment settings.

    ``config`` is the ``Params`` object built by ``utility.params.read_config_file``
    (nested attribute access, absent fields simply skipped). Returns "" when
    ``config`` is ``None`` or nothing useful could be read from it, so the caller
    can decide what to append.
    """
    if config is None:
        return ""
    try:
        ep = getattr(config, 'env_params', None)
        parts = []

        gym_type = getattr(ep, 'gym_type', None) if ep is not None else None
        seed = getattr(config, 'random_seed', None)
        head = []
        if gym_type:
            head.append(str(gym_type))
        if seed is not None:
            head.append(f"seed {seed}")
        if head:
            parts.append("scenario: " + ", ".join(head))

        comm = getattr(ep, 'communication', None) if ep is not None else None
        if comm is not None:
            strategy = getattr(comm, 'strategy', None)
            strategies = getattr(comm, 'strategies', None)
            use_coord = getattr(comm, 'use_coordinator', None)
            comm_bits = []
            if strategy:
                comm_bits.append(str(strategy))
            elif strategies:
                comm_bits.append("+".join(str(s) for s in strategies))
            if use_coord:
                comm_bits.append("coordinator")
            if comm_bits:
                parts.append("comm: " + " / ".join(comm_bits))

        net = getattr(ep, 'net_params', None) if ep is not None else None
        if net is not None:
            num_hosts = getattr(net, 'num_hosts', None)
            num_iots = getattr(net, 'num_iots', None)
            if num_hosts is not None:
                topo = f"{num_hosts} hosts"
                if num_iots:
                    topo += f" + {num_iots} iot"
                parts.append("topology: " + topo)

        if ep is not None:
            episodes = getattr(ep, 'episodes', None)
            test_episodes = getattr(ep, 'test_episodes', None)
            max_steps = getattr(ep, 'max_steps', None)
            ep_bits = []
            if episodes is not None:
                ep_bits.append(f"train {episodes} ep")
            if test_episodes is not None:
                ep_bits.append(f"test {test_episodes} ep")
            if max_steps is not None:
                ep_bits.append(f"{max_steps} steps/ep")
            if ep_bits:
                parts.append("episodes: " + ", ".join(ep_bits))

        agents = getattr(config, 'agents', None) or []
        algos = []
        for agent in agents:
            if not getattr(agent, 'enabled', True):
                continue
            algo = getattr(agent, 'algorithm', None)
            name = getattr(agent, 'name', None)
            algos.append(f"{algo}({name})" if algo and name else str(algo or name or '?'))
        if algos:
            parts.append("agents: " + ", ".join(algos))

        return "Params:\n" + "\n".join(parts) if parts else ""
    except Exception:
        return ""


def build_run_summary():
    """Compact per-agent accuracy / f1 line-up from the cached run summaries.

    Returns "" when nothing useful is cached (so the caller can decide what to
    append).
    """
    try:
        from utility.my_log import get_agent_summaries
        summaries = get_agent_summaries() or {}
    except Exception:
        return ""

    lines = []
    for agent_name, entry in summaries.items():
        if not isinstance(entry, dict):
            continue
        metrics = (entry.get("evaluation") or {}).get("latest_metrics") or {}
        accuracy = metrics.get("accuracy")
        f1_score = metrics.get("f1_score")
        parts = []
        if accuracy is not None:
            parts.append(f"acc {accuracy:.3f}")
        if f1_score is not None:
            parts.append(f"f1 {f1_score:.3f}")
        lines.append(f"- {agent_name}: {', '.join(parts) if parts else 'done'}")

    return "\n".join(lines)
