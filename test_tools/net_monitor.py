#!/usr/bin/env python3.12
"""
OVS per-host traffic monitor with Q-table bin display and statuses.json replay.

LIVE mode (default):
    sudo python3 test_tools/net_monitor.py [--bridge s1] [--interval 1]

REPLAY mode (fully standalone — no separate Mininet needed):
    sudo python3 test_tools/net_monitor.py \
        --replay _training/attacks_ho/XXXX/statuses.json --launch

    --launch  reads the host list from statuses.json, creates a single-switch
              Mininet topology automatically, runs the replay, then tears it down.
    Without --launch the script uses any already-running Mininet namespaces.

    Loads a statuses.json from a past experiment, generates the same traffic
    step-by-step via 'ip netns exec', reads OVS counters and compares
    observed vs expected traffic, flagging anomalies.

Columns (live and replay):
    TX pkt/s  TX B/s   RX pkt/s  RX B/s   BINS         TX total  RX total
    (TX = host transmits = port RX counter)
    (RX = host receives  = port TX counter)

BINS [TXp,TXb|RXp,RXb]:
    Each digit is the Q-table bin assigned to that value.
    0=idle  1=normal  2=attack-range  3=extreme(overflow)
    Default (attack_ho): high = thr_pkts * n_hosts (220 000) → bin3 needs ≥220 000 pkt/s.
    --per-host (MARL): high = thr_pkts (22 000) → bin3 needs ≥22 000 pkt/s (attack visible).
    Use --n-bins 5 for finer granularity inside the attack range.
"""

import subprocess, re, time, sys, argparse, math, json, os

# ── auto-reexec with the venv Python when Mininet / numpy are missing ─────────
def _ensure_venv():
    missing = []
    for mod in ("mininet", "numpy"):
        try:
            __import__(mod)
        except ModuleNotFoundError:
            missing.append(mod)
    if missing:
        _venv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             ".venv", "bin", "python3")
        if os.path.exists(_venv) and os.path.realpath(_venv) != os.path.realpath(sys.executable):
            os.execv(_venv, [_venv] + sys.argv)
        sys.exit(f"ERROR: {missing} not found. Run with: sudo .venv/bin/python3 {sys.argv[0]}")

_ensure_venv()

# ANSI colours
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"
_BIN_COL = {0: DIM, 1: GREEN, 2: YELLOW, 3: RED}


# ─────────────────────────────────────────────────────────────
# Pure-Python log-bin (mirrors attack_ho get_discretized_state)
# ─────────────────────────────────────────────────────────────

def _logspace(start_exp: float, stop_exp: float, n: int):
    if n <= 1:
        return [10.0 ** start_exp]
    step = (stop_exp - start_exp) / (n - 1)
    return [10.0 ** (start_exp + i * step) for i in range(n)]


def _digitize(val: float, bins: list) -> int:
    for i, edge in enumerate(bins):
        if val < edge:
            return i
    return len(bins)


def _log_bin(val: float, high: float, n_bins: int = 4) -> int:
    """
    bin 0 → val ≤ 0 or val < 10
    bins 1..n_bins-1 → log scale [10, high]
    """
    if val <= 0:
        return 0
    high_safe = max(float(high), 11.0)
    n    = n_bins - 1
    bins = _logspace(1.0, math.log10(high_safe), n)
    idx  = _digitize(val, bins) - 1 + 1
    return max(0, min(idx, n_bins - 1))


def _cbin(b: int) -> str:
    return f"{_BIN_COL.get(b, RESET)}{b}{RESET}"


def _bin_label(tx_p, tx_b, rx_p, rx_b, thr_p, thr_b, n_hosts, n_bins,
               per_host: bool = False) -> str:
    mult = 1 if per_host else n_hosts
    hp = thr_p * mult
    hb = thr_b * mult
    return (f"[{_cbin(_log_bin(tx_p, hp, n_bins))},{_cbin(_log_bin(tx_b, hb, n_bins))}"
            f"|{_cbin(_log_bin(rx_p, hp, n_bins))},{_cbin(_log_bin(rx_b, hb, n_bins))}]")


# ─────────────────────────────────────────────────────────────
# OVS helpers
# ─────────────────────────────────────────────────────────────

def run(cmd: str, timeout: int = 5) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout
    except Exception:
        return ""


def detect_bridge() -> str:
    out = run("ovs-vsctl list-br")
    bridges = [b.strip() for b in out.splitlines() if b.strip()]
    return bridges[0] if bridges else "s1"


def get_port_to_host(bridge: str) -> dict:
    """Both port-name and port-number keys → host name."""
    mapping = {}
    out = run(f"ovs-ofctl show {bridge}")
    for m in re.finditer(r'\s*(\d+)\(([^)]+)\):', out):
        port_no, port_name = m.group(1), m.group(2)
        if port_name.startswith(bridge + "-eth"):
            idx  = int(port_name.split("eth")[-1])
            host = f"h{idx}"
            mapping[port_name] = host
            mapping[port_no]   = host
    if not mapping:
        raw = run(f"ovs-ofctl dump-ports {bridge}")
        for m in re.finditer(r'port\s+"?([\w\-]+)"?:', raw):
            pname = m.group(1)
            if pname in ("local", "LOCAL"):
                continue
            if pname.startswith(bridge + "-eth"):
                mapping[pname] = f"h{int(pname.split('eth')[-1])}"
            elif pname.isdigit():
                mapping[pname] = f"h{pname}"
    return mapping


def parse_dump_ports(raw: str) -> dict:
    stats   = {}
    current = None
    PORT_RX = re.compile(
        r'port\s+"?(?P<port>[\w\-]+)"?:\s+'
        r'rx pkts=(?P<rx_pkts>\d+),\s*bytes=(?P<rx_bytes>\d+)'
    )
    PORT_TX = re.compile(r'\s+tx pkts=(?P<tx_pkts>\d+),\s*bytes=(?P<tx_bytes>\d+)')
    for line in raw.splitlines():
        m_rx = PORT_RX.search(line)
        m_tx = PORT_TX.search(line)
        if m_rx:
            pname = m_rx.group("port")
            if pname in ("local", "LOCAL"):
                current = None
                continue
            current = pname
            stats[pname] = {"rx_pkts": int(m_rx.group("rx_pkts")),
                            "rx_bytes": int(m_rx.group("rx_bytes")),
                            "tx_pkts": 0, "tx_bytes": 0}
        elif m_tx and current:
            stats[current]["tx_pkts"]  = int(m_tx.group("tx_pkts"))
            stats[current]["tx_bytes"] = int(m_tx.group("tx_bytes"))
            current = None
    return stats


def fmt_bytes(b: float) -> str:
    b = int(b)
    if b >= 1_048_576: return f"{b/1_048_576:.1f}MB"
    if b >= 1024:      return f"{b/1024:.1f}KB"
    return f"{b}B"


def show_flows(bridge: str):
    print(f"\n{BOLD}{CYAN}=== dump-flows {bridge} (priority=2) ==={RESET}")
    raw   = run(f"ovs-ofctl dump-flows {bridge}")
    found = False
    for line in raw.splitlines():
        if "priority=2" in line:
            print(f"  {DIM}{line.strip()}{RESET}")
            found = True
    if not found:
        print(f"  {YELLOW}No priority=2 flows.{RESET}")
    print()


def show_dpctl():
    print(f"\n{BOLD}{CYAN}=== ovs-dpctl dump-flows (first 20) ==={RESET}")
    raw   = run("ovs-dpctl dump-flows")
    lines = [l for l in raw.splitlines() if "eth(src=" in l]
    if not lines:
        print(f"  {YELLOW}No kernel datapath flows.{RESET}")
    for l in lines[:20]:
        print(f"  {DIM}{l.strip()}{RESET}")
    if len(lines) > 20:
        print(f"  {DIM}... ({len(lines)-20} more){RESET}")
    print()


def _print_bin_legend(thr_pkts, thr_bytes, n_hosts, n_bins, per_host: bool = False):
    mult   = 1 if per_host else n_hosts
    label  = "per-host" if per_host else f"×{n_hosts}"
    hp     = thr_pkts  * mult
    hb     = thr_bytes * mult
    bins_p = _logspace(1.0, math.log10(max(hp, 11)), n_bins - 1)
    bins_b = _logspace(1.0, math.log10(max(hb, 11)), n_bins - 1)
    print(f"{DIM}Pkt  log-bins ({label}={hp}): {[f'{v:.0f}' for v in bins_p]}{RESET}")
    print(f"{DIM}Byte log-bins ({label}={fmt_bytes(hb)}): {[fmt_bytes(v) for v in bins_b]}{RESET}")
    print(f"{DIM}BINS [TXp,TXb|RXp,RXb]  "
          f"{_BIN_COL[0]}0{RESET}=idle  {_BIN_COL[1]}1{RESET}=normal  "
          f"{_BIN_COL[2]}2{RESET}=attack  {_BIN_COL[3]}3{RESET}=extreme "
          f"(bin3 needs ≥{hp} pkt/s){RESET}\n")


# ─────────────────────────────────────────────────────────────
# Mininet auto-launch helpers
# ─────────────────────────────────────────────────────────────

def _extract_hosts_from_statuses(data: list) -> list:
    """Return sorted host names found in the statuses.json."""
    names = set()
    for step in data:
        for h in step.get("hostStatusesStructured", {}):
            names.add(h)
    return sorted(names)


def start_mininet(host_names: list, bridge: str = "s1") -> tuple:
    """
    Create and start a single-switch Mininet topology with the given hosts.
    Returns (net, {host_name: ip}).
    Caller must call net.stop() when done.
    """
    from mininet.net import Mininet
    from mininet.node import OVSController, OVSSwitch
    from mininet.log import setLogLevel
    setLogLevel("warning")

    net = Mininet(controller=OVSController, switch=OVSSwitch, autoSetMacs=True)
    net.addController("c0")
    sw = net.addSwitch(bridge)

    host_ips = {}
    for i, name in enumerate(host_names, start=1):
        ip = f"10.0.0.{i}/8"
        h  = net.addHost(name, ip=ip)
        net.addLink(h, sw)
        host_ips[name] = f"10.0.0.{i}"

    net.start()
    # Brief wait so OVS ports are ready
    time.sleep(0.5)
    return net, host_ips


# ─────────────────────────────────────────────────────────────
# Replay helpers
# ─────────────────────────────────────────────────────────────

def detect_host_ips() -> dict:
    """Returns {host_name: ip} for every Mininet namespace (h*, iot*)."""
    out = run("ip netns list")
    result = {}
    for line in out.splitlines():
        ns = line.split()[0] if line.strip() else ""
        if not re.match(r'^(h|iot)\d+$', ns):
            continue
        ip_out = run(f"ip netns exec {ns} hostname -I 2>/dev/null").strip()
        if ip_out:
            result[ns] = ip_out.split()[0]
    return result


def _ns_run(ns: str, cmd: str):
    """Fire-and-forget inside namespace (background)."""
    subprocess.Popen(f"ip netns exec {ns} {cmd}",
                     shell=True, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)


def kill_traffic_ns(host: str):
    run(f"ip netns exec {host} pkill -9 hping3 2>/dev/null")
    run(f"ip netns exec {host} pkill -9 ping   2>/dev/null")
    run(f"ip netns exec {host} pkill -9 iperf3 2>/dev/null")


def launch_traffic_step(step: dict, host_ips: dict) -> list:
    """
    Launch traffic matching one statuses.json step inside Mininet namespaces.
    Returns list of (host, kind, destination) that were started.
    """
    launched = []
    for host_name, info in step.get("hostStatusesStructured", {}).items():
        if host_name not in host_ips:
            continue
        status       = info.get("status",      "idle")
        task_type    = info.get("taskType",    "normal")
        traffic_type = info.get("trafficType", "none")
        destination  = info.get("destination")
        if not destination or destination not in host_ips:
            continue
        dest_ip = host_ips[destination]

        is_attack = (status == "attacking" or
                     task_type in ("short_attack", "long_attack", "SHORT_ATTACK", "LONG_ATTACK"))

        if is_attack:
            _ns_run(host_name, f"hping3 --flood --udp {dest_ip} >/dev/null 2>&1")
            launched.append((host_name, "ATTACK", destination))

        elif status == "normal" and traffic_type not in ("none", ""):
            if traffic_type == "ping":
                _ns_run(host_name, f"ping -f {dest_ip} >/dev/null 2>&1")
                launched.append((host_name, "PING", destination))
            elif traffic_type in ("udp", "tcp"):
                proto = "-u -b 200M" if traffic_type == "udp" else "-b 200M"
                _ns_run(destination, "iperf3 -s -1 >/dev/null 2>&1")
                time.sleep(0.05)
                _ns_run(host_name, f"iperf3 -c {dest_ip} {proto} >/dev/null 2>&1")
                launched.append((host_name, f"IPERF-{traffic_type.upper()}", destination))

    return launched


def _match_symbol(obs: float, exp: float) -> str:
    """Compare observed vs expected packet rate."""
    if exp < 5:
        return GREEN + "✓" + RESET if obs < 200 else YELLOW + "?" + RESET
    ratio = obs / exp
    if ratio < 0.05:
        return RED + "▼LOW" + RESET
    if ratio < 0.3:
        return YELLOW + "▼" + RESET
    if ratio > 10:
        return YELLOW + "▲" + RESET
    return GREEN + "✓" + RESET


# ─────────────────────────────────────────────────────────────
# REPLAY mode
# ─────────────────────────────────────────────────────────────

def replay_mode(statuses_file: str, bridge: str,
                thr_pkts: int, thr_bytes: int, n_bins: int,
                step_start: int, step_end: int, pause: bool,
                per_host: bool = False, launch: bool = False):

    with open(statuses_file) as f:
        data = json.load(f)

    mn_net = None  # will hold the Mininet object if we launched it ourselves

    host_ips = detect_host_ips()
    if not host_ips or launch:
        if host_ips and launch:
            print(f"{YELLOW}--launch requested: ignoring existing namespaces, "
                  f"starting fresh Mininet.{RESET}")
        host_names = _extract_hosts_from_statuses(data)
        if not host_names:
            print(f"{RED}Cannot extract host names from statuses.json.{RESET}")
            sys.exit(1)
        print(f"{BOLD}Launching Mininet{RESET} with hosts: {host_names}  "
              f"bridge: {bridge}")
        try:
            mn_net, host_ips = start_mininet(host_names, bridge=bridge)
            print(f"{GREEN}Mininet started.{RESET}  IPs: {host_ips}\n")
        except Exception as exc:
            print(f"{RED}Failed to start Mininet: {exc}{RESET}")
            sys.exit(1)

    n_hosts_actual = len(host_ips)
    port_to_host   = get_port_to_host(bridge)

    print(f"{BOLD}Replay:{RESET} {os.path.basename(statuses_file)}  "
          f"({len(data)} steps,  using {step_start}..{step_end or len(data)-1})")
    print(f"{BOLD}Hosts:{RESET} {sorted(host_ips.keys())}  "
          f"{BOLD}Bridge:{RESET} {bridge}  "
          f"{BOLD}n_bins:{RESET} {n_bins}  "
          f"{BOLD}bins:{RESET} {'per-host (MARL)' if per_host else 'global (attack_ho)'}")
    _print_bin_legend(thr_pkts, thr_bytes, n_hosts_actual, n_bins, per_host=per_host)

    steps = data[step_start: step_end + 1 if step_end else len(data)]
    CW = 8

    try:
        for rel_idx, step in enumerate(steps):
            abs_idx = step_start + rel_idx
            hosts_info = step.get("hostStatusesStructured", {})

            # Summarise what this step expects
            attackers = [(h, v.get("destination", "?"))
                         for h, v in hosts_info.items()
                         if v.get("status") == "attacking"]
            victims   = [h for h, v in hosts_info.items()
                         if v.get("status") == "under_attack"]

            print(f"\n{BOLD}{'─'*78}{RESET}")
            atk_str = (f"  {RED}ATK: {attackers}  victims: {victims}{RESET}"
                       if attackers else f"  {GREEN}normal{RESET}")
            print(f"{BOLD}Step {abs_idx+1}/{len(data)}{RESET}{atk_str}")

            # Kill leftovers, snapshot before
            for h in host_ips:
                kill_traffic_ns(h)
            time.sleep(0.1)
            stats_before = parse_dump_ports(run(f"ovs-ofctl dump-ports {bridge}"))
            t0 = time.time()

            # Launch traffic
            launched = launch_traffic_step(step, host_ips)
            if launched:
                print(f"  {DIM}Launched: {launched}{RESET}")

            # Wait for counters to fill
            time.sleep(1.0)

            stats_after = parse_dump_ports(run(f"ovs-ofctl dump-ports {bridge}"))
            elapsed = max(time.time() - t0, 0.001)

            # Kill traffic
            for h in host_ips:
                kill_traffic_ns(h)

            # Header
            print(f"\n  {'Host':<8}  {'Status':<20}  "
                  f"{'ExpTX':>{CW}}  {'ObsTX':>{CW}}  "
                  f"{'ExpRX':>{CW}}  {'ObsRX':>{CW}}  "
                  f"{'BINS':<14}  Chk")
            print(f"  {'-'*84}")

            seen: set = set()
            any_anomaly = False

            for port, host in sorted(port_to_host.items(), key=lambda x: x[1]):
                if host in seen:
                    continue
                seen.add(host)

                b = stats_before.get(port, {})
                a = stats_after.get(port, {})
                if not b or not a:
                    continue

                obs_tx_pps = max(0, a["rx_pkts"]  - b["rx_pkts"])  / elapsed
                obs_tx_bps = max(0, a["rx_bytes"] - b["rx_bytes"]) / elapsed
                obs_rx_pps = max(0, a["tx_pkts"]  - b["tx_pkts"])  / elapsed
                obs_rx_bps = max(0, a["tx_bytes"] - b["tx_bytes"]) / elapsed

                info      = hosts_info.get(host, {})
                exp_tx    = info.get("transmittedPackets", 0)
                exp_rx    = info.get("receivedPackets",    0)
                status    = info.get("status", "idle")

                bins = _bin_label(obs_tx_pps, obs_tx_bps, obs_rx_pps, obs_rx_bps,
                                  thr_pkts, thr_bytes, n_hosts_actual, n_bins,
                                  per_host=per_host)

                chk_tx = _match_symbol(obs_tx_pps, exp_tx)
                chk_rx = _match_symbol(obs_rx_pps, exp_rx)

                if "LOW" in chk_tx or "LOW" in chk_rx:
                    any_anomaly = True

                status_col = (RED    if status == "attacking"    else
                              YELLOW if status == "under_attack" else
                              GREEN  if status == "normal"       else DIM)

                print(f"  {host:<8}  {status_col}{status:<20}{RESET}  "
                      f"{exp_tx:>{CW}.0f}  {obs_tx_pps:>{CW}.0f}  "
                      f"{exp_rx:>{CW}.0f}  {obs_rx_pps:>{CW}.0f}  "
                      f"{bins}  {chk_tx}/{chk_rx}")

            if any_anomaly:
                print(f"\n  {RED}{BOLD}^^^ ANOMALY: observed traffic much lower than expected ^^^{RESET}")
                print(f"  {RED}Possible causes: drop rules active, hping3 not installed, "
                      f"or wrong namespace.{RESET}")

            if pause:
                try:
                    input(f"\n  {DIM}[Enter] next step, [Ctrl-C] stop ...{RESET}  ")
                except EOFError:
                    pass

    except KeyboardInterrupt:
        print(f"\n{DIM}Stopped.{RESET}")
    finally:
        for h in host_ips:
            kill_traffic_ns(h)
        if mn_net is not None:
            print(f"{DIM}Stopping Mininet...{RESET}")
            mn_net.stop()
            print(f"{DIM}Mininet stopped.{RESET}")


# ─────────────────────────────────────────────────────────────
# LIVE monitor
# ─────────────────────────────────────────────────────────────

def monitor(bridge: str, interval: float, show_fl: bool, show_dp: bool,
            raw_mode: bool, debug_mode: bool,
            thr_pkts: int, thr_bytes: int, n_hosts: int, n_bins: int,
            per_host: bool = False):

    port_to_host = get_port_to_host(bridge)
    if not port_to_host:
        print(f"{RED}Cannot detect port→host mapping for '{bridge}'.{RESET}")
        sys.exit(1)

    if debug_mode:
        print(f"{CYAN}port_to_host mapping:{RESET}")
        for k, v in sorted(port_to_host.items()):
            print(f"  {k!r:20} → {v}")
        print()

    print(f"{BOLD}Bridge:{RESET} {bridge}  "
          f"{BOLD}interval:{RESET} {interval}s  "
          f"{BOLD}hosts:{RESET} {len(port_to_host)//2}  "
          f"{BOLD}n_bins:{RESET} {n_bins}  "
          f"{BOLD}thr_pkts:{RESET} {thr_pkts}  "
          f"{BOLD}thr_bytes:{RESET} {fmt_bytes(thr_bytes)}/s  "
          f"{BOLD}bins:{RESET} {'per-host (MARL)' if per_host else 'global (attack_ho)'}")
    _print_bin_legend(thr_pkts, thr_bytes, n_hosts, n_bins, per_host=per_host)

    if show_fl: show_flows(bridge)
    if show_dp: show_dpctl()

    prev: dict       = {}
    prev_time: float = 0.0
    tick: int        = 0
    CW = 10

    try:
        while True:
            raw      = run(f"ovs-ofctl dump-ports {bridge}")
            now_time = time.time()
            curr     = parse_dump_ports(raw)

            if raw_mode:
                print(f"\n{DIM}--- raw dump-ports ---\n{raw}{RESET}")
            if debug_mode and tick == 0:
                print(f"{CYAN}parsed keys: {sorted(curr.keys())}{RESET}\n")

            if prev and (now_time - prev_time) > 0:
                elapsed = now_time - prev_time

                if tick % 25 == 0:
                    hdr = (f"\n{'Host':<7}  "
                           f"{'TX pkt/s':>{CW}}  {'TX B/s':>{CW}}  "
                           f"{'RX pkt/s':>{CW}}  {'RX B/s':>{CW}}  "
                           f"{'BINS':<14}  "
                           f"{'TX total':>{CW}}  {'RX total':>{CW}}")
                    print(f"{BOLD}{hdr}{RESET}")
                    print(f"{DIM}{'':7}  {'':>{CW}}  {'':>{CW}}  "
                          f"{'':>{CW}}  {'':>{CW}}  "
                          f"{'[TXp,TXb|RXp,RXb]':<14}{RESET}")
                    print("-" * (7 + 2 + (CW + 2) * 4 + 16 + (CW + 2) * 2))

                any_issue = False
                g         = dict(tx_pkts=0, tx_bytes=0, rx_pkts=0, rx_bytes=0)
                seen: set = set()

                for port, host in sorted(port_to_host.items(), key=lambda x: x[1]):
                    if host in seen:
                        continue
                    c = curr.get(port)
                    p = prev.get(port)
                    if not c or not p:
                        continue
                    seen.add(host)

                    d_rx_pkts  = max(0, c["rx_pkts"]  - p["rx_pkts"])
                    d_rx_bytes = max(0, c["rx_bytes"] - p["rx_bytes"])
                    d_tx_pkts  = max(0, c["tx_pkts"]  - p["tx_pkts"])
                    d_tx_bytes = max(0, c["tx_bytes"] - p["tx_bytes"])

                    host_tx_pps = d_rx_pkts  / elapsed
                    host_tx_bps = d_rx_bytes / elapsed
                    host_rx_pps = d_tx_pkts  / elapsed
                    host_rx_bps = d_tx_bytes / elapsed

                    g["tx_pkts"]  += d_rx_pkts;  g["tx_bytes"] += d_rx_bytes
                    g["rx_pkts"]  += d_tx_pkts;  g["rx_bytes"] += d_tx_bytes

                    active     = host_tx_pps > 0 or host_rx_pps > 0
                    asymmetric = active and (host_tx_pps == 0 or host_rx_pps == 0)
                    if asymmetric:
                        any_issue = True

                    row_col = RED if asymmetric else (GREEN if active else DIM)
                    bins    = _bin_label(host_tx_pps, host_tx_bps,
                                         host_rx_pps, host_rx_bps,
                                         thr_pkts, thr_bytes, n_hosts, n_bins,
                                         per_host=per_host)

                    line = (f"{host:<7}  "
                            f"{host_tx_pps:>{CW}.1f}  {fmt_bytes(host_tx_bps):>{CW}}  "
                            f"{host_rx_pps:>{CW}.1f}  {fmt_bytes(host_rx_bps):>{CW}}  "
                            f"{bins}  "
                            f"{fmt_bytes(c['rx_bytes']):>{CW}}  "
                            f"{fmt_bytes(c['tx_bytes']):>{CW}}")
                    print(f"{row_col}{line}{RESET}")

                print(f"{DIM}{'TOTAL':<7}  "
                      f"{g['tx_pkts']/elapsed:>{CW}.1f}  "
                      f"{fmt_bytes(g['tx_bytes']/elapsed):>{CW}}  "
                      f"{g['rx_pkts']/elapsed:>{CW}.1f}  "
                      f"{fmt_bytes(g['rx_bytes']/elapsed):>{CW}}{RESET}")

                if any_issue:
                    print(f"{RED}{BOLD}  ^^^ ANOMALY: host(s) TX without RX or vice versa ^^^{RESET}")

                tick += 1

            prev      = curr
            prev_time = now_time
            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\n{DIM}Stopped.{RESET}")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OVS per-host monitor + statuses.json replay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Common
    parser.add_argument("--bridge",    default=None,               help="OVS bridge (auto-detect)")
    parser.add_argument("--n-bins",    default=4,    type=int,     help="Q-table bins (default 4; use 5 for more granularity)")
    parser.add_argument("--thr-pkts",  default=22000, type=int,    help="packets threshold (default 22000)")
    parser.add_argument("--thr-bytes", default=423_000_000, type=int, help="bytes threshold (default 423000000)")
    parser.add_argument("--n-hosts",   default=10,   type=int,     help="number of hosts (default 10)")
    parser.add_argument("--per-host",  action="store_true",
                        help="use per-host thresholds for bins (MARL mode, no ×n_hosts multiplier)")

    # Live mode
    parser.add_argument("--interval",  default=1.0,  type=float,   help="sample interval s (default 1)")
    parser.add_argument("--flows",     action="store_true",         help="dump priority=2 flows once")
    parser.add_argument("--dpctl",     action="store_true",         help="dump kernel datapath once")
    parser.add_argument("--raw",       action="store_true",         help="print raw dump-ports each tick")
    parser.add_argument("--debug",     action="store_true",         help="print port mapping and parsed keys")

    # Replay mode
    parser.add_argument("--replay",    default=None, metavar="FILE",
                        help="statuses.json to replay (standalone mode)")
    parser.add_argument("--step-start", default=0,   type=int,     help="first step index (default 0)")
    parser.add_argument("--step-end",   default=None, type=int,    help="last step index inclusive (default: all)")
    parser.add_argument("--pause",      action="store_true",       help="pause between steps (press Enter)")
    parser.add_argument("--launch",     action="store_true",
                        help="auto-launch Mininet from statuses.json host list (no separate mn needed)")

    args = parser.parse_args()
    bridge = args.bridge or detect_bridge()

    if args.replay:
        replay_mode(
            statuses_file=args.replay,
            bridge=bridge,
            thr_pkts=args.thr_pkts,
            thr_bytes=args.thr_bytes,
            n_bins=args.n_bins,
            step_start=args.step_start,
            step_end=args.step_end,
            pause=args.pause,
            per_host=args.per_host,
            launch=args.launch,
        )
    else:
        monitor(
            bridge=bridge,
            interval=args.interval,
            show_fl=args.flows,
            show_dp=args.dpctl,
            raw_mode=args.raw,
            debug_mode=args.debug,
            thr_pkts=args.thr_pkts,
            thr_bytes=args.thr_bytes,
            n_hosts=args.n_hosts,
            n_bins=args.n_bins,
            per_host=args.per_host,
        )
