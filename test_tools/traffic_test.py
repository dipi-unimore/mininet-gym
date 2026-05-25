#!/usr/bin/env python3.12
"""
Standalone framework pipeline test — no external controller required.

Usage:
  sudo python3 test_tools/traffic_test.py \\
      --statuses _training/attacks_ho/20260523-075157_1_5_5/statuses.json \\
      --step 5 [--step-end 10] [--scenario attack_ho|marl] [--n-bins 4]

What it does:
  1. Reads host names from statuses.json
  2. Creates a Mininet single-switch topology (no controller, failMode=standalone)
  3. Calls framework's initialize_monitoring()
  4. For each selected step:
       - generates traffic via Mininet host.cmd() matching the step
       - reads traffic with framework's get_data_flow()
       - applies the same normalization / discretization as the real environment
       - compares observed state with expected values from statuses.json
  5. Stops Mininet and exits.
"""

import sys, os, time, json, argparse, copy

# ── auto-reexec with the venv Python if numpy is missing ─────────────────────
try:
    import numpy as np
except ModuleNotFoundError:
    _venv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         ".venv", "bin", "python3")
    if os.path.exists(_venv) and os.path.realpath(_venv) != os.path.realpath(sys.executable):
        os.execv(_venv, [_venv] + sys.argv)
    sys.exit("ERROR: numpy not found and venv Python not available. "
             "Run with: sudo .venv/bin/python3 " + sys.argv[0])

# ── make framework importable ────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── ANSI colours ─────────────────────────────────────────────────────────────
RED    = "\033[91m"; GREEN  = "\033[92m"; YELLOW = "\033[93m"
CYAN   = "\033[96m"; BOLD   = "\033[1m";  RESET  = "\033[0m"; DIM = "\033[2m"


# ─────────────────────────────────────────────────────────────────────────────
# Framework imports
# ─────────────────────────────────────────────────────────────────────────────
from utility.network_flows import get_data_flow, initialize_monitoring
from reinforcement_learning.network_env import (
    get_normalized_state,
    get_log_bin_index,
    get_linear_bin_index,
)
from reinforcement_learning.scenarios.attack_detect_host_observable.\
    network_env_attack_detect_per_host_observable import (
    discretize_attack_detect_ho_state,
)


# ─────────────────────────────────────────────────────────────────────────────
# Mininet helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_network(host_names: list, bridge: str = "s1"):
    """
    Create a single-switch Mininet topology without a remote controller.
    The switch runs in standalone/failsafe mode and uses 'actions=normal'
    so packets are forwarded without any controller.
    """
    from mininet.net import Mininet
    from mininet.node import OVSKernelSwitch
    from mininet.link import TCLink
    from mininet.clean import cleanup
    from functools import partial

    cleanup()

    StandaloneOVS = partial(OVSKernelSwitch, failMode='standalone',
                            protocols='OpenFlow10,OpenFlow13')

    net = Mininet(controller=None, switch=StandaloneOVS, link=TCLink)

    sw = net.addSwitch(bridge)
    for i, name in enumerate(host_names, start=1):
        h = net.addHost(name, ip=f"10.0.0.{i}/24")
        net.addLink(h, sw)

    net.hosts_links  = {h.name: {} for h in net.hosts}
    net.blocked_hosts = []

    net.start()

    # Ensure both OF versions are enabled (mirrors create_network in network_configurator)
    sw.cmd(f"ovs-vsctl set bridge {bridge} protocols=OpenFlow10,OpenFlow13")
    # Default forwarding flow so packets are forwarded without a controller
    sw.cmd(f"ovs-ofctl add-flow {bridge} priority=1,actions=normal")
    time.sleep(0.3)
    return net


# ─────────────────────────────────────────────────────────────────────────────
# Traffic generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_traffic(step: dict, net):
    """
    Launch traffic for one step using Mininet host.cmd() (background).
    Returns list of (src_host_obj, desc) that were started.
    """
    host_map = {h.name: h for h in net.hosts}
    launched = []

    for src_name, info in step.get("hostStatusesStructured", {}).items():
        src = host_map.get(src_name)
        if src is None:
            continue

        status       = info.get("status",      "idle")
        task_type    = info.get("taskType",    "normal")
        destination  = info.get("destination")
        dst          = host_map.get(destination) if destination else None

        is_attack = (status == "attacking" or
                     task_type in ("short_attack", "long_attack",
                                   "SHORT_ATTACK", "LONG_ATTACK"))

        if is_attack and dst:
            # setsid + timeout so hping3 auto-terminates (same fix as network_attacks.py)
            src.cmd(f"setsid timeout 2s hping3 --flood --udp {dst.IP()} >/dev/null 2>&1 &")
            launched.append((src_name, f"ATTACK → {destination}"))

        elif status == "normal" and dst:
            traffic_type = info.get("trafficType", "")
            if traffic_type == "ping":
                # Framework uses ping -c <duration> (count, not flood) — 1 pkt per step
                src.cmd(f"ping -c 1 {dst.IP()} > /dev/null 2>&1 &")
                launched.append((src_name, f"PING → {destination}"))
            elif traffic_type in ("udp", "tcp"):
                # Framework uses iperf (v2), not iperf3
                srv_flag = "-u" if traffic_type == "udp" else ""
                cli_flag = "-u -b 200M" if traffic_type == "udp" else ""
                dst.cmd(f"iperf -s {srv_flag} -p 15000 > /dev/null 2>&1 &")
                time.sleep(0.1)
                src.cmd(f"iperf -c {dst.IP()} {cli_flag} -p 15000 -t 1 > /dev/null 2>&1 &")
                launched.append((src_name, f"IPERF-{traffic_type.upper()} → {destination}"))

    return launched


def kill_all_traffic(net):
    for h in net.hosts:
        h.cmd("pkill -9 hping3 2>/dev/null; pkill -9 ping 2>/dev/null; "
              "pkill -9 iperf3 2>/dev/null")


# ─────────────────────────────────────────────────────────────────────────────
# Framework read pipeline  (mirrors read_from_network in NetworkEnv)
# ─────────────────────────────────────────────────────────────────────────────

def read_host_states(net, prev_totals: dict) -> tuple:
    """
    Call get_data_flow() and compute per-host increments exactly as
    NetworkEnv.read_from_network() does.

    Returns:
        (host_states, new_totals)
        host_states: {host_name: np.array(8)} — same layout as global_state.host_states
        new_totals:  {host_name: np.array(4)} — cumulative [RX_pkt, RX_B, TX_pkt, TX_B]
    """
    flows = get_data_flow(net)
    if not flows:
        return {}, prev_totals

    host_states  = {}
    new_totals   = {}

    for host in net.hosts:
        rx_pkt = rx_b = tx_pkt = tx_b = 0

        for flow in flows.get("flows", []):
            if flow.get("src_name") != host.name and flow.get("dst_name") != host.name:
                continue
            direction = flow.get("direction")
            if direction == "TX":          # switch→host  = host receives
                rx_pkt += flow.get("packets", 0)
                rx_b   += flow.get("bytes",   0)
            elif direction == "RX":        # host→switch  = host transmits
                tx_pkt += flow.get("packets", 0)
                tx_b   += flow.get("bytes",   0)
            else:                          # legacy table flows
                if flow.get("src_name") == host.name:
                    tx_pkt += flow.get("packets", 0)
                    tx_b   += flow.get("bytes",   0)
                elif flow.get("dst_name") == host.name:
                    rx_pkt += flow.get("packets", 0)
                    rx_b   += flow.get("bytes",   0)

        prev = prev_totals.get(host.name, np.zeros(4))
        prev_rx_pkt, prev_rx_b, prev_tx_pkt, prev_tx_b = (
            max(0, rx_pkt - prev[0]),
            max(0, rx_b   - prev[1]),
            max(0, tx_pkt - prev[2]),
            max(0, tx_b   - prev[3]),
        )

        host_states[host.name] = np.array([
            prev_rx_pkt, 0.0, prev_rx_b, 0.0,
            prev_tx_pkt, 0.0, prev_tx_b, 0.0,
        ], dtype=np.float32)
        new_totals[host.name] = np.array([rx_pkt, rx_b, tx_pkt, tx_b],
                                          dtype=np.float32)

    return host_states, new_totals


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation / discretisation wrappers
# ─────────────────────────────────────────────────────────────────────────────

def make_bounds(scenario: str, thr_pkts, thr_var_pkts, thr_bytes,
                thr_var_bytes, num_hosts):
    """Return (low, high) arrays matching what the env uses."""
    if scenario == "marl":
        # host_agent_env.py — per-host (fixed)
        low  = np.array([0, -thr_var_pkts, 0, -thr_var_bytes,
                         0, -thr_var_pkts, 0, -thr_var_bytes, 0])
        high = np.array([thr_pkts, thr_var_pkts, thr_bytes, thr_var_bytes,
                         thr_pkts, thr_var_pkts, thr_bytes, thr_var_bytes,
                         num_hosts])
    else:
        # network_env_attack_detect_per_host_observable.py — per-host (no ×num_hosts)
        low  = np.array([0, -thr_var_pkts, 0, -thr_var_bytes,
                         0, -thr_var_pkts, 0, -thr_var_bytes])
        high = np.array([thr_pkts, thr_var_pkts,
                         thr_bytes, thr_var_bytes,
                         thr_pkts, thr_var_pkts,
                         thr_bytes, thr_var_bytes])
    return low, high


def discretize(state8, low, high, n_bins, scenario):
    """
    Discretize an 8-element host state using the same logic as the real env.
    attack_ho → discretize_attack_detect_ho_state()
    marl      → host_agent_env.get_discretized_state() logic
    """
    if scenario == "attack_ho":
        return discretize_attack_detect_ho_state(state8, low[:8], high[:8], n_bins)
    # MARL: same algorithm as host_agent_env.get_discretized_state
    from reinforcement_learning.network_env import get_custom_bin_index
    disc = []
    for i, val in enumerate(state8):
        if i in (1, 3):
            b = get_linear_bin_index(val, low[i], high[i], n_bins - 1) + 1
        elif i == 0:
            b = get_custom_bin_index(val, low[i], high[i], n_bins)
        else:
            b = get_linear_bin_index(val, low[i], high[i], n_bins)
        disc.append(int(b))
    return tuple(disc)


# ─────────────────────────────────────────────────────────────────────────────
# Pretty print
# ─────────────────────────────────────────────────────────────────────────────

_STATUS_COL = {
    "attacking":    RED,
    "under_attack": YELLOW,
    "normal":       GREEN,
    "idle":         DIM,
}

def _col(status: str) -> str:
    return _STATUS_COL.get(status, RESET)


def _kb(val: float) -> str:
    """Format bytes as KB with one decimal, compact."""
    return f"{val/1000:.1f}k"


def print_step_result(step_idx, step, host_states, low, high, n_bins, scenario):
    hosts_info = step.get("hostStatusesStructured", {})

    print(f"\n{BOLD}{'─'*120}{RESET}")
    atk = [(h, v.get("destination","?")) for h,v in hosts_info.items()
           if v.get("status") == "attacking"]
    vic = [h for h,v in hosts_info.items() if v.get("status") == "under_attack"]
    lbl = f"{RED}ATK {atk}  VIC {vic}{RESET}" if atk else f"{GREEN}normal{RESET}"
    print(f"{BOLD}Step {step_idx+1}{RESET}  {lbl}")

    # table header — packets (p) and bytes (b), RX=received TX=transmitted
    WP = 8   # packet column width
    WB = 9   # byte column width (KB)
    print(f"\n  {'Host':<7}  {'Status':<14}  "
          f"{'ExpRXp':>{WP}}  {'ObsRXp':>{WP}}  "
          f"{'ExpRXb':>{WB}}  {'ObsRXb':>{WB}}  "
          f"{'ExpTXp':>{WP}}  {'ObsTXp':>{WP}}  "
          f"{'ExpTXb':>{WB}}  {'ObsTXb':>{WB}}  "
          f"  {'NRXp':>6}  {'NTXp':>6}  "
          f"  BINS[RXp,RXb|TXp,TXb]")
    print(f"  {'-'*120}")

    for host_name in sorted(host_states.keys()):
        s8    = host_states[host_name]
        info  = hosts_info.get(host_name, {})
        status = info.get("status", "idle")

        exp_rx_p = info.get("receivedPackets",    0)
        exp_rx_b = info.get("receivedBytes",      0)
        exp_tx_p = info.get("transmittedPackets", 0)
        exp_tx_b = info.get("transmittedBytes",   0)
        obs_rx_p = s8[0]   # received packets
        obs_rx_b = s8[2]   # received bytes
        obs_tx_p = s8[4]   # transmitted packets
        obs_tx_b = s8[6]   # transmitted bytes

        # normalised
        norm = get_normalized_state(s8, low[:8], high[:8])
        norm_rx_p = norm[0]
        norm_tx_p = norm[4]

        # discretised
        disc = discretize(s8, low, high, n_bins, scenario)
        bin_str = f"[{disc[0]},{disc[2]}|{disc[4]},{disc[6]}]"

        # check (packets only — bytes are secondary)
        rx_ok = abs(obs_rx_p - exp_rx_p) < max(exp_rx_p * 0.3, 5)
        tx_ok = abs(obs_tx_p - exp_tx_p) < max(exp_tx_p * 0.3, 5)
        chk = (f"{GREEN}✓{RESET}" if rx_ok and tx_ok else f"{RED}✗{RESET}")

        print(f"  {host_name:<7}  "
              f"{_col(status)}{status:<14}{RESET}  "
              f"{exp_rx_p:>{WP}.0f}  {obs_rx_p:>{WP}.0f}  "
              f"{_kb(exp_rx_b):>{WB}}  {_kb(obs_rx_b):>{WB}}  "
              f"{exp_tx_p:>{WP}.0f}  {obs_tx_p:>{WP}.0f}  "
              f"{_kb(exp_tx_b):>{WB}}  {_kb(obs_tx_b):>{WB}}  "
              f"  {norm_rx_p:>6.3f}  {norm_tx_p:>6.3f}  "
              f"  {bin_str}  {chk}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

_SCENARIO_YAML = {
    "attack_ho": os.path.join(_ROOT, "reinforcement_learning", "scenarios",
                              "attack_detect_host_observable", "scenario_env_param.yaml"),
    "marl":      os.path.join(_ROOT, "reinforcement_learning", "scenarios",
                              "marl", "scenario_env_param.yaml"),
}


def _load_yaml_thresholds(scenario: str) -> dict:
    """Load threshold values from the scenario YAML (requires PyYAML)."""
    yaml_path = _SCENARIO_YAML.get(scenario)
    if not yaml_path or not os.path.exists(yaml_path):
        return {}
    try:
        import yaml
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        thr = cfg.get("attacks", {}).get("thresholds", {})
        return thr
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser(
        description="Framework pipeline test (no external controller)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--statuses", required=True, metavar="FILE",
                    help="statuses.json from a past experiment")
    ap.add_argument("--step",     default=0,    type=int, help="first step to test (default 0)")
    ap.add_argument("--step-end", default=None, type=int,
                    help="last step inclusive (default: end of file when --pause, else same as --step)")
    ap.add_argument("--scenario", default="attack_ho", choices=["attack_ho", "marl"],
                    help="which env's normalisation/discretization to use (default attack_ho)")
    ap.add_argument("--n-bins",   default=4, type=int, help="Q-table bins (default 4)")
    ap.add_argument("--thr-pkts",      default=None, type=int,
                    help="packet threshold (default: read from scenario YAML)")
    ap.add_argument("--thr-var-pkts",  default=None, type=int)
    ap.add_argument("--thr-bytes",     default=None, type=int)
    ap.add_argument("--thr-var-bytes", default=None, type=int)
    ap.add_argument("--bridge",   default="s1")
    ap.add_argument("--pause",    action="store_true", help="pause between steps (Enter=next, Ctrl-C=stop)")
    args = ap.parse_args()

    # ── resolve thresholds: CLI > YAML > built-in fallback ──────────────────
    yaml_thr = _load_yaml_thresholds(args.scenario)
    thr_pkts      = args.thr_pkts      if args.thr_pkts      is not None else yaml_thr.get("packets",     12000)
    thr_var_pkts  = args.thr_var_pkts  if args.thr_var_pkts  is not None else yaml_thr.get("var_packets",    50)
    thr_bytes     = args.thr_bytes     if args.thr_bytes     is not None else yaml_thr.get("bytes",   423_000_000)
    thr_var_bytes = args.thr_var_bytes if args.thr_var_bytes is not None else yaml_thr.get("var_bytes",      30)
    if yaml_thr:
        print(f"{DIM}Loaded thresholds from YAML ({args.scenario}): "
              f"pkts={thr_pkts}  var_pkts={thr_var_pkts}  "
              f"bytes={thr_bytes}  var_bytes={thr_var_bytes}{RESET}")

    # ── load statuses ────────────────────────────────────────────────────────
    with open(args.statuses) as f:
        data = json.load(f)

    # When --pause and no --step-end: iterate all remaining steps interactively
    if args.step_end is not None:
        step_end = args.step_end
    elif args.pause:
        step_end = len(data) - 1
    else:
        step_end = args.step
    steps    = data[args.step : step_end + 1]
    if not steps:
        print(f"{RED}No steps in range {args.step}..{step_end}{RESET}")
        sys.exit(1)

    # ── extract host names from the first step that has them ────────────────
    host_names = []
    for s in data:
        hns = sorted(s.get("hostStatusesStructured", {}).keys())
        if hns:
            host_names = hns
            break
    if not host_names:
        print(f"{RED}Cannot find host names in statuses.json{RESET}")
        sys.exit(1)

    num_hosts = len(host_names)
    low, high = make_bounds(args.scenario, thr_pkts, thr_var_pkts,
                             thr_bytes, thr_var_bytes, num_hosts)

    # ── build Mininet ────────────────────────────────────────────────────────
    print(f"{BOLD}Building Mininet{RESET} (no controller)  hosts: {host_names}")
    net = build_network(host_names, bridge=args.bridge)
    print(f"{GREEN}Network started.{RESET}")

    print(f"{BOLD}Initializing monitoring…{RESET}")
    if not initialize_monitoring(net, bridge_name=args.bridge):
        print(f"{RED}initialize_monitoring failed.{RESET}")
        net.stop()
        sys.exit(1)
    print(f"{GREEN}Monitoring ready.{RESET}")
    print(f"\nScenario: {BOLD}{args.scenario}{RESET}  n_bins: {args.n_bins}  "
          f"thr_pkts: {thr_pkts}  thr_bytes: {thr_bytes}\n"
          f"low:  {low}\nhigh: {high}\n")

    # Baseline read (flush initial counters)
    prev_totals: dict = {}
    read_host_states(net, prev_totals)   # discard first read

    try:
        for rel_idx, step in enumerate(steps):
            abs_idx = args.step + rel_idx

            # Baseline snapshot before traffic
            _, prev_totals = read_host_states(net, prev_totals)

            # Generate traffic
            kill_all_traffic(net)
            time.sleep(0.1)
            launched = generate_traffic(step, net)
            if launched:
                print(f"  {DIM}Launched: {launched}{RESET}")
            else:
                print(f"  {DIM}No traffic launched (idle step){RESET}")

            # Wait exactly 1s (mirrors the env's time.sleep(1) in step())
            time.sleep(1.0)

            # Read with framework pipeline
            host_states, prev_totals = read_host_states(net, prev_totals)

            # Kill traffic
            kill_all_traffic(net)

            # Display
            print_step_result(abs_idx, step, host_states,
                              low, high, args.n_bins, args.scenario)

            if args.pause:
                sys.stdout.write(f"\n  {DIM}── [Enter] next step  [Ctrl-C] stop ──{RESET}  ")
                sys.stdout.flush()
                try:
                    sys.stdin.readline()
                except (EOFError, KeyboardInterrupt):
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        print(f"\n{DIM}Interrupted.{RESET}")
    finally:
        kill_all_traffic(net)
        print(f"{DIM}Stopping Mininet…{RESET}")
        net.stop()
        print(f"{DIM}Done.{RESET}")


if __name__ == "__main__":
    main()
