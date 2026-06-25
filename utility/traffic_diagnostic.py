#!/usr/bin/env python3.12
"""
Traffic Diagnostic Tool — real environment, real scenario, no RL agents.

Instantiates the actual NetworkEnvAttackDetectPerHostObservable (or the
scenario specified with --scenario), generates a scenario with the same
generator used in training, replays steps via _apply_scenario_step, reads
OVS port counters, and prints per-host packet deltas with colour coding:

  GREEN  = normal traffic step   (all host rows printed normally)
  PINK   = attacker host row     (entire row in pink)
  YELLOW = victim host row       (entire row in yellow)

Each host row shows discretized bin indices [b0..b7] derived from the
current config thresholds and n_bins, exactly as the RL agents see them.

At the end a threshold analysis section shows:
  • current bin boundaries for packets and bytes
  • observed min/mean/max per role (victim during attack, all hosts normal)
  • suggested optimal thresholds for clean bin separation

Usage:
    sudo .venv/bin/python3 utility/traffic_diagnostic.py \\
        [--scenario attacks_ho]     # default
        [--config path/to/config.yaml]
        [--steps N]                 # limit to first N steps (default: all)
        [--episodes N]              # episodes to generate (default: 1)
"""

import sys, os, re, time, copy, tempfile, argparse

# ── project root on sys.path ────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── auto-reexec with venv python if numpy is missing ────────────────────────
try:
    import numpy as np
except ModuleNotFoundError:
    _venv = os.path.join(_ROOT, ".venv", "bin", "python3")
    # Use abspath (not realpath) so the venv symlink path differs from the
    # system python path even when both point to the same binary.
    if os.path.exists(_venv) and os.path.abspath(_venv) != os.path.abspath(sys.executable):
        os.execv(_venv, [_venv] + sys.argv)
    sys.exit("ERROR: numpy not found. Run with: sudo .venv/bin/python3 " + sys.argv[0])

# ── ANSI colours ─────────────────────────────────────────────────────────────
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
PINK   = "\033[95m"   # bright magenta / pink — attacker row
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"

# ── constants used across the module ─────────────────────────────────────────
_DEFAULT_SCENARIO = "attacks_ho"
_SCENARIO_CHOICES = ["attacks_ho", "marl_attacks"]

# Feature labels for the 8-element per-host state vector
_FEAT_LABELS = [
    "rx_pkts", "Δrx%", "rx_bytes", "Δrx_b%",
    "tx_pkts", "Δtx%", "tx_bytes", "Δtx_b%",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def fmt_pkts(n):
    n = int(n)
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000:     return f"{n/1_000:.1f}k"
    return str(n)


def fmt_num(n):
    """Format any number with commas for readability."""
    try:
        return f"{int(n):,}"
    except Exception:
        return str(n)


def _find_default_config():
    """Return the first config.yaml found in the usual locations."""
    candidates = [
        os.path.join(_ROOT, "config", "default.yaml"),
        os.path.join(_ROOT, "base_config.yaml"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def load_config(config_path, scenario):
    """
    Read the YAML config, set gym_type to the requested scenario, and fill in
    the few attributes that main.py normally sets at runtime.
    """
    from utility.params import read_config_file

    config, _ = read_config_file(config_path)
    config.env_params.gym_type = scenario

    # training_execution_directory is needed by generate_and_save_scenario
    # to know where to write scenario.json; we point it at a temp dir.
    tmp_dir = tempfile.mkdtemp(prefix="diag_scenario_")
    config.training_execution_directory = tmp_dir

    # data_traffic_file is read in NetworkEnv.__init__ — set to empty string
    if not hasattr(config.env_params, 'data_traffic_file'):
        config.env_params.data_traffic_file = ''

    return config


# ─────────────────────────────────────────────────────────────────────────────
# Standalone Mininet network (no remote controller / ODL)
# ─────────────────────────────────────────────────────────────────────────────

def build_standalone_net(net_params):
    """
    Build a Mininet network in failMode=standalone so OVS forwards packets
    without a remote controller.  Returns the Mininet net object ready for
    use as existing_net in the environment constructor.
    """
    from mininet.net import Mininet
    from mininet.node import OVSKernelSwitch
    from mininet.link import TCLink
    from mininet.clean import cleanup
    from functools import partial

    cleanup()

    num_hosts = getattr(net_params, 'num_hosts', 5)
    num_iots  = getattr(net_params, 'num_iots',  0)

    StandaloneOVS = partial(OVSKernelSwitch,
                            failMode='standalone',
                            protocols='OpenFlow10,OpenFlow13')
    net = Mininet(controller=None, switch=StandaloneOVS, link=TCLink)

    sw = net.addSwitch('s1')

    for i in range(1, num_hosts + 1):
        h = net.addHost(f'h{i}', ip=f'10.0.0.{i}/24')
        net.addLink(h, sw)

    for i in range(1, num_iots + 1):
        iot = net.addHost(f'iot{i}',
                          ip=f'10.0.0.{num_hosts + i}/24')
        net.addLink(iot, sw)

    # Attributes the framework expects on the net object
    net.hosts_links          = {h.name: {} for h in net.hosts}
    net.blocked_hosts        = []
    net.traffic_types        = getattr(net_params, 'traffic_types',
                                       ['none', 'ping', 'udp', 'tcp'])
    net.switches             = [sw]
    net.total_packets_received    = 0
    net.total_bytes_received      = 0
    net.total_packets_transmitted = 0
    net.total_bytes_transmitted   = 0

    net.start()

    sw.cmd('ovs-vsctl set bridge s1 protocols=OpenFlow10,OpenFlow13')
    sw.cmd('ovs-ofctl add-flow s1 priority=1,actions=normal')
    time.sleep(0.5)
    return net


# ─────────────────────────────────────────────────────────────────────────────
# Discretization helpers (mirror the env's get_discretized_state logic)
# ─────────────────────────────────────────────────────────────────────────────

def _log_bin_boundaries(high, n_bins):
    """
    Return the n_bins-1 boundary values used by get_log_bin_index.
    get_log_bin_index(val, 1.0, high, n_bins-1) produces:
        bins = np.logspace(1.0, log10(high), n_bins-1)
    Values below bins[0]=10 fall in bin 0 together with zero.
    """
    if high <= 1.0:
        return []
    return list(np.logspace(1.0, np.log10(max(high, 10.0)), n_bins - 1))


def _linear_bin_boundaries(low, high, n_bins):
    """
    Boundaries for get_linear_bin_index(val, low, high, n_bins-1).
    bins = np.linspace(low, high, n_bins-1)
    """
    return list(np.linspace(low, high, n_bins - 1))


def _discretize_host(hs, low, high, n_bins):
    """
    Compute per-host discretized bins for an 8-element state slice.
    Mirrors NetworkEnvAttackDetectPerHostObservable.get_discretized_state.
    """
    from reinforcement_learning.network_env import get_log_bin_index, get_linear_bin_index
    n = max(2, int(n_bins))
    counter_idx   = {0, 2, 4, 6}
    variation_idx = {1, 3, 5, 7}
    disc = []
    for i, val in enumerate(hs):
        if i in variation_idx:
            b = get_linear_bin_index(val, low[i], high[i], n - 1) + 1
        elif i in counter_idx:
            if val <= 0:
                b = 0
            else:
                h_safe = max(float(high[i]), 1.0)
                b = get_log_bin_index(val, 1.0, h_safe, n - 1) + 1
        else:
            b = get_linear_bin_index(val, low[i], high[i], n)
        disc.append(int(b))
    return disc


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_step_header(global_step, episode, ep_step, mode, attacker, victim):
    color = RED if mode == 'attack' else GREEN
    label = "ATTACK" if mode == 'attack' else "NORMAL"
    ep_info = f"ep={episode} step={ep_step}"
    atk_info = f"  {attacker} → {victim}" if mode == 'attack' else ""
    print(f"\n{color}{BOLD}[g={global_step:03d} {ep_info}] {label:6s}{RESET}"
          f"{color}{atk_info}{RESET}")


def _print_host_line(host_name, hs, disc, line_color):
    """
    Print one host row with traffic stats and discretized bin vector.

    hs        : 8-element array [rx_pkts, rx_pct, rx_bytes, rx_b_pct,
                                  tx_pkts, tx_pct, tx_bytes, tx_b_pct]
    disc      : list of 8 bin indices
    line_color: ANSI colour applied to the entire line (RESET for normal hosts)
    """
    rx_pkts  = int(hs[0])
    rx_bytes = int(hs[2])
    tx_pkts  = int(hs[4])
    tx_bytes = int(hs[6])
    bins_str = ",".join(str(b) for b in disc)
    print(f"{line_color}  {host_name:6s}"
          f"  RX:{fmt_pkts(rx_pkts):>8}"
          f"  TX:{fmt_pkts(tx_pkts):>8}"
          f"  rx_B:{fmt_pkts(rx_bytes):>8}"
          f"  [{bins_str}]{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# Threshold analysis
# ─────────────────────────────────────────────────────────────────────────────

def _bin_range_str(boundaries, bin_idx, n_bins):
    """Human-readable range for one bin index."""
    if bin_idx == 0:
        lo = boundaries[0] if boundaries else 0
        return f"< {fmt_num(lo)}"
    if bin_idx >= n_bins - 1:
        lo = boundaries[-1] if boundaries else "?"
        return f"≥ {fmt_num(lo)}"
    lo = boundaries[bin_idx - 1]
    hi = boundaries[bin_idx]
    return f"{fmt_num(lo)} … <{fmt_num(hi)}"


def _print_threshold_analysis(thr_pkts, thr_var_pkts, thr_bytes, thr_var_bytes,
                               n_bins,
                               victim_rx_pkts, victim_rx_bytes,
                               attacker_tx_pkts, attacker_tx_bytes,
                               normal_rx_pkts,  normal_rx_bytes,
                               normal_tx_pkts,  normal_tx_bytes):
    """Print the full threshold / bin-boundary / optimal-suggestion section."""

    W = 72
    div = "─" * W

    print(f"\n{BOLD}{CYAN}{div}")
    print(f"  THRESHOLDS & DISCRETIZATION")
    print(f"{div}{RESET}")

    # ── current config values ─────────────────────────────────────────────
    print(f"\n  {BOLD}Config:{RESET}"
          f"  n_bins={n_bins}"
          f"  |  packets={fmt_num(thr_pkts)}"
          f"  var_packets=±{thr_var_pkts}"
          f"  |  bytes={fmt_num(thr_bytes)}"
          f"  var_bytes=±{thr_var_bytes}")

    # ── packet counter bin boundaries ─────────────────────────────────────
    pkt_bounds = _log_bin_boundaries(thr_pkts, n_bins)
    print(f"\n  {BOLD}Packet counter bins (log scale, high={fmt_num(thr_pkts)}):{RESET}")
    for b in range(n_bins):
        rng = _bin_range_str(pkt_bounds, b, n_bins)
        label = ""
        if b == 0:           label = "  ← zero / idle"
        elif b == n_bins-1:  label = "  ← HIGH / attack zone"
        print(f"    bin {b}  :  {rng}{label}")

    # ── bytes counter bin boundaries ──────────────────────────────────────
    byte_bounds = _log_bin_boundaries(thr_bytes, n_bins)
    print(f"\n  {BOLD}Byte counter bins (log scale, high={fmt_num(thr_bytes)}):{RESET}")
    for b in range(n_bins):
        rng = _bin_range_str(byte_bounds, b, n_bins)
        label = ""
        if b == 0:           label = "  ← zero / idle"
        elif b == n_bins-1:  label = "  ← HIGH / attack zone"
        print(f"    bin {b}  :  {rng}{label}")

    # variation bin boundaries (linear, symmetric)
    var_pkt_bounds = _linear_bin_boundaries(-thr_var_pkts, thr_var_pkts, n_bins)
    print(f"\n  {BOLD}Variation bins (linear ±{thr_var_pkts} for pkts, ±{thr_var_bytes} for bytes):{RESET}")
    for b in range(n_bins):
        rng = _bin_range_str(var_pkt_bounds, b, n_bins)
        print(f"    bin {b}  :  {rng}")

    # ── observed stats ────────────────────────────────────────────────────
    print(f"\n{BOLD}{CYAN}{div}")
    print(f"  OBSERVED VALUES BY ROLE")
    print(f"{div}{RESET}")

    def stat_row(label, values, bounds, n_b, color):
        if not values:
            print(f"  {label:30s}  {DIM}no data{RESET}")
            return
        mn = np.min(values);  mx = np.max(values);  av = np.mean(values)
        b_min = int(np.digitize(mn, bounds))  if bounds else 0
        b_max = int(np.digitize(mx, bounds))  if bounds else 0
        b_avg = int(np.digitize(av, bounds))  if bounds else 0
        print(f"  {color}{label:30s}"
              f"  min={fmt_pkts(mn):>8} (bin {b_min})"
              f"  mean={fmt_pkts(av):>8} (bin {b_avg})"
              f"  max={fmt_pkts(mx):>8} (bin {b_max}){RESET}")

    print(f"\n  {BOLD}rx_pkts  (victim during attacks  /  all hosts during normal):{RESET}")
    stat_row("attack VICTIM rx_pkts",  victim_rx_pkts,  pkt_bounds, n_bins, PINK)
    stat_row("normal ALL    rx_pkts",  normal_rx_pkts,  pkt_bounds, n_bins, GREEN)

    print(f"\n  {BOLD}tx_pkts  (attacker during attacks  /  all hosts during normal):{RESET}")
    stat_row("attack ATTACKER tx_pkts", attacker_tx_pkts, pkt_bounds, n_bins, PINK)
    stat_row("normal ALL      tx_pkts", normal_tx_pkts,   pkt_bounds, n_bins, GREEN)

    print(f"\n  {BOLD}rx_bytes (victim during attacks  /  all hosts during normal):{RESET}")
    stat_row("attack VICTIM rx_bytes",  victim_rx_bytes,  byte_bounds, n_bins, PINK)
    stat_row("normal ALL    rx_bytes",  normal_rx_bytes,  byte_bounds, n_bins, GREEN)

    # ── optimal thresholds ────────────────────────────────────────────────
    print(f"\n{BOLD}{CYAN}{div}")
    print(f"  OPTIMAL THRESHOLD SUGGESTIONS")
    print(f"{div}{RESET}")

    def suggest_threshold(label, atk_vals, nrm_vals, cur_thr, unit="pkts"):
        print(f"\n  {BOLD}{label}{RESET}  (current: {fmt_num(cur_thr)})")
        if not atk_vals or not nrm_vals:
            print(f"    {YELLOW}⚠ not enough data (need both attack and normal steps){RESET}")
            return None
        max_nrm = float(np.max(nrm_vals))
        min_atk = float(np.min(atk_vals))
        mean_atk = float(np.mean(atk_vals))
        if min_atk > max_nrm:
            gap_lo = max_nrm
            gap_hi = min_atk
            # geometric mean keeps the threshold in the middle of the log gap
            opt = int(np.sqrt(gap_lo * gap_hi)) if gap_lo > 0 else int(min_atk / 2)
            opt = max(opt, int(max_nrm) + 1)
            print(f"    {GREEN}✓ Clean separation: max_normal={fmt_pkts(max_nrm)}"
                  f"  <  min_attack={fmt_pkts(min_atk)}{RESET}")
            print(f"    → Optimal {unit} threshold: {fmt_num(opt)}"
                  f"  (geometric mean of gap; attack will be in highest bin)")
        else:
            overlap = max_nrm - min_atk
            print(f"    {RED}✗ Overlap: max_normal={fmt_pkts(max_nrm)}"
                  f"  ≥  min_attack={fmt_pkts(min_atk)}"
                  f"  (overlap={fmt_pkts(overlap)}){RESET}")
            # Still suggest: just above mean attack so most attack steps are detected
            opt = int(mean_atk * 0.5)
            print(f"    → Suggest {unit} threshold: {fmt_num(opt)}"
                  f"  (50%% of mean attack; partial detection until traffic improves)")
        return opt

    opt_pkts  = suggest_threshold("threshold_packets",
                                   victim_rx_pkts, normal_rx_pkts,
                                   thr_pkts, "packets")
    opt_bytes = suggest_threshold("threshold_bytes",
                                   victim_rx_bytes, normal_rx_bytes,
                                   thr_bytes, "bytes")

    # Print YAML snippet for easy copy-paste
    print(f"\n  {BOLD}Suggested YAML update (attacks.thresholds):{RESET}")
    opt_pkts_out  = opt_pkts  if opt_pkts  else thr_pkts
    opt_bytes_out = opt_bytes if opt_bytes else thr_bytes
    print(f"    thresholds:")
    print(f"      packets  : {opt_pkts_out}")
    print(f"      var_packets: {thr_var_pkts}")
    print(f"      bytes    : {opt_bytes_out}")
    print(f"      var_bytes: {thr_var_bytes}")


# ─────────────────────────────────────────────────────────────────────────────
# Main diagnostic loop
# ─────────────────────────────────────────────────────────────────────────────

def run_diagnostics(config_path, scenario, max_steps, episodes):
    from utility.constants import SHORT_ATTACK, LONG_ATTACK, NORMAL
    from utility.network_flows import initialize_monitoring
    from utility.scenario_generator import generate_and_save_scenario
    from reinforcement_learning.scenarios.attack_detect_host_observable.\
        network_env_attack_detect_per_host_observable import (
        NetworkEnvAttackDetectPerHostObservable,
    )

    print(f"\n{BOLD}{CYAN}=== Traffic Diagnostic Tool ==={RESET}")
    print(f"  Scenario : {scenario}")
    print(f"  Config   : {config_path}")
    print(f"  Episodes : {episodes}  |  Max steps/episode: {max_steps or 'all'}\n")

    # ── 1. Load config ────────────────────────────────────────────────────────
    config = load_config(config_path, scenario)

    ep_cfg = config.env_params
    # Override episodes/steps if requested from CLI
    if episodes:
        ep_cfg.episodes      = episodes
        ep_cfg.test_episodes = 0
    if max_steps:
        ep_cfg.max_steps = max_steps

    # Read thresholds and n_bins from config
    attacks_cfg   = getattr(ep_cfg, 'attacks', None)
    thr_obj       = getattr(attacks_cfg, 'thresholds', None)
    thr_pkts      = getattr(thr_obj, 'packets',     120000)      if thr_obj else 120000
    thr_var_pkts  = getattr(thr_obj, 'var_packets', 50)          if thr_obj else 50
    thr_bytes     = getattr(thr_obj, 'bytes',       4_230_000_000) if thr_obj else 4_230_000_000
    thr_var_bytes = getattr(thr_obj, 'var_bytes',   30)          if thr_obj else 30
    n_bins        = getattr(ep_cfg, 'n_bins', 4)
    train_likely  = getattr(attacks_cfg, 'likely_train', 0.5) if attacks_cfg else 0.5

    # Build env low/high arrays for discretization (mirrors NetworkEnv init)
    env_low  = np.array([0, -thr_var_pkts,  0, -thr_var_bytes,
                          0, -thr_var_pkts,  0, -thr_var_bytes], dtype=np.float32)
    env_high = np.array([thr_pkts,  thr_var_pkts,  thr_bytes,  thr_var_bytes,
                          thr_pkts,  thr_var_pkts,  thr_bytes,  thr_var_bytes],
                         dtype=np.float32)

    print(f"  Hosts : {ep_cfg.net_params.num_hosts} regular + "
          f"{ep_cfg.net_params.num_iots} IoT")
    print(f"  Attack likely (train target fraction): {train_likely}"
          f"  |  n_bins={n_bins}")
    print(f"  threshold_packets={fmt_num(thr_pkts)}"
          f"  threshold_bytes={fmt_num(thr_bytes)}")

    # ── 2. Build standalone Mininet (no ODL) ─────────────────────────────────
    print(f"\n{DIM}Building standalone Mininet network...{RESET}")
    net = build_standalone_net(ep_cfg.net_params)

    # initialize_monitoring installs per-host OVS tracking flows
    ok = initialize_monitoring(net, 's1')
    if not ok:
        print(f"{YELLOW}  ⚠ initialize_monitoring failed — counters may be zero{RESET}")
    else:
        print(f"  {GREEN}✓ OVS monitoring flows installed{RESET}")

    # ── 3. Create environment (reuse existing net, no controller needed) ──────
    print(f"{DIM}Creating environment...{RESET}")
    server_user = getattr(config, 'server_user', 'mininet-gym')
    base_env = NetworkEnvAttackDetectPerHostObservable(ep_cfg, server_user,
                                                       existing_net=net)
    base_env.stop_event  = __import__('threading').Event()
    base_env.pause_event = __import__('threading').Event()

    # ── 4. Generate scenario ──────────────────────────────────────────────────
    print(f"{DIM}Generating scenario...{RESET}")
    scenario_path = os.path.join(config.training_execution_directory, "scenario.json")
    full_scenario = generate_and_save_scenario(
        base_env, config, scenario_path,
        train_attack_likely=train_likely,
        eval_attack_likely=0.3,
    )
    training_steps = list(full_scenario["training"])
    total = len(training_steps)
    print(f"  {GREEN}✓ Scenario generated: {total} training steps{RESET}")

    # ── 5. Replay steps ───────────────────────────────────────────────────────
    base_env.df = list(training_steps)   # env pops from df in update_state()

    # Accumulators for threshold analysis
    victim_rx_pkts    = []
    victim_rx_bytes   = []
    attacker_tx_pkts  = []
    attacker_tx_bytes = []
    normal_rx_pkts    = []
    normal_rx_bytes   = []
    normal_tx_pkts    = []
    normal_tx_bytes   = []
    attack_victim_rx_per_step = []  # for separation ratio
    normal_all_rx_per_step    = []

    host_names = [h.name for h in base_env.hosts]
    print(f"\n  Hosts in env: {', '.join(host_names)}")
    print(f"\n  Row colours:  {PINK}■ attacker{RESET}  {YELLOW}■ victim{RESET}"
          f"  normal (no colour)")
    print(f"  Bins format:  [rx_pkts, Δrx%, rx_bytes, Δrx_b%, "
          f"tx_pkts, Δtx%, tx_bytes, Δtx_b%]")
    print(f"\n{BOLD}{'─'*72}{RESET}")

    steps_done = 0
    while base_env.df:
        # Peek at next step to know ground truth before popping
        step_plan = base_env.df[0]
        global_step = step_plan.get("step",        steps_done)
        episode     = step_plan.get("episode",     0)
        ep_step     = step_plan.get("episode_step", 0)

        attacker_name = victim_name = None
        step_mode     = 'normal'
        for hname in host_names:
            hplan = step_plan.get(hname, {})
            if hplan.get("traffic_type") == "attack":
                step_mode     = 'attack'
                attacker_name = hname
                victim_name   = hplan.get("destination")
                break

        _print_step_header(global_step, episode, ep_step,
                           step_mode, attacker_name, victim_name)

        # Run the real traffic generation + OVS read
        base_env.update_state()  # pops from df, calls _apply_scenario_step, reads OVS

        # Read per-host deltas from global_state.host_states
        # indices: 0=rx_pkts, 1=rx_pct, 2=rx_bytes, 3=rx_bytes_pct,
        #          4=tx_pkts, 5=tx_pct, 6=tx_bytes, 7=tx_bytes_pct
        step_victim_rx  = 0
        step_all_rx     = 0

        for hname in host_names:
            hs = base_env.global_state.host_states.get(hname)
            if hs is None:
                continue

            # Compute discretized bins for this host
            host_arr = np.array([float(x) for x in hs], dtype=np.float32)
            disc = _discretize_host(host_arr, env_low, env_high, n_bins)

            is_atk = (hname == attacker_name)
            is_vic = (hname == victim_name)

            if step_mode == 'attack':
                line_color = PINK if is_atk else (YELLOW if is_vic else RESET)
            else:
                line_color = RESET

            _print_host_line(hname, host_arr, disc, line_color)

            # Accumulate stats
            rx_pkts  = float(hs[0])
            rx_bytes = float(hs[2])
            tx_pkts  = float(hs[4])
            tx_bytes = float(hs[6])

            step_all_rx += rx_pkts

            if step_mode == 'attack':
                if is_vic:
                    victim_rx_pkts.append(rx_pkts)
                    victim_rx_bytes.append(rx_bytes)
                    step_victim_rx = rx_pkts
                if is_atk:
                    attacker_tx_pkts.append(tx_pkts)
                    attacker_tx_bytes.append(tx_bytes)
            else:
                normal_rx_pkts.append(rx_pkts)
                normal_rx_bytes.append(rx_bytes)
                normal_tx_pkts.append(tx_pkts)
                normal_tx_bytes.append(tx_bytes)

        if step_mode == 'attack':
            attack_victim_rx_per_step.append(step_victim_rx)
        else:
            normal_all_rx_per_step.append(step_all_rx / max(len(host_names), 1))

        steps_done += 1
        if max_steps and steps_done >= max_steps:
            break

    # ── 6. Summary ────────────────────────────────────────────────────────────
    W = 72
    print(f"\n{BOLD}{CYAN}{'─'*W}")
    print(f"  SUMMARY  ({steps_done} steps replayed)")
    print(f"{'─'*W}{RESET}")

    mean_atk_rx = np.mean(attack_victim_rx_per_step) if attack_victim_rx_per_step else 0.0
    mean_nrm_rx = np.mean(normal_all_rx_per_step)    if normal_all_rx_per_step    else 0.0
    ratio        = mean_atk_rx / max(mean_nrm_rx, 1.0)

    n_atk = sum(1 for s in training_steps[:steps_done]
                if any(s.get(h, {}).get("traffic_type") == "attack"
                       for h in host_names))
    n_nrm = steps_done - n_atk

    print(f"\n  Steps : {n_atk} attack  /  {n_nrm} normal")
    print(f"\n  Victim RX pkts per step:")
    print(f"    {PINK}Attack  mean={fmt_pkts(mean_atk_rx):>8}  "
          f"min={fmt_pkts(min(attack_victim_rx_per_step, default=0)):>8}  "
          f"max={fmt_pkts(max(attack_victim_rx_per_step, default=0)):>8}{RESET}")
    print(f"    {GREEN}Normal  mean={fmt_pkts(mean_nrm_rx):>8}  "
          f"min={fmt_pkts(min(normal_all_rx_per_step, default=0)):>8}  "
          f"max={fmt_pkts(max(normal_all_rx_per_step, default=0)):>8}{RESET}")

    sep_color = GREEN if ratio >= 5 else (YELLOW if ratio >= 2 else RED)
    print(f"\n  Separation ratio (victim attack / normal mean): "
          f"{sep_color}{BOLD}{ratio:.1f}x{RESET}")

    if ratio >= 5:
        print(f"  {GREEN}✓ Attacks clearly distinguishable — agent should learn detection{RESET}")
    elif ratio >= 2:
        print(f"  {YELLOW}⚠ Weak separation — agent will struggle{RESET}")
    else:
        print(f"  {RED}✗ No separation — agent CANNOT learn detection{RESET}")
        print(f"    → Check: hping3 start errors above, iperf failures")

    # ── 7. Threshold & optimal analysis ──────────────────────────────────────
    _print_threshold_analysis(
        thr_pkts, thr_var_pkts, thr_bytes, thr_var_bytes, n_bins,
        victim_rx_pkts,   victim_rx_bytes,
        attacker_tx_pkts, attacker_tx_bytes,
        normal_rx_pkts,   normal_rx_bytes,
        normal_tx_pkts,   normal_tx_bytes,
    )

    base_env.stop()
    print(f"\n{BOLD}Done.{RESET}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset mode — replay from statuses.json (no Mininet needed)
# ─────────────────────────────────────────────────────────────────────────────

_SCENARIO_YAML_MAP = {
    "attacks_ho":   os.path.join(_ROOT, "reinforcement_learning", "scenarios",
                                 "attack_detect_host_observable", "scenario_env_param.yaml"),
    "marl_attacks": os.path.join(_ROOT, "reinforcement_learning", "scenarios",
                                 "marl", "scenario_env_param.yaml"),
}


def _load_thresholds_from_yaml(scenario):
    yaml_path = _SCENARIO_YAML_MAP.get(scenario)
    if not yaml_path or not os.path.exists(yaml_path):
        return {}
    try:
        import yaml
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("attacks", {}).get("thresholds", {})
    except Exception:
        return {}


def run_dataset_diagnostics(dataset_path, scenario, max_steps=None, step_start=0,
                             pause=False, thr_pkts_override=None,
                             thr_var_pkts_override=None, thr_bytes_override=None,
                             thr_var_bytes_override=None, n_bins_override=None):
    """
    Show the same per-step diagnostic table as run_diagnostics(), but reading
    recorded data from a past experiment's statuses.json instead of generating
    live traffic.  Does not require root or Mininet.

    dataset_path : path to the experiment folder (contains statuses.json)
                   or directly to a statuses.json file.
    """
    import json

    # ── locate statuses.json ──────────────────────────────────────────────────
    if os.path.isdir(dataset_path):
        statuses_path = os.path.join(dataset_path, "statuses.json")
    else:
        statuses_path = dataset_path

    if not os.path.exists(statuses_path):
        sys.exit(f"ERROR: statuses.json not found at {statuses_path}")

    with open(statuses_path) as f:
        data = json.load(f)

    # ── resolve thresholds: CLI override > scenario YAML > fallback ───────────
    yaml_thr      = _load_thresholds_from_yaml(scenario)
    thr_pkts      = thr_pkts_override      if thr_pkts_override      is not None \
                    else yaml_thr.get("packets",      12000)
    thr_var_pkts  = thr_var_pkts_override  if thr_var_pkts_override  is not None \
                    else yaml_thr.get("var_packets",     50)
    thr_bytes     = thr_bytes_override     if thr_bytes_override     is not None \
                    else yaml_thr.get("bytes",   423_000_000)
    thr_var_bytes = thr_var_bytes_override if thr_var_bytes_override is not None \
                    else yaml_thr.get("var_bytes",       30)
    n_bins        = n_bins_override if n_bins_override else 4

    env_low  = np.array([0, -thr_var_pkts, 0, -thr_var_bytes,
                          0, -thr_var_pkts, 0, -thr_var_bytes], dtype=np.float32)
    env_high = np.array([thr_pkts, thr_var_pkts, thr_bytes, thr_var_bytes,
                          thr_pkts, thr_var_pkts, thr_bytes, thr_var_bytes],
                         dtype=np.float32)

    print(f"\n{BOLD}{CYAN}=== Traffic Diagnostic — Dataset Mode ==={RESET}")
    print(f"  Source   : {statuses_path}")
    print(f"  Scenario : {scenario}  |  n_bins={n_bins}")
    print(f"  threshold_packets={fmt_num(thr_pkts)}"
          f"  threshold_bytes={fmt_num(thr_bytes)}")
    if yaml_thr:
        print(f"  {DIM}(thresholds loaded from scenario YAML){RESET}")

    # ── extract host names from first non-empty step ──────────────────────────
    host_names = []
    for s in data:
        hns = sorted(s.get("hostStatusesStructured", {}).keys())
        if hns:
            host_names = hns
            break
    if not host_names:
        sys.exit("ERROR: no hostStatusesStructured found in statuses.json")

    print(f"  Hosts    : {', '.join(host_names)}")
    print(f"\n  Row colours:  {PINK}■ attacker{RESET}  {YELLOW}■ victim{RESET}"
          f"  normal (no colour)")
    print(f"  Bins format:  [rx_pkts, Δrx%, rx_bytes, Δrx_b%, "
          f"tx_pkts, Δtx%, tx_bytes, Δtx_b%]")
    print(f"\n{BOLD}{'─'*72}{RESET}")

    # ── accumulators for threshold analysis (same as real-time mode) ──────────
    victim_rx_pkts    = []
    victim_rx_bytes   = []
    attacker_tx_pkts  = []
    attacker_tx_bytes = []
    normal_rx_pkts    = []
    normal_rx_bytes   = []
    normal_tx_pkts    = []
    normal_tx_bytes   = []
    attack_victim_rx_per_step = []
    normal_all_rx_per_step    = []

    # ── iterate steps ─────────────────────────────────────────────────────────
    steps = data[step_start:]
    if max_steps:
        steps = steps[:max_steps]

    steps_done = 0
    try:
        for step_idx, step in enumerate(steps):
            hosts_info = step.get("hostStatusesStructured", {})
            if not hosts_info:
                continue

            abs_idx = step_start + step_idx

            # Determine mode and roles
            attacker_name = victim_name = None
            step_mode = 'normal'
            for hname, info in hosts_info.items():
                st = info.get("status", "")
                if st == "attacking":
                    step_mode     = 'attack'
                    attacker_name = hname
                    victim_name   = info.get("destination")
                elif st == "under_attack" and victim_name is None:
                    victim_name   = hname

            _print_step_header(
                abs_idx,
                step.get("episode",      0),
                step.get("episode_step", abs_idx),
                step_mode, attacker_name, victim_name,
            )

            step_victim_rx = 0
            step_all_rx    = 0

            for hname in host_names:
                info = hosts_info.get(hname)
                if info is None:
                    continue

                # Build the same 8-element state vector the env produces
                host_arr = np.array([
                    float(info.get("receivedPackets",                  0)),
                    float(info.get("receivedPacketsPercentageChange",   0)),
                    float(info.get("receivedBytes",                    0)),
                    float(info.get("receivedBytesPercentageChange",     0)),
                    float(info.get("transmittedPackets",               0)),
                    float(info.get("transmittedPacketsPercentageChange",0)),
                    float(info.get("transmittedBytes",                 0)),
                    float(info.get("transmittedBytesPercentageChange",  0)),
                ], dtype=np.float32)

                disc = _discretize_host(host_arr, env_low, env_high, n_bins)

                is_atk = (hname == attacker_name)
                is_vic = (hname == victim_name)
                if step_mode == 'attack':
                    line_color = PINK if is_atk else (YELLOW if is_vic else RESET)
                else:
                    line_color = RESET

                _print_host_line(hname, host_arr, disc, line_color)

                rx_pkts  = float(host_arr[0])
                rx_bytes = float(host_arr[2])
                tx_pkts  = float(host_arr[4])
                tx_bytes = float(host_arr[6])
                step_all_rx += rx_pkts

                if step_mode == 'attack':
                    if is_vic:
                        victim_rx_pkts.append(rx_pkts)
                        victim_rx_bytes.append(rx_bytes)
                        step_victim_rx = rx_pkts
                    if is_atk:
                        attacker_tx_pkts.append(tx_pkts)
                        attacker_tx_bytes.append(tx_bytes)
                else:
                    normal_rx_pkts.append(rx_pkts)
                    normal_rx_bytes.append(rx_bytes)
                    normal_tx_pkts.append(tx_pkts)
                    normal_tx_bytes.append(tx_bytes)

            if step_mode == 'attack':
                attack_victim_rx_per_step.append(step_victim_rx)
            else:
                normal_all_rx_per_step.append(step_all_rx / max(len(host_names), 1))

            steps_done += 1

            if pause:
                sys.stdout.write(
                    f"\n  {DIM}── [Enter] next step  [Ctrl-C] stop ──{RESET}  ")
                sys.stdout.flush()
                sys.stdin.readline()

    except KeyboardInterrupt:
        print(f"\n{DIM}Interrupted.{RESET}")

    # ── summary ───────────────────────────────────────────────────────────────
    W = 72
    print(f"\n{BOLD}{CYAN}{'─'*W}")
    print(f"  SUMMARY  ({steps_done} steps from dataset)")
    print(f"{'─'*W}{RESET}")

    mean_atk_rx = np.mean(attack_victim_rx_per_step) if attack_victim_rx_per_step else 0.0
    mean_nrm_rx = np.mean(normal_all_rx_per_step)    if normal_all_rx_per_step    else 0.0
    ratio        = mean_atk_rx / max(mean_nrm_rx, 1.0)
    n_atk        = len(attack_victim_rx_per_step)
    n_nrm        = len(normal_all_rx_per_step)

    print(f"\n  Steps : {n_atk} attack  /  {n_nrm} normal")
    print(f"\n  Victim RX pkts per step:")
    print(f"    {PINK}Attack  mean={fmt_pkts(mean_atk_rx):>8}  "
          f"min={fmt_pkts(min(attack_victim_rx_per_step, default=0)):>8}  "
          f"max={fmt_pkts(max(attack_victim_rx_per_step, default=0)):>8}{RESET}")
    print(f"    {GREEN}Normal  mean={fmt_pkts(mean_nrm_rx):>8}  "
          f"min={fmt_pkts(min(normal_all_rx_per_step, default=0)):>8}  "
          f"max={fmt_pkts(max(normal_all_rx_per_step, default=0)):>8}{RESET}")

    sep_color = GREEN if ratio >= 5 else (YELLOW if ratio >= 2 else RED)
    print(f"\n  Separation ratio (victim attack / normal mean): "
          f"{sep_color}{BOLD}{ratio:.1f}x{RESET}")

    if ratio >= 5:
        print(f"  {GREEN}✓ Attacks clearly distinguishable{RESET}")
    elif ratio >= 2:
        print(f"  {YELLOW}⚠ Weak separation — agent will struggle{RESET}")
    else:
        print(f"  {RED}✗ No separation — agent CANNOT learn detection{RESET}")

    _print_threshold_analysis(
        thr_pkts, thr_var_pkts, thr_bytes, thr_var_bytes, n_bins,
        victim_rx_pkts,   victim_rx_bytes,
        attacker_tx_pkts, attacker_tx_bytes,
        normal_rx_pkts,   normal_rx_bytes,
        normal_tx_pkts,   normal_tx_bytes,
    )

    print(f"\n{BOLD}Done.{RESET}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Traffic diagnostic: real-time mode (Mininet) or dataset mode (statuses.json)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--scenario', default=_DEFAULT_SCENARIO,
                        choices=_SCENARIO_CHOICES,
                        help='Scenario name (gym_type)')

    # ── dataset mode ──────────────────────────────────────────────────────────
    parser.add_argument('--dataset', default=None, metavar='PATH',
                        help='Experiment folder or statuses.json path — enables dataset mode '
                             '(no Mininet, no root required)')
    parser.add_argument('--step',    type=int, default=0,
                        help='First step to show in dataset mode')
    parser.add_argument('--pause',   action='store_true',
                        help='Pause between steps (Enter=next, Ctrl-C=stop)')

    # ── threshold overrides (both modes) ─────────────────────────────────────
    parser.add_argument('--thr-pkts',      type=int, default=None,
                        help='Override threshold_packets (default: from scenario YAML)')
    parser.add_argument('--thr-var-pkts',  type=int, default=None)
    parser.add_argument('--thr-bytes',     type=int, default=None)
    parser.add_argument('--thr-var-bytes', type=int, default=None)
    parser.add_argument('--n-bins',        type=int, default=None,
                        help='Override Q-table bins (default: 4)')

    # ── real-time mode ────────────────────────────────────────────────────────
    parser.add_argument('--config',   default=None,
                        help='Path to config.yaml (real-time mode only)')
    parser.add_argument('--steps',    type=int, default=None,
                        help='Max steps to run/show (default: all)')
    parser.add_argument('--episodes', type=int, default=1,
                        help='Episodes to generate (real-time mode only)')

    args = parser.parse_args()

    if args.dataset:
        # Dataset mode — no root required
        run_dataset_diagnostics(
            dataset_path        = args.dataset,
            scenario            = args.scenario,
            max_steps           = args.steps,
            step_start          = args.step,
            pause               = args.pause,
            thr_pkts_override   = args.thr_pkts,
            thr_var_pkts_override = args.thr_var_pkts,
            thr_bytes_override  = args.thr_bytes,
            thr_var_bytes_override = args.thr_var_bytes,
            n_bins_override     = args.n_bins,
        )
    else:
        # Real-time mode — requires root and Mininet
        if os.geteuid() != 0:
            sys.exit("ERROR: real-time mode must be run as root (sudo).\n"
                     "       For dataset mode use: --dataset <path>")

        config_path = args.config or _find_default_config()
        if not config_path:
            sys.exit("ERROR: no config.yaml found. Pass --config <path>")

        run_diagnostics(
            config_path = config_path,
            scenario    = args.scenario,
            max_steps   = args.steps,
            episodes    = args.episodes,
        )
