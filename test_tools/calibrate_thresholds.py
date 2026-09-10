#!/usr/bin/env python3.12
"""
Threshold calibrator — measures this server's actual normal vs. attack
packet/byte rates on a throwaway Mininet topology and recommends
`attacks.thresholds.packets` / `.bytes` values for the target scenario's
scenario_env_param.yaml.

Why this exists: `packets`/`bytes` thresholds gate the log-bin discretization
used by tabular agents (see discretize_attack_detect_ho_state /
get_log_bin_index). They are static numbers in
reinforcement_learning/scenarios/<scenario>/scenario_env_param.yaml, but the
actual pps/Bps Mininet can push in the fixed per-step window
(env_params.wait_after_read, default 1s) depends on THIS machine's CPU
(veth pairs + the OVS software switch both run on the host CPU). Move to a
faster/slower server and the same traffic generator commands produce a
different packet/byte count in that window — the thresholds don't move with
it. Too-low thresholds for a fast server make normal AND attack traffic
saturate into the same top bin, erasing the signal tabular agents need.

See also `net_params.link_bw_mbps` and `net_params.normal_traffic_bw_mbps` in
config/default.yaml: capping host links (tc/HTB) and the "normal" udp/tcp
iperf clients makes both the attack ceiling and the normal-traffic floor
reproducible across servers — this script is the calibration step that
comes after choosing those two values, not a replacement for them.

Usage:
  sudo .venv/bin/python3 test_tools/calibrate_thresholds.py \\
      [--gym-type marl_pz] [--num-hosts 5] [--num-iots 5] \\
      [--rounds 6] [--attack-types udp_flood,syn_flood] \\
      [--link-bw 100] [--write]

What it does:
  1. Builds a single-switch Mininet topology (no controller), sized like
     config/default.yaml's net_params (or --num-hosts/--num-iots).
  2. Normal phase: for --rounds rounds, every host exchanges real traffic
     (ping/udp/tcp, rotating) with a neighbour, exactly as the framework's
     _apply_scenario_step does (fire-and-forget + fixed wait_after_read
     window), then samples per-host RX/TX packets & bytes via the
     framework's real get_data_flow().
  3. Attack phase: for --rounds rounds, one attacker floods one victim
     (attack type rotating through --attack-types) using the framework's
     own launch_attack_smart(), rest of the hosts idle; samples the
     attacker's TX and victim's RX packets & bytes.
  4. Prints percentiles for both pools and a recommended threshold sitting
     at the log-midpoint between "typical normal ceiling" (p95) and
     "weakest observed attack" (p10) — warns if the two pools overlap
     (this server can't cleanly separate them with the current settings).
  5. With --write, patches `packets:`/`bytes:` under `thresholds:` in the
     target scenario_env_param.yaml in place (backs up the original as
     `.bak` first). `var_packets`/`var_bytes` are left untouched — this
     script does not calibrate the percentage-variation features.

This module is also imported by the web app (app/routes/calibration_routes.py)
via run_calibration(), so the same logic backs both the CLI and the
"Calibrate Thresholds" button in the system configuration panel.
"""

import argparse
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

try:
    import numpy as np
except ModuleNotFoundError:
    _venv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         ".venv", "bin", "python3")
    if os.path.exists(_venv) and os.path.realpath(_venv) != os.path.realpath(sys.executable):
        os.execv(_venv, [_venv] + sys.argv)
    sys.exit("ERROR: numpy not found and venv Python not available. "
             "Run with: sudo .venv/bin/python3 " + sys.argv[0])

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from test_tools.traffic_test import (
    RED, GREEN, YELLOW, CYAN, BOLD, RESET, DIM,
    build_network, kill_all_traffic, read_host_states,
)
from utility.network_flows import initialize_monitoring
from utility.params import read_config_file
from utility.scenario_params import (
    gym_type_to_scenario_folder, SCENARIOS_BASE_DIR, SCENARIO_PARAM_FILE,
)
from reinforcement_learning.network_env import get_log_bin_index


# ─────────────────────────────────────────────────────────────────────────────
# Sampling phases
# ─────────────────────────────────────────────────────────────────────────────

def run_normal_phase(net, host_names, host_objs, traffic_types, rounds, duration,
                     prev_totals, executor, normal_bw_mbps=1, log=print):
    from reinforcement_learning.agents.adversarial_agent import _generate_normal_traffic_once

    pkts, byts = [], []
    n = len(host_names)
    for r in range(rounds):
        _, prev_totals = read_host_states(net, prev_totals)     # baseline
        kill_all_traffic(net)
        time.sleep(0.1)

        ttype = traffic_types[r % len(traffic_types)]
        for i, name in enumerate(host_names):
            src = host_objs[name]
            dst = host_objs[host_names[(i + 1) % n]]
            executor.submit(_generate_normal_traffic_once, src, dst, ttype, duration, normal_bw_mbps)

        time.sleep(duration)                                    # fixed per-step window
        host_states, prev_totals = read_host_states(net, prev_totals)
        kill_all_traffic(net)

        for s8 in host_states.values():
            pkts.append(float(s8[0])); pkts.append(float(s8[4]))   # RX pkt, TX pkt
            byts.append(float(s8[2])); byts.append(float(s8[6]))   # RX bytes, TX bytes

        log(f"  {DIM}normal round {r + 1}/{rounds} ({ttype}){RESET}")

    return pkts, byts, prev_totals


def run_attack_phase(net, host_names, host_objs, attack_types, rounds, duration,
                     prev_totals, executor, normal_bw_mbps=1, log=print):
    from reinforcement_learning.agents.adversarial_agent import _launch_attack_once, _name_to_attack_type
    from reinforcement_learning.agents.adversarial_agent import _generate_normal_traffic_once
    import random

    pkts, byts = [], []
    n = len(host_names)
    if n < 2:
        log(f"  {YELLOW}Need at least 2 hosts to run the attack phase, skipping.{RESET}")
        return pkts, byts, prev_totals
    
    # Each attack type is run for a fixed duration (e.g., 5s) to get a stable measurement.
    # During the attack, other hosts generate normal background traffic.
    attack_duration = 5.0

    for attack_idx, attack_name in enumerate(attack_types):
        # Choose a random attacker/victim pair for this attack type
        attacker_name, victim_name = random.sample(host_names, 2)
        attack_type = _name_to_attack_type(attack_name)

        _, prev_totals = read_host_states(net, prev_totals)      # baseline
        kill_all_traffic(net)
        time.sleep(0.1)

        # 1. Launch the main attack
        executor.submit(_launch_attack_once, host_objs[attacker_name],
                        host_objs[victim_name], attack_type, attack_duration)

        # 2. Launch background normal traffic from all other hosts
        other_hosts = [h for h in host_names if h not in (attacker_name, victim_name)]
        log(f"  {DIM}attack round: {attacker_name}->{victim_name} ({attack_name}), "
            f"background traffic from {len(other_hosts)} other hosts.{RESET}")

        for i, name in enumerate(other_hosts):
            src = host_objs[name]
            # Simple round-robin target for background traffic
            dst_idx = (i + 1) % len(other_hosts)
            dst = host_objs[other_hosts[dst_idx]]
            # Use a mix of normal traffic types for background noise
            ttype = ["ping", "udp", "tcp"][i % 3]
            executor.submit(_generate_normal_traffic_once, src, dst, ttype, attack_duration, normal_bw_mbps)

        time.sleep(attack_duration)
        host_states, prev_totals = read_host_states(net, prev_totals)
        kill_all_traffic(net)

        attacker_s8 = host_states.get(attacker_name)
        victim_s8 = host_states.get(victim_name)
        if attacker_s8 is not None:
            pkts.append(float(attacker_s8[4])); byts.append(float(attacker_s8[6]))  # attacker TX
        if victim_s8 is not None:
            pkts.append(float(victim_s8[0])); byts.append(float(victim_s8[2]))      # victim RX

        log(f"  {DIM}attack type {attack_idx + 1}/{len(attack_types)}  "
            f"{attacker_name}->{victim_name} ({attack_name}){RESET}")
    return pkts, byts, prev_totals


# ─────────────────────────────────────────────────────────────────────────────
# Statistics / recommendation
# ─────────────────────────────────────────────────────────────────────────────

def _pct(arr, q):
    return float(np.percentile(arr, q)) if len(arr) else 0.0


def summarize(name, normal_vals, attack_vals, safety_factor, log=print):
    normal_arr = np.asarray(normal_vals, dtype=float)
    attack_arr = np.asarray(attack_vals, dtype=float)

    n_p50, n_p95, n_max = _pct(normal_arr, 50), _pct(normal_arr, 95), (float(normal_arr.max()) if len(normal_arr) else 0.0)
    a_p10, a_p50, a_max = _pct(attack_arr, 10), _pct(attack_arr, 50), (float(attack_arr.max()) if len(attack_arr) else 0.0)

    ceiling = max(n_p95, 1.0)
    floor   = max(a_p10, 1.0)
    ratio   = floor / ceiling
    ok      = ratio >= safety_factor
    recommended = float(np.sqrt(ceiling * floor))

    log(f"\n{BOLD}{name}{RESET}")
    log(f"  normal   n={len(normal_arr):<4d}  p50={n_p50:>12,.0f}  p95={n_p95:>12,.0f}  max={n_max:>12,.0f}")
    log(f"  attack   n={len(attack_arr):<4d}  p10={a_p10:>12,.0f}  p50={a_p50:>12,.0f}  max={a_max:>12,.0f}")
    ratio_col = GREEN if ok else RED
    log(f"  separation ratio (attack p10 / normal p95): {ratio_col}{ratio:.2f}x{RESET}  "
        f"(need >= {safety_factor:.1f}x)")
    if ok:
        log(f"  {GREEN}Recommended threshold: {recommended:,.0f}{RESET}")
    else:
        log(f"  {RED}WARNING: normal and attack traffic overlap on this server "
            f"— no single threshold cleanly separates them.{RESET}")
        log(f"  {YELLOW}Best-effort threshold (log-midpoint): {recommended:,.0f} "
            f"— expect some normal traffic to reach the attack bins, or vice versa.{RESET}")
        log(f"  {DIM}Remedies: increase rounds for more stable percentiles, lower the normal "
            f"traffic rate in the scenario, use a stronger/longer attack burst, or reduce "
            f"n_bins so the classes still separate coarsely.{RESET}")

    return {
        "normal_p50": n_p50, "normal_p95": n_p95, "normal_max": n_max,
        "attack_p10": a_p10, "attack_p50": a_p50, "attack_max": a_max,
        "ratio": ratio, "ok": ok, "recommended": recommended,
    }


def _bin_of(val, threshold, n_bins):
    if val <= 0:
        return 0
    return int(get_log_bin_index(val, 1.0, max(float(threshold), 1.0), n_bins - 1)) + 1


def print_bin_check(label, threshold, n_bins, normal_p95, attack_p50, attack_max, log=print):
    log(f"  {DIM}bin check @ n_bins={n_bins}, threshold={threshold:,.0f}:{RESET} "
        f"normal p95 -> bin {_bin_of(normal_p95, threshold, n_bins)}   "
        f"attack p50 -> bin {_bin_of(attack_p50, threshold, n_bins)}   "
        f"attack max -> bin {_bin_of(attack_max, threshold, n_bins)}  (of 0..{n_bins - 1})")


# ─────────────────────────────────────────────────────────────────────────────
# YAML patching
# ─────────────────────────────────────────────────────────────────────────────

def patch_yaml_threshold(text, key, new_value, comment):
    """Replace the first `key: <value>` line found after the first
    'thresholds:' marker in `text`. Returns the patched text."""
    marker_idx = text.find("thresholds:")
    if marker_idx == -1:
        raise ValueError("No 'thresholds:' section found in the YAML file")
    split_at = marker_idx + len("thresholds:")
    head, tail = text[:split_at], text[split_at:]

    line_re = re.compile(rf'^(?P<indent>[ \t]*){re.escape(key)}:\s*[^\n#]*(?:#.*)?$', re.MULTILINE)
    m = line_re.search(tail)
    if not m:
        raise ValueError(f"No '{key}:' line found after 'thresholds:'")

    new_line = f"{m.group('indent')}{key}: {int(new_value)}  # {comment}"
    tail = tail[:m.start()] + new_line + tail[m.end():]
    return head + tail


def write_recommendations(yaml_path, packets, bytes_, source_note):
    with open(yaml_path) as f:
        text = f.read()

    backup_path = yaml_path + ".bak"
    shutil.copy2(yaml_path, backup_path)

    text = patch_yaml_threshold(text, "packets", packets, source_note)
    text = patch_yaml_threshold(text, "bytes",   bytes_,  source_note)

    with open(yaml_path, "w") as f:
        f.write(text)

    return backup_path


# ─────────────────────────────────────────────────────────────────────────────
# Reusable calibration entry point (CLI + web app both call this)
# ─────────────────────────────────────────────────────────────────────────────

def run_calibration(config_path="config/default.yaml", gym_type=None, num_hosts=None,
                    num_iots=None, rounds=6, duration=None, normal_types=None,
                    normal_bw_mbps=None, attack_types=None, n_bins=None,
                    safety_factor=3.0, bridge="s1", link_bw_mbps=None, log=print):
    """
    Run one full calibration pass (build topology, sample normal + attack
    traffic, compute recommended thresholds) and return a JSON-serializable
    result dict. Raises on setup failure (bad gym_type, missing YAML,
    Mininet/monitoring failure) — callers should catch and report.

    Any parameter left None falls back to the corresponding value in
    config_path's env_params/net_params, mirroring the CLI's --option
    defaults.
    """
    config, _ = read_config_file(config_path)
    gym_type  = gym_type or config.env_params.gym_type
    folder    = gym_type_to_scenario_folder(gym_type)
    if folder is None:
        raise ValueError(f"Cannot map gym_type '{gym_type}' to a scenario folder.")
    yaml_path = os.path.join(SCENARIOS_BASE_DIR, folder, SCENARIO_PARAM_FILE)
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Scenario YAML not found: {yaml_path}")

    net_params     = config.env_params.net_params
    num_hosts      = num_hosts if num_hosts is not None else net_params.num_hosts
    num_iots       = num_iots  if num_iots  is not None else getattr(net_params, "num_iots", 0)
    duration       = duration if duration is not None else getattr(config.env_params, "wait_after_read", 1.0)
    n_bins         = n_bins   if n_bins   is not None else getattr(config.env_params, "n_bins", 4)
    link_bw_mbps   = link_bw_mbps if link_bw_mbps is not None else getattr(net_params, "link_bw_mbps", None)
    normal_bw_mbps = normal_bw_mbps if normal_bw_mbps is not None else getattr(net_params, "normal_traffic_bw_mbps", 1)
    normal_types   = normal_types or [t for t in net_params.traffic_types if t != "none"]
    attack_types   = attack_types or ["udp_flood"]

    host_names = [f"h{i}" for i in range(1, num_hosts + 1)] + [f"iot{i}" for i in range(1, num_iots + 1)]

    bw_label = f"{link_bw_mbps} Mbit/s" if link_bw_mbps else "unshaped (legacy, CPU-bound)"
    log(f"{BOLD}Calibrating thresholds{RESET}  gym_type={gym_type}  scenario_yaml={yaml_path}")
    log(f"hosts: {host_names}  duration/round: {duration}s  rounds: {rounds}  "
        f"normal_types: {normal_types} (bw={normal_bw_mbps}Mbit/s)  "
        f"attack_types: {attack_types}  links: {bw_label}\n")
    if not link_bw_mbps:
        log(f"{YELLOW}No link_bw_mbps set — recommended thresholds below are only valid on "
            f"THIS machine's CPU. Set net_params.link_bw_mbps in {config_path} (or pass "
            f"--link-bw) to make them portable across servers.{RESET}\n")

    net = build_network(host_names, bridge=bridge, link_bw_mbps=link_bw_mbps)
    log(f"{GREEN}Mininet started.{RESET}")
    if not initialize_monitoring(net, bridge_name=bridge):
        net.stop()
        raise RuntimeError("initialize_monitoring failed.")
    log(f"{GREEN}Monitoring ready.{RESET}\n")

    host_objs   = {h.name: h for h in net.hosts}
    prev_totals: dict = {}
    read_host_states(net, prev_totals)   # discard first read

    try:
        with ThreadPoolExecutor(max_workers=len(host_names) + 2, thread_name_prefix="calib") as executor:
            log(f"{BOLD}Phase 1/2 — normal traffic{RESET}")
            normal_pkts, normal_byts, prev_totals = run_normal_phase(
                net, host_names, host_objs, normal_types, rounds, duration,
                prev_totals, executor, normal_bw_mbps=normal_bw_mbps, log=log,
            )

            log(f"\n{BOLD}Phase 2/2 — attack traffic{RESET}")
            attack_pkts, attack_byts, prev_totals = run_attack_phase(
                net, host_names, host_objs, attack_types, rounds,
                duration, prev_totals, executor,
                normal_bw_mbps=normal_bw_mbps, log=log,
            )
    except KeyboardInterrupt:
        log(f"\n{DIM}Interrupted.{RESET}")
        normal_pkts = normal_byts = attack_pkts = attack_byts = []
    finally:
        kill_all_traffic(net)
        net.stop()
        log(f"{DIM}Mininet stopped.{RESET}")

    pkt_stats  = summarize("Packets", normal_pkts, attack_pkts, safety_factor, log=log)
    byte_stats = summarize("Bytes",   normal_byts, attack_byts, safety_factor, log=log)

    log(f"\n{BOLD}Bin sanity check (current n_bins={n_bins}){RESET}")
    print_bin_check("Packets", pkt_stats["recommended"], n_bins,
                    pkt_stats["normal_p95"], pkt_stats["attack_p50"], pkt_stats["attack_max"], log=log)
    print_bin_check("Bytes",   byte_stats["recommended"], n_bins,
                    byte_stats["normal_p95"], byte_stats["attack_p50"], byte_stats["attack_max"], log=log)

    log(f"\n{BOLD}Recommended thresholds for {yaml_path}:{RESET}")
    log(f"  packets: {pkt_stats['recommended']:,.0f}")
    log(f"  bytes:   {byte_stats['recommended']:,.0f}")
    log(f"  {DIM}(var_packets / var_bytes are not calibrated by this script — left unchanged){RESET}")

    return {
        "gym_type": gym_type,
        "yaml_path": yaml_path,
        "host_names": host_names,
        "duration": duration,
        "n_bins": n_bins,
        "link_bw_mbps": link_bw_mbps,
        "normal_bw_mbps": normal_bw_mbps,
        "packets": pkt_stats,
        "bytes": byte_stats,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Calibrate attack-detection packet/byte thresholds for this server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--config", default="config/default.yaml", help="base config (default: config/default.yaml)")
    ap.add_argument("--gym-type", default=None, help="override env_params.gym_type to pick the scenario YAML")
    ap.add_argument("--num-hosts", type=int, default=None, help="override net_params.num_hosts")
    ap.add_argument("--num-iots", type=int, default=None, help="override net_params.num_iots")
    ap.add_argument("--rounds", type=int, default=6, help="sampling rounds per phase (default 6)")
    ap.add_argument("--duration", type=float, default=None,
                    help="traffic burst / wait window in seconds (default: env_params.wait_after_read)")
    ap.add_argument("--normal-types", default=None,
                    help="comma list, default: net_params.traffic_types minus 'none'")
    ap.add_argument("--normal-bw", type=float, default=None,
                    help="override net_params.normal_traffic_bw_mbps (iperf -b target for "
                         "normal udp/tcp traffic, Mbit/s, default 1)")
    ap.add_argument("--attack-types", default="udp_flood",
                    help="comma list from {udp_flood,syn_flood,icmp_flood} (default: udp_flood)")
    ap.add_argument("--n-bins", type=int, default=None, help="override env_params.n_bins for the bin sanity check")
    ap.add_argument("--safety-factor", type=float, default=3.0,
                    help="minimum attack-p10 / normal-p95 ratio considered a clean separation (default 3.0)")
    ap.add_argument("--bridge", default="s1")
    ap.add_argument("--link-bw", type=float, default=None, metavar="MBPS",
                    help="override net_params.link_bw_mbps (tc/HTB cap on every host link, "
                         "Mbit/s) — see config/default.yaml for why this matters")
    ap.add_argument("--write", action="store_true",
                    help="patch packets/bytes thresholds into the scenario YAML (.bak backup kept)")
    args = ap.parse_args()

    try:
        result = run_calibration(
            config_path=args.config,
            gym_type=args.gym_type,
            num_hosts=args.num_hosts,
            num_iots=args.num_iots,
            rounds=args.rounds,
            duration=args.duration,
            normal_types=(args.normal_types.split(",") if args.normal_types else None),
            normal_bw_mbps=args.normal_bw,
            attack_types=args.attack_types.split(","),
            n_bins=args.n_bins,
            safety_factor=args.safety_factor,
            bridge=args.bridge,
            link_bw_mbps=args.link_bw,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        sys.exit(f"{RED}{exc}{RESET}")

    if args.write:
        timestr  = time.strftime("%Y-%m-%d")
        link_bw  = result["link_bw_mbps"]
        bw_note  = f"link_bw={link_bw}Mbit/s" if link_bw else "link unshaped"
        try:
            backup = write_recommendations(
                result["yaml_path"],
                packets=result["packets"]["recommended"],
                bytes_=result["bytes"]["recommended"],
                source_note=f"calibrated {timestr} via calibrate_thresholds.py, {bw_note}",
            )
            print(f"\n{GREEN}Wrote thresholds to {result['yaml_path']} (backup: {backup}){RESET}")
        except ValueError as exc:
            print(f"\n{RED}Could not patch YAML: {exc}{RESET}")
    else:
        print(f"\n{DIM}Re-run with --write to patch {result['yaml_path']} in place "
              f"(a .bak backup is kept).{RESET}")


if __name__ == "__main__":
    main()
