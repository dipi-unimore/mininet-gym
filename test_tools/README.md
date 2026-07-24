# Test tools

Diagnostic and visualization scripts used to debug the traffic-reading /
detection pipeline outside of a full training run. These are **not** unit
tests (nothing here is picked up by pytest) — they are standalone scripts you
run manually against a live Mininet instance or a saved `statuses.json` from
a past experiment.

None of these tools calibrates thresholds automatically. They all *consume*
the thresholds already set in `config/default.yaml` /
`reinforcement_learning/scenarios/<scenario>/scenario_env_param.yaml`
(`attacks.thresholds.packets|var_packets|bytes|var_bytes`) — see
"Thresholds & server speed" below.

## Scripts

### `traffic_test.py` — end-to-end pipeline replay (no controller)

Builds a throwaway single-switch Mininet topology (standalone OVS, no
controller needed), replays the host tasks recorded in a `statuses.json`
step by step, generates the matching real traffic (`hping3`, `ping`,
`iperf`), reads it back through the framework's real `get_data_flow()` /
`initialize_monitoring()`, and prints expected vs. observed packets/bytes
plus the resulting normalized + discretized bin for each host.

Use this to check whether a given experiment's thresholds/bins produce
sensible discretization on *this* machine.

```bash
sudo .venv/bin/python3 test_tools/traffic_test.py \
    --statuses _training/attacks_ho/20260523-075157_1_5_5/statuses.json \
    --step 5 --step-end 10 \
    --scenario attack_ho \
    --n-bins 4
```

Key flags:
- `--scenario {attack_ho,marl}` — which env's normalization/discretization rules to mirror.
- `--thr-pkts / --thr-var-pkts / --thr-bytes / --thr-var-bytes` — override thresholds
  (default: read from the scenario's `scenario_env_param.yaml`).
- `--pause` — step through interactively (Enter = next step, Ctrl-C = stop).

The printed `✓`/`✗` column flags steps where observed packets deviate from
the expected value by more than 30% (or 5 packets, whichever is larger) —
a quick signal that traffic generation or reading is off on this server.

### `net_monitor.py` — live OVS monitor + statuses.json replay

Two modes in one script:

**Live mode** (default) — polls `ovs-ofctl dump-ports` every `--interval`
seconds and prints per-host TX/RX pps, bytes/s, and the Q-table bin each
value falls into, using whatever Mininet topology is *already running*.

```bash
sudo .venv/bin/python3 test_tools/net_monitor.py --bridge s1 --interval 1
```

**Replay mode** (`--replay`) — like `traffic_test.py` but works at the raw
`ovs-ofctl dump-ports` level (via `ip netns exec`) instead of importing the
framework's monitoring code. Add `--launch` to have it spin up its own
Mininet topology from the host names found in the statuses file (fully
standalone, no separate `sudo mn` needed).

```bash
sudo .venv/bin/python3 test_tools/net_monitor.py \
    --replay _training/attacks_ho/20260523-075157_1_5_5/statuses.json \
    --launch --pause
```

Useful flags:
- `--thr-pkts`, `--thr-bytes` — thresholds used only to color-code bins (does not affect the framework).
- `--per-host` — use per-host thresholds (MARL, no `×n_hosts` multiplier) instead of the global attack_ho multiplier.
- `--n-bins` — bins for the printed `[TXp,TXb|RXp,RXb]` column.
- `--flows` / `--dpctl` — dump OpenFlow priority=2 rules / kernel datapath flows once at startup (useful when counters look frozen).
- `--debug` — print the raw port→host mapping (useful when host columns show as "Unknown" or stay at zero).

The "ANOMALY" warnings this script prints (TX without RX, or observed ≪
expected) are the fastest way to notice a monitoring-flow / port-mapping
problem before blaming the RL agent.

### `qtable_coverage.py` — discretized-bin occupancy plot

Library (no `__main__`) — import and call from a notebook/script/shell. Loads
a `statuses.json`, runs every host-step through the real
`discretize_attack_detect_ho_state()` (or a simplified per-feature quantile
version), and plots one bar-chart subplot per ground-truth action
(`normal` / `attack_in` / `attack_out`) showing how many samples land in
each bin, per feature.

```python
from test_tools.qtable_coverage import plot_qtable_coverage_from_statuses
plot_qtable_coverage_from_statuses(
    statuses_path="_training/attacks_ho/20260523-075157_1_5_5/statuses.json",
    output_dir="_training/_debug",
)
```

**This is the tool to reach for when you suspect bin saturation** (see
below) — if the `attack_in`/`attack_out` bars are all piled on the last bin
while `normal` also has mass there, your thresholds are too low for the
traffic this server actually generates and the agent has no signal left to
separate the two classes.

### `plot_host_states_timeline.py` — host status Gantt chart + traffic curves

CLI wrapper around `helpers.plot_host_states_evolution` /
`plot_host_traffic_evolution`. Auto-detects the latest
`_training/attacks_ho/*` experiment if you don't pass one.

```bash
python3 test_tools/plot_host_states_timeline.py \
    --experiment_path _training/attacks_ho/20260523-075157_1_5_5/ \
    --output_dir _training/_debug
```

Produces:
- `host_states_evolution.png` — per-host status timeline (normal / attacking / under_attack / blocked), good for spotting collapsed/frozen traffic episodes.
- `host_traffic_evolution_packets.png` / `_bytes.png` — RX/TX curves per host over the whole experiment.

Flags: `--no-states`, `--no-traffic`, `--metric {packets,bytes}`,
`--statuses_file <path>` (bypass experiment auto-detect).

### `test_chart_predictions.py` — agent test-result chart exploration

Compares alternative ways of visualizing prediction-vs-ground-truth success
rate from a `TEST*/test.json` (ground_truth + predicted arrays), including a
synthetic-data mode for trying chart layouts without a real experiment.

```bash
python3 test_tools/test_chart_predictions.py \
    --experiment_path _training/attacks_ho/20260523-075157_1_5_5/ \
    --output_dir _training/_debug
# or, for a quick layout sketch with fake data:
python3 test_tools/test_chart_predictions.py --generate_test --output_dir _training/_debug
```

### `helpers.py` — shared library, not a script

Functions used by other tools and by ad-hoc debugging sessions:
`plot_host_states_evolution`, `plot_host_traffic_evolution`,
`test_controller_delay`, `test_controller_traffic_audit`,
`test_create_traffic[_no_synchronize]`, `test_deep_learning`,
`test_results_from_saved_data`, `test_generate_attacks`,
`test_generate_classification_traffic`. Import what you need rather than
running it directly.

## Running with sudo + the venv

`traffic_test.py` and `net_monitor.py` need real network namespaces
(Mininet/OVS), so they must run as root. They both auto-re-exec themselves
with `.venv/bin/python3` if `numpy`/`mininet` aren't importable under
`sudo python3`, but it's more reliable to just call the venv interpreter
directly:

```bash
sudo .venv/bin/python3 test_tools/net_monitor.py ...
```

If a previous run crashed and left namespaces/OVS bridges behind, `mn -c`
(or the `cleanup()` call at the top of `traffic_test.build_network`) clears
them before the next run.

## Thresholds & server speed

Most of the scripts in this folder only *read* the static threshold values
from `reinforcement_learning/scenarios/<scenario>/scenario_env_param.yaml`
(`attacks.thresholds: {packets, var_packets, bytes, var_bytes}`), with a
CLI override for manual experiments (`traffic_test.py --thr-pkts ...`,
`net_monitor.py --thr-pkts ...`). `calibrate_thresholds.py` (below) is the
exception — it measures this machine and recommends values for that file.

This matters because the actual pps/Bps a host can push through Mininet
(veth pairs + the OVS software switch, both running on the host CPU) is a
function of **this machine's CPU throughput**, not of the traffic generator
command. `hping3 --flood`, `iperf`, etc. all just "go as fast as the CPU
lets them" for the fixed windows the framework uses
(`time.sleep(1.0)` in `_apply_scenario_step` /
`step()`, see `reinforcement_learning/scenarios/attack_detect_host_observable/network_env_attack_detect_per_host_observable.py:364` and `:540`).
Move to a faster or slower server and the packets/bytes counted in that same
1-second window shift up or down — but the thresholds in the YAML don't
move with it.

Two failure modes follow directly from this:
- **Thresholds too high for a slow server**: normal and attack traffic both
  stay below `threshold`, everything piles into low bins, attacks look like
  normal traffic.
- **Thresholds too low for a fast server** (what you're describing): normal
  traffic already saturates near/above `threshold`, so both normal and
  attack traffic get clipped into the same top bin — this is exactly what
  `qtable_coverage.py` will show as mass piling on the last bin for every
  action. `docs/analysis_attacks_ho_single_agent.md` documents the same
  symptom independently (§"Aumentare n_bins selettivamente": *"Con
  n_bins=4 il bin 3 copre tutto il range [threshold, +∞). Un attaccante a
  100k pkt/s e uno a 500k pkt/s hanno lo stesso bin TX=3"*).

### `calibrate_thresholds.py` — measure this server and recommend thresholds

Builds the same kind of standalone Mininet topology as `traffic_test.py`,
sized from `config/default.yaml`'s `net_params` (or `--num-hosts`/`--num-iots`),
then:

1. **Normal phase** — for `--rounds` rounds, every host exchanges real
   traffic (ping/udp/tcp, rotating) with a neighbour using the framework's
   own `_generate_normal_traffic_once`, with the exact fire-and-forget +
   fixed-wait pattern `_apply_scenario_step` uses in production
   (`env_params.wait_after_read`, default 1s). Samples every host's RX/TX
   packets & bytes via the real `get_data_flow()`.
2. **Attack phase** — for `--rounds` rounds, one attacker floods one victim
   (attack type rotating through `--attack-types`) via the framework's own
   `launch_attack_smart()`, rest of the hosts idle. Samples the attacker's
   TX and the victim's RX packets & bytes.
3. Prints percentiles for both pools (`normal p95`, `attack p10/p50/max`)
   and a **recommended threshold**: the log-midpoint between "typical
   normal ceiling" (p95) and "weakest observed attack" (p10). Warns loudly
   if `attack p10 / normal p95` is below `--safety-factor` (default 3x) —
   that's the direct, on-this-machine version of the top-bin-saturation
   symptom described above: if the ratio is too small, no threshold can
   separate the two classes here.
4. A "bin check" line shows which discretized bin (`get_log_bin_index`,
   same formula as `discretize_attack_detect_ho_state`) normal-p95 and
   attack-p50/max would land in at the recommended threshold, for the
   scenario's current `n_bins`.

```bash
sudo .venv/bin/python3 test_tools/calibrate_thresholds.py \
    --gym-type marl_pz --rounds 8 --attack-types udp_flood,syn_flood
```

Add `--write` to patch `packets:`/`bytes:` under `thresholds:` directly into
the target scenario's `scenario_env_param.yaml` (a `.bak` backup of the
original is kept next to it; `var_packets`/`var_bytes` are left untouched —
this script only calibrates the counter thresholds, not the
percentage-variation ones). Without `--write` it only prints the
recommendation so you can copy it in by hand.

Re-run this any time you move to a different server, change the topology
size, or change `n_bins` — the recommended values are specific to this
machine's measured throughput, not portable to another one.

> Related tool outside this folder: `utility/traffic_diagnostic.py` runs the
> **real** `NetworkEnvAttackDetectPerHostObservable` through a generated
> scenario (not a controlled round-robin) and prints a similar threshold
> suggestion at the end, based on observed attacker/victim/normal traffic
> across whatever mix of steps the scenario generator produced. Use it to
> sanity-check calibrate_thresholds.py's recommendation against a realistic
> episode mix; use calibrate_thresholds.py for a faster, cleaner, isolated
> measurement of this server's raw normal/attack ceiling. Neither script
> accounted for link bandwidth shaping (below) before this session.

## Link bandwidth shaping — making Mininet behave the same across servers

Calibrating thresholds (above) fixes symptoms on *this* server. The
underlying cause is that Mininet host links are, by default, **unshaped**:
`net.addLink(host, switch)` has no `bw=` — every host's veth pair pushes
packets exactly as fast as the CPU running Mininet/OVS allows. That has two
very different consequences depending on the traffic:

- **Normal traffic is already mostly portable.** `generate_normal_traffic()`
  uses `ping -c N` (packet-count bounded) and plain `iperf -u`/`iperf` with
  no `-b` flag for the client, which defaults to iperf's own ~1 Mbit/s UDP
  target — deterministic and CPU-independent on any machine that can
  saturate 1 Mbit/s (trivial).
- **Attack traffic is not.** `hping3 --flood` (`utility/network_attacks.py`)
  has no rate cap at all — it sends as fast as the CPU/OVS softswitch allow.
  This is the actual variable behind "cambiando server cambia la velocità di
  lettura del traffico": the same attack produces a different pkt/s ceiling
  on every machine, so a `packets`/`bytes` threshold tuned on one server is
  wrong on another (too low → attacks saturate into the same bin as normal
  traffic, exactly the symptom in `docs/analysis_attacks_ho_single_agent.md`).

**Fix**: `net_params.link_bw_mbps` in `config/default.yaml` (default `null`
= legacy unshaped behaviour). When set, every host<->switch link is created
with Mininet's `TCLink(bw=link_bw_mbps)`, which installs a `tc`/HTB qdisc
capping throughput at that rate regardless of CPU speed. As long as the
machine's CPU can saturate the configured rate (true for any reasonably
modern server at typical values like 50–100 Mbit/s), both normal and attack
traffic top out at the *same* tc-enforced ceiling on every server — turning
the server-dependent variable into a config-dependent constant.

This is wired through every place that builds a Mininet topology:
- `utility/network_configurator.py::create_network` — the real training/testing path.
- `utility/traffic_diagnostic.py::build_standalone_net` — reads the same `net_params.link_bw_mbps`.
- `test_tools/traffic_test.py::build_network` — `--link-bw MBPS` flag (default: config value).
- `test_tools/net_monitor.py::start_mininet` — `--link-bw MBPS` flag, only used with `--launch`.
- `test_tools/calibrate_thresholds.py` — `--link-bw MBPS` flag; the value used is now recorded in the `--write` comment for traceability.

**Why the default is `null` (off)**: enabling it changes the pkt/s and B/s
your experiments actually see, so it would silently invalidate comparisons
against thresholds/results collected before this change. Turning it on is a
one-time decision, not a silent behavior change:

1. Pick a rate (start with `100`; lower it if even a slow target server
   can trivially saturate 100 Mbit/s and you want more headroom below the
   attack ceiling, raise it if 100 Mbit/s ever turns out too close to what
   normal traffic can already reach).
2. Set `net_params.link_bw_mbps: 100` in `config/default.yaml`.
3. Re-run `calibrate_thresholds.py --write` for every scenario you use —
   the recommended thresholds will now be the same on any server whose CPU
   can keep up with that rate.
4. Only revisit steps 1–3 if you change `link_bw_mbps`, `num_hosts`, or `n_bins`.
