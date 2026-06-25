# my_ho_statistics.py
"""
Plotting utilities specific to the per-host observable attack detection scenario
(ATTACKS_HO / ATTACKS_HO_FROM_DATASET).

Each episode_status entry produced by the PerHostScanWrapper has this structure:
    {
        "id":     [host_status_id],               # list with one int 0/1/2
        "status": [host_status_text],             # list with one string
        "action_choosen":  int,                   # 0/1/2
        "traffic_type":    int,                   # 0/1/2  (ground truth scalar)
        "received_packets":                       float,
        "received_packets_percentage_change":     float,
        "received_bytes":                         float,
        "received_bytes_percentage_change":       float,
        "transmitted_packets":                    float,
        "transmitted_packets_percentage_change":  float,
        "transmitted_bytes":                      float,
        "transmitted_bytes_percentage_change":    float,
    }

Each status entry in base_env.statuses (network tick) has this structure:
    {
        "id":     [s0, s1, ..., sN],              # list of int, one per host
        "status": [str, ...],                     # list of strings, one per host
        "packets":                  int,
        "bytes":                    int,
        "packetsPercentageChange":  float,
        "bytesPercentageChange":    float,
        "hostStatusesStructured":   dict,
    }

ground_truth / predicted lists contain int scalars 0/1/2:
    0 → NORMAL
    1 → UNDER_ATTACK
    2 → ATTACKING

Raw statuses may also contain:
    3 → INCOMING_BLOCKED_ATTACK
    4 → OUT_ATTACK_BLOCKED

When building confusion matrices / GT plots:
- Status 3 (incoming_blocked_attack) is mapped to class 0 (NORMAL)
- Status 4 (out_attack_blocked) is mapped to class 2 (ATTACKING)
The raw values 3 and 4 are still preserved in statuses.json.
"""

import traceback
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import metrics as sk_metrics
from colorama import Fore

from utility.my_log import error


# ------------------------------------------------------------------
# Confusion matrix rendering helper
# ------------------------------------------------------------------

from utility.my_statistics import _plot_cm_pct

# ------------------------------------------------------------------
# Color / legend helpers — 3-class host status palette
# ------------------------------------------------------------------

HO_STATUS_COLORS = {0: "cyan", 1: "orange", 2: "red"}


def map_ho_status_id_to_class(value):
    """
    Map raw host status ids to the 3 confusion-matrix classes.

    3 (incoming_blocked_attack) is preserved in raw saved statuses, but belongs to
    the NORMAL class for ground-truth / confusion-matrix purposes.
    
    4 (out_attack_blocked) is preserved in raw saved statuses, but belongs to
    the ATTACKING class for ground-truth / confusion-matrix purposes.
    """
    try:
        value = int(value)
    except (TypeError, ValueError):
        return 0

    if value == 3:
        return 0
    if value == 4:
        return 2
    if value in (0, 1, 2):
        return value
    return 0


def _get_colors_for_ho_types(items):
    """Map host status ids (0/1/2) to display colours."""
    return [HO_STATUS_COLORS.get(map_ho_status_id_to_class(v), "gray") for v in items]


def _get_legend_for_ho_types():
    return [
        mpatches.Patch(color="red",    label="2 - Attacking"),
        mpatches.Patch(color="orange", label="1 - Under Attack"),
        mpatches.Patch(color="cyan",   label="0 - Normal"),
    ]


def _get_colors_for_ho_predictions(actions, ground_truths):
    """Green when prediction matches ground truth, red otherwise."""
    return [
        "green" if int(a) == int(g) else "red"
        for a, g in zip(actions, ground_truths)
    ]


def _get_legend_for_ho_predictions():
    return [
        mpatches.Patch(color="green", label="Correct"),
        mpatches.Patch(color="red",   label="Wrong"),
    ]


def _safe_scalar(value):
    """Convert numpy scalars / single-element lists/arrays to a plain float."""
    if isinstance(value, (list, tuple)):
        value = value[0]
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _id_list_to_scalar(id_field):
    """
    The 'id' field in network tick statuses is a list [s0, s1, ..., sN].
    Return a single representative scalar:
      - 2 (attacking) if any host is attacking
      - 1 (under_attack) if any host is under attack
      - 0 (normal) otherwise
    """
    if isinstance(id_field, (int, float, np.integer, np.floating)):
        return map_ho_status_id_to_class(id_field)
    ids = [map_ho_status_id_to_class(v) for v in id_field]
    if 2 in ids:
        return 2
    if 1 in ids:
        return 1
    return 0


# ------------------------------------------------------------------
# Confusion matrix (training)
# ------------------------------------------------------------------

def plot_ho_agent_execution_confusion_matrix(indicators, dir_name,
                                              must_print=True, title=''):
    """
    Build and optionally save a 3-class confusion matrix from training indicators.

    Classes: 0=NORMAL, 1=UNDER_ATTACK, 2=ATTACKING.

    traffic_type (ground truth) and action_choosen (predicted) are int scalars
    stored by CustomCallback.manage_step_data for the ATTACKS_HO scenario.

    Args:
        indicators:  List of per-episode dicts produced by CustomCallback.
        dir_name:    Directory where the PNG is saved.
        must_print:  If True, save the plot to disk.
        title:       Optional title suffix.

    Returns:
        np.ndarray: The 3x3 confusion matrix, or None on error.
    """
    ground_truth = []
    predicted = []
    for item in indicators:
        ep_data = item['episode_statuses']
        if isinstance(ep_data, np.ndarray):
            ground_truth.extend(map_ho_status_id_to_class(int(v)) for v in ep_data[:, 0])
            predicted.extend(int(v) for v in ep_data[:, 1])
        else:
            for s in ep_data:
                if isinstance(s, (list, tuple)):
                    ground_truth.append(map_ho_status_id_to_class(int(s[0])))
                    predicted.append(int(s[1]))
                else:
                    ground_truth.append(map_ho_status_id_to_class(s['traffic_type']))
                    predicted.append(int(s['action_choosen']))

    if not ground_truth:
        error(Fore.RED + "plot_ho_agent_execution_confusion_matrix: "
              "no step data found in indicators.\n" + Fore.WHITE)
        return None

    try:
        cm = sk_metrics.confusion_matrix(
            ground_truth, predicted, labels=[0, 1, 2]
        )
        if must_print:
            _plot_cm_pct(
                cm,
                display_labels=["Normal", "Under Attack", "Attacking"],
                title=f"{title} Confusion Matrix" if title else "Confusion Matrix",
                filepath=f"{dir_name}/matrix.png",
            )
        return cm
    except Exception as e:
        error(Fore.RED + f"Error building confusion matrix!\n"
              f"{e}\n{traceback.format_exc()}\n" + Fore.WHITE)
        return None


# ------------------------------------------------------------------
# Episode statuses (training) — per-microstep traffic + GT + predictions
# ------------------------------------------------------------------

def plot_ho_agent_execution_statuses(indicators, dir_name, title=''):
    """
    Plot per-episode aggregate training metrics across all episodes.

    Each episode is summarised by a single data point (mean traffic,
    attack ratio, accuracy), which gives a faithful view of the learning
    curve without needing to store or render tens of thousands of micro-steps.

    Layout (2 columns × 2 rows):
      [0,0] Mean RX/TX packets per episode (log scale)
      [0,1] Mean RX/TX bytes per episode   (log scale)
      [1,0] Attack ratio per episode (% micro-steps with label > 0)
      [1,1] Prediction accuracy per episode (% correct micro-steps)

    Args:
        indicators: List of per-episode dicts produced by CustomCallback.
        dir_name:   Directory where the PNG is saved.
        title:      Optional title prefix.
    """
    episodes      = []
    mean_rx_pkt   = []
    mean_rx_bytes = []
    mean_tx_pkt   = []
    mean_tx_bytes = []
    attack_ratio  = []
    accuracy      = []

    for item in indicators:
        ep_data = item.get('episode_statuses')
        if ep_data is None:
            continue

        # numpy array (live training): cols traffic_type(0) action_choosen(1)
        #   rx_pkt(2) rx_pkt_pct(3) rx_bytes(4) rx_bytes_pct(5)
        #   tx_pkt(6) tx_pkt_pct(7) tx_bytes(8) tx_bytes_pct(9)
        if isinstance(ep_data, np.ndarray):
            if ep_data.ndim != 2 or len(ep_data) == 0:
                continue
            gt_arr   = np.vectorize(map_ho_status_id_to_class)(ep_data[:, 0].astype(int))
            pred_arr = ep_data[:, 1].astype(int)
            rx_pkt   = ep_data[:, 2]
            rx_byt   = ep_data[:, 4]
            tx_pkt   = ep_data[:, 6]
            tx_byt   = ep_data[:, 8]
        elif ep_data:
            gt_l, pred_l, rxp_l, rxb_l, txp_l, txb_l = [], [], [], [], [], []
            for s in ep_data:
                if isinstance(s, (list, tuple)):
                    gt_l.append(map_ho_status_id_to_class(int(s[0])))
                    pred_l.append(int(s[1]))
                    rxp_l.append(float(s[2]));  rxb_l.append(float(s[4]))
                    txp_l.append(float(s[6]));  txb_l.append(float(s[8]))
                else:
                    gt_l.append(map_ho_status_id_to_class(s['traffic_type']))
                    pred_l.append(int(s['action_choosen']))
                    rxp_l.append(_safe_scalar(s.get('received_packets', 0)))
                    rxb_l.append(_safe_scalar(s.get('received_bytes', 0)))
                    txp_l.append(_safe_scalar(s.get('transmitted_packets', 0)))
                    txb_l.append(_safe_scalar(s.get('transmitted_bytes', 0)))
            gt_arr   = np.array(gt_l, dtype=int)
            pred_arr = np.array(pred_l, dtype=int)
            rx_pkt   = np.array(rxp_l); rx_byt = np.array(rxb_l)
            tx_pkt   = np.array(txp_l); tx_byt = np.array(txb_l)
        else:
            continue

        episodes.append(item.get('episode', len(episodes) + 1))
        mean_rx_pkt.append(float(np.mean(rx_pkt)))
        mean_rx_bytes.append(float(np.mean(rx_byt)))
        mean_tx_pkt.append(float(np.mean(tx_pkt)))
        mean_tx_bytes.append(float(np.mean(tx_byt)))
        attack_ratio.append(float(np.mean(gt_arr > 0)) * 100)
        accuracy.append(float(np.mean(pred_arr == gt_arr)) * 100)

    if not episodes:
        error(Fore.RED + "plot_ho_agent_execution_statuses: "
              "no step data found in indicators.\n" + Fore.WHITE)
        return

    ep = np.array(episodes)
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))

    # ── [0,0] Mean packets per episode ────────────────────────────
    axs[0][0].set_yscale("log")
    axs[0][0].plot(ep, mean_tx_pkt,   label='Mean TX packets', color='purple',   linewidth=1.5)
    axs[0][0].plot(ep, mean_rx_pkt,   label='Mean RX packets', color='cyan',     linewidth=1.5)
    axs[0][0].set_title(f'{title} Mean TX/RX Packets per Episode')
    axs[0][0].set_xlabel('Episode')
    axs[0][0].set_ylabel('Mean packets (log scale)')
    axs[0][0].legend()
    axs[0][0].grid(True, alpha=0.3)

    # ── [0,1] Mean bytes per episode ──────────────────────────────
    axs[0][1].set_yscale("log")
    axs[0][1].plot(ep, mean_tx_bytes, label='Mean TX bytes',   color='royalblue', linewidth=1.5)
    axs[0][1].plot(ep, mean_rx_bytes, label='Mean RX bytes',   color='green',     linewidth=1.5)
    axs[0][1].set_title(f'{title} Mean TX/RX Bytes per Episode')
    axs[0][1].set_xlabel('Episode')
    axs[0][1].set_ylabel('Mean bytes (log scale)')
    axs[0][1].legend()
    axs[0][1].grid(True, alpha=0.3)

    # ── [1,0] Attack ratio per episode ────────────────────────────
    axs[1][0].plot(ep, attack_ratio, color='orange', linewidth=1.5, label='% attack micro-steps')
    axs[1][0].fill_between(ep, attack_ratio, alpha=0.15, color='orange')
    axs[1][0].set_ylim(0, 105)
    axs[1][0].set_title(f'{title} Attack Ratio per Episode')
    axs[1][0].set_xlabel('Episode')
    axs[1][0].set_ylabel('% micro-steps with attack label')
    axs[1][0].legend()
    axs[1][0].grid(True, alpha=0.3)

    # ── [1,1] Prediction accuracy per episode ─────────────────────
    axs[1][1].plot(ep, accuracy, color='green', linewidth=1.5, label='Accuracy %')
    axs[1][1].fill_between(ep, accuracy, alpha=0.15, color='green')
    axs[1][1].set_ylim(0, 105)
    axs[1][1].set_title(f'{title} Prediction Accuracy per Episode')
    axs[1][1].set_xlabel('Episode')
    axs[1][1].set_ylabel('Accuracy (%)')
    axs[1][1].legend()
    axs[1][1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{dir_name}/episode_statuses.png", dpi=100)
    plt.close()


# ------------------------------------------------------------------
# Environment execution statuses (network ticks from base_env.statuses)
# ------------------------------------------------------------------

def _make_legend_toggle(fig, ax, lines):
    """
    Wire up legend line picks to toggle the corresponding data line visibility.

    Works only with interactive matplotlib backends (e.g. TkAgg, Qt5Agg).
    With the Agg (PNG) backend the saved image always shows all lines.
    """
    leg = ax.legend()
    if leg is None:
        return
    lined = {}
    for leg_line, data_line in zip(leg.get_lines(), lines):
        leg_line.set_picker(5)
        lined[leg_line] = data_line

    def on_pick(event):
        leg_line = event.artist
        if leg_line not in lined:
            return
        data_line = lined[leg_line]
        visible = not data_line.get_visible()
        data_line.set_visible(visible)
        leg_line.set_alpha(1.0 if visible else 0.2)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", on_pick)


def plot_ho_enviroment_execution_statutes(statutes, dir_name, title=''):
    """
    Plot global network traffic and host status labels over all recorded
    network ticks (base_env.statuses).

    For ATTACKS_HO the 'id' field in each status is a list [s0, s1, ..., sN]
    (one entry per host). This function reduces it to a single representative
    scalar per tick using _id_list_to_scalar:
        2 if any host is attacking
        1 if any host is under attack
        0 otherwise

    TX/RX totals are aggregated from hostStatusesStructured when available.
    Legend lines are pick-enabled for interactive toggling (interactive backends only).

    Args:
        statutes: List of network tick dicts from base_env.statuses.
        dir_name: Directory where the PNG is saved.
        title:    Optional title prefix.
    """
    if not statutes:
        error(Fore.RED + "plot_ho_enviroment_execution_statutes: "
              "empty statutes list.\n" + Fore.WHITE)
        return

    episodes          = range(1, len(statutes) + 1)
    packets_total     = [item['packets']                for item in statutes]
    packets_variation = [item['packetsPercentageChange'] for item in statutes]
    bytes_total       = [item['bytes']                  for item in statutes]
    bytes_variation   = [item['bytesPercentageChange']  for item in statutes]

    # Aggregate per-host TX/RX from hostStatusesStructured when present
    tx_packets = []
    rx_packets = []
    tx_bytes   = []
    rx_bytes   = []
    has_txrx   = False
    for item in statutes:
        hs = item.get('hostStatusesStructured', {})
        if hs:
            has_txrx = True
            tx_packets.append(sum(h.get('transmittedPackets', 0) for h in hs.values()))
            rx_packets.append(sum(h.get('receivedPackets',    0) for h in hs.values()))
            tx_bytes.append(  sum(h.get('transmittedBytes',   0) for h in hs.values()))
            rx_bytes.append(  sum(h.get('receivedBytes',      0) for h in hs.values()))
        else:
            tx_packets.append(0); rx_packets.append(0)
            tx_bytes.append(0);   rx_bytes.append(0)

    # 'id' is a list per tick — reduce to one scalar per tick
    ids = [_id_list_to_scalar(item['id']) for item in statutes]

    x = min(80, 10 + 3 * int(len(statutes) / 200))
    y = min(40, 15 + int(len(statutes) / 200))
    fig, axs = plt.subplots(4, 1, figsize=(x, y))
    ep = list(episodes)

    # Precompute attack tick positions (status 1=under_attack, 2=attacking)
    attack_idx = [i for i, s in enumerate(ids) if s > 0]

    # ── [0] Packets ────────────────────────────────────────────────
    axs[0].set_yscale("log")
    l_total, = axs[0].plot(ep, packets_total, label='Total Packets', color='blue',   linewidth=1.5)
    pkt_lines = [l_total]
    if has_txrx:
        l_tx, = axs[0].plot(ep, tx_packets, label='TX Packets', color='red',    linewidth=1, linestyle='--')
        l_rx, = axs[0].plot(ep, rx_packets, label='RX Packets', color='#00AAFF', linewidth=1, linestyle='--')
        pkt_lines += [l_tx, l_rx]
    if attack_idx:
        atk_x = [ep[i] for i in attack_idx]
        atk_y = [packets_total[i] for i in attack_idx]
        l_atk_pkt = axs[0].scatter(atk_x, atk_y, color='red', marker='x',
                                    s=20, linewidths=1, zorder=5, label='Attack')
        pkt_lines.append(l_atk_pkt)
    axs[0].set_title(f'{title} Packets Count')
    axs[0].legend()
    _make_legend_toggle(fig, axs[0], pkt_lines)

    # ── [1] Bytes ──────────────────────────────────────────────────
    axs[1].set_yscale("log")
    l_btotal, = axs[1].plot(ep, bytes_total, label='Total Bytes', color='black',  linewidth=1.5)
    byte_lines = [l_btotal]
    if has_txrx:
        l_btx, = axs[1].plot(ep, tx_bytes, label='TX Bytes', color='darkred',  linewidth=1, linestyle='--')
        l_brx, = axs[1].plot(ep, rx_bytes, label='RX Bytes', color='steelblue', linewidth=1, linestyle='--')
        byte_lines += [l_btx, l_brx]
    if attack_idx:
        atk_yb = [bytes_total[i] for i in attack_idx]
        l_atk_byt = axs[1].scatter(atk_x, atk_yb, color='red', marker='x',
                                    s=20, linewidths=1, zorder=5, label='Attack')
        byte_lines.append(l_atk_byt)
    axs[1].set_title(f'{title} Bytes Count')
    axs[1].legend()
    _make_legend_toggle(fig, axs[1], byte_lines)

    # ── [2] % Variations ───────────────────────────────────────────
    axs[2].set_yscale("log")
    axs[2].plot(ep, packets_variation, label='% var packets', color='purple')
    axs[2].plot(ep, bytes_variation,   label='% var bytes',   color='cyan')
    axs[2].set_title(f'{title} % Variations')
    axs[2].legend()

    # ── [3] Host Status ────────────────────────────────────────────
    colors = _get_colors_for_ho_types(ids)
    axs[3].scatter(ep, ids, label='Host Status', c=colors, s=3)
    axs[3].set_yticks([0, 1, 2])
    axs[3].set_yticklabels(["Normal", "Under Attack", "Attacking"])
    axs[3].set_title(f'{title} Network Event Types')
    axs[3].set_xlabel('Network ticks')
    axs[3].set_ylabel('Worst host status')
    axs[3].legend(handles=_get_legend_for_ho_types(), loc='upper right')

    plt.tight_layout()
    plt.savefig(f"{dir_name}/statuses.png")
    plt.close()


# ------------------------------------------------------------------
# Test confusion matrix (3-class)
# ------------------------------------------------------------------

def plot_ho_test_confusion_matrix(dir_name, ground_truth, predicted, agent):
    """
    Save a 3-class confusion matrix for the evaluation phase.

    Args:
        dir_name:     Directory where the PNG is saved.
        ground_truth: List of int scalars 0/1/2.
        predicted:    List of int scalars 0/1/2.
        agent:        Agent name used in the filename.
    """
    try:
        ground_truth = [map_ho_status_id_to_class(v) for v in ground_truth]
        cm = sk_metrics.confusion_matrix(
            ground_truth, predicted, labels=[0, 1, 2]
        )
        _plot_cm_pct(
            cm,
            display_labels=["Normal", "Under Attack", "Attacking"],
            title=f"{agent} — Test Confusion Matrix",
            filepath=f"{dir_name}/{agent}_matrix.png",
        )
    except Exception as e:
        error(Fore.RED + f"Error plotting test confusion matrix for {agent}!\n"
              f"{e}\n{traceback.format_exc()}\n" + Fore.WHITE)


# ------------------------------------------------------------------
# Test results: ground truth vs predictions over evaluation micro-steps
# ------------------------------------------------------------------

def plot_ho_agent_test(test, dir_name, title=''):
    """
    Plot ground truth and predictions (int scalars 0/1/2) over evaluation
    micro-steps.

    Args:
        test:     Dict with keys 'ground_truth' (list[int]) and
                  'predicted' (dict {agent_name: list[int]}).
        dir_name: Directory where the PNG is saved.
        title:    Optional title prefix.
    """
    ground_truth = [map_ho_status_id_to_class(v) for v in test["ground_truth"]]
    n_points = len(ground_truth)

    # Limit points to avoid memory issues with large datasets
    max_points = 5000
    if n_points > max_points:
        step = n_points // max_points
        ground_truth = ground_truth[::step]
        step_size = step
    else:
        step_size = 1

    episodes = range(1, len(ground_truth) + 1)

    plt.figure(figsize=(14, 6), dpi=80)
    plt.plot(episodes, ground_truth, label='Ground Truth',
             color='black', linewidth=1.0)

    for agent_name, preds in test["predicted"].items():
        # Apply same step to predicted data
        preds_subset = preds[::step_size]
        plt.plot(episodes, preds_subset, label=agent_name, linestyle='--', linewidth=0.8)

    plt.yticks([0, 1, 2], ["Normal", "Under Attack", "Attacking"])
    plt.title(f"{title} Predictions vs Ground Truth")
    plt.xlabel("Evaluation micro-steps")
    plt.ylabel("Host Status")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{dir_name}/test_episodes.png", dpi=100)
    plt.close()


def plot_ho_agent_test_errors(test, dir_name, title=''):
    """
    Plot ground truth vs predictions and squared error per agent over
    evaluation micro-steps.

    Args:
        test:     Dict with keys 'ground_truth' (list[int]) and
                  'predicted' (dict {agent_name: list[int]}).
        dir_name: Directory where the PNG is saved.
        title:    Optional title prefix.
    """
    ground_truth = np.array(
        [map_ho_status_id_to_class(v) for v in test["ground_truth"]],
        dtype=float,
    )
    episodes     = np.arange(1, len(ground_truth) + 1)
    num_agents   = len(test["predicted"])

    fig, axs = plt.subplots(num_agents, 2,
                             figsize=(14, 4 * max(num_agents, 1)))
    if num_agents == 1:
        axs = np.array([axs])

    for idx, (agent_name, preds) in enumerate(test["predicted"].items()):
        pred_values = np.array(preds, dtype=float)
        errors      = (pred_values - ground_truth) ** 2

        axs[idx, 0].plot(episodes, ground_truth,
                         label='Ground Truth', color='black')
        axs[idx, 0].plot(episodes, pred_values,
                         label=f'{agent_name} Prediction', linestyle='--')
        axs[idx, 0].set_yticks([0, 1, 2])
        axs[idx, 0].set_yticklabels(["Normal", "Under Attack", "Attacking"])
        axs[idx, 0].set_title(f'{agent_name} — Predictions vs Ground Truth')
        axs[idx, 0].set_xlabel('Micro-step')
        axs[idx, 0].set_ylabel('Host Status')
        axs[idx, 0].legend()
        axs[idx, 0].grid(True)

        axs[idx, 1].plot(episodes, errors,
                         label='Squared Error', color='red')
        axs[idx, 1].set_title(f'{agent_name} — Squared Error')
        axs[idx, 1].set_xlabel('Micro-step')
        axs[idx, 1].set_ylabel('Error')
        axs[idx, 1].legend()
        axs[idx, 1].grid(True)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(f"{dir_name}/test_quadratic_errors.png")
    plt.close()


def plot_ho_attack_mitigation_stats(mitigation_history, dir_name, title=''):
    """
    Plot per-episode mitigation diagnostics collected by PerHostScanWrapper.

    Args:
        mitigation_history: List of dicts with keys:
            episode, under_attack_count,
            mitigated_under_attack_count, mitigated_under_attack_ratio.
        dir_name: Directory where the PNG is saved.
        title: Optional title prefix.
    """
    if not mitigation_history:
        return

    episodes = [int(item.get("episode", idx + 1)) for idx, item in enumerate(mitigation_history)]
    under_attack_counts = [int(item.get("under_attack_count", 0)) for item in mitigation_history]
    mitigated_counts = [int(item.get("mitigated_under_attack_count", 0)) for item in mitigation_history]
    mitigated_ratio = [float(item.get("mitigated_under_attack_ratio", 0.0)) for item in mitigation_history]

    fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axs[0].plot(episodes, under_attack_counts, marker='o', label='Under attack count', color='darkorange')
    axs[0].plot(episodes, mitigated_counts, marker='o', label='Mitigated count', color='seagreen')
    axs[0].set_ylabel('Count')
    axs[0].set_title(f"{title} Mitigation Counts" if title else "Mitigation Counts")
    axs[0].grid(True, alpha=0.4)
    axs[0].legend()

    axs[1].plot(episodes, mitigated_ratio, marker='o', color='royalblue', label='Mitigated ratio')
    axs[1].axhline(y=0.0, color='gray', linewidth=0.8)
    axs[1].axhline(y=1.0, color='gray', linewidth=0.8)
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Ratio [0..1]')
    axs[1].set_ylim(0, 1.05)
    axs[1].set_title(f"{title} Mitigation Ratio" if title else "Mitigation Ratio")
    axs[1].grid(True, alpha=0.4)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(f"{dir_name}/test_attack_mitigation.png")
    plt.close()


# ------------------------------------------------------------------
# Re-export metric plots from my_statistics (HO uses the same format)
# ------------------------------------------------------------------

from utility.my_statistics import (  # noqa: E402
    plot_metrics,
    plot_combined_performance_over_time,
    plot_comparison_bar_charts,
    plot_radar_chart,
    plot_metrics_kfold,
    plot_metrics_violin,
)


# ------------------------------------------------------------------
# Q-table coverage plot (tabular agents)
# ------------------------------------------------------------------

def plot_qtable_coverage(indicators, coverage_history, dir_name, agent_name=''):
    """
    Plot Q-table coverage over training for tabular agents (Q-Learning/SARSA).

    Two sub-plots:
      Top:    coverage % over global steps (sampled every N steps)
      Bottom: coverage % at end of each episode

    Args:
        indicators:       List of per-episode indicator dicts (from base_agent).
        coverage_history: List of {global_step, episode, step, coverage_pct}
                          dicts recorded during training.
        dir_name:         Directory where the PNG is saved.
        agent_name:       Agent name for the plot title.
    """
    if not coverage_history and not indicators:
        return

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))

    # ── Top: coverage over global steps ───────────────────────────
    if coverage_history:
        steps    = [r['global_step'] for r in coverage_history]
        coverage = [r['coverage_pct'] for r in coverage_history]
        axs[0].plot(steps, coverage, color='steelblue', linewidth=1.5)
        axs[0].set_xlabel('Global micro-step')
        axs[0].set_ylabel('Q-table coverage (%)')
        axs[0].set_title(f'{agent_name} — Q-table Coverage Over Steps')
        axs[0].set_ylim(0, max(coverage) * 1.1)
        axs[0].grid(True, alpha=0.4)
        # Mark episode boundaries
        ep_changes = [r['global_step'] for r in coverage_history
                      if r['step'] == 0 and r['episode'] > 0]
        for x in ep_changes:
            axs[0].axvline(x=x, color='gray', linestyle='--',
                           linewidth=0.5, alpha=0.5)

    # ── Bottom: coverage at end of each episode ────────────────────
    ep_covs = [(ind['episode'], ind.get('qtable_coverage_pct', 0))
               for ind in indicators
               if 'qtable_coverage_pct' in ind]
    if ep_covs:
        eps, covs = zip(*ep_covs)
        axs[1].plot(eps, covs, marker='o', color='darkorange',
                    linewidth=1.5, markersize=4)
        axs[1].set_xlabel('Episode')
        axs[1].set_ylabel('Q-table coverage (%)')
        axs[1].set_title(f'{agent_name} — Q-table Coverage per Episode')
        axs[1].set_ylim(0, max(covs) * 1.1)
        axs[1].grid(True, alpha=0.4)

        # Annotate final coverage
        axs[1].annotate(
            f'Final: {covs[-1]:.1f}%',
            xy=(eps[-1], covs[-1]),
            xytext=(-40, 10),
            textcoords='offset points',
            fontsize=9,
            arrowprops=dict(arrowstyle='->', color='black'),
        )

    plt.tight_layout()
    plt.savefig(f"{dir_name}/qtable_coverage.png")
    plt.close()


def plot_discrete_feature_bin_coverage(indicators, dir_name, agent_name='', n_bins=4):
    """Plot discretized feature/bin coverage with one subplot per action.

    Input data comes from ATTACKS_HO training indicators
    (`episode_statuses` produced by PerHostScanWrapper callback).
    """
    if not indicators:
        return

    features = [
        "received_packets",
        "received_packets_percentage_change",
        "received_bytes",
        "received_bytes_percentage_change",
        "transmitted_packets",
        "transmitted_packets_percentage_change",
        "transmitted_bytes",
        "transmitted_bytes_percentage_change",
    ]
    n_bins = max(2, int(n_bins))
    packet_byte_features = {
        "received_packets",
        "received_bytes",
        "transmitted_packets",
        "transmitted_bytes",
    }

    def _action_label(v):
        mapping = {0: "normal", 1: "attack_in", 2: "attack_out"}
        try:
            return mapping.get(int(v), str(v))
        except Exception:
            return str(v)

    # Column order in compact numpy arrays (must match CustomCallback._ho_step_buf layout)
    _FEAT_COL = {
        "received_packets":                      2,
        "received_packets_percentage_change":    3,
        "received_bytes":                        4,
        "received_bytes_percentage_change":      5,
        "transmitted_packets":                   6,
        "transmitted_packets_percentage_change": 7,
        "transmitted_bytes":                     8,
        "transmitted_bytes_percentage_change":   9,
    }

    rows = []
    for episode in indicators:
        ep_data = episode.get("episode_statuses", [])
        if isinstance(ep_data, np.ndarray):
            for row in ep_data:
                action = _action_label(int(row[0]))
                for feature in features:
                    col = _FEAT_COL.get(feature)
                    if col is None:
                        continue
                    rows.append({"feature": feature, "value": float(row[col]), "action": action})
        else:
            for s in ep_data:
                if isinstance(s, (list, tuple)):
                    action = _action_label(int(s[0]))
                    for feature in features:
                        col = _FEAT_COL.get(feature)
                        if col is None:
                            continue
                        rows.append({"feature": feature, "value": float(s[col]), "action": action})
                else:
                    action = _action_label(s.get("traffic_type", "unknown"))
                    for feature in features:
                        if feature not in s:
                            continue
                        try:
                            value = float(s.get(feature, 0))
                        except Exception:
                            continue
                        rows.append({"feature": feature, "value": value, "action": action})

    if not rows:
        return

    import pandas as pd
    df = pd.DataFrame(rows)
    df["bin"] = -1

    for feature in features:
        feature_mask = df["feature"] == feature
        feature_values = df.loc[feature_mask, "value"]
        if feature_values.empty:
            continue

        if feature in packet_byte_features:
            zero_mask = feature_values <= 0
            df.loc[feature_mask & zero_mask, "bin"] = 0
            positive_values = feature_values[~zero_mask]
            if positive_values.empty:
                continue

            positive_bins = max(1, n_bins - 1)
            transformed = np.log1p(positive_values)
            if transformed.nunique() > 1:
                cap = transformed.quantile(0.995)
                transformed = transformed.clip(upper=cap)

            try:
                q_labels = pd.qcut(
                    transformed,
                    q=positive_bins,
                    labels=False,
                    duplicates="drop",
                )
                df.loc[q_labels.index, "bin"] = q_labels.astype(int) + 1
            except Exception:
                tmin = float(transformed.min())
                tmax = float(transformed.max())
                if tmax == tmin:
                    df.loc[transformed.index, "bin"] = 1
                else:
                    scaled = (transformed - tmin) / (tmax - tmin)
                    bins = np.floor(scaled * positive_bins).astype(int)
                    df.loc[transformed.index, "bin"] = bins.clip(0, positive_bins - 1) + 1
            continue

        if feature_values.nunique() <= 1:
            df.loc[feature_mask, "bin"] = 0
            continue

        vmin = float(feature_values.min())
        vmax = float(feature_values.max())
        if vmax == vmin:
            df.loc[feature_mask, "bin"] = 0
        else:
            normalized = (feature_values - vmin) / (vmax - vmin)
            bins = np.floor(normalized * n_bins).astype(int)
            df.loc[feature_mask, "bin"] = bins.clip(0, n_bins - 1)

    counts = (
        df.groupby(["action", "feature", "bin"], sort=False)
        .size()
        .reset_index(name="count")
        .sort_values(["action", "feature", "bin"], ascending=[True, True, True])
    )
    total_records = int(len(df))

    preferred_actions = ["normal", "attack_in", "attack_out"]
    available_actions = [str(action) for action in pd.unique(df["action"]).tolist()]
    action_order = [action for action in preferred_actions if action in available_actions]
    for action in available_actions:
        if action not in action_order:
            action_order.append(action)

    block_gap = 1
    block_width = n_bins + block_gap
    x_rows = []
    for action in action_order:
        for feature_idx, feature in enumerate(features):
            for bin_idx in range(n_bins):
                x_rows.append({
                    "action": action,
                    "feature": feature,
                    "bin": bin_idx,
                    "x": feature_idx * block_width + bin_idx,
                })
    full_grid = pd.DataFrame(x_rows)
    counts = full_grid.merge(counts, on=["action", "feature", "bin"], how="left")
    counts["count"] = counts["count"].fillna(0).astype(int)

    n_actions = max(1, len(action_order))
    fig, axes = plt.subplots(
        n_actions,
        1,
        figsize=(24, max(6, 4.8 * n_actions)),
        constrained_layout=True,
        squeeze=False,
    )
    cmap = plt.get_cmap("tab10", n_bins)
    bin_colors = [cmap(i) for i in range(n_bins)]

    block_centers = [feature_idx * block_width + (n_bins - 1) / 2 for feature_idx in range(len(features))]
    feature_labels = features

    for axis, action in zip(axes.flatten(), action_order):
        action_counts = counts[counts["action"] == action].sort_values("x")
        max_count = max(1, int(action_counts["count"].max()))
        zero_label_y = max_count * 0.03
        positive_label_pad = max_count * 0.01

        non_zero = action_counts[action_counts["count"] > 0]
        bar_colors = [bin_colors[int(bin_idx)] for bin_idx in non_zero["bin"].tolist()]
        axis.bar(non_zero["x"], non_zero["count"], color=bar_colors, width=0.85)

        for _, row in action_counts.iterrows():
            count_value = int(row["count"])
            is_zero = count_value == 0
            axis.text(
                row["x"],
                zero_label_y if is_zero else count_value + positive_label_pad,
                str(count_value),
                color="red" if is_zero else "black",
                ha="center",
                va="bottom",
                fontsize=6,
            )

        for feature_idx in range(1, len(features)):
            separator_x = feature_idx * block_width - 0.5
            axis.axvline(separator_x, color="#BBBBBB", linestyle="--", linewidth=0.6)

        axis.set_title(f"{agent_name} - discrete feature/bin coverage - action: {action}")
        axis.set_ylabel("Occurrences")
        axis.set_xticks(block_centers)
        axis.set_xticklabels(feature_labels, rotation=25, ha="right")
        axis.set_ylim(0, max_count * 1.08)
        axis.grid(axis="y", alpha=0.25)

    axes[-1][0].set_xlabel(f"Feature blocks with {n_bins} discretized bins each")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=bin_colors[bin_idx], label=f"bin {bin_idx}")
        for bin_idx in range(n_bins)
    ]
    axes[0][0].legend(
        handles=legend_handles,
        loc="upper right",
        ncol=min(n_bins, 8),
        fontsize=8,
        title="Bin colors",
    )

    plt.suptitle(
        f"{agent_name} discrete feature/bin coverage | total traffics considered: {total_records}"
    )
    plt.savefig(f"{dir_name}/discrete_feature_bin_coverage.png")
    plt.close()


# ------------------------------------------------------------------
# Policy exploration metrics plot (SB3 deep agents)
# ------------------------------------------------------------------

def plot_policy_exploration(indicators, exploration_history,
                             dir_name, agent_name=''):
    """
    Plot policy exploration metrics over training for SB3 agents.

    DQN: exploration_rate + Q-values std
    PPO/A2C: policy entropy

    Two sub-plots:
      Top:    exploration metric over global steps (sampled every N steps)
      Bottom: exploration metric at end of each episode

    Args:
        indicators:           List of per-episode indicator dicts.
        exploration_history:  List of metric records from CustomCallback.
        dir_name:             Directory where the PNG is saved.
        agent_name:           Agent name for the plot title.
    """
    if not exploration_history and not indicators:
        return

    # Detect metric type from first record
    first = exploration_history[0] if exploration_history else {}
    has_eps  = 'exploration_rate' in first
    has_qstd = 'q_values_std'     in first
    has_entr = 'policy_entropy'   in first

    n_plots = 1 + (1 if has_qstd else 0)
    fig, axs = plt.subplots(n_plots, 1, figsize=(12, 5 * n_plots))
    if n_plots == 1:
        axs = [axs]

    steps = [r['global_step'] for r in exploration_history]

    # ── Primary metric (exploration_rate or policy_entropy) ────────
    if has_eps:
        values = [r['exploration_rate'] for r in exploration_history]
        label  = 'Exploration rate (ε)'
        color  = 'steelblue'
    elif has_entr:
        values = [r['policy_entropy'] for r in exploration_history]
        label  = 'Policy entropy'
        color  = 'purple'
    else:
        values = []
        label  = ''
        color  = 'gray'

    if values:
        axs[0].plot(steps, values, color=color, linewidth=1.5, label=label)
        axs[0].set_xlabel('Global step')
        axs[0].set_ylabel(label)
        axs[0].set_title(f'{agent_name} — {label} Over Steps')
        axs[0].grid(True, alpha=0.4)
        axs[0].legend()
        axs[0].set_ylim(0, max(values) * 1.1)

    # ── Q-values std (DQN only) ────────────────────────────────────
    if has_qstd and n_plots > 1:
        q_std = [r['q_values_std'] for r in exploration_history]
        axs[1].plot(steps, q_std, color='darkorange', linewidth=1.5,
                    label='Q-values std')
        axs[1].set_xlabel('Global step')
        axs[1].set_ylabel('Q-values std')
        axs[1].set_title(f'{agent_name} — Q-values Std (policy differentiation)')
        axs[1].grid(True, alpha=0.4)
        axs[1].legend()
        axs[1].set_ylim(0, max(q_std) * 1.1)
    plt.ylim(0, max(max(values), max(q_std)) * 1.1 ) 
    plt.tight_layout()
    plt.savefig(f"{dir_name}/policy_exploration.png")
    plt.close()


# ------------------------------------------------------------------
# Alternative success-rate visualizations (Task 22)
# ------------------------------------------------------------------

def plot_ho_success_rate_alternatives(test, dir_name, title=''):
    """
    Generate multiple alternative visualizations for prediction success rate.
    
    Creates 4 different chart types to clearly show how many predictions succeed:
    1. Success/Fail scatter plot with cumulative accuracy line
    2. Stacked bar chart (Correct/Wrong) with window-based summary
    3. Success rate gauge with trend line
    4. Heatmap of success rate over time windows
    
    Args:
        test:     Dict with keys 'ground_truth' (list[int]) and
                  'predicted' (dict {agent_name: list[int]}).
        dir_name: Directory where the PNGs are saved.
        title:    Optional title prefix.
    """
    ground_truth = np.array(
        [map_ho_status_id_to_class(v) for v in test["ground_truth"]],
        dtype=int,
    )
    episodes = np.arange(1, len(ground_truth) + 1)
    num_agents = len(test["predicted"])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Chart 1: Scatter plot with cumulative accuracy
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    fig, axs = plt.subplots(num_agents, 1, figsize=(14, 3 * max(num_agents, 1)))
    if num_agents == 1:
        axs = [axs]

    for idx, (agent_name, preds) in enumerate(test["predicted"].items()):
        pred_values = np.array(preds, dtype=int)
        matches = (pred_values == ground_truth).astype(int)
        cumulative_accuracy = np.cumsum(matches) / np.arange(1, len(matches) + 1)

        colors = ["green" if m else "red" for m in matches]
        axs[idx].scatter(episodes, matches, c=colors, s=30, alpha=0.6, label="Prediction result")
        axs[idx].plot(episodes, cumulative_accuracy, color='blue', linewidth=2,
                     label=f'Cumulative accuracy', marker='o', markersize=3)

        # Add percentage annotations
        final_accuracy = cumulative_accuracy[-1] * 100
        axs[idx].text(0.98, 0.95, f'Final accuracy: {final_accuracy:.1f}%',
                     transform=axs[idx].transAxes, ha='right', va='top',
                     fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        axs[idx].set_ylabel('Match (1=Correct, 0=Wrong)', fontsize=10)
        axs[idx].set_title(f'{agent_name} — Prediction Success Timeline', fontsize=12)
        axs[idx].set_ylim(-0.15, 1.15)
        axs[idx].set_yticks([0, 1])
        axs[idx].set_yticklabels(['Wrong', 'Correct'])
        axs[idx].grid(True, alpha=0.3, axis='y')
        axs[idx].legend(loc='lower right')

    fig.suptitle(f'{title} — Success/Fail Scatter with Cumulative Accuracy' if title else
                 'Success/Fail Scatter with Cumulative Accuracy', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(f"{dir_name}/success_scatter.png", dpi=100)
    plt.close()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Chart 2: Stacked bar chart with window-based summary
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    window_size = max(10, len(ground_truth) // 10)
    n_windows = (len(ground_truth) + window_size - 1) // window_size

    fig, axs = plt.subplots(num_agents, 1, figsize=(14, 3 * max(num_agents, 1)))
    if num_agents == 1:
        axs = [axs]

    for idx, (agent_name, preds) in enumerate(test["predicted"].items()):
        pred_values = np.array(preds, dtype=int)
        matches = (pred_values == ground_truth).astype(int)

        correct_counts = []
        wrong_counts = []
        window_labels = []

        for w in range(n_windows):
            start_idx = w * window_size
            end_idx = min((w + 1) * window_size, len(matches))
            window_data = matches[start_idx:end_idx]
            correct_counts.append(np.sum(window_data))
            wrong_counts.append(len(window_data) - np.sum(window_data))
            window_labels.append(f'{start_idx+1}-{end_idx}')

        x = np.arange(len(window_labels))
        width = 0.6
        axs[idx].bar(x, correct_counts, width, label='Correct', color='green', alpha=0.7)
        axs[idx].bar(x, wrong_counts, width, bottom=correct_counts, label='Wrong', color='red', alpha=0.7)

        # Add percentage labels on bars
        for i, (correct, wrong) in enumerate(zip(correct_counts, wrong_counts)):
            total = correct + wrong
            pct = (correct / total * 100) if total > 0 else 0
            axs[idx].text(i, total + 0.3, f'{pct:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

        axs[idx].set_ylabel('Count', fontsize=10)
        axs[idx].set_title(f'{agent_name} — Prediction Success by Time Window (window={window_size})', fontsize=12)
        axs[idx].set_xticks(x)
        axs[idx].set_xticklabels(window_labels, rotation=45, ha='right')
        axs[idx].legend(loc='upper left')
        axs[idx].grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'{title} — Stacked Success/Fail Bars by Window' if title else
                 'Stacked Success/Fail Bars by Window', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(f"{dir_name}/success_stacked_bars.png", dpi=100)
    plt.close()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Chart 3: Success rate gauge + trend line
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    fig = plt.figure(figsize=(14, 5 * num_agents))

    for idx, (agent_name, preds) in enumerate(test["predicted"].items()):
        pred_values = np.array(preds, dtype=int)
        matches = (pred_values == ground_truth).astype(int)
        cumulative_accuracy = np.cumsum(matches) / np.arange(1, len(matches) + 1)

        ax = fig.add_subplot(num_agents, 2, idx * 2 + 1, projection='polar')
        
        # Gauge (simplified version without polar)
        ax_gauge = fig.add_subplot(num_agents, 2, idx * 2 + 1)
        final_accuracy_pct = cumulative_accuracy[-1] * 100
        
        # Draw gauge arc
        theta = np.linspace(0, np.pi, 100)
        r = 1
        ax_gauge.plot(r * np.cos(theta), r * np.sin(theta), 'k-', linewidth=2)
        
        # Color zones
        low_theta = np.linspace(0, np.pi/3, 50)
        ax_gauge.fill_between(r * np.cos(low_theta), 0, r * np.sin(low_theta), 
                             alpha=0.3, color='red', label='0-33% (Bad)')
        mid_theta = np.linspace(np.pi/3, 2*np.pi/3, 50)
        ax_gauge.fill_between(r * np.cos(mid_theta), 0, r * np.sin(mid_theta),
                             alpha=0.3, color='orange', label='33-67% (Medium)')
        high_theta = np.linspace(2*np.pi/3, np.pi, 50)
        ax_gauge.fill_between(r * np.cos(high_theta), 0, r * np.sin(high_theta),
                             alpha=0.3, color='green', label='67-100% (Good)')
        
        # Needle
        needle_angle = (final_accuracy_pct / 100) * np.pi
        ax_gauge.arrow(0, 0, 0.8 * np.cos(needle_angle), 0.8 * np.sin(needle_angle),
                      head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        # Center circle
        circle = plt.Circle((0, 0), 0.1, color='black')
        ax_gauge.add_patch(circle)
        ax_gauge.text(0, -0.3, f'{final_accuracy_pct:.1f}%', ha='center', fontsize=14, fontweight='bold')
        
        ax_gauge.set_xlim(-1.2, 1.2)
        ax_gauge.set_ylim(-0.5, 1.2)
        ax_gauge.set_aspect('equal')
        ax_gauge.axis('off')
        ax_gauge.set_title(f'{agent_name} — Final Accuracy Gauge', fontsize=11)
        ax_gauge.legend(loc='upper left', fontsize=8)
        
        # Trend line
        ax_trend = fig.add_subplot(num_agents, 2, idx * 2 + 2)
        ax_trend.plot(episodes, cumulative_accuracy * 100, color='blue', linewidth=2, marker='o', markersize=3)
        ax_trend.axhline(y=final_accuracy_pct, color='red', linestyle='--', alpha=0.7, label='Final')
        ax_trend.fill_between(episodes, cumulative_accuracy * 100, alpha=0.2, color='blue')
        ax_trend.set_ylabel('Accuracy (%)', fontsize=10)
        ax_trend.set_xlabel('Micro-step', fontsize=10)
        ax_trend.set_title(f'{agent_name} — Accuracy Trend', fontsize=11)
        ax_trend.set_ylim(0, 105)
        ax_trend.grid(True, alpha=0.3)
        ax_trend.legend()

    fig.suptitle(f'{title} — Accuracy Gauge & Trend' if title else
                 'Accuracy Gauge & Trend', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(f"{dir_name}/success_gauge_trend.png", dpi=100)
    plt.close()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Chart 4: Success rate heatmap over time windows
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if num_agents <= 5:
        fig, ax = plt.subplots(figsize=(14, 3 + num_agents * 0.8))
        
        heatmap_data = []
        agent_names_list = []
        
        for agent_name, preds in test["predicted"].items():
            pred_values = np.array(preds, dtype=int)
            matches = (pred_values == ground_truth).astype(int)
            
            agent_window_accuracy = []
            for w in range(n_windows):
                start_idx = w * window_size
                end_idx = min((w + 1) * window_size, len(matches))
                window_data = matches[start_idx:end_idx]
                accuracy = np.sum(window_data) / len(window_data) * 100 if len(window_data) > 0 else 0
                agent_window_accuracy.append(accuracy)
            
            heatmap_data.append(agent_window_accuracy)
            agent_names_list.append(agent_name)
        
        heatmap_array = np.array(heatmap_data)
        im = ax.imshow(heatmap_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        
        ax.set_xticks(np.arange(n_windows))
        ax.set_xticklabels(window_labels, rotation=45, ha='right')
        ax.set_yticks(np.arange(len(agent_names_list)))
        ax.set_yticklabels(agent_names_list)
        ax.set_xlabel('Time Window', fontsize=11)
        ax.set_ylabel('Agent', fontsize=11)
        
        # Add percentage annotations
        for i in range(len(agent_names_list)):
            for j in range(n_windows):
                value = heatmap_array[i, j]
                text_color = 'white' if value < 50 else 'black'
                ax.text(j, i, f'{value:.0f}%', ha='center', va='center',
                       color=text_color, fontsize=9, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Accuracy (%)', fontsize=10)
        
        ax.set_title(f'Success Rate Heatmap by Agent and Time Window (window={window_size})', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{dir_name}/success_heatmap.png", dpi=100)
        plt.close()
