"""Macro-averaged *training* metrics, rebuilt from <agent_dir>/data.json.

The per-episode `train_metrics` arrays logged by the agents (accuracy, precision,
recall, f1_score) are class-frequency-weighted, so they are dominated by the
~83% NORMAL class and can look healthy for a predictor that never outputs
VICTIM or ATTACKER (see experiments9.tex, sec:exp_setup and the DQN
failure analysis). The paper commits to macro-averaged metrics, so this module
recomputes them from the raw per-step records in `train_indicators`.

Each `train_indicators[e]["episode_statuses"]` row carries `traffic_type`
(the ground-truth host status id, set from `infos["action_correct"]`) and
`action_choosen` (the agent's action). The ground-truth id is mapped to the
3-class label space exactly as `MarlPzEnv._is_action_correct` does:
    0 normal, 3 incoming_blocked_attack, -1 idle, anything else -> NORMAL(0)
    1 under_attack                                              -> VICTIM(1)
    2 attacking, 4 out_attack_blocked                           -> ATTACKER(2)
Checked on the tau-sweep runs: the recomputed accuracy equals
`correct_predictions / steps` in every episode.

Per-episode macro-F1 is not defined (mitigation leaves ~0-1 VICTIM rows per
60-step episode), so callers pool confusion matrices over several episodes
(rolling window, or the whole training run) before averaging over classes.
"""
import json
from pathlib import Path

import numpy as np

N_CLASSES = 3


def gt_class(status_id: int) -> int:
    if status_id == 1:
        return 1
    if status_id in (2, 4):
        return 2
    return 0


def train_confusion_per_episode(data_json: Path) -> np.ndarray:
    """Return an (episodes, 3, 3) array of confusion counts (rows = ground
    truth, columns = predicted) from a data.json's `train_indicators`."""
    with open(data_json, "r") as f:
        data = json.load(f)
    episodes = data["train_indicators"]
    cms = np.zeros((len(episodes), N_CLASSES, N_CLASSES), dtype=np.int64)
    for i, ep in enumerate(episodes):
        for row in ep["episode_statuses"]:
            cms[i, gt_class(int(row["traffic_type"])), int(row["action_choosen"])] += 1
    return cms


def macro_prf_from_confusion(cm: np.ndarray):
    """Macro precision, recall and F1 (unweighted mean over the 3 classes; a
    class with no support and no predictions scores 0, as in the test-set
    macro-F1 of compare_episodes_metrics.precision_recall_f1_macro)."""
    precisions, recalls, f1s = [], [], []
    for c in range(N_CLASSES):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s))


def rolling_macro_f1(cms: np.ndarray, window: int) -> np.ndarray:
    """Macro-F1 of the confusion counts pooled over a trailing window of
    `window` episodes. Element k covers episodes k-window+1..k (1-indexed
    episode k+1); the first window-1 episodes are omitted, so the result has
    len(cms) - window + 1 entries."""
    out = []
    for end in range(window, len(cms) + 1):
        pooled = cms[end - window:end].sum(axis=0)
        out.append(macro_prf_from_confusion(pooled)[2])
    return np.array(out)
