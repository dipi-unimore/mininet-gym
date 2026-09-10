#!/usr/bin/env python3
"""
Operational detection / mitigation ("IPS") metrics for the Paper 9 cells.

The paper reports macro-F1 and per-class recall. This script re-cuts the *same*
per-host ``ground_truth`` / ``predicted`` vectors in ``TEST_<agent>/test.json`` as
security-operational quantities, so the paper can speak as an intrusion-*prevention*
system, not just a classifier.

The 300 evaluation "episodes" are ONE continuous per-host trace: ``env.reset()``
does not touch ``global_state``, the OVS flow-drop rules, or ``status_links`` (only
the anti-oscillation counters), and the scenario advances continuously -- an
attacker keeps attacking, a block stays in effect, across episode boundaries. So an
attack is a contiguous run of ``gt == attacker`` for a host, spanning
``short_attack_duration`` (~5) or ``long_attack_duration`` (~25-29) episodes, and we
can measure how quickly and how durably the agent shuts it down.

Pointwise rates (over all per-host decisions):
  collateral_block_rate  P(pred=attacker | gt=normal)   -- benign hosts blocked
  block_precision        P(gt=attacker | pred=attacker) -- of blocks issued, fraction right
  victim_block_rate      P(pred=attacker | gt=victim)   -- mitigation misfired onto the victim

Temporal quantities (over attack windows = contiguous gt==attacker runs per host):
  mitigation_coverage    fraction of attack windows blocked at least once
  median_ttm / mean_ttm  episodes from attack onset to the first block (mitigated windows)
  attacker_exposure      attacker-active episodes with no block asserted / all such episodes
  post_block_hold        after the first block, fraction of the rest of the window still blocked
                         (1.0 = the block holds; < 1.0 = it flaps as the attacker's
                         dropped traffic makes the host look benign, then unblocks)

Usage:
    python3 test_tools/security_metrics_table.py test_tools/compare_config_security.yaml
    python3 test_tools/security_metrics_table.py cfg_a.yaml cfg_b.yaml --output-dir docs/.../tables
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from generate_ranking_table import (
    get_num_hosts,
    normalize_algorithm_name,
    read_test_payload,
)

NORMAL, VICTIM, ATTACKER = 0, 1, 2


def _runs(mask: np.ndarray):
    """Yield (start, length) of every maximal True run in a 1-D boolean array."""
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            yield i, j - i
            i = j
        else:
            i += 1


def _pointwise(gt: np.ndarray, pred: np.ndarray) -> dict:
    is_norm, is_vic, is_att = gt == NORMAL, gt == VICTIM, gt == ATTACKER
    blk = pred == ATTACKER
    n_blocks = int(blk.sum())
    return {
        "attacker_recall": float((blk & is_att).sum() / max(1, is_att.sum())),
        "collateral_block_rate": float((blk & is_norm).sum() / max(1, is_norm.sum())),
        "victim_block_rate": (
            float((blk & is_vic).sum() / is_vic.sum()) if is_vic.sum() else float("nan")
        ),
        "block_precision": (
            float((blk & is_att).sum() / n_blocks) if n_blocks else float("nan")
        ),
        "n_attacker": int(is_att.sum()),
        "n_normal": int(is_norm.sum()),
        "n_victim": int(is_vic.sum()),
    }


def _temporal(gt2d: np.ndarray, pred2d: np.ndarray) -> dict:
    """gt2d/pred2d shaped [n_episodes, n_hosts]."""
    ttms, exposures, holds = [], [], []
    n_windows = n_mitigated = 0
    att_eps_total = att_eps_unblocked = 0

    for h in range(gt2d.shape[1]):
        gcol, pcol = gt2d[:, h], pred2d[:, h]
        for start, length in _runs(gcol == ATTACKER):
            n_windows += 1
            win_pred_blk = pcol[start:start + length] == ATTACKER
            att_eps_total += length
            att_eps_unblocked += int((~win_pred_blk).sum())
            blocked_offsets = np.flatnonzero(win_pred_blk)
            if blocked_offsets.size:
                n_mitigated += 1
                ttm = int(blocked_offsets[0])
                ttms.append(ttm)
                if ttm < length - 1:
                    holds.append(float(win_pred_blk[ttm + 1:].mean()))
            exposures.append(length)

    return {
        "n_windows": n_windows,
        "mitigation_coverage": n_mitigated / max(1, n_windows),
        "median_ttm": float(np.median(ttms)) if ttms else float("nan"),
        "mean_ttm": float(np.mean(ttms)) if ttms else float("nan"),
        "attacker_exposure": att_eps_unblocked / max(1, att_eps_total),
        "post_block_hold": float(np.mean(holds)) if holds else float("nan"),
    }


def collect(experiment_path: str, excluded: set) -> dict:
    exp = Path(experiment_path)
    n_hosts = get_num_hosts(exp)
    out: dict = {}
    for test_dir in sorted(exp.iterdir()):
        if not test_dir.is_dir() or not test_dir.name.startswith("TEST"):
            continue
        tj = test_dir / "test.json"
        if not tj.exists():
            continue
        agent = test_dir.name[5:] if test_dir.name.startswith("TEST_") else test_dir.name
        algo = normalize_algorithm_name(agent)
        if algo in excluded:
            continue
        try:
            gt, pred = read_test_payload(tj, agent, n_hosts)
        except Exception as exc:  # noqa: BLE001
            print(f"  warn: skip {tj}: {exc}")
            continue
        if not gt.size or n_hosts is None or gt.size % n_hosts:
            continue
        m = _pointwise(gt, pred)
        m.update(_temporal(gt.reshape(-1, n_hosts), pred.reshape(-1, n_hosts)))
        out[algo] = m
    return out


def _load_experiments(config_paths):
    experiments, excluded, prefixes, seen = [], set(), [], set()
    for cp in config_paths:
        with open(cp) as fh:
            cfg = yaml.safe_load(fh)
        prefixes.append(cfg.get("output_prefix", Path(cp).stem))
        for a in cfg.get("excluded_algorithms", []) or []:
            excluded.add(normalize_algorithm_name(a))
        for e in cfg.get("experiments", []):
            if e["path"] in seen:
                continue
            seen.add(e["path"])
            experiments.append(e)
    prefix = prefixes[0] if len(set(prefixes)) == 1 else "security"
    return experiments, excluded, prefix


# (key, latex header)
COLS = [
    ("attacker_recall", r"Neutr."),
    ("median_ttm", r"TTM"),
    ("attacker_exposure", r"Expo."),
    ("post_block_hold", r"Hold"),
    ("collateral_block_rate", r"Collat."),
    ("block_precision", r"Prec."),
]


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{nd}f}"


def write_latex(rows, prefix, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"security_metrics_{prefix}.tex"
    header = " & ".join(["Cell", "Alg."] + [c[1] for c in COLS]) + r" \\"
    lines = [
        "% generated by test_tools/security_metrics_table.py -- do not edit by hand",
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \small",
        r"    \setlength{\tabcolsep}{4.5pt}",
        r"    \begin{tabular}{@{}ll" + "c" * len(COLS) + r"@{}}",
        r"        \toprule",
        "        " + header,
        r"        \midrule",
    ]
    for label, algo, m in rows:
        cells = " & ".join(
            _fmt(m[k], 1 if k == "median_ttm" else 3) for k, _ in COLS
        )
        lines.append(f"        {label} & {algo} & {cells} " + r"\\")
    lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
        r"    \caption{Operational detection/mitigation, recomputed from the per-host "
        r"\texttt{ground\_truth}/\texttt{predicted} traces of "
        r"Sections~\ref{sec:exp_baseline}--\ref{sec:exp_tier_b} (an \textsc{attacker} "
        r"label is an actual OVS flow-drop; the 300 evaluation episodes form one "
        r"continuous per-host trace). \emph{Neutralized}: fraction of attacker "
        r"host-steps blocked. \emph{TTM}: median episodes from attack onset to the "
        r"first block. \emph{Exposure}: attacker-active episodes with no block "
        r"asserted, over all attacker-active episodes. \emph{Block hold}: after the "
        r"first block, fraction of the remaining attack window still blocked. "
        r"\emph{Collateral}: benign host-steps blocked. \emph{Block prec.}: fraction "
        r"of issued blocks that hit a real attacker.}",
        rf"    \label{{tab:security_metrics_{prefix}}}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--output-dir", default="_training/_debug/generate")
    args = ap.parse_args()

    experiments, excluded, prefix = _load_experiments(args.configs)
    if excluded:
        print(f"excluded algorithms: {sorted(excluded)}")

    order = {"Q-Learning": 0, "SARSA": 1, "DQN": 2, "PPO": 3, "A2C": 4}
    rows = []
    hdr = (f"\n{'cell':<11}{'alg':<11}{'neutr':>7}{'medTTM':>7}{'expos':>7}"
           f"{'hold':>7}{'collat':>7}{'blkPrc':>7}   (nWin  n_att/n_norm/n_vic)")
    print(hdr)
    for e in experiments:
        if not Path(e["path"]).exists():
            print(f"  warn: missing {e['path']}")
            continue
        for algo, m in sorted(collect(e["path"], excluded).items(),
                              key=lambda kv: order.get(kv[0], 9)):
            rows.append((e["label"], algo, m))
            print(f"{e['label']:<11}{algo:<11}"
                  f"{_fmt(m['attacker_recall']):>7}{_fmt(m['median_ttm'],1):>7}"
                  f"{_fmt(m['attacker_exposure']):>7}{_fmt(m['post_block_hold']):>7}"
                  f"{_fmt(m['collateral_block_rate']):>7}{_fmt(m['block_precision']):>7}"
                  f"   ({m['n_windows']:>3}  {m['n_attacker']}/{m['n_normal']}/{m['n_victim']})")

    out_dir = Path(args.output_dir)
    tex = write_latex(rows, prefix, out_dir)
    (out_dir / f"security_metrics_{prefix}.json").write_text(
        json.dumps([{"cell": l, "algorithm": a, **m} for l, a, m in rows], indent=2) + "\n"
    )
    print(f"\nwrote {tex}\nwrote {out_dir / f'security_metrics_{prefix}.json'}")


if __name__ == "__main__":
    main()
