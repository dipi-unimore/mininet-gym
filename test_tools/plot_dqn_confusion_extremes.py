#!/usr/bin/env python3
"""Side-by-side DQN confusion matrices for two Tier B conditions, chosen to make
the Experience-Sharing-driven collapse visually obvious: S1+S2 (mild, still
majority-class-collapsed but onto NORMAL) vs. S1+S4 (severe, collapsed onto
ATTACKING instead, UNDER_ATTACK essentially disappears from both axes).

Not part of the generic compare_* pipeline (DQN is excluded there) -- reuses
generate_confusion_plots.render_confusion_matrix/get_class_ids/class_names for
visual consistency with the rest of paper9's confusion-matrix figures.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_confusion_plots import class_names, get_class_ids, render_confusion_matrix

BASE = Path("_training/marl_pz")

# (panel title, run directory, agent dir)
PANELS = [
    ("S1 + S2\n(federated_sync)", "20260816-174823_1_5_5", "DQN_quick"),
    ("S1 + S4\n(experience_sharing)", "20260818-150129_1_5_5", "DQN_quick"),
]


def load_df(run_dir: str, agent_dir: str) -> pd.DataFrame:
    test_json = BASE / run_dir / f"TEST_{agent_dir}" / "test.json"
    with open(test_json) as f:
        t = json.load(f)
    gt = t["ground_truth"]
    pred = t["predicted"][agent_dir]
    return pd.DataFrame({"true_label": gt, "predicted_label": pred})


def main():
    dfs = [load_df(run_dir, agent_dir) for _, run_dir, agent_dir in PANELS]
    class_ids = get_class_ids({f"panel{i}": df for i, df in enumerate(dfs)})
    class_labels = class_names(class_ids)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, (title, _, _), df in zip(axes, PANELS, dfs):
        render_confusion_matrix(ax, df, title, class_ids, class_labels)

    fig.suptitle(
        "DQN confusion matrices under Tier B: S2 (mild collapse) vs. S4 (severe collapse), both under S1/UAQ",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out_path = "docs/papers/9-draft of future work in MARL_Coordination__Strategies/images/dqn_confusion_extremes.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
