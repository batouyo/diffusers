"""Aggregate held-out joint controls and test sparse-candidate superiority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from aggregate_results import cluster_macro, stratified_bootstrap


def load_joint(run_root: Path) -> pd.DataFrame:
    rows = []
    for meta_path in sorted((run_root / "joint").rglob("*.json")):
        if meta_path.name.endswith(".eval.json"):
            continue
        eval_path = meta_path.with_suffix(".eval.json")
        if not eval_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
        quality = evaluation.get("quality", {})
        rows.append(
            {
                **meta,
                "s_edit": evaluation.get("s_edit"),
                "s_preserve": evaluation.get("dino_similarity"),
                "lpips_distance": evaluation.get("lpips_distance"),
                "bad_image": bool(
                    (not quality.get("finite", True))
                    or quality.get("all_black", False)
                    or quality.get("all_white", False)
                    or quality.get("severe_saturation", False)
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    keys = ["sample_id", "seed", "resolution"]
    baseline = frame[frame["arm"] == "baseline"][keys + ["s_edit", "s_preserve", "lpips_distance"]].drop_duplicates(keys)
    baseline = baseline.rename(
        columns={
            "s_edit": "s_edit_base",
            "s_preserve": "s_preserve_base",
            "lpips_distance": "lpips_distance_base",
        }
    )
    frame = frame.merge(baseline, on=keys, how="left")
    frame["semantic_gain"] = frame["s_edit"] - frame["s_edit_base"]
    frame["preservation_delta"] = frame["s_preserve"] - frame["s_preserve_base"]
    frame["lpips_cost"] = frame["lpips_distance"] - frame["lpips_distance_base"]
    return frame


def paired_difference(frame: pd.DataFrame, left_arm: str, right_arm: str, config: dict) -> dict:
    keys = ["sample_id", "seed", "category"]
    left = frame[frame["arm"] == left_arm][keys + ["s_edit"]].rename(columns={"s_edit": "left"})
    right = frame[frame["arm"] == right_arm][keys + ["s_edit"]].rename(columns={"s_edit": "right"})
    paired = left.merge(right, on=keys)
    paired["difference"] = paired["left"] - paired["right"]
    low, high = stratified_bootstrap(
        paired,
        "difference",
        config["statistics"]["bootstrap_samples"],
        config["statistics"]["random_seed"],
    )
    return {"mean": cluster_macro(paired, "difference"), "ci_low": low, "ci_high": high, "n": len(paired)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="probe_config.yaml")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_id = args.run_id or config["project"]["run_id"]
    run_root = Path(config["project"]["output_root"]) / run_id
    frame = load_joint(run_root)
    if frame.empty:
        raise RuntimeError("no evaluated joint-validation outputs")
    frame.to_csv(run_root / "joint_metrics.csv", index=False)
    summaries = []
    for arm, group in frame.groupby("arm"):
        gain_low, gain_high = stratified_bootstrap(
            group, "semantic_gain", config["statistics"]["bootstrap_samples"], config["statistics"]["random_seed"]
        )
        preserve_low, preserve_high = stratified_bootstrap(
            group,
            "preservation_delta",
            config["statistics"]["bootstrap_samples"],
            config["statistics"]["random_seed"] + 1,
        )
        summaries.append(
            {
                "arm": arm,
                "semantic_gain": cluster_macro(group, "semantic_gain"),
                "semantic_gain_ci_low": gain_low,
                "semantic_gain_ci_high": gain_high,
                "preservation_delta": cluster_macro(group, "preservation_delta"),
                "preservation_ci_low": preserve_low,
                "preservation_ci_high": preserve_high,
                "lpips_cost": cluster_macro(group, "lpips_cost"),
                "bad_image_rate": float(group["bad_image"].mean()),
            }
        )
    summary = pd.DataFrame(summaries).sort_values("semantic_gain", ascending=False)
    summary.to_csv(run_root / "joint_summary.csv", index=False)
    lookup = summary.set_index("arm")
    candidate_gain = float(lookup.loc["candidate_combo", "semantic_gain"])
    random_values = summary[summary["arm"].str.startswith("random_")]["semantic_gain"].to_numpy()
    empirical_p = float((1 + np.sum(random_values >= candidate_gain)) / (1 + len(random_values)))
    comparisons = {
        arm: paired_difference(frame, "candidate_combo", arm, config)
        for arm in [
            "baseline",
            "all_blocks",
            "all_blocks_budget_matched",
            "textailor_flux1dev_control",
        ]
        if arm in set(frame["arm"])
    }
    candidate = lookup.loc["candidate_combo"]
    success = bool(
        empirical_p <= 1 / 21
        and all(value["ci_low"] > 0 for value in comparisons.values())
        and candidate["preservation_ci_low"] >= config["statistics"]["dino_noninferiority_margin"]
        and candidate["bad_image_rate"] <= config["statistics"]["bad_image_rate_max"]
    )
    result = {
        "status": "validated" if success else "not_validated",
        "candidate_combo_semantic_gain": candidate_gain,
        "random_empirical_p": empirical_p,
        "comparisons": comparisons,
        "preservation_ci_low": float(candidate["preservation_ci_low"]),
        "bad_image_rate": float(candidate["bad_image_rate"]),
    }
    (run_root / "joint_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot = summary[
        summary["arm"].isin(
            [
                "candidate_combo",
                "all_blocks",
                "all_blocks_budget_matched",
                "textailor_flux1dev_control",
            ]
        )
        | summary["arm"].str.startswith("random_")
    ]
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 6))
    sns.barplot(data=plot, x="arm", y="semantic_gain", color="#4C78A8")
    plt.xticks(rotation=75, ha="right")
    plt.tight_layout()
    plots = run_root / "plots"
    plots.mkdir(exist_ok=True)
    plt.savefig(plots / "candidate_vs_random_and_all.png", dpi=180)
    plt.close()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
