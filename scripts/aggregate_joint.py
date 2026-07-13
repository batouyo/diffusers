"""Aggregate held-out joint controls and test sparse-candidate superiority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from aggregate_results import cluster_macro, stratified_bootstrap
from evaluators import evaluation_hash, reusable_evaluation
from probe_flux_kontext_blocks import file_sha256
try:
    from run_joint_validation import arms, joint_hash
except ModuleNotFoundError:  # package import during tests
    from scripts.run_joint_validation import arms, joint_hash


def load_joint(run_root: Path, expected_evaluation_hash: str) -> pd.DataFrame:
    rows = []
    for meta_path in sorted((run_root / "joint").rglob("*.json")):
        if meta_path.name.endswith(".eval.json"):
            continue
        eval_path = meta_path.with_suffix(".eval.json")
        if not eval_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
        if not reusable_evaluation(
            evaluation, meta, expected_evaluation_hash, require_vlm=True
        ):
            raise RuntimeError(f"joint evaluation is stale or incomplete: {eval_path}")
        output_path = Path(meta.get("output_path", ""))
        if not output_path.exists() or file_sha256(output_path) != meta.get("output_sha256"):
            raise RuntimeError(f"joint image is missing or checksum-invalid: {output_path}")
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
                "vlm_parse_ok": evaluation.get("vlm_parse_ok"),
                "evaluation_hash": evaluation.get("evaluation_hash"),
                "evaluation_output_sha256": evaluation.get("output_sha256"),
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


def validate_joint_pairing(frame: pd.DataFrame) -> None:
    """Require every joint arm to share source and initial latent hashes with its baseline."""
    keys = ["sample_id", "seed", "resolution"]
    baseline = frame[frame["arm"] == "baseline"]
    if baseline.duplicated(keys).any():
        raise RuntimeError("duplicate joint baselines make latent/source pairing ambiguous")
    required = ["latent_hash", "source_sha256"]
    if any(column not in frame.columns for column in required):
        raise RuntimeError("joint metadata is missing latent/source hashes")
    lookup = baseline[keys + required].rename(
        columns={column: f"{column}_base" for column in required}
    )
    paired = frame.merge(lookup, on=keys, how="left")
    if paired[[f"{column}_base" for column in required]].isna().to_numpy().any():
        raise RuntimeError("joint arm is missing its paired baseline hashes")
    for column in required:
        if (paired[column] != paired[f"{column}_base"]).any():
            raise RuntimeError(f"joint baseline/arm {column} mismatch")


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
    expected_evaluation_hash = evaluation_hash(config)
    frame = load_joint(run_root, expected_evaluation_hash)
    if frame.empty:
        raise RuntimeError("no evaluated joint-validation outputs")
    validate_joint_pairing(frame)
    invalid_metrics = frame[
        (frame["evaluation_hash"] != expected_evaluation_hash)
        | (frame["vlm_parse_ok"] != True)
        | frame[["s_edit", "s_preserve", "lpips_distance"]].isna().any(axis=1)
    ]
    if not invalid_metrics.empty:
        raise RuntimeError(
            f"joint validation has {len(invalid_metrics)} stale or invalid evaluations"
        )
    selection = json.loads((run_root / "selected_blocks.json").read_text(encoding="utf-8"))
    candidates = sorted(int(value) for value in selection.get("selected_global_blocks", []))
    if not candidates:
        raise RuntimeError("joint aggregation requires a non-empty independently selected candidate set")
    dataset = [
        json.loads(line)
        for line in Path(config["project"]["dataset_manifest"]).read_text(encoding="utf-8").splitlines()
    ]
    heldout_rows = [row for row in dataset if row["split"] == "heldout"]
    heldout_by_id = {row["id"]: row for row in heldout_rows}
    expected_per_arm = len(heldout_rows) * len(config["inference"]["seeds"])
    selected_alpha_path = run_root / "selected_alpha.json"
    if not selected_alpha_path.exists():
        raise RuntimeError("selected_alpha.json is required to bind joint validation to alpha selection")
    selected_alpha = float(json.loads(selected_alpha_path.read_text(encoding="utf-8"))["alpha"])
    structure = json.loads(
        (Path(config["project"]["output_root"]) / "preflight" / "structure_report.json").read_text(
            encoding="utf-8"
        )
    )
    transformer_shape = SimpleNamespace(
        transformer_blocks=[None] * int(structure["double_block_count"]),
        single_transformer_blocks=[None] * int(structure["single_block_count"]),
    )
    arm_list = arms(
        transformer_shape,
        candidates,
        selected_alpha,
        int(config["probing"]["random_control_sets"]),
        int(config["statistics"]["random_seed"]),
        [int(value) for value in config["probing"]["forbidden_prior_blocks"]],
    )
    arm_specs = {
        name: {"intervention_mode": mode, "block_indices": tuple(blocks), "alpha": float(alpha)}
        for name, mode, blocks, alpha in arm_list
    }
    expected_arms = set(arm_specs)
    observed_arms = set(frame["arm"])
    if observed_arms != expected_arms:
        raise RuntimeError(
            f"joint arm mismatch: missing={sorted(expected_arms - observed_arms)}, "
            f"unexpected={sorted(observed_arms - expected_arms)}"
        )
    arm_counts = frame["arm"].value_counts().to_dict()
    incomplete = {arm: arm_counts.get(arm, 0) for arm in expected_arms if arm_counts.get(arm, 0) != expected_per_arm}
    if incomplete:
        raise RuntimeError(f"joint evaluation incomplete; expected {expected_per_arm} per arm: {incomplete}")
    key_columns = ["arm", "sample_id", "seed", "resolution"]
    if frame.duplicated(key_columns).any():
        raise RuntimeError("duplicate evaluated joint jobs detected")
    resolution = int(config["inference"]["resolution"])
    expected_keys = {
        (arm, row["id"], int(seed), resolution)
        for arm in expected_arms
        for row in heldout_rows
        for seed in config["inference"]["seeds"]
    }
    observed_keys = {
        (str(row["arm"]), str(row["sample_id"]), int(row["seed"]), int(row["resolution"]))
        for _, row in frame.iterrows()
    }
    if observed_keys != expected_keys:
        raise RuntimeError(
            f"joint exact job matrix mismatch: missing={len(expected_keys - observed_keys)} "
            f"unexpected={len(observed_keys - expected_keys)}"
        )
    expected_fingerprint = joint_hash(
        config, candidates, selected_alpha, arm_list, "heldout", resolution
    )
    protocol_errors = []
    for _, row in frame.iterrows():
        spec = arm_specs[str(row["arm"])]
        source = heldout_by_id.get(str(row["sample_id"]))
        if (
            row.get("status") != "complete"
            or row.get("split") != "heldout"
            or source is None
            or row.get("category") != source["category"]
            or row.get("intervention_mode") != spec["intervention_mode"]
            or tuple(row.get("block_indices", [])) != spec["block_indices"]
            or not np.isclose(float(row.get("alpha")), spec["alpha"])
            or row.get("joint_hash") != expected_fingerprint
            or row.get("evaluation_output_sha256") != row.get("output_sha256")
        ):
            protocol_errors.append(
                (str(row.get("arm")), str(row.get("sample_id")), int(row.get("seed")))
            )
    if protocol_errors:
        raise RuntimeError(f"joint protocol mismatch in {len(protocol_errors)} jobs: {protocol_errors[:3]}")
    joint_hashes = set(frame["joint_hash"].dropna())
    if joint_hashes != {expected_fingerprint}:
        raise RuntimeError(
            f"joint fingerprint mismatch: observed={sorted(joint_hashes)} expected={expected_fingerprint}"
        )
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
    category_summary = (
        frame.groupby(["arm", "category"], as_index=False)
        .agg(
            semantic_gain=("semantic_gain", "mean"),
            preservation_delta=("preservation_delta", "mean"),
            lpips_cost=("lpips_cost", "mean"),
            bad_image_rate=("bad_image", "mean"),
            n=("sample_id", "size"),
        )
    )
    category_summary.to_csv(run_root / "joint_category_summary.csv", index=False)
    seed_summary = (
        frame.groupby(["arm", "seed"], as_index=False)
        .agg(
            semantic_gain=("semantic_gain", "mean"),
            preservation_delta=("preservation_delta", "mean"),
            lpips_cost=("lpips_cost", "mean"),
            bad_image_rate=("bad_image", "mean"),
            n=("sample_id", "size"),
        )
    )
    seed_summary.to_csv(run_root / "joint_seed_summary.csv", index=False)
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
    candidate_seed_means = {
        str(int(row["seed"])): float(row["semantic_gain"])
        for _, row in seed_summary[seed_summary["arm"] == "candidate_combo"].iterrows()
    }
    candidate_category_means = {
        str(row["category"]): float(row["semantic_gain"])
        for _, row in category_summary[category_summary["arm"] == "candidate_combo"].iterrows()
    }
    all_seed_means_positive = len(candidate_seed_means) == len(config["inference"]["seeds"]) and all(
        value > 0 for value in candidate_seed_means.values()
    )
    success = bool(
        empirical_p <= 1 / (config["probing"]["random_control_sets"] + 1)
        and all(value["ci_low"] > 0 for value in comparisons.values())
        and candidate["preservation_ci_low"] >= config["statistics"]["dino_noninferiority_margin"]
        and candidate["bad_image_rate"] <= config["statistics"]["bad_image_rate_max"]
        and all_seed_means_positive
    )
    result = {
        "execution_status": "complete",
        "status": "validated" if success else "not_validated",
        "protocol_fingerprint": next(iter(joint_hashes)),
        "expected_protocol_fingerprint": expected_fingerprint,
        "exact_job_matrix_verified": True,
        "image_checksums_verified": True,
        "expected_per_arm": expected_per_arm,
        "expected_total": expected_per_arm * len(expected_arms),
        "evaluated_total": int(len(frame)),
        "arm_counts": {arm: int(arm_counts[arm]) for arm in sorted(arm_counts)},
        "candidate_combo_semantic_gain": candidate_gain,
        "random_empirical_p": empirical_p,
        "comparisons": comparisons,
        "preservation_ci_low": float(candidate["preservation_ci_low"]),
        "bad_image_rate": float(candidate["bad_image_rate"]),
        "candidate_seed_semantic_gain": candidate_seed_means,
        "all_seed_means_positive": all_seed_means_positive,
        "candidate_category_semantic_gain": candidate_category_means,
        "positive_category_count": sum(value > 0 for value in candidate_category_means.values()),
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
