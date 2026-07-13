"""Aggregate per-image metrics, calculate paired block statistics, and select candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy.stats import wilcoxon


def read_records(run_root: Path) -> pd.DataFrame:
    rows = []
    for meta_path in sorted((run_root / "images").rglob("*.json")):
        if meta_path.name.endswith(".eval.json"):
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if meta.get("status") != "complete":
            continue
        eval_path = meta_path.with_suffix(".eval.json")
        evaluation = {}
        if eval_path.exists():
            evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
        quality = evaluation.get("quality", {})
        rows.append(
            {
                **meta,
                "s_edit": evaluation.get("s_edit"),
                "s_preserve": evaluation.get("dino_similarity"),
                "lpips_distance": evaluation.get("lpips_distance"),
                "vlm_parse_ok": evaluation.get("vlm_parse_ok"),
                "bad_image": bool(
                    (not quality.get("finite", True))
                    or quality.get("all_black", False)
                    or quality.get("all_white", False)
                    or quality.get("severe_saturation", False)
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or frame["s_edit"].isna().all():
        return frame
    keys = ["sample_id", "seed", "resolution"]
    base = frame[frame["mode"] == "baseline"][keys + ["s_edit", "s_preserve", "lpips_distance"]].drop_duplicates(keys)
    base = base.rename(
        columns={
            "s_edit": "s_edit_base",
            "s_preserve": "s_preserve_base",
            "lpips_distance": "lpips_distance_base",
        }
    )
    merged = frame.merge(base, on=keys, how="left")
    merged["semantic_gain"] = np.where(
        merged["mode"] == "enhance_text", merged["s_edit"] - merged["s_edit_base"], np.nan
    )
    merged["semantic_drop"] = np.where(
        merged["mode"] == "disable_text", merged["s_edit_base"] - merged["s_edit"], np.nan
    )
    merged["removal_edit_drop"] = np.where(
        merged["mode"] == "remove_block", merged["s_edit_base"] - merged["s_edit"], np.nan
    )
    merged["preservation_cost"] = np.where(
        merged["mode"] == "enhance_text", merged["s_preserve_base"] - merged["s_preserve"], np.nan
    )
    merged["lpips_cost"] = np.where(
        merged["mode"] == "enhance_text", merged["lpips_distance"] - merged["lpips_distance_base"], np.nan
    )
    merged["removal_preservation_cost"] = np.where(
        merged["mode"] == "remove_block", merged["s_preserve_base"] - merged["s_preserve"], np.nan
    )
    return merged


def cluster_macro(values: pd.DataFrame, column: str) -> float:
    available = values.dropna(subset=[column])
    if available.empty:
        return float("nan")
    per_sample = available.groupby(["category", "sample_id"], as_index=False)[column].mean()
    per_category = per_sample.groupby("category")[column].mean()
    return float(per_category.mean())


def stratified_bootstrap(values: pd.DataFrame, column: str, draws: int, seed: int) -> tuple[float, float]:
    available = values.dropna(subset=[column])
    if available.empty:
        return float("nan"), float("nan")
    per_sample = available.groupby(["category", "sample_id"], as_index=False)[column].mean()
    groups = {category: group[column].to_numpy() for category, group in per_sample.groupby("category")}
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        means = [rng.choice(array, size=len(array), replace=True).mean() for array in groups.values()]
        estimates[draw] = np.mean(means)
    return tuple(np.quantile(estimates, [0.025, 0.975]).tolist())


def one_sided_paired_p(values: pd.DataFrame, column: str) -> float:
    """One-sided paired Wilcoxon p-value after averaging repeated seeds per sample."""
    available = values.dropna(subset=[column])
    if available.empty:
        return float("nan")
    per_sample = available.groupby(["category", "sample_id"])[column].mean().to_numpy()
    if not len(per_sample) or np.allclose(per_sample, 0):
        return 1.0
    try:
        return float(wilcoxon(per_sample, alternative="greater", zero_method="zsplit").pvalue)
    except ValueError:
        return float("nan")


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    """Return monotone BH-adjusted q-values while preserving missing entries."""
    result = pd.Series(np.nan, index=values.index, dtype=float)
    available = values.dropna().astype(float)
    if available.empty:
        return result
    order = np.argsort(available.to_numpy())
    ordered = available.to_numpy()[order]
    adjusted = ordered * len(ordered) / np.arange(1, len(ordered) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    ordered_indices = available.index.to_numpy()[order]
    result.loc[ordered_indices] = adjusted
    return result


def summarize_blocks(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    stats = config["statistics"]
    summaries = []
    block_rows = frame[frame["global_block_index"].notna()].copy()
    if block_rows.empty:
        return pd.DataFrame()
    for global_index, group in block_rows.groupby("global_block_index"):
        enhance = group[
            (group["mode"] == "enhance_text")
            & np.isclose(group["alpha"].astype(float), float(config["inference"]["alpha"]))
        ]
        disable = group[group["mode"] == "disable_text"]
        remove = group[group["mode"] == "remove_block"]
        gain = cluster_macro(enhance, "semantic_gain")
        drop = cluster_macro(disable, "semantic_drop")
        removal_drop = cluster_macro(remove, "removal_edit_drop")
        cost = cluster_macro(enhance, "preservation_cost")
        removal_cost = cluster_macro(remove, "removal_preservation_cost")
        gain_lo, gain_hi = stratified_bootstrap(
            enhance, "semantic_gain", stats["bootstrap_samples"], stats["random_seed"] + int(global_index)
        )
        drop_lo, drop_hi = stratified_bootstrap(
            disable, "semantic_drop", stats["bootstrap_samples"], stats["random_seed"] + 1000 + int(global_index)
        )
        cost_lo, cost_hi = stratified_bootstrap(
            enhance, "preservation_cost", stats["bootstrap_samples"], stats["random_seed"] + 2000 + int(global_index)
        )
        removal_drop_lo, removal_drop_hi = stratified_bootstrap(
            remove, "removal_edit_drop", stats["bootstrap_samples"], stats["random_seed"] + 3000 + int(global_index)
        )
        removal_cost_lo, removal_cost_hi = stratified_bootstrap(
            remove,
            "removal_preservation_cost",
            stats["bootstrap_samples"],
            stats["random_seed"] + 4000 + int(global_index),
        )
        gain_p = one_sided_paired_p(enhance, "semantic_gain")
        drop_p = one_sided_paired_p(disable, "semantic_drop")
        category_gain = (
            enhance.groupby(["category", "sample_id"])["semantic_gain"].mean().groupby("category").mean().to_dict()
        )
        per_sample_gain = enhance.groupby("sample_id")["semantic_gain"].mean()
        per_seed_gain = enhance.groupby("seed")["semantic_gain"].mean().to_dict()
        address = group["block_address"].dropna().iloc[0] if group["block_address"].notna().any() else {}
        positive_categories = sum(value > 0 for value in category_gain.values())
        preservation_limit = -float(stats["dino_noninferiority_margin"])
        enhance_bad_rate = float(enhance["bad_image"].mean()) if len(enhance) else np.nan
        disable_bad_rate = float(disable["bad_image"].mean()) if len(disable) else np.nan
        remove_bad_rate = float(remove["bad_image"].mean()) if len(remove) else np.nan
        if (np.isfinite(enhance_bad_rate) and enhance_bad_rate > stats["bad_image_rate_max"]) or (
            np.isfinite(cost) and cost > preservation_limit
        ):
            block_class = "D_destructive_or_unstable"
        elif gain >= stats["universal_gain_min"] and drop >= stats["universal_drop_min"]:
            block_class = "A_high_gain_high_disable_drop"
        elif gain >= stats["universal_gain_min"]:
            block_class = "B_high_gain_low_disable_drop"
        elif drop >= stats["universal_drop_min"]:
            block_class = "C_low_gain_high_disable_drop"
        else:
            block_class = "other_low_signal"
        summaries.append(
            {
                "global_block_index": int(global_index),
                "local_block_index": address.get("local_index") if isinstance(address, dict) else None,
                "block_type": address.get("block_type") if isinstance(address, dict) else None,
                "semantic_gain": gain,
                "semantic_gain_ci_low": gain_lo,
                "semantic_gain_ci_high": gain_hi,
                "semantic_gain_p_one_sided": gain_p,
                "semantic_drop": drop,
                "semantic_drop_ci_low": drop_lo,
                "semantic_drop_ci_high": drop_hi,
                "semantic_drop_p_one_sided": drop_p,
                "removal_edit_drop": removal_drop,
                "removal_edit_drop_ci_low": removal_drop_lo,
                "removal_edit_drop_ci_high": removal_drop_hi,
                "preservation_cost": cost,
                "preservation_cost_ci_low": cost_lo,
                "preservation_cost_ci_high": cost_hi,
                "removal_preservation_cost": removal_cost,
                "removal_preservation_cost_ci_low": removal_cost_lo,
                "removal_preservation_cost_ci_high": removal_cost_hi,
                "positive_sample_rate": float((per_sample_gain > 0).mean()) if len(per_sample_gain) else np.nan,
                "positive_sample_count": int((per_sample_gain > 0).sum()),
                "sample_count": int(len(per_sample_gain)),
                "positive_categories": positive_categories,
                "category_count": int(len(category_gain)),
                "all_seed_means_positive": bool(per_seed_gain) and all(value > 0 for value in per_seed_gain.values()),
                "seed_gain_mean": float(np.mean(list(per_seed_gain.values()))) if per_seed_gain else np.nan,
                "seed_gain_std": float(np.std(list(per_seed_gain.values()), ddof=0)) if per_seed_gain else np.nan,
                "seed_gain_json": json.dumps(per_seed_gain, sort_keys=True),
                "bad_image_rate": enhance_bad_rate,
                "disable_bad_image_rate": disable_bad_rate,
                "remove_bad_image_rate": remove_bad_rate,
                "block_class": block_class,
                "enhance_n": int(len(enhance)),
                "disable_n": int(len(disable)),
                "remove_n": int(len(remove)),
                "category_gain_json": json.dumps(category_gain, sort_keys=True),
            }
        )
    result = pd.DataFrame(summaries).sort_values("global_block_index")
    result["semantic_gain_q_bh"] = benjamini_hochberg(result["semantic_gain_p_one_sided"])
    result["semantic_drop_q_bh"] = benjamini_hochberg(result["semantic_drop_p_one_sided"])
    return result


def select_candidates(summary: pd.DataFrame, frame: pd.DataFrame, config: dict) -> dict:
    stats = config["statistics"]
    if summary.empty:
        return {
            "status": "insufficient_data",
            "stage2_blocks": [],
            "universal_blocks": [],
            "category_specific_blocks": [],
        }
    ranked = summary.sort_values(["semantic_gain_ci_low", "semantic_gain"], ascending=False)
    stage2 = [
        int(value)
        for value in ranked.head(int(config["probing"]["stage2_blocks"]))["global_block_index"]
    ]
    preserve_limit = -float(stats["dino_noninferiority_margin"])
    eligible = summary[
        (summary["semantic_gain"] >= stats["universal_gain_min"])
        & (summary["semantic_gain_ci_low"] > 0)
        & (summary["semantic_gain_q_bh"] <= stats["bh_q"])
        & (summary["positive_categories"] >= stats["universal_positive_categories"])
        & (summary["positive_sample_rate"] >= stats["universal_positive_sample_rate"])
        & summary["all_seed_means_positive"]
        & (summary["semantic_drop"] >= stats["universal_drop_min"])
        & (summary["semantic_drop_ci_low"] > 0)
        & (summary["semantic_drop_q_bh"] <= stats["bh_q"])
        & (summary["preservation_cost_ci_high"] <= preserve_limit)
        & (summary["bad_image_rate"] <= stats["bad_image_rate_max"])
    ].copy()
    eligible = eligible.sort_values(["semantic_gain_ci_low", "semantic_gain"], ascending=False)
    enhance = frame[
        (frame["mode"] == "enhance_text")
        & np.isclose(frame["alpha"].astype(float), float(config["inference"]["alpha"]))
    ]
    response_vectors = {
        int(index): group.groupby("sample_id")["semantic_gain"].mean()
        for index, group in enhance.groupby("global_block_index")
    }

    def redundant_with_selected(index: int, selected_indices: list[int]) -> bool:
        for prior in selected_indices:
            if abs(index - prior) > stats["adjacent_distance"]:
                continue
            pair = pd.concat(
                [response_vectors.get(index), response_vectors.get(prior)], axis=1, join="inner"
            ).dropna()
            if len(pair) < 4:
                continue
            correlation = float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman"))
            if np.isfinite(correlation) and correlation >= stats["redundancy_spearman"]:
                return True
        return False

    universal = []
    for _, row in eligible.iterrows():
        index = int(row["global_block_index"])
        if not redundant_with_selected(index, universal):
            universal.append(index)
        if len(universal) >= config["probing"]["max_candidates"]:
            break

    category_specific = {}
    verified_blocks = set(
        int(value)
        for value in summary.loc[summary["semantic_drop"].notna(), "global_block_index"].tolist()
    )
    category_tests = []
    for (global_index, category), group in enhance.groupby(["global_block_index", "category"]):
        category_tests.append(
            {
                "global_block_index": int(global_index),
                "category": category,
                "p": one_sided_paired_p(group, "semantic_gain"),
            }
        )
    category_test_frame = pd.DataFrame(category_tests)
    category_q_lookup = {}
    if not category_test_frame.empty:
        for category, indices in category_test_frame.groupby("category").groups.items():
            adjusted = benjamini_hochberg(category_test_frame.loc[indices, "p"])
            for row_index, q_value in adjusted.items():
                row = category_test_frame.loc[row_index]
                category_q_lookup[(int(row["global_block_index"]), category)] = float(q_value)
    for (global_index, category), group in enhance.groupby(["global_block_index", "category"]):
        if int(global_index) not in verified_blocks:
            continue
        per_sample = group.groupby("sample_id")["semantic_gain"].mean().dropna()
        if len(per_sample) < 2:
            continue
        mean = float(per_sample.mean())
        low, _ = stratified_bootstrap(group, "semantic_gain", stats["bootstrap_samples"], stats["random_seed"] + int(global_index))
        block_row = summary.loc[summary["global_block_index"] == global_index].iloc[0]
        per_seed = group.groupby("seed")["semantic_gain"].mean().dropna()
        category_q = category_q_lookup.get((int(global_index), category), float("nan"))
        safe = (
            block_row["semantic_drop_ci_low"] > 0
            and block_row["semantic_drop_q_bh"] <= stats["bh_q"]
            and block_row["preservation_cost_ci_high"] <= preserve_limit
            and block_row["bad_image_rate"] <= stats["bad_image_rate_max"]
        )
        if (
            safe
            and category_q <= stats["bh_q"]
            and mean >= stats["category_gain_min"]
            and low > 0
            and float((per_sample > 0).mean()) >= stats["category_positive_sample_rate"]
            and len(per_seed) == len(config["inference"]["seeds"])
            and bool((per_seed > 0).all())
        ):
            current = category_specific.get(category)
            candidate = {
                "global_block_index": int(global_index),
                "gain": mean,
                "ci_low": low,
                "category_gain_q_bh": category_q,
            }
            if current is None or candidate["ci_low"] > current["ci_low"]:
                category_specific[category] = candidate
    selected = list(universal)
    retained_category_specific = {}
    for category, item in sorted(
        category_specific.items(), key=lambda pair: pair[1]["ci_low"], reverse=True
    ):
        index = item["global_block_index"]
        if index not in selected and redundant_with_selected(index, selected):
            continue
        if index not in selected and len(selected) >= config["probing"]["max_candidates"]:
            continue
        if index not in selected:
            selected.append(index)
        retained_category_specific[category] = item
    category_specific = retained_category_specific
    selected = sorted(selected)
    has_disable = summary["semantic_drop"].notna().any()
    return {
        "status": ("selected" if selected else "no_go") if has_disable else "awaiting_disable_text",
        "stage2_blocks": stage2[: config["probing"]["stage2_blocks"]],
        "universal_blocks": universal,
        "category_specific_blocks": category_specific,
        "selected_global_blocks": selected,
        "candidate_count": len(selected),
        "preservation_cost_limit": preserve_limit,
        "redundancy_rule": {
            "maximum_global_index_distance": stats["adjacent_distance"],
            "minimum_spearman": stats["redundancy_spearman"],
        },
        "note": "TexTailor FLUX.1-Dev indices were not used for selection; no minimum candidate count is forced.",
    }


def plot_curves(summary: pd.DataFrame, frame: pd.DataFrame, config: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        return
    sns.set_theme(style="whitegrid")
    for column, name in [
        ("semantic_gain", "semantic_gain"),
        ("semantic_drop", "semantic_drop"),
        ("preservation_cost", "preservation_cost"),
        ("removal_edit_drop", "removal_edit_drop"),
        ("removal_preservation_cost", "removal_preservation_cost"),
    ]:
        plt.figure(figsize=(12, 4))
        sns.lineplot(data=summary, x="global_block_index", y=column, marker="o")
        double_rows = summary[summary["block_type"] == "double"]
        boundary = float(double_rows["global_block_index"].max()) + 0.5 if not double_rows.empty else 18.5
        plt.axvline(boundary, color="black", linestyle="--", label="double/single boundary")
        plt.axhline(0, color="gray", linewidth=1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output / f"{name}_vs_global_block.png", dpi=180)
        plt.close()
    all_enhance = frame[frame["mode"] == "enhance_text"].dropna(subset=["semantic_gain"])
    enhance = all_enhance[
        np.isclose(all_enhance["alpha"].astype(float), float(config["inference"]["alpha"]))
    ]
    if not enhance.empty:
        category_curve = enhance.groupby(["category", "global_block_index"], as_index=False)["semantic_gain"].mean()
        plt.figure(figsize=(14, 7))
        sns.lineplot(
            data=category_curve,
            x="global_block_index",
            y="semantic_gain",
            hue="category",
            marker="o",
        )
        plt.axvline(boundary, color="black", linestyle="--")
        plt.axhline(0, color="gray", linewidth=1)
        plt.tight_layout()
        plt.savefig(output / "category_block_response_curves.png", dpi=180)
        plt.close()
        alpha_curve = all_enhance.groupby(["global_block_index", "alpha"], as_index=False)["semantic_gain"].mean()
        if alpha_curve["alpha"].nunique() > 1:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=alpha_curve, x="alpha", y="semantic_gain", hue="global_block_index", marker="o")
            plt.axhline(0, color="gray", linewidth=1)
            plt.tight_layout()
            plt.savefig(output / "alpha_sensitivity_curves.png", dpi=180)
            plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="probe_config.yaml")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_id = args.run_id or config["project"]["run_id"]
    run_root = Path(config["project"]["output_root"]) / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    frame = paired_metrics(read_records(run_root))
    frame.to_csv(run_root / "raw_metrics.csv", index=False)
    summary = summarize_blocks(frame, config)
    summary.to_csv(run_root / "block_summary.csv", index=False)
    if not summary.empty:
        stream_summary = (
            summary.groupby("block_type", as_index=False)
            .agg(
                block_count=("global_block_index", "count"),
                mean_semantic_gain=("semantic_gain", "mean"),
                mean_semantic_drop=("semantic_drop", "mean"),
                mean_removal_edit_drop=("removal_edit_drop", "mean"),
                mean_preservation_cost=("preservation_cost", "mean"),
                mean_seed_gain_std=("seed_gain_std", "mean"),
                bad_image_rate=("bad_image_rate", "mean"),
            )
        )
        stream_summary.to_csv(run_root / "stream_summary.csv", index=False)
    selected = select_candidates(summary, frame, config)
    (run_root / "selected_blocks.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    alpha_rows = frame[frame["mode"] == "enhance_text"].dropna(subset=["semantic_gain"])
    if not alpha_rows.empty:
        alpha_summary = (
            alpha_rows.groupby(["global_block_index", "alpha"], as_index=False)
            .agg(
                semantic_gain=("semantic_gain", "mean"),
                preservation_cost=("preservation_cost", "mean"),
                positive_rate=("semantic_gain", lambda values: float((values > 0).mean())),
                bad_image_rate=("bad_image", "mean"),
            )
        )
        alpha_summary.to_csv(run_root / "alpha_summary.csv", index=False)
    plot_curves(summary, frame, config, run_root / "plots")
    print(json.dumps(selected, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
