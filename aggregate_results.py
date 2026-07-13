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
    base = frame[frame["mode"] == "baseline"][keys + ["s_edit", "s_preserve"]].drop_duplicates(keys)
    base = base.rename(columns={"s_edit": "s_edit_base", "s_preserve": "s_preserve_base"})
    merged = frame.merge(base, on=keys, how="left")
    merged["semantic_gain"] = np.where(
        merged["mode"] == "enhance_text", merged["s_edit"] - merged["s_edit_base"], np.nan
    )
    merged["semantic_drop"] = np.where(
        merged["mode"] == "disable_text", merged["s_edit_base"] - merged["s_edit"], np.nan
    )
    merged["preservation_cost"] = np.where(
        merged["mode"] == "enhance_text", merged["s_preserve_base"] - merged["s_preserve"], np.nan
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


def summarize_blocks(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    stats = config["statistics"]
    summaries = []
    block_rows = frame[frame["global_block_index"].notna()].copy()
    if block_rows.empty:
        return pd.DataFrame()
    for global_index, group in block_rows.groupby("global_block_index"):
        enhance = group[group["mode"] == "enhance_text"]
        disable = group[group["mode"] == "disable_text"]
        gain = cluster_macro(enhance, "semantic_gain")
        drop = cluster_macro(disable, "semantic_drop")
        cost = cluster_macro(enhance, "preservation_cost")
        gain_lo, gain_hi = stratified_bootstrap(
            enhance, "semantic_gain", stats["bootstrap_samples"], stats["random_seed"] + int(global_index)
        )
        drop_lo, drop_hi = stratified_bootstrap(
            disable, "semantic_drop", stats["bootstrap_samples"], stats["random_seed"] + 1000 + int(global_index)
        )
        category_gain = (
            enhance.groupby(["category", "sample_id"])["semantic_gain"].mean().groupby("category").mean().to_dict()
        )
        per_sample_gain = enhance.groupby("sample_id")["semantic_gain"].mean()
        per_seed_gain = enhance.groupby("seed")["semantic_gain"].mean().to_dict()
        address = group["block_address"].dropna().iloc[0] if group["block_address"].notna().any() else {}
        positive_categories = sum(value > 0 for value in category_gain.values())
        summaries.append(
            {
                "global_block_index": int(global_index),
                "local_block_index": address.get("local_index") if isinstance(address, dict) else None,
                "block_type": address.get("block_type") if isinstance(address, dict) else None,
                "semantic_gain": gain,
                "semantic_gain_ci_low": gain_lo,
                "semantic_gain_ci_high": gain_hi,
                "semantic_drop": drop,
                "semantic_drop_ci_low": drop_lo,
                "semantic_drop_ci_high": drop_hi,
                "preservation_cost": cost,
                "positive_sample_rate": float((per_sample_gain > 0).mean()) if len(per_sample_gain) else np.nan,
                "positive_categories": positive_categories,
                "all_seed_means_positive": bool(per_seed_gain) and all(value > 0 for value in per_seed_gain.values()),
                "bad_image_rate": float(group["bad_image"].mean()),
                "category_gain_json": json.dumps(category_gain, sort_keys=True),
            }
        )
    return pd.DataFrame(summaries).sort_values("global_block_index")


def select_candidates(summary: pd.DataFrame, frame: pd.DataFrame, config: dict) -> dict:
    stats = config["statistics"]
    if summary.empty:
        return {"status": "insufficient_data", "universal_blocks": [], "category_specific_blocks": []}
    eligible = summary[
        (summary["semantic_gain"] >= stats["universal_gain_min"])
        & (summary["semantic_gain_ci_low"] > 0)
        & (summary["positive_categories"] >= stats["universal_positive_categories"])
        & (summary["positive_sample_rate"] >= stats["universal_positive_sample_rate"])
        & summary["all_seed_means_positive"]
        & (summary["semantic_drop"] >= stats["universal_drop_min"])
        & (summary["semantic_drop_ci_low"] > 0)
        & (summary["bad_image_rate"] <= stats["bad_image_rate_max"])
    ].copy()
    eligible = eligible.sort_values(["semantic_gain_ci_low", "semantic_gain"], ascending=False)
    universal = []
    for _, row in eligible.iterrows():
        index = int(row["global_block_index"])
        redundant = any(abs(index - prior) <= stats["adjacent_distance"] for prior in universal)
        if not redundant:
            universal.append(index)
        if len(universal) >= config["probing"]["max_candidates"]:
            break

    category_specific = {}
    enhance = frame[frame["mode"] == "enhance_text"]
    for (global_index, category), group in enhance.groupby(["global_block_index", "category"]):
        per_sample = group.groupby("sample_id")["semantic_gain"].mean().dropna()
        if len(per_sample) < 2:
            continue
        mean = float(per_sample.mean())
        low, _ = stratified_bootstrap(group, "semantic_gain", stats["bootstrap_samples"], stats["random_seed"] + int(global_index))
        if mean >= stats["category_gain_min"] and low > 0 and float((per_sample > 0).mean()) >= stats["category_positive_sample_rate"]:
            current = category_specific.get(category)
            candidate = {"global_block_index": int(global_index), "gain": mean, "ci_low": low}
            if current is None or candidate["ci_low"] > current["ci_low"]:
                category_specific[category] = candidate
    selected = sorted(set(universal) | {item["global_block_index"] for item in category_specific.values()})
    if len(selected) > config["probing"]["max_candidates"]:
        selected = selected[: config["probing"]["max_candidates"]]
    return {
        "status": "selected" if selected else "no_go",
        "universal_blocks": universal,
        "category_specific_blocks": category_specific,
        "selected_global_blocks": selected,
        "candidate_count": len(selected),
        "note": "TexTailor FLUX.1-Dev indices were not used for selection.",
    }


def plot_curves(summary: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        return
    sns.set_theme(style="whitegrid")
    for column, name in [
        ("semantic_gain", "semantic_gain"),
        ("semantic_drop", "semantic_drop"),
        ("preservation_cost", "preservation_cost"),
    ]:
        plt.figure(figsize=(12, 4))
        sns.lineplot(data=summary, x="global_block_index", y=column, marker="o")
        plt.axvline(18.5, color="black", linestyle="--", label="double/single boundary")
        plt.axhline(0, color="gray", linewidth=1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output / f"{name}_vs_global_block.png", dpi=180)
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
    selected = select_candidates(summary, frame, config)
    (run_root / "selected_blocks.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_curves(summary, run_root / "plots")
    print(json.dumps(selected, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
