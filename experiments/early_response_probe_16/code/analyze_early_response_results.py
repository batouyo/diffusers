"""Independent descriptive audit of the completed early-response probe."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).parent / "full_results"
MODES = ["first_only", "first_two"]


def finite_summary(series: pd.Series) -> dict[str, float | int]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    values = values[np.isfinite(values)]
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)) if values.size else float("nan"),
        "median": float(np.median(values)) if values.size else float("nan"),
        "min": float(np.min(values)) if values.size else float("nan"),
        "max": float(np.max(values)) if values.size else float("nan"),
    }


def main() -> None:
    branch = pd.read_csv(ROOT / "early_response_results.csv")
    corr = pd.read_csv(ROOT / "sample_correlations.csv")
    anti = pd.read_csv(ROOT / "antithetic_response_stats.csv")
    pair = pd.read_csv(ROOT / "antithetic_pair_responses.csv")
    ranges = pd.read_csv(ROOT / "sample_response_ranges.csv")
    reversal = pd.read_csv(ROOT / "sample_reversal_stats.csv")
    subgroup = pd.read_csv(ROOT / "subgroup_summary.csv")
    config = json.loads((ROOT / "run_config.json").read_text(encoding="utf-8"))

    print("VALIDATION")
    print({
        "branch_rows": len(branch),
        "unique_samples": branch.sample_id.nunique(),
        "sample_mode_units": branch[["sample_id", "timestep_mode"]].drop_duplicates().shape[0],
        "signed_per_unit_min": int(branch.groupby(["sample_id", "timestep_mode"]).size().min()),
        "signed_per_unit_max": int(branch.groupby(["sample_id", "timestep_mode"]).size().max()),
        "pair_rows": len(pair),
        "corr_rows": len(corr),
        "degenerate_units": int(corr.degenerate_clip.sum()),
        "nonfinite_branch_scores": int((~np.isfinite(branch[["q_early_clip", "q_final_clip"]].to_numpy(float))).sum()),
        "dino_status": config.get("encoder_status", {}).get("dino", {}),
    })

    print("\nMODE_SUMMARY")
    mode_rows = []
    for mode in MODES:
        c = corr[corr.timestep_mode == mode]
        a = anti[anti.timestep_mode == mode]
        r = ranges[ranges.timestep_mode == mode]
        v = reversal[reversal.timestep_mode == mode]
        mode_rows.append({
            "mode": mode,
            "mean_spearman": c.spearman_clip.mean(),
            "median_spearman": c.spearman_clip.median(),
            "median_pearson": c.pearson_clip.median(),
            "positive_spearman_ratio": (c.spearman_clip > 0).mean(),
            "spearman_ge_0.3_ratio": (c.spearman_clip >= .3).mean(),
            "spearman_ge_0.5_ratio": (c.spearman_clip >= .5).mean(),
            "median_reversal_ratio": v.reversal_ratio_clip.median(),
            "pooled_reversal_ratio": v.reversal_pairs_clip.sum() / v.valid_unordered_pairs_clip.sum(),
            "median_sign_agreement": a.sign_agreement_ratio_clip.median(),
            "pooled_sign_agreement": (pair.loc[(pair.timestep_mode == mode) & (pair.valid_clip == 1), "sign_agree_clip"].mean()),
            "median_delta_spearman": a.delta_spearman_clip.median(),
            "median_delta_pearson": a.delta_pearson_clip.median(),
            "median_range_early": r.range_q_early_clip.median(),
            "median_range_final": r.range_q_final_clip.median(),
            "median_abs_delta_early": r.median_abs_delta_q_early_clip.median(),
            "median_abs_delta_final": r.median_abs_delta_q_final_clip.median(),
            "valid_antithetic_pairs": int(a.valid_pairs_clip.sum()),
        })
    print(pd.DataFrame(mode_rows).to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nPAIRED_MODE_COMPARISON")
    metrics = [
        (corr, "spearman_clip", "higher"),
        (corr, "pearson_clip", "higher"),
        (anti, "sign_agreement_ratio_clip", "higher"),
        (anti, "delta_spearman_clip", "higher"),
        (reversal, "reversal_ratio_clip", "lower"),
        (ranges, "range_q_early_clip", "context"),
        (ranges, "range_q_final_clip", "context"),
    ]
    paired_rows = []
    for frame, metric, preferred in metrics:
        wide = frame.pivot(index="sample_id", columns="timestep_mode", values=metric)
        difference = wide.first_two - wide.first_only
        paired_rows.append({
            "metric": metric,
            "preferred": preferred,
            "median_first_only": wide.first_only.median(),
            "median_first_two": wide.first_two.median(),
            "median_difference_two_minus_only": difference.median(),
            "first_two_higher_samples": int((difference > 0).sum()),
            "first_only_higher_samples": int((difference < 0).sum()),
            "ties": int((difference == 0).sum()),
        })
    print(pd.DataFrame(paired_rows).to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nSUBGROUP")
    print(subgroup.sort_values(["timestep_mode", "subgroup"]).to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nPER_SAMPLE_OVERALL")
    per = corr[["sample_id", "subgroup", "timestep_mode", "spearman_clip", "pearson_clip"]].merge(
        anti[["sample_id", "timestep_mode", "sign_agreement_ratio_clip", "delta_spearman_clip"]],
        on=["sample_id", "timestep_mode"],
    ).merge(
        reversal[["sample_id", "timestep_mode", "reversal_ratio_clip"]],
        on=["sample_id", "timestep_mode"],
    ).merge(
        ranges[["sample_id", "timestep_mode", "range_q_early_clip", "range_q_final_clip"]],
        on=["sample_id", "timestep_mode"],
    )
    print(per.sort_values(["timestep_mode", "spearman_clip"], ascending=[True, False]).to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nEXTREMES")
    for mode in MODES:
        subset = per[per.timestep_mode == mode]
        print(mode)
        print("best_spearman", subset.nlargest(3, "spearman_clip")[["sample_id", "spearman_clip", "reversal_ratio_clip", "sign_agreement_ratio_clip"]].to_dict("records"))
        print("worst_spearman", subset.nsmallest(3, "spearman_clip")[["sample_id", "spearman_clip", "reversal_ratio_clip", "sign_agreement_ratio_clip"]].to_dict("records"))
        print("smallest_final_range", subset.nsmallest(3, "range_q_final_clip")[["sample_id", "range_q_early_clip", "range_q_final_clip", "spearman_clip"]].to_dict("records"))

    print("\nQ_DISTRIBUTION")
    for mode in MODES:
        subset = branch[branch.timestep_mode == mode]
        print(mode, {
            "q_early": finite_summary(subset.q_early_clip),
            "q_final": finite_summary(subset.q_final_clip),
        })


if __name__ == "__main__":
    main()
