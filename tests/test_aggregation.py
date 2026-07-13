from __future__ import annotations

import numpy as np
import pandas as pd

from aggregate_results import benjamini_hochberg, paired_metrics, select_candidates, summarize_blocks


def test_paired_metrics_include_disable_and_remove_effects() -> None:
    common = {
        "sample_id": "s1",
        "seed": 42,
        "resolution": 512,
        "latent_hash": "same-latent",
        "source_sha256": "same-source",
    }
    frame = pd.DataFrame(
        [
            {**common, "mode": "baseline", "s_edit": 0.5, "s_preserve": 0.9, "lpips_distance": 0.1},
            {**common, "mode": "enhance_text", "s_edit": 0.8, "s_preserve": 0.85, "lpips_distance": 0.2},
            {**common, "mode": "disable_text", "s_edit": 0.2, "s_preserve": 0.88, "lpips_distance": 0.12},
            {**common, "mode": "remove_block", "s_edit": 0.1, "s_preserve": 0.6, "lpips_distance": 0.4},
        ]
    )
    result = paired_metrics(frame).set_index("mode")
    assert np.isclose(result.loc["enhance_text", "semantic_gain"], 0.3)
    assert np.isclose(result.loc["disable_text", "semantic_drop"], 0.3)
    assert np.isclose(result.loc["remove_block", "removal_edit_drop"], 0.4)
    assert np.isclose(result.loc["enhance_text", "preservation_cost"], 0.05)
    assert np.isclose(result.loc["remove_block", "removal_preservation_cost"], 0.3)


def test_paired_metrics_reject_latent_or_source_mismatch() -> None:
    common = {
        "sample_id": "s1",
        "seed": 42,
        "resolution": 512,
        "source_sha256": "source",
        "s_preserve": 0.9,
        "lpips_distance": 0.1,
    }
    frame = pd.DataFrame(
        [
            {**common, "mode": "baseline", "s_edit": 0.5, "latent_hash": "baseline-latent"},
            {**common, "mode": "enhance_text", "s_edit": 0.8, "latent_hash": "different-latent"},
        ]
    )
    with np.testing.assert_raises_regex(RuntimeError, "latent_hash mismatch"):
        paired_metrics(frame)


def test_candidate_selection_enforces_preservation_and_correlated_adjacency_dedup() -> None:
    summary = pd.DataFrame(
        [
            {
                "global_block_index": 0,
                "semantic_gain": 0.12,
                "semantic_gain_ci_low": 0.06,
                "semantic_gain_q_bh": 0.01,
                "semantic_drop": 0.10,
                "semantic_drop_ci_low": 0.04,
                "semantic_drop_q_bh": 0.01,
                "preservation_cost_ci_high": 0.015,
                "bad_image_rate": 0.0,
                "positive_categories": 1,
                "positive_sample_rate": 1.0,
                "all_seed_means_positive": True,
                "category_gain_json": '{"color": 0.12}',
            },
            {
                "global_block_index": 1,
                "semantic_gain": 0.11,
                "semantic_gain_ci_low": 0.05,
                "semantic_gain_q_bh": 0.01,
                "semantic_drop": 0.09,
                "semantic_drop_ci_low": 0.03,
                "semantic_drop_q_bh": 0.01,
                "preservation_cost_ci_high": 0.014,
                "bad_image_rate": 0.0,
                "positive_categories": 1,
                "positive_sample_rate": 1.0,
                "all_seed_means_positive": True,
                "category_gain_json": '{"color": 0.11}',
            },
            {
                "global_block_index": 4,
                "semantic_gain": 0.20,
                "semantic_gain_ci_low": 0.08,
                "semantic_gain_q_bh": 0.01,
                "semantic_drop": 0.12,
                "semantic_drop_ci_low": 0.05,
                "semantic_drop_q_bh": 0.01,
                "preservation_cost_ci_high": 0.08,
                "bad_image_rate": 0.0,
                "positive_categories": 1,
                "positive_sample_rate": 1.0,
                "all_seed_means_positive": True,
                "category_gain_json": '{"color": 0.20}',
            },
        ]
    )
    rows = []
    for block, scale in [(0, 1.0), (1, 0.95), (4, 1.2)]:
        for sample_index in range(6):
            for seed in [42, 1234, 2025]:
                rows.append(
                    {
                        "mode": "enhance_text",
                        "alpha": 1.5,
                        "global_block_index": block,
                        "category": "color",
                        "sample_id": f"s{sample_index}",
                        "seed": seed,
                        "semantic_gain": scale * (0.08 + 0.01 * sample_index),
                    }
                )
    config = {
        "inference": {"alpha": 1.5, "seeds": [42, 1234, 2025]},
        "dataset": {"categories": ["color"]},
        "probing": {"stage2_blocks": 3, "max_candidates": 8},
        "statistics": {
            "bootstrap_samples": 200,
            "random_seed": 7,
            "universal_gain_min": 0.05,
            "universal_drop_min": 0.05,
            "universal_positive_categories": 1,
            "universal_positive_sample_rate": 0.6,
            "category_gain_min": 0.05,
            "category_positive_sample_rate": 0.6,
            "bad_image_rate_max": 0.01,
            "dino_noninferiority_margin": -0.02,
            "adjacent_distance": 2,
            "redundancy_spearman": 0.9,
            "bh_q": 0.05,
        },
    }
    selected = select_candidates(summary, pd.DataFrame(rows), config)
    assert selected["universal_blocks"] == [0]
    assert selected["selected_global_blocks"] == [0]
    assert 4 not in selected["selected_global_blocks"]
    assert selected["preservation_cost_limit"] == 0.02
    assert selected["category_specific_blocks"]["color"]["category_gain_q_bh"] <= 0.05


def test_benjamini_hochberg_is_monotone_and_preserves_missing() -> None:
    values = pd.Series([0.01, 0.04, 0.03, np.nan], index=[10, 11, 12, 13])
    adjusted = benjamini_hochberg(values)
    assert np.isclose(adjusted.loc[10], 0.03)
    assert np.isclose(adjusted.loc[11], 0.04)
    assert np.isclose(adjusted.loc[12], 0.04)
    assert np.isnan(adjusted.loc[13])


def test_stage2_is_strict_global_ranking_not_category_order_injection() -> None:
    summary = pd.DataFrame(
        [
            {
                "global_block_index": index,
                "semantic_gain": 1.0 - index / 100,
                "semantic_gain_ci_low": 0.9 - index / 100,
                "semantic_gain_q_bh": 1.0,
                "semantic_drop": np.nan,
                "semantic_drop_ci_low": np.nan,
                "semantic_drop_q_bh": np.nan,
                "preservation_cost_ci_high": 0.0,
                "bad_image_rate": 0.0,
                "positive_categories": 0,
                "positive_sample_rate": 0.0,
                "all_seed_means_positive": False,
                "category_gain_json": '{"late_category": 99.0}' if index == 15 else "{}",
            }
            for index in range(16)
        ]
    )
    frame = pd.DataFrame(
        columns=["mode", "alpha", "global_block_index", "category", "sample_id", "seed", "semantic_gain"]
    )
    config = {
        "inference": {"alpha": 1.5, "seeds": [42, 1234, 2025]},
        "dataset": {"categories": ["late_category"]},
        "probing": {"stage2_blocks": 15, "max_candidates": 8},
        "statistics": {
            "bootstrap_samples": 20,
            "random_seed": 7,
            "universal_gain_min": 2.0,
            "universal_drop_min": 2.0,
            "universal_positive_categories": 1,
            "universal_positive_sample_rate": 0.6,
            "category_gain_min": 2.0,
            "category_positive_sample_rate": 0.6,
            "bad_image_rate_max": 0.01,
            "dino_noninferiority_margin": -0.02,
            "adjacent_distance": 2,
            "redundancy_spearman": 0.9,
            "bh_q": 0.05,
        },
    }
    selected = select_candidates(summary, frame, config)
    assert selected["stage2_blocks"] == list(range(15))
    assert 15 not in selected["stage2_blocks"]


def test_block_summary_reports_positive_sample_and_category_counts() -> None:
    rows = []
    for sample_id, category, gain in [("s1", "color", 0.2), ("s2", "shape", -0.1)]:
        rows.append(
            {
                "global_block_index": 0,
                "mode": "enhance_text",
                "alpha": 1.5,
                "category": category,
                "sample_id": sample_id,
                "seed": 42,
                "semantic_gain": gain,
                "semantic_drop": np.nan,
                "removal_edit_drop": np.nan,
                "preservation_cost": 0.0,
                "removal_preservation_cost": np.nan,
                "bad_image": False,
                "block_address": {"global_index": 0, "local_index": 0, "block_type": "double"},
            }
        )
    config = {
        "inference": {"alpha": 1.5},
        "statistics": {
            "bootstrap_samples": 20,
            "random_seed": 7,
            "dino_noninferiority_margin": -0.02,
            "bad_image_rate_max": 0.01,
            "universal_gain_min": 0.05,
            "universal_drop_min": 0.05,
        },
    }
    summary = summarize_blocks(pd.DataFrame(rows), config).iloc[0]
    assert summary["positive_sample_count"] == 1
    assert summary["sample_count"] == 2
    assert summary["positive_categories"] == 1
    assert summary["category_count"] == 2
