from scripts.expected_counts import compute_counts


def test_counts_use_runtime_blocks_and_observed_dataset_rows() -> None:
    config = {
        "dataset": {"categories": ["color", "lighting"], "pilot_per_category": 5},
        "inference": {"seeds": [1, 2, 3]},
        "probing": {"stage2_blocks": 2, "stage3_blocks": 1},
    }
    rows = []
    for category in config["dataset"]["categories"]:
        rows.extend({"category": category, "split": "discovery"} for _ in range(7))
        rows.extend({"category": category, "split": "heldout"} for _ in range(2))
    counts = compute_counts(config, {"total_block_count": 3}, rows)
    assert counts["pilot_samples"] == 10
    assert counts["pilot_stage1_jobs"] == 40
    assert counts["formal_baseline_jobs"] == 42
    assert counts["formal_enhance_jobs"] == 126
    assert counts["formal_disable_jobs"] == 84
    assert counts["formal_remove_jobs"] == 42
