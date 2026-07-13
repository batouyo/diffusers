from pathlib import Path

from scripts.make_calibration_bundle import bundle_current, row_identity_hash
from scripts.make_calibration_manifest import balanced_category_items


def item(sample: str, mode: str, block: int | None = None):
    return Path(f"{sample}_{mode}_{block}.json"), {
        "sample_id": sample,
        "mode": mode,
        "global_block_index": block,
    }


def test_calibration_halves_each_have_one_baseline_and_four_enhanced():
    items = [item("base-a", "baseline"), item("base-b", "baseline")]
    items += [item(f"sample-{index}", "enhance_text", index) for index in range(20)]
    selected = balanced_category_items(items)
    assert len(selected) == 10
    for subset in ["prompt_calibration", "locked_validation"]:
        rows = [meta for _, meta, observed_subset in selected if observed_subset == subset]
        assert len(rows) == 5
        assert sum(row["mode"] == "baseline" for row in rows) == 1
        assert sum(row["mode"] == "enhance_text" for row in rows) == 4


def test_blinded_row_identity_ignores_only_human_response_fields():
    row = {
        "calibration_id": "cal_000",
        "subset": "locked_validation",
        "category": "object",
        "instruction": "change it",
        "target_description": "changed",
        "source_image": "images/source.png",
        "output_image": "images/output.png",
        "human_score_0_to_4": "",
        "human_evidence": "",
    }
    original = row_identity_hash(row)
    row["human_score_0_to_4"] = 4
    row["human_evidence"] = "visible"
    assert row_identity_hash(row) == original
    row["instruction"] = "different"
    assert row_identity_hash(row) != original


def test_missing_bundle_is_not_current(tmp_path):
    assert not bundle_current(tmp_path)
