from __future__ import annotations

import json

import pandas as pd
import pytest
import torch

from probe_flux_kontext_blocks import file_sha256
from scripts.aggregate_joint import load_joint, validate_joint_pairing
from scripts.run_joint_validation import arms, joint_hash, random_controls


class ToyTransformer:
    def __init__(self):
        self.transformer_blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(4)])
        self.single_transformer_blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(6)])


def test_random_controls_match_stream_counts_and_exclude_candidates():
    candidates = [0, 2, 5]
    controls = random_controls(10, 4, candidates, count=5, seed=7)
    assert len(controls) == len(set(controls)) == 5
    for control in controls:
        assert len(control) == 3
        assert sum(index < 4 for index in control) == 2
        assert not set(control) & set(candidates)


def test_arms_include_required_controls_and_budget_match():
    values = arms(
        ToyTransformer(),
        [0, 5],
        alpha=1.5,
        random_sets=3,
        seed=9,
        textailor_control_blocks=[2, 7, 12, 17, 22],
    )
    lookup = {name: (mode, blocks, alpha) for name, mode, blocks, alpha in values}
    assert "candidate_combo" in lookup
    assert "all_blocks" in lookup
    assert "all_blocks_budget_matched" in lookup
    assert "textailor_flux1dev_control" in lookup
    assert "candidate_disable_g000" in lookup
    assert len([name for name in lookup if name.startswith("random_")]) == 3
    assert lookup["candidate_disable_g000"][0] == "disable_text"
    assert lookup["all_blocks_budget_matched"][2] == 1.0 + 2 / 10 * 0.5


def test_joint_hash_changes_when_random_arm_definition_changes():
    config = {
        "project": {"dataset_manifest": "missing-test-manifest.jsonl"},
        "inference": {"seeds": [42, 1234, 2025]},
    }
    common = [("baseline", "none", tuple(), 1.0)]
    first = common + [("random_00", "enhance_text", (1, 4), 1.5)]
    second = common + [("random_00", "enhance_text", (2, 5), 1.5)]
    first_hash = joint_hash(config, [0, 6], 1.5, first, "heldout", 512)
    second_hash = joint_hash(config, [0, 6], 1.5, second, "heldout", 512)
    assert first_hash != second_hash


def test_joint_loader_requires_current_evaluation_and_matching_image_checksum(tmp_path):
    folder = tmp_path / "joint" / "heldout" / "object" / "sample"
    folder.mkdir(parents=True)
    image = folder / "seed42_baseline.png"
    image.write_bytes(b"joint-image")
    metadata_path = image.with_suffix(".json")
    metadata = {
        "status": "complete",
        "output_path": str(image),
        "output_sha256": file_sha256(image),
        "sample_id": "sample",
        "seed": 42,
        "resolution": 512,
        "arm": "baseline",
        "category": "object",
        "joint_hash": "fingerprint",
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    evaluation_path = metadata_path.with_suffix(".eval.json")
    evaluation = {
        "output_sha256": metadata["output_sha256"],
        "evaluation_hash": "current-evaluator",
        "dino_similarity": 0.9,
        "lpips_distance": 0.1,
        "quality": {"finite": True},
        "vlm_parse_ok": True,
        "s_edit": 2.0,
    }
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    assert len(load_joint(tmp_path, "current-evaluator")) == 1

    evaluation["evaluation_hash"] = "stale-evaluator"
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    with pytest.raises(RuntimeError, match="stale or incomplete"):
        load_joint(tmp_path, "current-evaluator")


def test_joint_pairing_rejects_latent_mismatch():
    common = {"sample_id": "sample", "seed": 42, "resolution": 512, "source_sha256": "source"}
    frame = pd.DataFrame(
        [
            {**common, "arm": "baseline", "latent_hash": "latent-a"},
            {**common, "arm": "candidate_combo", "latent_hash": "latent-b"},
        ]
    )
    with pytest.raises(RuntimeError, match="latent_hash mismatch"):
        validate_joint_pairing(frame)
