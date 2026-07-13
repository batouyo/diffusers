from __future__ import annotations

from evaluators import parse_json_object, reusable_evaluation


def test_parse_json_corrects_inconsistent_sum() -> None:
    value = parse_json_object(
        '{"target_present": 2, "correct_object": 1, "localized_as_requested": 1, "score_0_to_4": 2}'
    )
    assert value["score_0_to_4"] == 4
    assert value["score_corrected"] is True


def test_failed_vlm_cache_is_not_reusable() -> None:
    meta = {"output_sha256": "pixels"}
    prior = {
        "output_sha256": "pixels",
        "evaluation_hash": "protocol",
        "dino_similarity": 0.9,
        "lpips_distance": 0.1,
        "quality": {"finite": True},
        "s_edit": None,
        "vlm_parse_ok": False,
    }
    assert not reusable_evaluation(prior, meta, "protocol", require_vlm=True)
    prior.update({"s_edit": 0.75, "vlm_parse_ok": True})
    assert reusable_evaluation(prior, meta, "protocol", require_vlm=True)
    assert not reusable_evaluation(prior, meta, "different-protocol", require_vlm=True)
