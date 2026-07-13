"""Validate the 80-example human calibration and compare it with the VLM."""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from make_calibration_bundle import PROTOCOL_VERSION, row_identity_hash
from probe_flux_kontext_blocks import file_sha256


def main() -> None:
    root = Path("/data15/hyp/project_storage/flux-kontext-block-probing/main_512/calibration")
    labels = pd.read_csv(root / "blinded_labels.csv")
    bundle = json.loads((root / "bundle_manifest.json").read_text(encoding="utf-8"))
    expected_ids = {f"cal_{index:03d}" for index in range(80)}
    observed_ids = set(labels["calibration_id"].astype(str))
    if len(labels) != 80 or not labels["calibration_id"].is_unique or observed_ids != expected_ids:
        raise RuntimeError("calibration file must contain each blinded ID cal_000 through cal_079 exactly once")
    subset_counts = labels["subset"].value_counts().to_dict()
    if subset_counts != {"prompt_calibration": 40, "locked_validation": 40}:
        raise RuntimeError(f"unexpected calibration subset counts: {subset_counts}")
    if bundle.get("status") != "complete" or bundle.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("calibration bundle manifest is missing or uses a stale protocol")
    if set(bundle.get("rows", {})) != expected_ids:
        raise RuntimeError("calibration bundle manifest IDs do not match the blinded labels")
    for _, row in labels.iterrows():
        calibration_id = str(row["calibration_id"])
        expected = bundle["rows"][calibration_id]
        row_dict = row.to_dict()
        if row_identity_hash(row_dict) != expected["immutable_row_sha256"]:
            raise RuntimeError(f"immutable blinded fields changed for {calibration_id}")
        if file_sha256(root / str(row["source_image"])) != expected["source_sha256"]:
            raise RuntimeError(f"source checksum mismatch for {calibration_id}")
        if file_sha256(root / str(row["output_image"])) != expected["output_sha256"]:
            raise RuntimeError(f"output checksum mismatch for {calibration_id}")
    if labels["human_score_0_to_4"].isna().any():
        missing = labels.loc[labels["human_score_0_to_4"].isna(), "calibration_id"].tolist()
        raise RuntimeError(f"human labels incomplete: {missing[:10]} ({len(missing)} missing)")
    scores = pd.to_numeric(labels["human_score_0_to_4"], errors="coerce")
    if scores.isna().any() or not scores.between(0, 4).all() or not (scores % 1 == 0).all():
        raise ValueError("human scores must be integers from 0 to 4")
    labels["human_score_0_to_4"] = scores.astype(int)
    if labels["human_evidence"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("every human score must include brief visible evidence")
    key = json.loads((root / "sealed_key.json").read_text(encoding="utf-8"))
    if set(key) != expected_ids:
        raise RuntimeError("sealed key IDs do not exactly match the blinded calibration file")
    if any(key[item]["output_sha256"] != bundle["rows"][item]["output_sha256"] for item in expected_ids):
        raise RuntimeError("sealed key output hashes do not match the calibration bundle")
    labels["vlm_score_0_to_4"] = labels["calibration_id"].map(lambda value: key[value]["vlm_score_0_to_4"])
    results = {}
    for subset, frame in [("all", labels), *labels.groupby("subset")]:
        rho, p_value = spearmanr(frame["human_score_0_to_4"], frame["vlm_score_0_to_4"])
        results[subset] = {
            "n": len(frame),
            "spearman": float(rho) if math.isfinite(float(rho)) else None,
            "p_value": float(p_value) if math.isfinite(float(p_value)) else None,
        }
    locked = results["locked_validation"]
    results["gate"] = {"metric": "locked_validation_spearman", "threshold": 0.7}
    results["gate_pass"] = bool(locked["spearman"] is not None and locked["spearman"] >= 0.7)
    results["labels_sha256"] = hashlib.sha256((root / "blinded_labels.csv").read_bytes()).hexdigest()
    results["scoring_protocol_hash"] = file_sha256(__file__)
    (root / "calibration_report.json").write_text(
        json.dumps(results, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))
    if not results["gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
