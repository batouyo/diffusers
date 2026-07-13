"""Validate the 80-example human calibration and compare it with the VLM."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


def main() -> None:
    root = Path("/data15/hyp/project_storage/flux-kontext-block-probing/main_512/calibration")
    labels = pd.read_csv(root / "blinded_labels.csv")
    if labels["human_score_0_to_4"].isna().any():
        missing = labels.loc[labels["human_score_0_to_4"].isna(), "calibration_id"].tolist()
        raise RuntimeError(f"human labels incomplete: {missing[:10]} ({len(missing)} missing)")
    if not labels["human_score_0_to_4"].between(0, 4).all():
        raise ValueError("human scores must be integers from 0 to 4")
    key = json.loads((root / "sealed_key.json").read_text(encoding="utf-8"))
    labels["vlm_score_0_to_4"] = labels["calibration_id"].map(lambda value: key[value]["vlm_score_0_to_4"])
    results = {}
    for subset, frame in [("all", labels), *labels.groupby("subset")]:
        rho, p_value = spearmanr(frame["human_score_0_to_4"], frame["vlm_score_0_to_4"])
        results[subset] = {"n": len(frame), "spearman": float(rho), "p_value": float(p_value)}
    locked = results["locked_validation"]
    results["gate_pass"] = bool(locked["spearman"] >= 0.7)
    (root / "calibration_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    if not results["gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
