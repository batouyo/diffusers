"""Verify the exact pilot job/evaluation set and write a hash-bound stage sentinel."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from evaluators import evaluation_hash, reusable_evaluation
from expected_counts import ROOT, load_counts
from probe_flux_kontext_blocks import experiment_hash, load_config


SENTINEL_NAME = "pilot_pipeline_complete.json"


def sentinel_current(root: Path = ROOT) -> bool:
    config = load_config(root / "probe_config.yaml")
    run_root = Path(config["project"]["output_root"]) / config["project"]["run_id"]
    path = run_root / SENTINEL_NAME
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        value.get("status") == "complete"
        and value.get("expected_jobs") == load_counts(root)["pilot_stage1_jobs"]
        and value.get("generation_hash") == experiment_hash(config)
        and value.get("evaluation_hash") == evaluation_hash(config)
        and value.get("valid_evaluations") == value.get("expected_jobs")
    )


def verify(root: Path = ROOT, *, write: bool = True) -> dict:
    config = load_config(root / "probe_config.yaml")
    run_root = Path(config["project"]["output_root"]) / config["project"]["run_id"]
    dataset = [
        json.loads(line)
        for line in Path(config["project"]["dataset_manifest"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pilot_ids = set()
    for category in config["dataset"]["categories"]:
        rows = sorted(
            (row for row in dataset if row["split"] == "discovery" and row["category"] == category),
            key=lambda row: row["id"],
        )
        pilot_ids.update(row["id"] for row in rows[: config["dataset"]["pilot_per_category"]])

    expected_jobs = load_counts(root)["pilot_stage1_jobs"]
    generation_hash = experiment_hash(config)
    evaluator_hash = evaluation_hash(config)
    records = []
    for meta_path in (run_root / "images").rglob("*.json"):
        if meta_path.name.endswith(".eval.json"):
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            meta.get("status") == "complete"
            and meta.get("sample_id") in pilot_ids
            and meta.get("seed") == config["inference"]["pilot_seed"]
            and meta.get("mode") in {"baseline", "enhance_text"}
            and np.isclose(float(meta.get("alpha")), float(config["inference"]["alpha"]))
        ):
            records.append((meta_path, meta))
    keys = {
        (meta["sample_id"], meta["seed"], meta["mode"], meta.get("global_block_index"))
        for _, meta in records
    }
    if len(records) != expected_jobs or len(keys) != expected_jobs:
        raise RuntimeError(
            f"pilot generation incomplete or duplicated: records={len(records)} unique={len(keys)} expected={expected_jobs}"
        )
    stale_generation = [meta["output_path"] for _, meta in records if meta.get("config_hash") != generation_hash]
    if stale_generation:
        raise RuntimeError(f"pilot contains {len(stale_generation)} stale generation records")
    valid_evaluations = 0
    for meta_path, meta in records:
        eval_path = meta_path.with_suffix(".eval.json")
        if not eval_path.exists():
            continue
        try:
            evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if reusable_evaluation(evaluation, meta, evaluator_hash, require_vlm=True):
            valid_evaluations += 1
    if valid_evaluations != expected_jobs:
        raise RuntimeError(f"pilot current valid evaluations={valid_evaluations}, expected={expected_jobs}")
    aggregate_files = ["raw_metrics.csv", "block_summary.csv", "selected_blocks.json"]
    missing = [name for name in aggregate_files if not (run_root / name).exists()]
    if missing:
        raise RuntimeError(f"pilot aggregate artifacts missing: {missing}")
    result = {
        "status": "complete",
        "expected_jobs": expected_jobs,
        "generation_records": len(records),
        "valid_evaluations": valid_evaluations,
        "generation_hash": generation_hash,
        "evaluation_hash": evaluator_hash,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "aggregate_files": aggregate_files,
    }
    if write:
        path = run_root / SENTINEL_NAME
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
        temporary.replace(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(write=not args.check_only), indent=2))


if __name__ == "__main__":
    main()
