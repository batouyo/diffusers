"""Verify pilot top-15 disable and top-10 remove stages before downstream work."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

from evaluators import evaluation_hash, reusable_evaluation
from expected_counts import ROOT, load_counts
from probe_flux_kontext_blocks import experiment_hash, file_sha256, load_config


SENTINEL_NAME = "pilot_followup_complete.json"


def _state(root: Path):
    config = load_config(root / "probe_config.yaml")
    derived = copy.deepcopy(config)
    derived["project"]["dataset_manifest"] = str(
        Path(config["project"]["output_root"]) / "preflight" / "pilot_dataset.jsonl"
    )
    derived["inference"]["seeds"] = [derived["inference"]["pilot_seed"]]
    run_root = Path(config["project"]["output_root"]) / config["project"]["run_id"]
    selection = json.loads((run_root / "selected_blocks.json").read_text(encoding="utf-8"))
    stage3 = json.loads((run_root / "stage3_blocks.json").read_text(encoding="utf-8"))
    stage2_blocks = [int(value) for value in selection.get("stage2_blocks", [])]
    stage3_blocks = [int(value) for value in stage3.get("stage3_blocks", [])]
    return config, derived, run_root, stage2_blocks, stage3_blocks


def sentinel_current(root: Path = ROOT) -> bool:
    try:
        config, derived, run_root, stage2_blocks, stage3_blocks = _state(root)
        value = json.loads((run_root / SENTINEL_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, KeyError):
        return False
    return bool(
        value.get("status") == "complete"
        and value.get("generation_hash") == experiment_hash(derived)
        and value.get("evaluation_hash") == evaluation_hash(config)
        and value.get("verification_protocol_hash") == file_sha256(__file__)
        and value.get("stage2_blocks") == stage2_blocks
        and value.get("stage3_blocks") == stage3_blocks
        and value.get("valid_disable_evaluations") == value.get("expected_disable_jobs")
        and value.get("valid_remove_evaluations") == value.get("expected_remove_jobs")
    )


def verify(root: Path = ROOT, *, write: bool = True) -> dict:
    config, derived, run_root, stage2_blocks, stage3_blocks = _state(root)
    if len(stage2_blocks) != int(config["probing"]["stage2_blocks"]):
        raise RuntimeError(f"stage2 block list incomplete: {stage2_blocks}")
    if len(stage3_blocks) != int(config["probing"]["stage3_blocks"]):
        raise RuntimeError(f"stage3 block list incomplete: {stage3_blocks}")
    pilot_rows = [
        json.loads(line)
        for line in Path(derived["project"]["dataset_manifest"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pilot_ids = {row["id"] for row in pilot_rows}
    pilot_samples = load_counts(root)["pilot_samples"]
    if len(pilot_ids) != pilot_samples:
        raise RuntimeError(f"pilot manifest has {len(pilot_ids)} unique samples; expected {pilot_samples}")
    expected_disable = pilot_samples * len(stage2_blocks)
    expected_remove = pilot_samples * len(stage3_blocks)
    generation_hash = experiment_hash(derived)
    evaluator_hash = evaluation_hash(config)
    records = {"disable_text": [], "remove_block": []}
    for meta_path in (run_root / "images").rglob("*.json"):
        if meta_path.name.endswith(".eval.json"):
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        mode = meta.get("mode")
        allowed = stage2_blocks if mode == "disable_text" else stage3_blocks if mode == "remove_block" else []
        if (
            allowed
            and meta.get("status") == "complete"
            and meta.get("sample_id") in pilot_ids
            and meta.get("seed") == config["inference"]["pilot_seed"]
            and int(meta.get("global_block_index")) in allowed
        ):
            records[mode].append((meta_path, meta))
    for mode, expected in [("disable_text", expected_disable), ("remove_block", expected_remove)]:
        unique = {
            (meta["sample_id"], meta["seed"], int(meta["global_block_index"]))
            for _, meta in records[mode]
        }
        if len(records[mode]) != expected or len(unique) != expected:
            raise RuntimeError(
                f"{mode} incomplete or duplicated: records={len(records[mode])} unique={len(unique)} expected={expected}"
            )
        stale = [meta["output_path"] for _, meta in records[mode] if meta.get("config_hash") != generation_hash]
        if stale:
            raise RuntimeError(f"{mode} contains {len(stale)} stale generation records")
    valid = {}
    for mode, items in records.items():
        count = 0
        for meta_path, meta in items:
            image_path = Path(meta.get("output_path", ""))
            if not image_path.exists() or file_sha256(image_path) != meta.get("output_sha256"):
                raise RuntimeError(f"{mode} image missing or checksum-invalid: {image_path}")
            eval_path = meta_path.with_suffix(".eval.json")
            if not eval_path.exists():
                continue
            try:
                evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if reusable_evaluation(evaluation, meta, evaluator_hash, require_vlm=True):
                count += 1
        valid[mode] = count
    if valid["disable_text"] != expected_disable or valid["remove_block"] != expected_remove:
        raise RuntimeError(f"follow-up valid evaluations incomplete: {valid}")
    result = {
        "status": "complete",
        "stage2_blocks": stage2_blocks,
        "stage3_blocks": stage3_blocks,
        "expected_disable_jobs": expected_disable,
        "valid_disable_evaluations": valid["disable_text"],
        "expected_remove_jobs": expected_remove,
        "valid_remove_evaluations": valid["remove_block"],
        "generation_hash": generation_hash,
        "evaluation_hash": evaluator_hash,
        "verification_protocol_hash": file_sha256(__file__),
        "images_rehashed": True,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
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
    try:
        result = verify(write=not args.check_only)
    except (OSError, json.JSONDecodeError, KeyError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "not_ready", "reason": str(exc)}, indent=2))
        sys.exit(1)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
