"""Verify the exact hash-current formal discovery/evaluation matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from evaluators import evaluation_hash, reusable_evaluation
from expected_counts import ROOT
from probe_flux_kontext_blocks import (
    experiment_hash,
    file_sha256,
    generate_jobs,
    job_output_paths,
    load_config,
    load_dataset,
)


SENTINEL_NAME = "formal_discovery_complete.json"
REQUIRED_AGGREGATES = ["raw_metrics.csv", "block_summary.csv", "stream_summary.csv", "selected_blocks.json"]


def job_key(value) -> tuple:
    """Return the exact logical key shared by ProbeJob and metadata dictionaries."""
    if not isinstance(value, dict):
        value = asdict(value)
    return (
        value.get("sample_id"),
        int(value.get("seed")),
        value.get("mode"),
        value.get("global_block_index"),
        float(value.get("alpha")),
        int(value.get("resolution")),
        value.get("split"),
    )


def expected_formal_jobs(config: dict, dataset: list[dict], total_blocks: int, stage2: list[int], stage3: list[int]):
    return {
        "baseline": generate_jobs(config, dataset, "baseline", [], "discovery"),
        "enhance_text": generate_jobs(config, dataset, "enhance_text", list(range(total_blocks)), "discovery"),
        "disable_text": generate_jobs(config, dataset, "disable_text", stage2, "discovery"),
        "remove_block": generate_jobs(config, dataset, "remove_block", stage3, "discovery"),
    }


def _state(root: Path):
    config = load_config(root / "probe_config.yaml")
    output_root = Path(config["project"]["output_root"])
    run_root = output_root / config["project"]["run_id"]
    structure = json.loads((output_root / "preflight" / "structure_report.json").read_text(encoding="utf-8"))
    dataset = load_dataset(config["project"]["dataset_manifest"])
    selection = json.loads((run_root / "selected_blocks.json").read_text(encoding="utf-8"))
    stage3_value = json.loads((run_root / "stage3_blocks.json").read_text(encoding="utf-8"))
    stage2 = [int(value) for value in selection.get("stage2_blocks", [])]
    stage3 = [int(value) for value in stage3_value.get("stage3_blocks", [])]
    return config, run_root, structure, dataset, stage2, stage3


def sentinel_current(root: Path = ROOT) -> bool:
    try:
        config, run_root, structure, dataset, stage2, stage3 = _state(root)
        value = json.loads((run_root / SENTINEL_NAME).read_text(encoding="utf-8"))
        jobs = expected_formal_jobs(config, dataset, int(structure["total_block_count"]), stage2, stage3)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False
    expected_counts = {mode: len(items) for mode, items in jobs.items()}
    return bool(
        value.get("status") == "complete"
        and value.get("generation_hash") == experiment_hash(config)
        and value.get("evaluation_hash") == evaluation_hash(config)
        and value.get("verification_protocol_hash") == file_sha256(__file__)
        and value.get("stage2_blocks") == stage2
        and value.get("stage3_blocks") == stage3
        and value.get("valid_mode_counts") == expected_counts
        and value.get("valid_total") == sum(expected_counts.values())
    )


def verify(root: Path = ROOT, *, write: bool = True, rehash_images: bool = True) -> dict:
    config, run_root, structure, dataset, stage2, stage3 = _state(root)
    if len(stage2) != int(config["probing"]["stage2_blocks"]) or len(set(stage2)) != len(stage2):
        raise RuntimeError(f"formal stage2 block list invalid: {stage2}")
    if len(stage3) != int(config["probing"]["stage3_blocks"]) or len(set(stage3)) != len(stage3):
        raise RuntimeError(f"formal stage3 block list invalid: {stage3}")
    total_blocks = int(structure["total_block_count"])
    if any(index < 0 or index >= total_blocks for index in stage2 + stage3):
        raise RuntimeError("formal diagnostic block index is outside the runtime structure")
    if not set(stage3).issubset(stage2):
        raise RuntimeError("formal stage3 blocks must be a subset of stage2 blocks")

    groups = expected_formal_jobs(config, dataset, total_blocks, stage2, stage3)
    expected_keys = [job_key(job) for items in groups.values() for job in items]
    if len(expected_keys) != len(set(expected_keys)):
        raise RuntimeError("formal expected job matrix contains duplicate logical keys")
    generation = experiment_hash(config)
    evaluator = evaluation_hash(config)
    valid_counts = Counter()
    for mode, jobs in groups.items():
        for job in jobs:
            image_path, meta_path = job_output_paths(config, job)
            if not image_path.exists() or not meta_path.exists():
                raise RuntimeError(f"formal artifact missing: {meta_path}")
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"formal metadata unreadable: {meta_path}") from exc
            if meta.get("status") != "complete" or job_key(meta) != job_key(job):
                raise RuntimeError(f"formal metadata key/status mismatch: {meta_path}")
            if meta.get("config_hash") != generation:
                raise RuntimeError(f"formal metadata has stale generation hash: {meta_path}")
            if Path(meta.get("output_path", "")).resolve() != image_path.resolve():
                raise RuntimeError(f"formal metadata output path mismatch: {meta_path}")
            if rehash_images and file_sha256(image_path) != meta.get("output_sha256"):
                raise RuntimeError(f"formal image checksum mismatch: {image_path}")
            eval_path = meta_path.with_suffix(".eval.json")
            if not eval_path.exists():
                raise RuntimeError(f"formal evaluation missing: {eval_path}")
            try:
                evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"formal evaluation unreadable: {eval_path}") from exc
            if not reusable_evaluation(evaluation, meta, evaluator, require_vlm=True):
                raise RuntimeError(f"formal evaluation is stale or incomplete: {eval_path}")
            valid_counts[mode] += 1

    missing_aggregates = [name for name in REQUIRED_AGGREGATES if not (run_root / name).exists()]
    if missing_aggregates:
        raise RuntimeError(f"formal aggregate artifacts missing: {missing_aggregates}")
    expected_counts = {mode: len(items) for mode, items in groups.items()}
    if dict(valid_counts) != expected_counts:
        raise RuntimeError(f"formal mode counts differ: valid={dict(valid_counts)} expected={expected_counts}")
    result = {
        "status": "complete",
        "valid_mode_counts": dict(valid_counts),
        "valid_total": sum(valid_counts.values()),
        "stage2_blocks": stage2,
        "stage3_blocks": stage3,
        "generation_hash": generation,
        "evaluation_hash": evaluator,
        "verification_protocol_hash": file_sha256(__file__),
        "images_rehashed": rehash_images,
        "aggregate_files": REQUIRED_AGGREGATES,
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
    parser.add_argument("--skip-image-rehash", action="store_true")
    args = parser.parse_args()
    try:
        result = verify(write=not args.check_only, rehash_images=not args.skip_image_rehash)
    except (OSError, json.JSONDecodeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "not_ready", "reason": str(exc)}, indent=2))
        sys.exit(1)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
