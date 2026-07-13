"""Verify exact pilot or formal alpha-grid generation and evaluation coverage."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from collections import Counter
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


def _key(alpha: float) -> str:
    return str(float(alpha))


def _state(root: Path, scope: str):
    base = load_config(root / "probe_config.yaml")
    output_root = Path(base["project"]["output_root"])
    run_root = output_root / base["project"]["run_id"]
    if scope == "pilot":
        stage3 = json.loads((run_root / "stage3_blocks.json").read_text(encoding="utf-8"))
        candidates = [int(value) for value in stage3.get("stage3_blocks", [])[:5]]
        sentinel = run_root / "pilot_alpha_complete.json"
    else:
        selection = json.loads((run_root / "selected_blocks.json").read_text(encoding="utf-8"))
        candidates = [int(value) for value in selection.get("selected_global_blocks", [])]
        sentinel = run_root / "formal_alpha_complete.json"
    return base, run_root, candidates, sentinel


def expected_alpha_jobs(root: Path, scope: str, candidates: list[int]):
    base = load_config(root / "probe_config.yaml")
    full_dataset = load_dataset(base["project"]["dataset_manifest"])
    jobs = {}
    configs = {}
    base_alpha = float(base["inference"]["alpha"])
    for value in base["inference"]["alpha_grid"]:
        alpha = float(value)
        config = copy.deepcopy(base)
        config["inference"]["alpha"] = alpha
        if scope == "pilot" and alpha == base_alpha:
            items = [
                job
                for job in generate_jobs(base, full_dataset, "pilot", candidates, "discovery")
                if job.mode == "enhance_text"
            ]
            config = base
        elif scope == "pilot":
            pilot_manifest = Path(base["project"]["output_root"]) / "preflight" / "pilot_dataset.jsonl"
            if not pilot_manifest.exists():
                raise RuntimeError(f"pilot dataset manifest is missing: {pilot_manifest}")
            config["project"]["dataset_manifest"] = str(pilot_manifest)
            config["inference"]["seeds"] = [config["inference"]["pilot_seed"]]
            items = generate_jobs(
                config,
                load_dataset(config["project"]["dataset_manifest"]),
                "enhance_text",
                candidates,
                "discovery",
            )
        else:
            items = generate_jobs(config, full_dataset, "enhance_text", candidates, "discovery")
        jobs[_key(alpha)] = items
        configs[_key(alpha)] = config
    return jobs, configs


def sentinel_current(root: Path = ROOT, scope: str = "pilot") -> bool:
    try:
        base, _, candidates, sentinel_path = _state(root, scope)
        value = json.loads(sentinel_path.read_text(encoding="utf-8"))
        jobs, configs = expected_alpha_jobs(root, scope, candidates)
    except (OSError, json.JSONDecodeError, KeyError, RuntimeError, TypeError, ValueError):
        return False
    expected_counts = {alpha: len(items) for alpha, items in jobs.items()}
    generation_hashes = {alpha: experiment_hash(config) for alpha, config in configs.items()}
    return bool(
        value.get("status") == "complete"
        and value.get("scope") == scope
        and value.get("candidates") == candidates
        and value.get("alpha_grid") == [float(item) for item in base["inference"]["alpha_grid"]]
        and value.get("valid_counts_by_alpha") == expected_counts
        and value.get("generation_hashes") == generation_hashes
        and value.get("evaluation_hash") == evaluation_hash(base)
        and value.get("verification_protocol_hash") == file_sha256(__file__)
    )


def verify(root: Path = ROOT, *, scope: str, write: bool = True, rehash_images: bool = True) -> dict:
    base, run_root, candidates, sentinel_path = _state(root, scope)
    if not candidates or len(set(candidates)) != len(candidates):
        raise RuntimeError(f"{scope} alpha candidates are empty or duplicated: {candidates}")
    if scope == "pilot" and len(candidates) != 5:
        raise RuntimeError(f"pilot alpha scan requires five candidates, got {candidates}")
    jobs, configs = expected_alpha_jobs(root, scope, candidates)
    evaluator = evaluation_hash(base)
    valid = Counter()
    logical_keys = set()
    for alpha, items in jobs.items():
        generation = experiment_hash(configs[alpha])
        for job in items:
            key = (job.sample_id, job.seed, job.global_block_index, float(job.alpha), job.resolution)
            if key in logical_keys:
                raise RuntimeError(f"duplicate expected alpha job: {key}")
            logical_keys.add(key)
            image_path, meta_path = job_output_paths(configs[alpha], job)
            if not image_path.exists() or not meta_path.exists():
                raise RuntimeError(f"alpha artifact missing: {meta_path}")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (
                meta.get("status") != "complete"
                or meta.get("mode") != "enhance_text"
                or meta.get("sample_id") != job.sample_id
                or int(meta.get("seed")) != job.seed
                or int(meta.get("global_block_index")) != job.global_block_index
                or float(meta.get("alpha")) != float(job.alpha)
                or meta.get("config_hash") != generation
            ):
                raise RuntimeError(f"alpha metadata protocol mismatch: {meta_path}")
            if Path(meta.get("output_path", "")).resolve() != image_path.resolve():
                raise RuntimeError(f"alpha metadata output path mismatch: {meta_path}")
            if rehash_images and file_sha256(image_path) != meta.get("output_sha256"):
                raise RuntimeError(f"alpha image checksum mismatch: {image_path}")
            eval_path = meta_path.with_suffix(".eval.json")
            if not eval_path.exists():
                raise RuntimeError(f"alpha evaluation missing: {eval_path}")
            evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
            if not reusable_evaluation(evaluation, meta, evaluator, require_vlm=True):
                raise RuntimeError(f"alpha evaluation is stale or incomplete: {eval_path}")
            valid[alpha] += 1
    if not (run_root / "alpha_summary.csv").exists():
        raise RuntimeError("alpha_summary.csv is missing after alpha aggregation")
    expected_counts = {alpha: len(items) for alpha, items in jobs.items()}
    if dict(valid) != expected_counts:
        raise RuntimeError(f"alpha counts differ: valid={dict(valid)} expected={expected_counts}")
    result = {
        "status": "complete",
        "scope": scope,
        "candidates": candidates,
        "alpha_grid": [float(item) for item in base["inference"]["alpha_grid"]],
        "valid_counts_by_alpha": dict(valid),
        "valid_total": sum(valid.values()),
        "generation_hashes": {alpha: experiment_hash(config) for alpha, config in configs.items()},
        "evaluation_hash": evaluator,
        "verification_protocol_hash": file_sha256(__file__),
        "images_rehashed": rehash_images,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    }
    if write:
        temporary = sentinel_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
        temporary.replace(sentinel_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["pilot", "formal"], required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-image-rehash", action="store_true")
    args = parser.parse_args()
    try:
        result = verify(
            scope=args.scope,
            write=not args.check_only,
            rehash_images=not args.skip_image_rehash,
        )
    except (OSError, json.JSONDecodeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "not_ready", "scope": args.scope, "reason": str(exc)}, indent=2))
        sys.exit(1)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
