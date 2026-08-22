#!/usr/bin/env python
"""Build an Oracle manifest and choose layer controls from a real ablation CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from probe_utils import atomic_write_json, load_json, load_jsonl


def select_quick_samples(samples: list[dict], sample_groups: list[dict]) -> dict:
    """Select one deterministic sample for each configured quick-validation role."""
    selected: list[dict] = []
    missing: list[dict] = []
    selected_ids: set[str] = set()
    for group in sample_groups:
        role = str(group["role"])
        categories = [str(value) for value in group["categories"]]
        match = None
        for category in categories:
            match = next((sample for sample in samples if sample.get("category") == category), None)
            if match is not None:
                break
        if match is None:
            missing.append({"role": role, "categories": categories})
            continue
        sample_id = str(match["sample_id"])
        if sample_id in selected_ids:
            raise ValueError(f"quick sample {sample_id} was selected for more than one role")
        selected_ids.add(sample_id)
        selected.append({**match, "quick_role": role})
    return {
        "samples": selected,
        "missing_roles": missing,
        "complete": not missing and len(selected) == len(sample_groups),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_layers(config: dict) -> dict[str, str]:
    summary_path = Path(config["layer_summary_csv"])
    with summary_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    candidate = config["candidate_layers"]
    middle_start, middle_stop = candidate["middle_single_index_range"]
    middle = [
        row
        for row in rows
        if row["stream"] == "single" and middle_start <= int(row["index"]) <= middle_stop
    ]
    if not middle:
        raise ValueError("no middle single blocks are available")
    selected_middle = max(middle, key=lambda row: float(row["mean_dino_score"]))

    low = [
        row
        for row in rows
        if row["stream"] == candidate["negative_control_stream"]
        and row["classification"] == "not_critical"
        and int(row["critical_sample_count"]) == 0
    ]
    if not low:
        raise ValueError("no low-sensitivity control satisfies the configured policy")
    selected_low = min(low, key=lambda row: float(row["mean_dino_score"]))
    primary = candidate["primary"]
    secondary = candidate["secondary"]
    return {
        "primary": primary,
        "secondary": secondary,
        "middle_single": selected_middle["layer_id"],
        "negative_control": selected_low["layer_id"],
        "summary_csv_sha256": sha256_file(summary_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-ids", nargs="*")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    config = load_json(args.config)
    samples = load_jsonl(config["manifest_path"])
    if args.sample_ids:
        wanted = set(args.sample_ids)
        samples = [sample for sample in samples if sample["sample_id"] in wanted]
        if {sample["sample_id"] for sample in samples} != wanted:
            raise ValueError("one or more requested samples were not found")
    result = {
        "config_path": args.config,
        "layers": select_layers(config),
        "samples": samples,
    }
    if args.quick:
        quick = config.get("quick_validation", {})
        if not quick.get("enabled"):
            raise ValueError("--quick requires quick_validation.enabled=true")
        result["quick_selection"] = select_quick_samples(samples, quick["sample_groups"])
    atomic_write_json(args.output, result)


if __name__ == "__main__":
    main()
