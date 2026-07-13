"""Verify dataset provenance, split isolation, image integrity, and instruction constraints."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

from expected_counts import ROOT
from probe_flux_kontext_blocks import file_sha256, load_config


REPORT_NAME = "dataset_report.json"


def inspect_rows(config: dict, rows: list[dict]) -> dict:
    categories = list(config["dataset"]["categories"])
    expected_counts = {
        (category, "discovery"): int(config["dataset"]["discovery_per_category"])
        for category in categories
    } | {
        (category, "heldout"): int(config["dataset"]["heldout_per_category"])
        for category in categories
    }
    counts = Counter((row.get("category"), row.get("split")) for row in rows)
    errors = []
    if counts != Counter(expected_counts):
        errors.append(f"category/split counts differ: observed={dict(counts)} expected={expected_counts}")
    ids = [str(row.get("id")) for row in rows]
    source_hashes = [str(row.get("source_sha256")) for row in rows]
    prepared_hashes = [str(row.get("prepared_sha256")) for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("dataset IDs are not globally unique")
    if len(source_hashes) != len(set(source_hashes)) or any(len(value) != 64 for value in source_hashes):
        errors.append("source hashes are missing, malformed, or duplicated across splits")
    if len(prepared_hashes) != len(set(prepared_hashes)) or any(len(value) != 64 for value in prepared_hashes):
        errors.append("prepared hashes are missing, malformed, or duplicated across splits")
    observed_sizes = Counter()
    observed_modes = Counter()
    instruction_word_counts = []
    for row in rows:
        sample_id = str(row.get("id"))
        image_path = Path(str(row.get("image", "")))
        if not image_path.is_file():
            errors.append(f"prepared image missing: {sample_id} -> {image_path}")
            continue
        actual_hash = file_sha256(image_path)
        if actual_hash != row.get("prepared_sha256"):
            errors.append(f"prepared image checksum mismatch: {sample_id}")
        try:
            with Image.open(image_path) as image:
                image.load()
                observed_sizes[str(tuple(image.size))] += 1
                observed_modes[str(image.mode)] += 1
                if image.size != (512, 512) or image.mode != "RGB":
                    errors.append(f"prepared image format mismatch: {sample_id} size={image.size} mode={image.mode}")
        except Exception as exc:
            errors.append(f"prepared image decode failed: {sample_id}: {exc}")
        words = str(row.get("instruction", "")).split()
        instruction_word_counts.append(len(words))
        if not 4 <= len(words) <= 28:
            errors.append(f"instruction length outside 4..28 words: {sample_id}")
        if not str(row.get("target_description", "")).strip():
            errors.append(f"target description missing: {sample_id}")
        if row.get("qc_status") != "automatic_pass_pending_human_review":
            errors.append(f"unexpected QC provenance status: {sample_id}")
        reference = Path(str(row.get("reference_target_image", "")))
        if not reference.is_file():
            errors.append(f"reference target provenance path missing: {sample_id} -> {reference}")
    return {
        "status": "pass" if not errors else "fail",
        "row_count": len(rows),
        "category_split_counts": {
            f"{category}:{split}": counts[(category, split)]
            for category in categories
            for split in ["discovery", "heldout"]
        },
        "unique_ids": len(set(ids)),
        "unique_source_hashes": len(set(source_hashes)),
        "unique_prepared_hashes": len(set(prepared_hashes)),
        "observed_sizes": dict(observed_sizes),
        "observed_modes": dict(observed_modes),
        "instruction_words_min": min(instruction_word_counts) if instruction_word_counts else None,
        "instruction_words_max": max(instruction_word_counts) if instruction_word_counts else None,
        "qc_status": "automatic_pass_pending_human_review",
        "reference_targets_used_for_scoring": False,
        "errors": errors,
    }


def sentinel_current(root: Path = ROOT) -> bool:
    try:
        config = load_config(root / "probe_config.yaml")
        manifest = Path(config["project"]["dataset_manifest"])
        report_path = Path(config["project"]["output_root"]) / "preflight" / REPORT_NAME
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False
    return bool(
        report.get("status") == "pass"
        and report.get("manifest_sha256") == file_sha256(manifest)
        and report.get("verification_protocol_hash") == file_sha256(__file__)
        and report.get("row_count") == 240
        and report.get("unique_ids") == 240
        and report.get("unique_source_hashes") == 240
        and report.get("unique_prepared_hashes") == 240
    )


def main() -> None:
    config = load_config(ROOT / "probe_config.yaml")
    manifest = Path(config["project"]["dataset_manifest"])
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = inspect_rows(config, rows)
    report.update(
        {
            "manifest": str(manifest),
            "manifest_sha256": file_sha256(manifest),
            "verification_protocol_hash": file_sha256(__file__),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        }
    )
    output = Path(config["project"]["output_root"]) / "preflight" / REPORT_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "pass":
        sys.exit(1)


if __name__ == "__main__":
    main()
