#!/usr/bin/env python
"""Write lightweight H3 checkpoint summaries while shard logs are running."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pie_bench import CATEGORIES, decode_mask


ROOT = Path("/data15/hyp/experiments/flux_kontext_h3_branch_probe")
SHARDS = (ROOT / "pilot50_shard0", ROOT / "pilot50_shard1")
STEPS = (0, 1, 2, 3, 4, 5, 8, 14)


def read_rows() -> list[dict]:
    rows = []
    for shard in SHARDS:
        path = shard / "run.log"
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("{") and '"branching_step"' in line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def complete_samples(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["sample_id"]), []).append(row)
    result = {}
    for sample_id, values in grouped.items():
        by_step = {int(value["branching_step"]): value for value in values}
        if all(step in by_step for step in STEPS):
            result[sample_id] = [by_step[step] for step in STEPS]
    return result


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def mask_map() -> dict[tuple[str, str], np.ndarray]:
    import pyarrow.parquet as pq

    result = {}
    for category in CATEGORIES:
        files = sorted((Path("/data15/hyp/dataset/PIE-Bench") / category).glob("*.parquet"))
        if not files:
            continue
        for row in pq.read_table(files[0], columns=["id", "mask"]).to_pylist():
            result[(category, str(row["id"]))] = decode_mask(str(row["mask"]))
    return result


def preservation_metrics(sample_id: str, category: str, branching_step: int, shard: Path, masks: dict) -> tuple[float | None, float | None]:
    source_path = shard / "dataset" / "source_images" / category / f"{sample_id}.png"
    branch_root = shard / "images" / sample_id / f"k_{branching_step:02d}"
    files = sorted(branch_root.glob("s_*.png"))
    if not source_path.is_file() or len(files) != 5:
        return None, None
    source = np.asarray(Image.open(source_path).convert("RGB"), dtype=np.float32) / 255.0
    preserve = 1.0 - masks[(category, sample_id)].astype(np.float32)
    mask_image = Image.fromarray((preserve * 255).astype(np.uint8)).resize(Image.open(files[0]).size, Image.Resampling.NEAREST)
    preserve = np.asarray(mask_image, dtype=np.float32) / 255.0
    area = float(preserve.sum())
    if area == 0.0:
        return None, None
    source_image = Image.fromarray((source * 255).astype(np.uint8)).resize(Image.open(files[0]).size, Image.Resampling.LANCZOS)
    source = np.asarray(source_image, dtype=np.float32) / 255.0
    values = [np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0 for path in files]
    l1 = [float((np.abs(value - source) * preserve[..., None]).sum() / (area * 3.0)) for value in values]
    l2 = [float(np.sqrt(((value - source) ** 2 * preserve[..., None]).sum() / (area * 3.0))) for value in values]
    return float(np.mean(l1)), float(np.mean(l2))


def build_summary(samples: dict[str, list[dict]]) -> dict:
    masks = mask_map()
    by_k = {}
    for index, step in enumerate(STEPS):
        values = [rows[index] for rows in samples.values()]
        def nums(key: str) -> list[float]:
            return [float(row[key]) for row in values if row.get(key) is not None]
        preservation = []
        for row in values:
            category = str(row.get("category", {
                "4": "4_change_attribute_content_40",
                "6": "6_change_attribute_color_40",
                "7": "7_change_attribute_material_40",
                "8": "8_change_background_80",
                "9": "9_change_style_80",
            }.get(str(row["sample_id"])[0], "")))
            sample_id = str(row["sample_id"])
            if not category:
                continue
            shard = next((candidate for candidate in SHARDS if (candidate / "images" / sample_id).is_dir()), SHARDS[0])
            l1, l2 = preservation_metrics(sample_id, category, step, shard, masks)
            if l1 is not None:
                preservation.append((l1, l2))
        by_k[str(step)] = {
            "n": len(values),
            "edit_dynamic_range_mean": mean(nums("edit_dynamic_range")),
            "edit_dynamic_range_min": min(nums("edit_dynamic_range"), default=None),
            "edit_dynamic_range_max": max(nums("edit_dynamic_range"), default=None),
            "monotonicity_mean": mean(nums("monotonicity")),
            "normal_suffix_rms_mean": mean(nums("normal_suffix_rms_error")),
            "normal_suffix_rms_max": max(nums("normal_suffix_rms_error"), default=None),
            "preserve_l1_coverage": len(nums("preserve_l1_mean")),
            "preserve_l1_mean": mean(nums("preserve_l1_mean")),
            "preserve_l2_mean": mean([value[1] for value in preservation]),
            "preserve_l1_mean": mean([value[0] for value in preservation]),
            "preserve_coverage": len(preservation),
        }
    return {
        "completed_samples": len(samples),
        "sample_ids": sorted(samples),
        "branching_steps": list(STEPS),
        "by_branching_step": by_k,
    }


def write_checkpoint(summary: dict) -> None:
    count = int(summary["completed_samples"])
    output = ROOT / "checkpoints"
    output.mkdir(parents=True, exist_ok=True)
    (output / f"checkpoint_{count:03d}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [f"# H3 checkpoint: {count} complete samples", "", "| k | R_edit mean | R_edit range | monotonicity | parity RMS mean/max | preservation coverage |", "|---:|---:|---:|---:|---:|---:|"]
    for step in STEPS:
        row = summary["by_branching_step"][str(step)]
        r = row["edit_dynamic_range_mean"]
        rrange = f"{row['edit_dynamic_range_min']:.4f}..{row['edit_dynamic_range_max']:.4f}" if row["edit_dynamic_range_min"] is not None else "n/a"
        parity = f"{row['normal_suffix_rms_mean']:.4f}/{row['normal_suffix_rms_max']:.4f}" if row["normal_suffix_rms_mean"] is not None else "n/a"
        lines.append(f"| {step} | {r:.4f} | {rrange} | {row['monotonicity_mean']:.3f} | {parity} | {row['preserve_coverage']}/{count} |" if r is not None else f"| {step} | n/a | {rrange} | n/a | {parity} | {row['preserve_coverage']}/{count} |")
    (output / f"checkpoint_{count:03d}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    written: set[int] = set()
    while True:
        samples = complete_samples(read_rows())
        count = len(samples)
        for target in range(10, count + 1, 10):
            if target not in written:
                subset = {key: samples[key] for key in sorted(samples)[:target]}
                write_checkpoint(build_summary(subset))
                written.add(target)
        if count >= 50:
            return
        time.sleep(300)


if __name__ == "__main__":
    main()
