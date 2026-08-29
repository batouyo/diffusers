"""Fast category-balanced PIE-Bench manifest builder.

PIE-Bench stores source images and packed flat masks in one parquet per task
category, so no target-image reconstruction or full HQ-Edit scan is needed.
"""
from __future__ import annotations
import io, json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

def decode_image(value: object) -> Image.Image:
    if isinstance(value, dict): value = value.get("bytes", value.get("path"))
    if isinstance(value, memoryview): value = value.tobytes()
    if isinstance(value, (bytes, bytearray)): return Image.open(io.BytesIO(value)).convert("RGB")
    if isinstance(value, str): return Image.open(value).convert("RGB")
    raise TypeError(f"unsupported PIE image cell: {type(value).__name__}")

def parse_flat_mask(encoded: str, size: tuple[int, int]) -> Image.Image:
    width, height = size; flat = np.zeros(width * height, dtype=np.uint8)
    values = [int(x) for x in str(encoded).split() if x.strip()]
    if len(values) % 2: raise ValueError("PIE mask must contain start/length pairs")
    for start, length in zip(values[::2], values[1::2]):
        if start < 0 or length < 0 or start + length > flat.size: raise ValueError("PIE mask interval outside image")
        flat[start:start + length] = 255
    return Image.fromarray(flat.reshape(height, width), mode="L")

def select_pie_subset(root: str | Path, train_count: int = 16, test_count: int = 8, seed: int = 20260830, per_category: int = 3, min_area: float = 0.02, max_area: float = 0.40) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = np.random.default_rng(seed); candidates = []; all_valid = []
    for category_dir in sorted(Path(root).iterdir()):
        parquet = next(category_dir.glob("*.parquet"), None) if category_dir.is_dir() else None
        if parquet is None: continue
        frame = pd.read_parquet(parquet)
        valid_rows = []
        for row_index in range(len(frame)):
            row = frame.iloc[int(row_index)]; source = decode_image(row["image"])
            mask = parse_flat_mask(str(row["mask"]), source.size)
            area = float(np.asarray(mask).mean() / 255.0)
            if min_area <= area <= max_area:
                valid_rows.append({"sample_id": f"pie_{category_dir.name}_{row['id']}", "category": category_dir.name, "shard": str(parquet), "row_index": int(row_index), "source": source, "mask": mask, "mask_area": area, "instruction": str(row["target_prompt"]), "source_prompt": str(row["source_prompt"]), "target_description": str(row["target_prompt"])})
        chosen = [valid_rows[i] for i in rng.permutation(len(valid_rows))[:per_category].tolist()]
        candidates.extend(chosen); all_valid.extend(valid_rows)
    chosen_ids = {record["sample_id"] for record in candidates}
    remainder = [record for record in all_valid if record["sample_id"] not in chosen_ids]
    if len(candidates) < train_count + test_count:
        candidates.extend(remainder[i] for i in rng.permutation(len(remainder))[: train_count + test_count - len(candidates)].tolist())
    order = rng.permutation(len(candidates)).tolist(); selected = [candidates[i] for i in order[:train_count + test_count]]
    if len(selected) != train_count + test_count: raise RuntimeError(f"PIE-Bench contains only {len(selected)} selected records")
    return selected[:train_count], selected[train_count:]

def write_pie_manifest(path: str | Path, records: list[dict[str, object]]) -> None:
    fields = ("sample_id", "category", "shard", "row_index", "instruction", "source_prompt", "target_description", "mask_area")
    Path(path).write_text(json.dumps([{k: record[k] for k in fields} for record in records], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
