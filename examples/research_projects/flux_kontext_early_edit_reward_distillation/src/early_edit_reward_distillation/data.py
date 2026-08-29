"""Streaming HQ-Edit selection and diagnostic pseudo-mask generation."""
from __future__ import annotations
import hashlib, io, json
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageOps

def _decode(value: object) -> Image.Image:
    if isinstance(value, dict): value = value.get("bytes", value.get("path"))
    if isinstance(value, memoryview): value = value.tobytes()
    if isinstance(value, (bytes, bytearray)): return Image.open(io.BytesIO(value)).convert("RGB")
    if isinstance(value, str): return Image.open(value).convert("RGB")
    raise TypeError(f"unsupported image cell: {type(value).__name__}")

def pseudo_edit_mask(source: Image.Image, target: Image.Image, size: int = 512) -> Image.Image:
    source = ImageOps.fit(source.convert("RGB"), (size, size), method=Image.Resampling.LANCZOS)
    target = ImageOps.fit(target.convert("RGB"), (size, size), method=Image.Resampling.LANCZOS)
    diff = np.abs(np.asarray(source, dtype=np.float32) - np.asarray(target, dtype=np.float32)).mean(axis=2)
    blurred = np.asarray(Image.fromarray(np.clip(diff, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.0)), dtype=np.float32)
    threshold = max(float(np.quantile(blurred, 0.85)), 8.0)
    mask = Image.fromarray((blurred >= threshold).astype(np.uint8) * 255, mode="L")
    return mask.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3)).point(lambda value: 255 if value else 0)

def sample_id(shard: Path, row_index: int, source: Image.Image, target: Image.Image) -> str:
    digest = hashlib.sha256()
    for image in (source, target):
        buffer = io.BytesIO(); image.save(buffer, format="PNG"); digest.update(buffer.getvalue())
    return f"{shard.stem}_row{row_index:05d}_{digest.hexdigest()[:12]}"

def stream_local_pairs(root: str | Path) -> Iterable[dict[str, object]]:
    for shard in sorted(Path(root).glob("*.parquet")):
        frame = pd.read_parquet(shard, columns=["input_image", "output_image", "edit", "output"])
        for row_index, row in frame.iterrows():
            source, target = _decode(row["input_image"]), _decode(row["output_image"])
            yield {"sample_id": sample_id(shard, int(row_index), source, target), "shard": str(shard), "row_index": int(row_index), "source": source, "target": target, "instruction": str(row["edit"]), "target_description": str(row.get("output", ""))}

def select_local_subset(root: str | Path, train_count: int = 16, test_count: int = 8, seed: int = 20260830, min_area: float = 0.02, max_area: float = 0.40) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    candidates = []
    for record in stream_local_pairs(root):
        mask = pseudo_edit_mask(record["source"], record["target"])
        area = float(np.asarray(mask, dtype=np.uint8).mean() / 255.0)
        if min_area <= area <= max_area: candidates.append({**record, "mask": mask, "mask_area": area})
    order = np.random.default_rng(seed).permutation(len(candidates)).tolist()
    selected = [candidates[i] for i in order[:train_count + test_count]]
    if len(selected) != train_count + test_count: raise RuntimeError(f"only found {len(selected)} locality-valid samples")
    return selected[:train_count], selected[train_count:]

def write_manifest(path: str | Path, records: list[dict[str, object]]) -> None:
    fields = ("sample_id", "shard", "row_index", "instruction", "target_description", "mask_area")
    Path(path).write_text(json.dumps([{k: record[k] for k in fields} for record in records], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
