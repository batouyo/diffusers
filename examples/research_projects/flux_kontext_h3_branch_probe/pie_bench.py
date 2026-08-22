from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


CATEGORIES = (
    "4_change_attribute_content_40",
    "6_change_attribute_color_40",
    "7_change_attribute_material_40",
    "8_change_background_80",
    "9_change_style_80",
)


@dataclass(frozen=True)
class PIEBenchSample:
    sample_id: str
    category: str
    source_image: str
    source_prompt: str
    target_prompt: str
    mask: np.ndarray
    seed: int


def decode_mask(encoded: str, size: int = 512) -> np.ndarray:
    """Decode PIE-Bench++ flat (start, length) intervals."""
    values = [int(value) for value in str(encoded).split() if value]
    if len(values) % 2:
        raise ValueError("PIE-Bench mask must contain start/length pairs")
    total = size * size
    mask = np.zeros(total, dtype=np.uint8)
    for start, length in zip(values[::2], values[1::2]):
        if start < 0 or length < 0 or start + length > total:
            raise ValueError(f"mask interval {(start, length)} exceeds {total} pixels")
        mask[start : start + length] = 1
    return mask.reshape(size, size).astype(bool)


def _sample_seed(sample_id: str, base_seed: int) -> int:
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    return int((int(digest[:8], 16) + int(base_seed)) % (2**31 - 1))


def load_samples(
    dataset_root: str | Path,
    source_cache: str | Path,
    *,
    per_category: int = 10,
    categories: tuple[str, ...] = CATEGORIES,
    base_seed: int = 20260822,
) -> list[PIEBenchSample]:
    import pyarrow.parquet as pq

    dataset_root = Path(dataset_root)
    source_cache = Path(source_cache)
    samples: list[PIEBenchSample] = []
    for category in categories:
        files = sorted((dataset_root / category).glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no parquet file found for {category}")
        rows = pq.read_table(files[0]).to_pylist()
        rows.sort(key=lambda row: str(row["id"]))
        for row in rows[:per_category]:
            sample_id = str(row["id"])
            image_record: dict[str, Any] = row["image"]
            image = Image.open(io.BytesIO(image_record["bytes"])).convert("RGB")
            if image.size != (512, 512):
                image = image.resize((512, 512), Image.Resampling.LANCZOS)
            image_path = source_cache / category / f"{sample_id}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            if not image_path.is_file():
                image.save(image_path)
            samples.append(
                PIEBenchSample(
                    sample_id=sample_id,
                    category=category,
                    source_image=str(image_path),
                    source_prompt=str(row["source_prompt"]),
                    target_prompt=str(row["target_prompt"]),
                    mask=decode_mask(str(row["mask"])),
                    seed=_sample_seed(sample_id, base_seed),
                )
            )
    return samples
