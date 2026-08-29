"""Independent lightweight metrics used after reward selection."""
from __future__ import annotations
import numpy as np
from PIL import Image

def region_l1(a: Image.Image, b: Image.Image, mask: Image.Image, preserve: bool = False) -> float:
    first = np.asarray(a.convert("RGB").resize(mask.size), dtype=np.float32) / 255.0
    second = np.asarray(b.convert("RGB").resize(mask.size), dtype=np.float32) / 255.0
    m = np.asarray(mask.convert("L"), dtype=np.float32) > 127
    if preserve: m = ~m
    if not m.any(): return float("nan")
    return float(np.abs(first - second)[m].mean())
