"""Versioned teacher cache I/O."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import torch

SCHEMA_VERSION = 1

def save_teacher_record(root: str | Path, sample_id: str, tensors: dict[str, torch.Tensor], metadata: dict[str, Any]) -> None:
    folder = Path(root) / sample_id; folder.mkdir(parents=True, exist_ok=True)
    torch.save({name: value.detach().cpu() for name, value in tensors.items()}, folder / "tensors.pt")
    payload = {"schema_version": SCHEMA_VERSION, "sample_id": sample_id, **metadata}
    (folder / "metadata.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

def load_teacher_record(root: str | Path, sample_id: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    folder = Path(root) / sample_id
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION: raise ValueError("unsupported teacher cache schema")
    tensors = torch.load(folder / "tensors.pt", map_location="cpu", weights_only=True)
    return tensors, metadata
