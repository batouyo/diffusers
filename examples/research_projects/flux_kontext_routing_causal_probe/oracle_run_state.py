"""Reproducible fingerprints, status files, and checkpoint helpers for the Oracle run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from probe_utils import atomic_write_json, file_sha256, load_json


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def code_fingerprint(paths: Iterable[str | Path]) -> dict[str, Any]:
    entries = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        entries.append({"path": str(path), "sha256": file_sha256(path)})
    return {"sha256": stable_hash(entries), "files": entries}


def _model_files(model_path: Path) -> list[Path]:
    suffixes = {".json", ".safetensors", ".bin"}
    return sorted(path for path in model_path.rglob("*") if path.is_file() and path.suffix in suffixes)


def model_fingerprint(model_path: str | Path, cache_path: str | Path) -> dict[str, Any]:
    root = Path(model_path).resolve()
    files = _model_files(root)
    if not files:
        raise RuntimeError(f"no model/config weight files found under {root}")
    manifest = [
        {"path": str(path.relative_to(root)), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in files
    ]
    manifest_hash = stable_hash(manifest)
    cache = Path(cache_path)
    if cache.is_file():
        saved = load_json(cache)
        if saved.get("manifest_sha256") == manifest_hash:
            return saved
    entries = []
    for index, path in enumerate(files, start=1):
        print(f"[model-hash] {index}/{len(files)} {path.relative_to(root)}", flush=True)
        entries.append(
            {
                "path": str(path.relative_to(root)),
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    result = {
        "model_path": str(root),
        "manifest_sha256": manifest_hash,
        "content_sha256": stable_hash(entries),
        "files": entries,
    }
    atomic_write_json(cache, result)
    return result


def tensor_state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def write_status(root: str | Path, *, state: str, stage: str, message: str, **details: Any) -> None:
    atomic_write_json(
        Path(root) / "status.json",
        {"state": state, "stage": stage, "message": message, "details": details},
    )


def require_fingerprint(path: str | Path, expected: str, artifact: str) -> dict[str, Any]:
    metadata = load_json(path)
    actual = metadata.get("fingerprint")
    if actual != expected:
        raise RuntimeError(
            f"refusing to resume incompatible {artifact}: expected fingerprint {expected}, found {actual}"
        )
    return metadata


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: str) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)
