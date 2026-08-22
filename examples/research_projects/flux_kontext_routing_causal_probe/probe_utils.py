from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def atomic_write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, path)


def atomic_torch_save(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(value, temp_name)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(path: str | Path) -> str:
    root = Path(path)
    digest = hashlib.sha256()
    files = sorted(
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file()
        and "__pycache__" not in candidate.parts
        and candidate.suffix in {".py", ".json", ".md"}
    )
    for candidate in files:
        digest.update(candidate.relative_to(root).as_posix().encode("utf-8"))
        digest.update(candidate.read_bytes())
    return digest.hexdigest()


def git_revision(repo: str | Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def ensure_run_config(path: str | Path, config: dict[str, Any]) -> None:
    """Create an immutable run config with an O_EXCL process-safe lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = load_json(path)
        if canonical_hash(existing) != canonical_hash(config):
            raise RuntimeError(f"run configuration mismatch at {path}; choose a new run directory")
        return
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        ensure_run_config(path, config)
        return
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def selected_step_indices(num_steps: int, configured: Iterable[int] | None = None) -> list[int]:
    if configured:
        indices = [int(x) for x in configured]
    else:
        indices = [0, 1, num_steps // 2, num_steps - 2]
    return sorted({x for x in indices if 0 <= x < num_steps})


def cross_bias_conditions(config: dict[str, Any]) -> list[dict[str, float]]:
    scan = config["bias_scan"]
    source_values = [float(x) for x in scan["source"]]
    text_values = [float(x) for x in scan["text"]]
    conditions = [{"b_source": 0.0, "b_text": 0.0, "b_target": 0.0}]
    conditions.extend(
        {"b_source": value, "b_text": 0.0, "b_target": 0.0}
        for value in source_values
        if value != 0.0
    )
    conditions.extend(
        {"b_source": 0.0, "b_text": value, "b_target": 0.0}
        for value in text_values
        if value != 0.0
    )
    return conditions


def choose_negative_controls(config: dict[str, Any]) -> list[str]:
    layer_config = config["layers"]
    csv_path = Path(layer_config["negative_control_csv"])
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))

    def pick(prefix: str, bounds: list[int]) -> str:
        candidates = []
        for row in rows:
            raw_id = row.get("layer_id") or row.get("layer") or ""
            normalized = raw_id.replace("transformer_blocks.", "dual.").replace(
                "single_transformer_blocks.", "single."
            )
            try:
                stream, index_text = normalized.split(".", 1)
                index = int(index_text)
            except (ValueError, TypeError):
                continue
            hit_text = (
                row.get("critical_sample_count")
                or row.get("hit_sample_count")
                or row.get("critical_count")
                or "0"
            )
            mean_text = (
                row.get("mean_main_score")
                or row.get("mean_dino_score")
                or row.get("mean_score")
                or row.get("mean_distance")
                or "inf"
            )
            if (
                stream == prefix
                and bounds[0] <= index <= bounds[1]
                and (not layer_config.get("require_zero_hits", True) or int(float(hit_text)) == 0)
            ):
                candidates.append((float(mean_text), index))
        if not candidates:
            raise RuntimeError(f"no eligible {prefix} negative-control layer in {csv_path}")
        _, index = min(candidates)
        return f"{prefix}.{index:02d}"

    return [
        pick("dual", layer_config["dual_middle_range"]),
        pick("single", layer_config["single_middle_range"]),
    ]


def configured_layers(config: dict[str, Any]) -> list[str]:
    base = config["layers"]["primary"] + config["layers"]["task_sensitive"]
    return list(dict.fromkeys(base + choose_negative_controls(config)))


def move_tensors(value: Any, device: torch.device | str) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_tensors(item, device) for item in value)
    if isinstance(value, list):
        return [move_tensors(item, device) for item in value]
    return value


def velocity_metrics(controlled: torch.Tensor, reference: torch.Tensor, epsilon: float = 1e-12) -> dict[str, Any]:
    controlled_fp32 = controlled.float()
    reference_fp32 = reference.float()
    delta = controlled_fp32 - reference_fp32
    delta_l2 = float(torch.linalg.vector_norm(delta).item())
    reference_l2 = float(torch.linalg.vector_norm(reference_fp32).item())
    relative_l2 = delta_l2 / (reference_l2 + epsilon)
    cosine = float(
        torch.nn.functional.cosine_similarity(controlled_fp32.flatten(), reference_fp32.flatten(), dim=0).item()
    )
    return {
        "delta_l2": delta_l2,
        "relative_l2": relative_l2,
        "cosine": cosine,
        "velocity_rms": float(controlled_fp32.square().mean().sqrt().item()),
        "velocity_max_abs": float(controlled_fp32.abs().max().item()),
        "finite": bool(torch.isfinite(controlled_fp32).all().item()),
    }


def token_norm_map(delta: torch.Tensor, grid_height: int, grid_width: int) -> np.ndarray:
    values = torch.linalg.vector_norm(delta.float(), dim=-1).mean(dim=0)
    if values.numel() != grid_height * grid_width:
        raise ValueError("velocity token count cannot be reshaped to the requested grid")
    return values.reshape(grid_height, grid_width).cpu().numpy()
