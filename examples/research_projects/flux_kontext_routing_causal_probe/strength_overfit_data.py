"""Dataset, cache, contract and run-provenance utilities for strength overfit."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image


REQUIRED_FIELDS = {
    "sample_id", "source_image", "full_prompt", "neutral_prompt",
    "target_edit_description", "edit_type", "target_phrase", "notes",
}
EDIT_TYPES = {"attribute", "pose", "addition", "removal", "material", "season"}


@dataclass(frozen=True)
class StrengthSample:
    sample_id: str
    source_image: str
    full_prompt: str
    neutral_prompt: str
    target_edit_description: str
    edit_type: str
    target_phrase: str
    notes: str
    category_group: str = ""
    imagegen_prompt: str = ""
    image_sha256: str = ""
    train_noise_seeds: tuple[int, ...] = (1101, 1102, 1103, 1104)
    validation_noise_seeds: tuple[int, ...] = (2101, 2102)
    rollout_seeds: tuple[int, ...] = (3101, 3102, 3103)
    resolved_width: int | None = None
    resolved_height: int | None = None


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from error
    return rows


def load_metadata(path: str | Path) -> list[StrengthSample]:
    samples: list[StrengthSample] = []
    seen: set[str] = set()
    for row in read_jsonl(path):
        missing = REQUIRED_FIELDS - set(row)
        if missing:
            raise ValueError(f"metadata sample missing fields: {sorted(missing)}")
        sample = StrengthSample(
            sample_id=str(row["sample_id"]),
            source_image=str(row["source_image"]),
            full_prompt=str(row["full_prompt"]),
            neutral_prompt=str(row["neutral_prompt"]),
            target_edit_description=str(row["target_edit_description"]),
            edit_type=str(row["edit_type"]),
            target_phrase=str(row["target_phrase"]),
            notes=str(row["notes"]),
            category_group=str(row.get("category_group", row["edit_type"])),
            imagegen_prompt=str(row.get("imagegen_prompt", "")),
            image_sha256=str(row.get("image_sha256", "")),
            train_noise_seeds=tuple(int(x) for x in row.get("train_noise_seeds", [1101, 1102, 1103, 1104])),
            validation_noise_seeds=tuple(int(x) for x in row.get("validation_noise_seeds", [2101, 2102])),
            rollout_seeds=tuple(int(x) for x in row.get("rollout_seeds", [3101, 3102, 3103])),
            resolved_width=row.get("resolved_width"),
            resolved_height=row.get("resolved_height"),
        )
        if sample.sample_id in seen:
            raise ValueError(f"duplicate sample_id {sample.sample_id}")
        if sample.edit_type not in EDIT_TYPES:
            raise ValueError(f"{sample.sample_id}: unsupported edit_type {sample.edit_type}")
        if sample.neutral_prompt != "":
            raise ValueError(f"{sample.sample_id}: neutral_prompt must be the shared empty prompt")
        if not Path(sample.source_image).exists():
            raise FileNotFoundError(f"{sample.sample_id}: source image not found: {sample.source_image}")
        seen.add(sample.sample_id)
        samples.append(sample)
    if not samples:
        raise ValueError("metadata is empty")
    return samples


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_fingerprint(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().float().cpu()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def input_contract(
    *,
    target_latents: torch.Tensor,
    source_latents: torch.Tensor,
    timestep: torch.Tensor | float,
    sigma: torch.Tensor | float,
    text_ids: torch.Tensor,
    image_ids: torch.Tensor,
    seed: int,
) -> dict[str, Any]:
    return {
        "target_latents": tensor_fingerprint(target_latents),
        "source_latents": tensor_fingerprint(source_latents),
        "timestep": float(torch.as_tensor(timestep).float().flatten()[0].item()),
        "sigma": float(torch.as_tensor(sigma).float().flatten()[0].item()),
        "text_ids": tensor_fingerprint(text_ids),
        "image_ids": tensor_fingerprint(image_ids),
        "seed": int(seed),
    }


def assert_same_contract(first: dict[str, Any], second: dict[str, Any]) -> None:
    if first != second:
        differing = {key: (first.get(key), second.get(key)) for key in set(first) | set(second) if first.get(key) != second.get(key)}
        raise RuntimeError(f"teacher/student input contract mismatch: {differing}")


def native_working_size(image: Image.Image, max_area: int = 1024 * 1024, multiple: int = 16) -> tuple[int, int]:
    width, height = image.size
    scale = min(1.0, (max_area / max(width * height, 1)) ** 0.5)
    width = max(multiple, int(round(width * scale / multiple)) * multiple)
    height = max(multiple, int(round(height * scale / multiple)) * multiple)
    return width, height


def save_preprocessed_image(image: Image.Image, destination: str | Path, width: int, height: int) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS).save(destination)
    return destination


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    parent = data.pop("base_config", None)
    if parent is None:
        return data
    base = load_config(path.parent / parent)
    return deep_merge(base, data)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def refuse_overwrite(run_dir: str | Path, *, resume: bool, fingerprint: dict[str, Any]) -> Path:
    run_dir = Path(run_dir)
    identity_path = run_dir / "run_fingerprint.json"
    if run_dir.exists() and any(run_dir.iterdir()):
        if not resume:
            raise FileExistsError(f"refusing to overwrite non-empty run directory {run_dir}")
        if not identity_path.exists():
            raise RuntimeError("resume requested but run fingerprint is absent")
        if json.loads(identity_path.read_text(encoding="utf-8")) != fingerprint:
            raise RuntimeError("resume fingerprint does not match existing run")
    run_dir.mkdir(parents=True, exist_ok=True)
    identity_path.write_text(json.dumps(fingerprint, indent=2, sort_keys=True), encoding="utf-8")
    return run_dir


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def environment_fingerprint(repo_root: str | Path) -> dict[str, Any]:
    import diffusers
    import transformers

    def command(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, cwd=repo_root, text=True).strip()
        except Exception:
            return ""

    return {
        "diffusers_file": str(Path(diffusers.__file__).resolve()),
        "diffusers_version": getattr(diffusers, "__version__", ""),
        "torch_version": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "git_commit": command(["git", "rev-parse", "HEAD"]),
        "git_branch": command(["git", "branch", "--show-current"]),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "pid": os.getpid(),
    }


def assert_diffusers_checkout(expected_root: str | Path) -> None:
    import diffusers

    actual = Path(diffusers.__file__).resolve()
    expected = Path(expected_root).resolve()
    if expected not in actual.parents:
        raise RuntimeError(f"wrong diffusers import: {actual}; expected a path under {expected}")


def sample_dict(sample: StrengthSample) -> dict[str, Any]:
    return asdict(sample)

