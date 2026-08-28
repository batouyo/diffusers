"""FLUX-Kontext coupling-as-strength four-sample mechanism micro-test.

This entry point is intentionally independent from the temporal/early-response
probes.  It implements the official rectified-flow equivalent SDE update with
explicit Brownian noise, keeps Kontext source-conditioning tokens immutable,
and confines variable Brownian coupling to the earliest three steps whose
diffusion coefficient is non-zero.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.stats import spearmanr

from diffusers import FluxKontextPipeline
from diffusers.pipelines.flux.pipeline_flux_kontext import calculate_shift, retrieve_timesteps
from transformers import AutoImageProcessor, Dinov2Model

from run_temporal_probe import decode, prepare, sigma_pair, velocity


SCHEMA_VERSION = 1
FORMULA_VERSION = "syncsde_rf_equivalent_v1_explicit_noise"
EXPERIMENT_SEED = 20260828
INITIAL_SEED_BASE = 202608280
IDENTITY_INSTRUCTION = "Keep the image unchanged."
RHOS = (1.0, 0.75, 0.50, 0.25, 0.0)
SEED_SLOTS = (0, 1, 2)
STEPS = 28
GUIDANCE = 3.5
RESOLUTION = 1024
ROI_SCALE = 1.125
ROI_FULL_MASK_RATIO = 0.60
TIE_EPS = 1e-8


@dataclass(frozen=True)
class Case:
    sample_id: str
    parquet_path: str
    row_index: int
    img_id: int
    turn_index: int
    instruction: str
    edit_type: str


@dataclass
class Assets:
    case: Case
    source: Image.Image
    target: Image.Image
    raw_mask: Image.Image
    edit_mask: np.ndarray
    preserve_mask: np.ndarray
    source_sha256: str
    target_sha256: str
    mask_sha256: str
    original_source_size: tuple[int, int]
    original_target_size: tuple[int, int]
    original_mask_size: tuple[int, int]
    mask_area_ratio: float
    mask_bbox: tuple[int, int, int, int]
    roi_bbox: tuple[int, int, int, int]
    roi_uses_full_image: bool


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".tmp", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(json_ready(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, rows: list[dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    fd, temp_name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".tmp", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
            writer.writeheader()
            writer.writerows([{k: json_ready(v) for k, v in row.items()} for row in rows])
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_ready(row), ensure_ascii=False, sort_keys=True) + "\n")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().float().contiguous().cpu().numpy()
    return sha256_bytes(array.tobytes())


def image_sha256(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return sha256_bytes(buffer.getvalue())


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_cases(path: Path) -> list[Case]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = [Case(**row) for row in payload["cases"]]
    if len(cases) != 4 or len({case.sample_id for case in cases}) != 4:
        raise RuntimeError("The locked manifest must contain exactly four unique samples")
    return cases


def decode_cell(value: Any) -> Image.Image:
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            value = value["bytes"]
        elif value.get("path"):
            return Image.open(value["path"]).copy()
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).copy()
    if isinstance(value, Image.Image):
        return value.copy()
    raise TypeError(f"Unsupported parquet image cell: {type(value).__name__}")


def expanded_roi(mask: np.ndarray) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], bool]:
    if mask.ndim != 2 or not np.any(mask):
        raise RuntimeError("Empty or malformed edit mask")
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    original = (x0, y0, x1, y1)
    if float(mask.mean()) >= ROI_FULL_MASK_RATIO:
        return original, (0, 0, width, height), True
    center_x, center_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    expanded_w, expanded_h = (x1 - x0) * ROI_SCALE, (y1 - y0) * ROI_SCALE
    rx0 = max(0, int(math.floor(center_x - expanded_w / 2.0)))
    ry0 = max(0, int(math.floor(center_y - expanded_h / 2.0)))
    rx1 = min(width, int(math.ceil(center_x + expanded_w / 2.0)))
    ry1 = min(height, int(math.ceil(center_y + expanded_h / 2.0)))
    if rx1 <= rx0 or ry1 <= ry0:
        raise RuntimeError("ROI expansion produced an empty crop")
    return original, (rx0, ry0, rx1, ry1), False


def load_assets(case: Case) -> Assets:
    frame = pd.read_parquet(case.parquet_path)
    if case.row_index < 0 or case.row_index >= len(frame):
        raise RuntimeError(f"Out-of-range row for {case.sample_id}: {case.row_index}")
    row = frame.iloc[case.row_index]
    actual_instruction = str(row["instruction"])
    if int(row["img_id"]) != case.img_id or int(row["turn_index"]) != case.turn_index:
        raise RuntimeError(f"Row identity mismatch for {case.sample_id}")
    if actual_instruction != case.instruction:
        raise RuntimeError(f"Instruction mismatch for {case.sample_id}: {actual_instruction!r}")
    source_original = decode_cell(row["source_img"]).convert("RGB")
    target_original = decode_cell(row["target_img"]).convert("RGB")
    mask_original = decode_cell(row["mask_img"]).convert("L")
    source = source_original.resize((RESOLUTION, RESOLUTION), Image.Resampling.LANCZOS)
    target = target_original.resize((RESOLUTION, RESOLUTION), Image.Resampling.LANCZOS)
    raw_mask = mask_original.resize((RESOLUTION, RESOLUTION), Image.Resampling.NEAREST)
    edit_mask = np.asarray(raw_mask, dtype=np.uint8) < 128
    preserve_mask = ~edit_mask
    bbox, roi, use_full = expanded_roi(edit_mask)
    return Assets(
        case=case, source=source, target=target, raw_mask=raw_mask,
        edit_mask=edit_mask, preserve_mask=preserve_mask,
        source_sha256=image_sha256(source_original), target_sha256=image_sha256(target_original),
        mask_sha256=image_sha256(mask_original), original_source_size=source_original.size,
        original_target_size=target_original.size, original_mask_size=mask_original.size,
        mask_area_ratio=float(edit_mask.mean()), mask_bbox=bbox, roi_bbox=roi,
        roi_uses_full_image=use_full,
    )


def save_overlay(image: Image.Image, mask: np.ndarray, path: Path) -> None:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    tint = np.zeros_like(base)
    tint[..., 0] = 255
    alpha = mask[..., None].astype(np.float32) * 0.35
    out = np.clip(base * (1 - alpha) + tint * alpha, 0, 255).astype(np.uint8)
    Image.fromarray(out).save(path)


def save_roi_overlay(image: Image.Image, bbox: tuple[int, int, int, int], path: Path) -> None:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle((bbox[0], bbox[1], bbox[2] - 1, bbox[3] - 1), outline=(255, 40, 40), width=6)
    out.save(path)


def materialize_assets(cases: list[Case], output: Path) -> tuple[list[Assets], list[dict[str, Any]]]:
    assets_list, rows = [], []
    for case in cases:
        assets = load_assets(case)
        assets_list.append(assets)
        folder = output / "samples" / case.sample_id
        folder.mkdir(parents=True, exist_ok=True)
        assets.source.save(folder / "source.png")
        assets.target.save(folder / "target_gt.png")
        assets.raw_mask.save(folder / "dataset_mask.png")
        Image.fromarray((assets.edit_mask * 255).astype(np.uint8)).save(folder / "edit_mask.png")
        Image.fromarray((assets.preserve_mask * 255).astype(np.uint8)).save(folder / "preserve_mask.png")
        save_overlay(assets.source, assets.edit_mask, folder / "source_mask_overlay.png")
        save_roi_overlay(assets.source, assets.roi_bbox, folder / "source_roi_overlay.png")
        assets.source.crop(assets.roi_bbox).save(folder / "source_roi.png")
        assets.target.crop(assets.roi_bbox).save(folder / "target_roi.png")
        rows.append({
            **asdict(case), "source_sha256": assets.source_sha256,
            "target_sha256": assets.target_sha256, "mask_sha256": assets.mask_sha256,
            "original_source_size": assets.original_source_size,
            "original_target_size": assets.original_target_size,
            "original_mask_size": assets.original_mask_size,
            "mask_edit_area_ratio": assets.mask_area_ratio,
            "mask_bbox": assets.mask_bbox, "roi_crop_coordinates": assets.roi_bbox,
            "roi_uses_full_image": assets.roi_uses_full_image,
        })
    atomic_json(output / "sample_manifest.json", {"schema_version": SCHEMA_VERSION, "cases": rows})
    return assets_list, rows


def rf_diffusion_coefficient(sigma: float, sigma_next: float, is_first_step: bool) -> float:
    if is_first_step:
        return 0.0
    if not 0.0 <= sigma < 1.0 or sigma_next > sigma:
        raise ValueError(f"Invalid RF sigma pair: {sigma}, {sigma_next}")
    value = 2.0 * sigma / (1.0 - sigma) * (sigma - sigma_next)
    return float(math.sqrt(max(value, 0.0)))


def rf_sde_step(
    sample: torch.Tensor,
    model_output: torch.Tensor,
    sigma: float | torch.Tensor,
    sigma_next: float | torch.Tensor,
    explicit_noise: torch.Tensor,
    is_first_step: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Official RF-equivalent SDE discretization with caller-supplied noise."""
    source_dtype = sample.dtype
    x, prediction, noise = sample.float(), model_output.float(), explicit_noise.float()
    s = float(sigma.item()) if isinstance(sigma, torch.Tensor) else float(sigma)
    sn = float(sigma_next.item()) if isinstance(sigma_next, torch.Tensor) else float(sigma_next)
    if sn > s:
        raise ValueError("Scheduler sigma must decrease")
    if is_first_step:
        drift = prediction
        coefficient = 0.0
        # Preserve the installed FlowMatchEulerDiscreteScheduler operation
        # order exactly. In particular, do not append a numerically redundant
        # ``+ 0 * noise`` operation and do not pre-upcast a bfloat16 model
        # output before its scalar multiplication.
        delta = sigma_next - sigma
        updated = sample.float() + delta * model_output
    else:
        if s >= 1.0:
            raise ValueError("Only the special first step may have sigma=1")
        drift = 2.0 * prediction + x / (1.0 - s)
        coefficient = rf_diffusion_coefficient(s, sn, False)
        updated = x + (sn - s) * drift + coefficient * noise
    finite = bool(torch.isfinite(updated).all().item())
    diagnostics = {
        "sigma": s, "sigma_next": sn, "delta_sigma": sn - s,
        "drift_norm": float(torch.linalg.vector_norm(drift).item()),
        "model_output_norm": float(torch.linalg.vector_norm(prediction).item()),
        "sample_norm": float(torch.linalg.vector_norm(x).item()),
        "diffusion_coeff": coefficient,
        "noise_mean": float(noise.mean().item()), "noise_std": float(noise.std().item()),
        "noise_norm": float(torch.linalg.vector_norm(noise).item()), "finite": finite,
    }
    if not finite:
        raise FloatingPointError("Non-finite RF SDE state")
    return updated.to(source_dtype), diagnostics


def derived_torch_seed(sample_index: int, seed_slot: int, step_index: int, stream_id: int) -> int:
    sequence = np.random.SeedSequence([EXPERIMENT_SEED, sample_index, seed_slot, step_index, stream_id])
    value = int(sequence.generate_state(1, dtype=np.uint64)[0])
    return value % (2**63 - 1)


def explicit_noise(shape: Sequence[int], device: torch.device, sample_index: int, seed_slot: int,
                   step_index: int, stream_id: int) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(
        derived_torch_seed(sample_index, seed_slot, step_index, stream_id)
    )
    return torch.randn(tuple(shape), generator=generator, device=device, dtype=torch.float32)


def coupled_noise(shared: torch.Tensor, independent: torch.Tensor, packed_edit_mask: torch.Tensor,
                  rho: float) -> torch.Tensor:
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [-1, 1]")
    mask = packed_edit_mask.to(device=shared.device, dtype=torch.float32)
    mixed_edit = rho * shared.float() + math.sqrt(max(0.0, 1.0 - rho * rho)) * independent.float()
    return (1.0 - mask) * shared.float() + mask * mixed_edit


def pack_edit_mask(edit_mask: np.ndarray, latent_channels: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    image = torch.from_numpy(edit_mask.astype(np.float32))[None, None]
    latent_side = RESOLUTION // 8
    latent = F.interpolate(image, size=(latent_side, latent_side), mode="nearest").repeat(1, latent_channels, 1, 1)
    packed = FluxKontextPipeline._pack_latents(latent, 1, latent_channels, latent_side, latent_side)
    if not torch.equal(torch.unique(packed), torch.tensor([0.0, 1.0])) and torch.unique(packed).numel() > 1:
        raise RuntimeError("Packed mask is not binary")
    return latent.to(device), packed.to(device)


def schedule_values(model_path: str) -> tuple[list[float], list[float], list[int]]:
    from diffusers import FlowMatchEulerDiscreteScheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_path, subfolder="scheduler", local_files_only=True
    )
    sigmas_input = np.linspace(1.0, 1.0 / STEPS, STEPS)
    mu = calculate_shift(
        4096, scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5), scheduler.config.get("max_shift", 1.15),
    )
    _, _ = retrieve_timesteps(scheduler, STEPS, torch.device("cpu"), sigmas=sigmas_input, mu=mu)
    sigmas = [float(x) for x in scheduler.sigmas]
    coeffs = [rf_diffusion_coefficient(sigmas[i], sigmas[i + 1], i == 0) for i in range(STEPS)]
    coupling = [i for i, value in enumerate(coeffs) if value > 0.0][:3]
    return sigmas, coeffs, coupling


def configuration(args: argparse.Namespace, manifest_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    clip_cache = Path(args.clip_cache_path)
    clip_status = {
        "model": "openai/clip-vit-large-patch14",
        "status": "available_offline" if clip_cache.exists() else "unavailable",
        "path": str(clip_cache),
        "reason": "local cache not found; no download attempted" if not clip_cache.exists() else "",
    }
    package_versions = {}
    for package in ("diffusers", "transformers", "scipy", "lpips", "pandas"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = "not-installed"
    payload = {
        "schema_version": SCHEMA_VERSION, "formula_version": FORMULA_VERSION,
        "model_path": str(Path(args.model_path).resolve()), "dino_model": args.dino_model,
        "steps": STEPS, "guidance": GUIDANCE, "resolution": RESOLUTION,
        "dtype": "bfloat16", "sde_update_dtype": "float32", "rhos": RHOS,
        "seed_slots": SEED_SLOTS, "experiment_seed": EXPERIMENT_SEED,
        "initial_seed_policy": "202608280 + sample_index",
        "brownian_seed_policy": "SeedSequence([20260828,sample_index,seed_slot,step_index,stream_id])",
        "identity_instruction": IDENTITY_INSTRUCTION, "branch_batch_size": 4,
        "mask_policy": "edit = grayscale(mask_img) < 128; nearest latent resize; official packing",
        "roi_scale": ROI_SCALE, "roi_full_mask_ratio": ROI_FULL_MASK_RATIO,
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "platform": platform.platform(), "packages": package_versions,
        },
        "optional_clip": clip_status,
        "provenance_hashes": {
            "runner_sha256": sha256_bytes(Path(__file__).read_bytes()),
            "manifest_file_sha256": sha256_bytes(Path(args.manifest).read_bytes()),
            "model_index_sha256": sha256_bytes((Path(args.model_path) / "model_index.json").read_bytes()),
            "scheduler_config_sha256": sha256_bytes(
                (Path(args.model_path) / "scheduler" / "scheduler_config.json").read_bytes()
            ),
        },
        "manifest": manifest_rows or [],
    }
    canonical = json.dumps(json_ready(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["configuration_fingerprint"] = sha256_bytes(canonical.encode("utf-8"))
    return payload


IMPLEMENTATION_NOTES = """# Implementation notes

The deterministic RF ODE convention used by the installed Kontext pipeline is

`x_next = x + (sigma_next - sigma) * model_output`, with sigma decreasing from 1 to 0.

The explicit-noise RF-equivalent SDE is implemented as follows. At `sigma=1`,
`drift=model_output` and `diffusion_coeff=0`. At every later step,
`drift=2*model_output + x/(1-sigma)` and
`diffusion_coeff=sqrt(2*sigma/(1-sigma)*(sigma-sigma_next))`. The update is
`x_next=x+(sigma_next-sigma)*drift+diffusion_coeff*noise`. Drift, coefficient,
noise mixing and update are evaluated in float32 before casting back to bfloat16.
The helper never calls a random-number generator; all noise is supplied explicitly.

The FLUX-Kontext transformer output is used directly. The `v_t=-noise_pred`
naming in a separate syncSDE pipeline is not applied a second time here.

RF-equivalent SDE 的第一个 `sigma=1` inference step 是 deterministic，
`diffusion_coeff=0`。因此，本实验中的 variable Brownian coupling 实际作用于
scheduler 中 earliest three non-zero-diffusion steps，而不是第一个 inference step。

This is a controlled adaptation of the syncSDE RF discretization to native
FLUX-Kontext source-image conditioning. The generation-latent prefix receives
the SDE update; source-conditioning latent tokens remain unchanged. This is not
claimed to be equivalent to syncSDE's RF-inversion pipeline.

Noise reuse is verified through deterministic SeedSequence-derived seeds,
tensor SHA-256 hashes, empirical correlations and a strict elementwise rho=1
identity check. Masks are resized to the VAE latent grid, expanded across latent
channels and passed through the pipeline's official packing operation.
"""


def validate_noise(args: argparse.Namespace) -> None:
    output, cases = Path(args.output), load_cases(Path(args.manifest))
    output.mkdir(parents=True, exist_ok=True)
    assets_list, manifest_rows = materialize_assets(cases, output)
    config = configuration(args, manifest_rows)
    atomic_json(output / "run_config.json", config)
    atomic_text(output / "implementation_notes.md", IMPLEMENTATION_NOTES)

    sigmas, coeffs, coupling = schedule_values(args.model_path)
    schedule_checks = {
        "sigmas_strictly_decreasing": all(sigmas[i + 1] < sigmas[i] for i in range(STEPS)),
        "terminal_sigma": sigmas[-1], "first_coefficient_exact_zero": coeffs[0] == 0.0,
        "coupling_step_indices": coupling, "coupling_sigmas": [sigmas[i] for i in coupling],
        "coupling_diffusion_coefficients": [coeffs[i] for i in coupling],
        "steps": [{"index": i, "sigma": sigmas[i], "sigma_next": sigmas[i + 1],
                   "diffusion_coeff": coeffs[i]} for i in range(STEPS)],
    }
    if not schedule_checks["sigmas_strictly_decreasing"] or not schedule_checks["first_coefficient_exact_zero"]:
        raise RuntimeError("Schedule validation failed")
    if len(coupling) != 3:
        raise RuntimeError(f"Expected at least three non-zero diffusion steps, got {coupling}")
    atomic_json(output / "schedule_validation.json", schedule_checks)

    shape = (1, 512, 64)
    half_mask = torch.zeros(shape, dtype=torch.float32)
    half_mask[:, :256] = 1.0
    noise_rows = []
    for rho in RHOS:
        shared = explicit_noise(shape, torch.device("cpu"), 0, 0, coupling[0], 0)
        independent = explicit_noise(shape, torch.device("cpu"), 0, 0, coupling[0], 1)
        target = coupled_noise(shared, independent, half_mask, rho)
        edit = half_mask.bool().reshape(-1)
        preserve = ~edit
        a, b = shared.reshape(-1).numpy(), target.reshape(-1).numpy()
        edit_corr = float(np.corrcoef(a[edit.numpy()], b[edit.numpy()])[0, 1])
        preserve_corr = float(np.corrcoef(a[preserve.numpy()], b[preserve.numpy()])[0, 1])
        row = {
            "rho": rho, "edit_empirical_correlation": edit_corr,
            "preserve_empirical_correlation": preserve_corr,
            "target_mean": float(target.mean()), "target_std": float(target.std()),
            "shared_hash": tensor_sha256(shared), "independent_hash": tensor_sha256(independent),
            "target_hash": tensor_sha256(target),
        }
        if abs(edit_corr - rho) >= 0.01 or abs(preserve_corr - 1.0) >= 0.01:
            raise RuntimeError(f"Noise correlation validation failed for rho={rho}: {row}")
        if abs(row["target_mean"]) >= 0.01 or abs(row["target_std"] - 1.0) >= 0.01:
            raise RuntimeError(f"Noise marginal validation failed for rho={rho}: {row}")
        if rho == 1.0 and not torch.equal(target, shared):
            raise RuntimeError("rho=1 is not elementwise identical to shared noise")
        noise_rows.append(row)
    repeat_a = explicit_noise(shape, torch.device("cpu"), 1, 2, 7, 0)
    repeat_b = explicit_noise(shape, torch.device("cpu"), 1, 2, 7, 0)
    distinct = explicit_noise(shape, torch.device("cpu"), 1, 2, 7, 1)
    noise_report = {
        "rows": noise_rows, "repeat_bitwise_identical": torch.equal(repeat_a, repeat_b),
        "different_stream_distinct": not torch.equal(repeat_a, distinct),
        "basis_hashes_reused_across_rho": len({row["shared_hash"] for row in noise_rows}) == 1
        and len({row["independent_hash"] for row in noise_rows}) == 1,
    }
    if not all([noise_report["repeat_bitwise_identical"], noise_report["different_stream_distinct"],
                noise_report["basis_hashes_reused_across_rho"]]):
        raise RuntimeError("Deterministic noise validation failed")
    atomic_json(output / "noise_coupling_validation.json", noise_report)

    mask_rows, roi_rows = [], []
    for index, assets in enumerate(assets_list):
        latent, packed = pack_edit_mask(assets.edit_mask, 16, torch.device("cpu"))
        reconstructed = FluxKontextPipeline._unpack_latents(packed, RESOLUTION, RESOLUTION, 8)
        folder = output / "samples" / assets.case.sample_id
        torch.save(latent.cpu(), folder / "vae_latent_mask.pt")
        torch.save(packed.cpu(), folder / "packed_edit_mask.pt")
        reconstructed_mask = reconstructed[0, 0].float().numpy()
        reconstruction_equal = bool(np.array_equal(reconstructed_mask, latent[0, 0].numpy()))
        if not reconstruction_equal:
            raise RuntimeError(f"Packed-mask reconstruction mismatch for {assets.case.sample_id}")
        Image.fromarray((reconstructed_mask * 255).astype(np.uint8)).resize(
            (RESOLUTION, RESOLUTION), Image.Resampling.NEAREST
        ).save(folder / "packed_mask_reconstruction.png")
        mask_rows.append({
            "sample_id": assets.case.sample_id, "image_mask_shape": assets.edit_mask.shape,
            "latent_mask_shape": tuple(latent.shape), "packed_mask_shape": tuple(packed.shape),
            "packed_token_count": packed.shape[1], "packed_channels": packed.shape[2],
            "binary_values": sorted(float(x) for x in torch.unique(packed)),
            "edit_preserve_complement": bool(np.all((assets.edit_mask ^ assets.preserve_mask))),
            "packed_reconstruction_exact": reconstruction_equal,
            "packed_hash": tensor_sha256(packed),
        })
        crop = assets.source.crop(assets.roi_bbox)
        roi_rows.append({
            "sample_id": assets.case.sample_id, "mask_bbox": assets.mask_bbox,
            "roi_bbox": assets.roi_bbox, "mask_area_ratio": assets.mask_area_ratio,
            "roi_uses_full_image": assets.roi_uses_full_image,
            "source_target_output_coordinates_identical_by_policy": True,
            "normal_rgb_crop": crop.mode == "RGB" and np.asarray(crop).std() > 0,
        })
    atomic_json(output / "mask_validation.json", {"samples": mask_rows})

    dino = DinoEncoder(args.dino_model, torch.device("cpu"))
    global_images: list[Image.Image] = []
    roi_images: list[Image.Image] = []
    for assets in assets_list:
        global_images.extend([assets.source, assets.target])
        roi_images.extend([assets.source.crop(assets.roi_bbox), assets.target.crop(assets.roi_bbox)])
    global_embeddings = dino.encode(global_images, batch_size=4)
    roi_embeddings = dino.encode(roi_images, batch_size=4)
    endpoint_rows = []
    for index, assets in enumerate(assets_list):
        gs, gt = global_embeddings[2 * index:2 * index + 2]
        rs, rt = roi_embeddings[2 * index:2 * index + 2]
        global_q, global_den, global_deg = directional_progress(gs, gt, np.stack([gs, gt]))
        roi_q, roi_den, roi_deg = directional_progress(rs, rt, np.stack([rs, rt]))
        row = {
            "sample_id": assets.case.sample_id,
            "global_source_progress": float(global_q[0]), "global_target_progress": float(global_q[1]),
            "global_denominator": global_den, "global_degenerate": global_deg,
            "roi_source_progress": float(roi_q[0]), "roi_target_progress": float(roi_q[1]),
            "roi_denominator": roi_den, "roi_degenerate": roi_deg,
        }
        if global_deg or roi_deg or abs(row["global_source_progress"]) > 1e-5 \
                or abs(row["global_target_progress"] - 1.0) > 1e-5 \
                or abs(row["roi_source_progress"]) > 1e-5 \
                or abs(row["roi_target_progress"] - 1.0) > 1e-5:
            raise RuntimeError(f"DINO endpoint validation failed: {row}")
        endpoint_rows.append(row)
    del dino
    atomic_json(output / "roi_validation.json", {"samples": roi_rows, "dino_endpoint_checks": endpoint_rows})
    print(json.dumps({"status": "VALIDATION_PASSED", "output": str(output), "coupling_indices": coupling}))


def make_prompt_state(pipe: FluxKontextPipeline, state: dict[str, Any], prompt: str,
                      device: torch.device) -> dict[str, Any]:
    embeds, pooled, text_ids = pipe.encode_prompt(
        prompt=prompt, device=device, num_images_per_prompt=1, max_sequence_length=512
    )
    result = dict(state)
    result.update(prompt_embeds=embeds, pooled=pooled, text_ids=text_ids)
    return result


def batch_state(state: dict[str, Any], batch: int) -> dict[str, Any]:
    # velocity() handles batch repeats for all prompt/source tensors.
    return state


def debug_step(path: Path, branch: str, sample_id: str, seed_slot: int, step_index: int,
               rho: float | None, diagnostics: dict[str, Any], shared_hash: str,
               independent_hash: str | None) -> None:
    append_jsonl(path, {
        "branch": branch, "sample_id": sample_id, "seed_slot": seed_slot,
        "step_index": step_index, "rho": rho, **diagnostics,
        "shared_noise_hash": shared_hash, "independent_noise_hash": independent_hash,
    })


@torch.inference_mode()
def run_trajectory_set(pipe: FluxKontextPipeline, assets: Assets, sample_index: int, seed_slot: int,
                       rhos: Sequence[float], output: Path, stage: str,
                       coupling_indices: list[int], device: torch.device,
                       prepared_edit_state: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    sample_folder = output / "samples" / assets.case.sample_id
    source_path = sample_folder / "source.png"
    initial_seed = INITIAL_SEED_BASE + sample_index
    edit_state = prepared_edit_state or prepare(
        pipe, str(source_path), assets.case.instruction, initial_seed, STEPS, GUIDANCE, device
    )
    reference_state = make_prompt_state(pipe, edit_state, IDENTITY_INSTRUCTION, device)
    base = edit_state["latents"].clone()
    initial_hash = tensor_sha256(base)
    conditioning_hash_before = tensor_sha256(edit_state["image_latents"])
    latent_channels = pipe.transformer.config.in_channels // 4
    latent_mask, packed_mask = pack_edit_mask(assets.edit_mask, latent_channels, device)
    if packed_mask.shape != base.shape:
        raise RuntimeError(f"Packed mask {packed_mask.shape} != generation latent {base.shape}")
    if stage in {"smoke", "preflight"}:
        latent_mask.cpu().to(torch.float32).numpy()

    noise_by_step: dict[int, tuple[torch.Tensor, torch.Tensor, str, str]] = {}
    for step_index in range(STEPS):
        shared = explicit_noise(base.shape, device, sample_index, seed_slot, step_index, 0)
        independent = explicit_noise(base.shape, device, sample_index, seed_slot, step_index, 1)
        noise_by_step[step_index] = (shared, independent, tensor_sha256(shared), tensor_sha256(independent))

    debug_path = output / "debug_steps.jsonl"
    current = base.clone()
    for step_index, timestep in enumerate(reference_state["timesteps"]):
        prediction = velocity(pipe, reference_state, current, timestep)
        sigma, sigma_next = sigma_pair(pipe, timestep, current.ndim)
        shared, _, shared_hash, _ = noise_by_step[step_index]
        current, diagnostics = rf_sde_step(current, prediction, sigma, sigma_next, shared, step_index == 0)
        debug_step(debug_path, "reference", assets.case.sample_id, seed_slot, step_index,
                   None, diagnostics, shared_hash, None)
    reference_image = decode(pipe, current, RESOLUTION, RESOLUTION)[0]
    seed_folder = sample_folder / f"seed_{seed_slot}"
    seed_folder.mkdir(parents=True, exist_ok=True)
    reference_path = seed_folder / "reference.png"
    reference_image.save(reference_path)

    rows: list[dict[str, Any]] = []
    for start in range(0, len(rhos), 4):
        rho_batch = list(rhos[start:start + 4])
        branch_count = len(rho_batch)
        current = base.repeat(branch_count, 1, 1)
        for step_index, timestep in enumerate(edit_state["timesteps"]):
            prediction = velocity(pipe, batch_state(edit_state, branch_count), current, timestep)
            sigma, sigma_next = sigma_pair(pipe, timestep, current.ndim)
            shared, independent, shared_hash, independent_hash = noise_by_step[step_index]
            noises = []
            for rho in rho_batch:
                effective_rho = rho if step_index in coupling_indices else 1.0
                noises.append(coupled_noise(shared, independent, packed_mask, effective_rho))
            batch_noise = torch.cat(noises, dim=0)
            current, diagnostics = rf_sde_step(
                current, prediction, sigma, sigma_next, batch_noise, step_index == 0
            )
            for branch_index, rho in enumerate(rho_batch):
                branch_diag = dict(diagnostics)
                branch_diag.update(
                    noise_mean=float(batch_noise[branch_index].mean()),
                    noise_std=float(batch_noise[branch_index].std()),
                    noise_norm=float(torch.linalg.vector_norm(batch_noise[branch_index]).item()),
                )
                debug_step(debug_path, "edit", assets.case.sample_id, seed_slot, step_index,
                           rho, branch_diag, shared_hash, independent_hash)
        images = decode(pipe, current, RESOLUTION, RESOLUTION)
        for rho, image in zip(rho_batch, images):
            rho_name = f"rho_{rho:.2f}"
            image_path = seed_folder / f"{rho_name}.png"
            image.save(image_path)
            rows.append({
                "schema_version": SCHEMA_VERSION, "formula_version": FORMULA_VERSION,
                "stage": stage, "sample_index": sample_index, "sample_id": assets.case.sample_id,
                "img_id": assets.case.img_id, "turn_index": assets.case.turn_index,
                "instruction": assets.case.instruction, "edit_type": assets.case.edit_type,
                "seed_slot": seed_slot, "initial_latent_seed": initial_seed,
                "initial_latent_hash": initial_hash, "rho": rho, "steps": STEPS,
                "source_conditioning_hash_before": conditioning_hash_before,
                "source_conditioning_hash_after": tensor_sha256(edit_state["image_latents"]),
                "source_conditioning_unchanged": tensor_sha256(edit_state["image_latents"]) == conditioning_hash_before,
                "guidance": GUIDANCE, "resolution": RESOLUTION,
                "coupling_step_indices": coupling_indices,
                "coupling_sigmas": [float(pipe.scheduler.sigmas[i]) for i in coupling_indices],
                "coupling_diffusion_coefficients": [rf_diffusion_coefficient(
                    float(pipe.scheduler.sigmas[i]), float(pipe.scheduler.sigmas[i + 1]), False
                ) for i in coupling_indices],
                "shared_basis_hashes": [noise_by_step[i][2] for i in coupling_indices],
                "independent_basis_hashes": [noise_by_step[i][3] for i in coupling_indices],
                "source_sha256": assets.source_sha256, "target_sha256": assets.target_sha256,
                "mask_sha256": assets.mask_sha256, "mask_edit_area_ratio": assets.mask_area_ratio,
                "roi_bbox": assets.roi_bbox, "roi_uses_full_image": assets.roi_uses_full_image,
                "reference_path": str(reference_path), "image_path": str(image_path),
            })
    marker = output / "checkpoints" / stage / assets.case.sample_id / f"seed_{seed_slot}.json"
    if any(not row["source_conditioning_unchanged"] for row in rows):
        raise RuntimeError("Source-conditioning latent tokens changed during generation")
    atomic_json(marker, {"status": "complete", "initial_latent_hash": initial_hash, "rows": rows})
    return rows


def load_pipeline(args: argparse.Namespace, device: torch.device) -> FluxKontextPipeline:
    if device.type != "cuda":
        raise RuntimeError("FLUX generation requires CUDA")
    pipe = FluxKontextPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, local_files_only=True
    ).to(device)
    pipe.set_progress_bar_config(disable=False)
    return pipe


def gpu_snapshot() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = []
    for line in completed.stdout.strip().splitlines():
        index, name, total_mib, free_mib, utilization = [part.strip() for part in line.split(",", 4)]
        rows.append({"index": int(index), "name": name, "total_mib": int(total_mib),
                     "free_mib": int(free_mib),
                     "utilization_percent": int(utilization)})
    return rows


def wait_for_resource_gate(output: Path, timeout_minutes: int) -> int:
    deadline = time.time() + timeout_minutes * 60
    first: tuple[int, dict[str, Any], float] | None = None
    history: list[dict[str, Any]] = []
    while True:
        snapshot = gpu_snapshot()
        now = time.time()
        history.append({"timestamp": now, "gpus": snapshot})
        # An advertised 80-GB H800 exposes less than 80 GiB after unit
        # conversion/driver reservation, so a literal 81,920-MiB requirement
        # can exceed physical capacity. Preserve the admission gate's intent:
        # require 80 GiB when hardware can expose it, otherwise require the card
        # to be essentially empty (within 1 GiB of its reported total).
        eligible = [row for row in snapshot
                    if row["free_mib"] >= min(80 * 1024, row["total_mib"] - 1024)
                    and row["utilization_percent"] <= 10]
        if eligible:
            chosen = max(eligible, key=lambda row: row["free_mib"])
            if first is not None and first[0] == chosen["index"] and now - first[2] >= 30:
                required_free = min(80 * 1024, chosen["total_mib"] - 1024)
                report = {"status": "PASSED", "gpu_index": chosen["index"],
                          "reported_total_mib": chosen["total_mib"],
                          "required_free_mib": required_free,
                          "nominal_requested_free_mib": 80 * 1024,
                          "hardware_capacity_adjustment": required_free < 80 * 1024,
                          "hardware_capacity_adjustment_policy": "min(81920 MiB, reported_total_mib - 1024 MiB)",
                          "required_utilization_percent": 10,
                          "observations": [first[1], chosen], "history_tail": history[-10:]}
                atomic_json(output / "resource_gate.json", report)
                return chosen["index"]
            first = (chosen["index"], chosen, now)
        else:
            first = None
        if time.time() >= deadline:
            atomic_json(output / "resource_gate.json", {
                "status": "TIMEOUT", "timeout_minutes": timeout_minutes, "history_tail": history[-10:]
            })
            raise RuntimeError("No GPU passed the 80-GiB/10%-utilization two-observation resource gate")
        time.sleep(30)


def generation_command(args: argparse.Namespace, stage: str) -> None:
    output, cases = Path(args.output), load_cases(Path(args.manifest))
    assets_list, manifest_rows = materialize_assets(cases, output)
    config = configuration(args, manifest_rows)
    existing = output / "run_config.json"
    if existing.exists():
        prior = json.loads(existing.read_text(encoding="utf-8"))
        if prior.get("configuration_fingerprint") != config["configuration_fingerprint"]:
            raise RuntimeError("Output fingerprint mismatch; refusing unsafe resume")
    atomic_json(existing, config)
    sigmas, coeffs, coupling = schedule_values(args.model_path)
    if len(coupling) != 3:
        raise RuntimeError("Could not identify three non-zero diffusion steps")
    if not args.skip_resource_gate:
        selected_gpu = wait_for_resource_gate(output, args.resource_wait_minutes)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
        device = torch.device("cuda:0")
    else:
        device = torch.device(args.device)
    cuda_index = device.index if device.index is not None else torch.cuda.current_device()
    # This PyTorch build requires the CUDA allocator/context to be initialized
    # before its peak counters can be reset.
    torch.empty(1, device=device)
    torch.cuda.reset_peak_memory_stats(cuda_index)
    pipe = load_pipeline(args, device)
    all_rows: list[dict[str, Any]] = []
    cuda_peak: dict[str, Any] = {}
    try:
        if stage == "smoke":
            units = [(0, 0, (1.0, 0.5, 0.0))]
        elif stage == "preflight":
            units = [(index, 0, (1.0,)) for index in range(4)]
        else:
            units = [(index, slot, RHOS) for index in range(4) for slot in SEED_SLOTS]
        cached_sample_index: int | None = None
        cached_edit_state: dict[str, Any] | None = None
        for sample_index, seed_slot, rhos in units:
            marker = output / "checkpoints" / stage / cases[sample_index].sample_id / f"seed_{seed_slot}.json"
            if marker.exists() and not args.force:
                payload = json.loads(marker.read_text(encoding="utf-8"))
                all_rows.extend(payload["rows"])
                continue
            if cached_sample_index != sample_index:
                sample_folder = output / "samples" / cases[sample_index].sample_id
                cached_edit_state = prepare(
                    pipe, str(sample_folder / "source.png"), cases[sample_index].instruction,
                    INITIAL_SEED_BASE + sample_index, STEPS, GUIDANCE, device,
                )
                cached_sample_index = sample_index
            rows = run_trajectory_set(
                pipe, assets_list[sample_index], sample_index, seed_slot, rhos,
                output, stage, coupling, device, cached_edit_state,
            )
            all_rows.extend(rows)
            torch.cuda.empty_cache()
    finally:
        cuda_peak = {
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(cuda_index)),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(cuda_index)),
            "max_memory_allocated_gib": float(torch.cuda.max_memory_allocated(cuda_index) / 1024**3),
            "max_memory_reserved_gib": float(torch.cuda.max_memory_reserved(cuda_index) / 1024**3),
        }
        del pipe
        torch.cuda.empty_cache()
    atomic_csv(output / f"{stage}_generation_records.csv", all_rows)
    objective_rows = evaluate_generated_stage(args, output, assets_list, all_rows, stage)
    if stage == "smoke":
        make_stage_grid(output, assets_list[0], 0, (1.0, 0.5, 0.0), output / "smoke_grid.png")
        atomic_json(output / "smoke_report.json", {
            "status": "METRICS_FINITE_PENDING_VISUAL_REVIEW", "rows": len(all_rows),
            "coupling_indices": coupling, "first_step_deterministic": coeffs[0] == 0.0,
            "objective_metric_rows": len(objective_rows),
            "all_metrics_finite": all(row["all_metrics_finite"] for row in objective_rows),
            "source_conditioning_unchanged": all(parse_bool(row["source_conditioning_unchanged"]) for row in all_rows),
            "initial_latent_hashes": sorted({row["initial_latent_hash"] for row in all_rows}),
            "shared_basis_hash_sets": sorted({json.dumps(row["shared_basis_hashes"]) for row in all_rows}),
            "independent_basis_hash_sets": sorted({json.dumps(row["independent_basis_hashes"]) for row in all_rows}),
            "cuda_peak_memory": cuda_peak,
        })
    elif stage == "preflight":
        for assets in assets_list:
            make_stage_grid(output, assets, 0, (1.0,),
                            output / "preflight" / f"{assets.case.sample_id}.png")
        review = {"samples": [{"sample_id": a.case.sample_id, "status": "PENDING"} for a in assets_list]}
        atomic_json(output / "reference_preflight_review.json", review)
        atomic_text(output / "reference_preflight.md", preflight_markdown(review))
    print(json.dumps({"status": f"{stage.upper()}_GENERATED", "rows": len(all_rows)}))


def make_stage_grid(output: Path, assets: Assets, seed_slot: int, rhos: Sequence[float], path: Path) -> None:
    folder = output / "samples" / assets.case.sample_id / f"seed_{seed_slot}"
    images = [assets.source, assets.target, Image.open(folder / "reference.png").convert("RGB")]
    labels = ["Source", "Target GT", "Reference"]
    for rho in rhos:
        images.append(Image.open(folder / f"rho_{rho:.2f}.png").convert("RGB"))
        labels.append(f"rho {rho:.2f}")
    save_grid(images, labels, path, columns=4)


def save_grid(images: Sequence[Image.Image], labels: Sequence[str], path: Path, columns: int = 4,
              thumb: int = 384) -> None:
    rows = math.ceil(len(images) / columns)
    header, gap = 32, 8
    canvas = Image.new("RGB", (columns * thumb + (columns + 1) * gap,
                               rows * (thumb + header) + (rows + 1) * gap), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (image, label) in enumerate(zip(images, labels)):
        row, col = divmod(i, columns)
        x, y = gap + col * (thumb + gap), gap + row * (thumb + header + gap)
        canvas.paste(image.resize((thumb, thumb), Image.Resampling.LANCZOS), (x, y + header))
        draw.text((x + 4, y + 8), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def preflight_markdown(review: dict[str, Any]) -> str:
    statuses = [row["status"] for row in review.get("samples", [])]
    pending = sum(value == "PENDING" for value in statuses)
    severe = sum(value == "SEVERE_FAIL" for value in statuses)
    passed = sum(value == "PASS" for value in statuses)
    if pending:
        status = "PENDING VISUAL REVIEW"
    elif severe >= 2:
        status = "REFERENCE CONSTRUCTION INCONCLUSIVE"
    elif passed >= 3:
        status = "REFERENCE PREFLIGHT PASSED"
    else:
        status = "INVALID PREFLIGHT RECORD"
    lines = ["# Reference preflight", "", f"Status: **{status}**", "",
             "| sample_id | status |", "|---|---|"]
    lines.extend(f"| {row['sample_id']} | {row['status']} |" for row in review.get("samples", []))
    lines.extend(["", "Gate: at least 3/4 samples must PASS; two or more SEVERE_FAIL results stop the experiment.", ""])
    return "\n".join(lines)


def validate_preflight_file(path: Path, cases: list[Case]) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    statuses = {row["sample_id"]: row["status"] for row in payload["samples"]}
    if set(statuses) != {case.sample_id for case in cases}:
        raise RuntimeError("Preflight status file does not match the locked four samples")
    if any(value not in {"PASS", "SEVERE_FAIL"} for value in statuses.values()):
        raise RuntimeError("Final preflight statuses must be PASS or SEVERE_FAIL")
    return statuses


class DinoEncoder:
    def __init__(self, model_path: str, device: torch.device) -> None:
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True, use_fast=False)
        except TypeError:
            self.processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = Dinov2Model.from_pretrained(model_path, local_files_only=True).to(device).eval()
        self.model.requires_grad_(False)
        self.device = device

    @torch.inference_mode()
    def encode(self, images: Sequence[Image.Image], batch_size: int = 8) -> np.ndarray:
        chunks = []
        for start in range(0, len(images), batch_size):
            inputs = self.processor(images=list(images[start:start + batch_size]), return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            embedding = self.model(**inputs).last_hidden_state[:, 0].float()
            chunks.append(F.normalize(embedding, dim=-1).cpu().numpy())
        return np.concatenate(chunks, axis=0)


def directional_progress(source: np.ndarray, target: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, float, bool]:
    direction = target - source
    denominator = float(np.dot(direction, direction))
    if denominator < TIE_EPS:
        return np.full((len(values),), np.nan), denominator, True
    return ((values - source) @ direction / (denominator + TIE_EPS)).astype(np.float64), denominator, False


def evaluate_generated_stage(args: argparse.Namespace, output: Path, assets_list: list[Assets],
                             generation_rows: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    """Objective post-generation checks for smoke and reference preflight."""
    if stage == "run":
        return []
    encoder = DinoEncoder(args.dino_model, torch.device("cpu"))
    results: list[dict[str, Any]] = []
    for assets in assets_list:
        sample_rows = [row for row in generation_rows if row["sample_id"] == assets.case.sample_id]
        if not sample_rows:
            continue
        reference_path = Path(sample_rows[0]["reference_path"])
        reference = Image.open(reference_path).convert("RGB")
        outputs = [Image.open(row["image_path"]).convert("RGB") for row in sample_rows]
        images = [assets.source, assets.target, reference] + outputs
        global_embeddings = encoder.encode(images, batch_size=4)
        roi_embeddings = encoder.encode([image.crop(assets.roi_bbox) for image in images], batch_size=4)
        global_q, global_den, global_deg = directional_progress(
            global_embeddings[0], global_embeddings[1], global_embeddings[2:]
        )
        roi_q, roi_den, roi_deg = directional_progress(
            roi_embeddings[0], roi_embeddings[1], roi_embeddings[2:]
        )
        ref_edit_l1, ref_preserve_l1 = l1_regions(reference, assets.source, assets.edit_mask)
        for index, row in enumerate(sample_rows):
            values = [float(global_q[0]), float(roi_q[0]), float(global_q[index + 1]), float(roi_q[index + 1])]
            result = {
                "stage": stage, "sample_id": assets.case.sample_id,
                "seed_slot": int(row["seed_slot"]), "rho": float(row["rho"]),
                "reference_dino_progress_global": values[0],
                "reference_dino_progress_roi": values[1],
                "output_dino_progress_global": values[2],
                "output_dino_progress_roi": values[3],
                "dino_denominator_global": global_den, "dino_denominator_roi": roi_den,
                "dino_degenerate_global": global_deg, "dino_degenerate_roi": roi_deg,
                "reference_edit_l1_vs_source": ref_edit_l1,
                "reference_preserve_l1_vs_source": ref_preserve_l1,
                "source_conditioning_unchanged": parse_bool(row["source_conditioning_unchanged"]),
                "all_metrics_finite": all(math.isfinite(value) for value in values)
                    and math.isfinite(ref_edit_l1) and math.isfinite(ref_preserve_l1)
                    and not global_deg and not roi_deg,
            }
            if not result["all_metrics_finite"] or not result["source_conditioning_unchanged"]:
                raise RuntimeError(f"{stage} objective post-generation validation failed: {result}")
            results.append(result)
        reference.crop(assets.roi_bbox).save(reference_path.with_name("reference_roi.png"))
    del encoder
    atomic_csv(output / f"{stage}_objective_metrics.csv", results)
    return results


def safe_spearman(x: Iterable[float], y: Iterable[float]) -> float:
    a, b = np.asarray(list(x), np.float64), np.asarray(list(y), np.float64)
    if len(a) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all() or np.ptp(b) <= TIE_EPS:
        return math.nan
    return float(spearmanr(a, b).statistic)


def orientation(value: float) -> str:
    if not math.isfinite(value) or value == 0:
        return "none"
    return "positive" if value > 0 else "negative"


def region_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if values.ndim == 3:
        values = values.mean(axis=2)
    selected = values[mask]
    return float(selected.mean()) if selected.size else math.nan


def l1_regions(first: Image.Image, second: Image.Image, edit_mask: np.ndarray) -> tuple[float, float]:
    a = np.asarray(first.convert("RGB"), dtype=np.float32) / 255.0
    b = np.asarray(second.convert("RGB"), dtype=np.float32) / 255.0
    difference = np.abs(a - b)
    return region_mean(difference, edit_mask), region_mean(difference, ~edit_mask)


def lpips_regions(model: Any, first: Image.Image, second: Image.Image, edit_mask: np.ndarray,
                  device: torch.device) -> tuple[float, float]:
    def tensor(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.inference_mode():
        spatial = model(tensor(first), tensor(second), normalize=False)
    mask = torch.from_numpy(edit_mask.astype(np.float32))[None, None].to(device)
    mask = F.interpolate(mask, size=spatial.shape[-2:], mode="nearest")
    edit = float((spatial * mask).sum() / mask.sum().clamp_min(1.0))
    preserve_mask = 1.0 - mask
    preserve = float((spatial * preserve_mask).sum() / preserve_mask.sum().clamp_min(1.0))
    return edit, preserve


def save_difference_maps(rho1: Image.Image, image: Image.Image, edit_mask: np.ndarray, folder: Path,
                         rho: float, gain: float = 4.0) -> None:
    a = np.asarray(rho1.convert("RGB"), dtype=np.float32) / 255.0
    b = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    raw = np.clip(np.abs(a - b) * gain, 0, 1)
    folder.mkdir(parents=True, exist_ok=True)
    for name, array in [
        ("raw", raw), ("edit", raw * edit_mask[..., None]),
        ("preserve", raw * (~edit_mask)[..., None]),
    ]:
        Image.fromarray((array * 255).astype(np.uint8)).save(folder / f"rho_{rho:.2f}_{name}.png")


def analyze(args: argparse.Namespace) -> None:
    import lpips

    output, cases = Path(args.output), load_cases(Path(args.manifest))
    assets_list = [load_assets(case) for case in cases]
    status_path = Path(args.preflight_status)
    statuses = validate_preflight_file(status_path, cases)
    severe = sum(value == "SEVERE_FAIL" for value in statuses.values())
    review = {"samples": [{"sample_id": case.sample_id, "status": statuses[case.sample_id]} for case in cases]}
    if severe >= 2:
        atomic_text(output / "reference_preflight_failed.md", preflight_markdown(review))
        atomic_json(output / "completion_report.json", {
            "status": "REFERENCE CONSTRUCTION INCONCLUSIVE", "preflight": review,
            "full_sweep_executed": False,
        })
        print(json.dumps({"status": "REFERENCE CONSTRUCTION INCONCLUSIVE"}))
        return
    atomic_text(output / "reference_preflight.md", preflight_markdown(review))

    generation_file = output / "run_generation_records.csv"
    if not generation_file.exists():
        raise RuntimeError("Full run records are absent; run the `run` subcommand first")
    with generation_file.open(newline="", encoding="utf-8") as handle:
        generation_rows = list(csv.DictReader(handle))
    if len(generation_rows) != 60:
        raise RuntimeError(f"Expected 60 generation rows, found {len(generation_rows)}")

    device = torch.device(args.eval_device)
    encoder = DinoEncoder(args.dino_model, device)
    lpips_model = lpips.LPIPS(net="alex", spatial=True).to(device).eval()
    lpips_model.requires_grad_(False)
    rows: list[dict[str, Any]] = []
    for sample_index, assets in enumerate(assets_list):
        subset = [row for row in generation_rows if row["sample_id"] == assets.case.sample_id]
        images = [Image.open(row["image_path"]).convert("RGB") for row in subset]
        global_images = [assets.source, assets.target] + images
        roi_images = [image.crop(assets.roi_bbox) for image in global_images]
        global_embeddings = encoder.encode(global_images)
        roi_embeddings = encoder.encode(roi_images)
        global_progress, global_denominator, global_degenerate = directional_progress(
            global_embeddings[0], global_embeddings[1], global_embeddings[2:]
        )
        roi_progress, roi_denominator, roi_degenerate = directional_progress(
            roi_embeddings[0], roi_embeddings[1], roi_embeddings[2:]
        )
        for local_index, (record, image) in enumerate(zip(subset, images)):
            slot, rho = int(record["seed_slot"]), float(record["rho"])
            seed_folder = output / "samples" / assets.case.sample_id / f"seed_{slot}"
            rho1 = Image.open(seed_folder / "rho_1.00.png").convert("RGB")
            reference = Image.open(seed_folder / "reference.png").convert("RGB")
            edit_rho1, preserve_rho1 = l1_regions(image, rho1, assets.edit_mask)
            _, preserve_source = l1_regions(image, assets.source, assets.edit_mask)
            lp_edit_rho1, lp_preserve_rho1 = lpips_regions(lpips_model, image, rho1, assets.edit_mask, device)
            _, lp_preserve_source = lpips_regions(lpips_model, image, assets.source, assets.edit_mask, device)
            leakage_l1 = preserve_rho1 / (edit_rho1 + TIE_EPS) if rho != 1.0 else math.nan
            leakage_lpips = lp_preserve_rho1 / (lp_edit_rho1 + TIE_EPS) if rho != 1.0 else math.nan
            row = dict(record)
            row.update({
                "rho": rho, "reference_preflight_status": statuses[assets.case.sample_id],
                "dino_progress_global": float(global_progress[local_index]),
                "dino_progress_roi": float(roi_progress[local_index]),
                "dino_denominator_global": global_denominator, "dino_denominator_roi": roi_denominator,
                "dino_degenerate_global": global_degenerate, "dino_degenerate_roi": roi_degenerate,
                "roi_x0": assets.roi_bbox[0], "roi_y0": assets.roi_bbox[1],
                "roi_x1": assets.roi_bbox[2], "roi_y1": assets.roi_bbox[3],
                "preserve_l1_vs_source": preserve_source,
                "preserve_lpips_vs_source": lp_preserve_source,
                "preserve_l1_vs_rho1": preserve_rho1,
                "preserve_lpips_vs_rho1": lp_preserve_rho1,
                "edit_l1_vs_rho1": edit_rho1, "edit_lpips_vs_rho1": lp_edit_rho1,
                "leakage_ratio_l1": leakage_l1, "leakage_ratio_lpips": leakage_lpips,
                "reference_path": str(seed_folder / "reference.png"),
            })
            rows.append(row)
            image.crop(assets.roi_bbox).save(seed_folder / f"rho_{rho:.2f}_roi.png")
            if rho != 1.0:
                save_difference_maps(rho1, image, assets.edit_mask, seed_folder / "diff_vs_rho1", rho)
        assets.source.crop(assets.roi_bbox).save(output / "samples" / assets.case.sample_id / "source_roi.png")
        assets.target.crop(assets.roi_bbox).save(output / "samples" / assets.case.sample_id / "target_roi.png")
        for slot in SEED_SLOTS:
            folder = output / "samples" / assets.case.sample_id / f"seed_{slot}"
            reference = Image.open(folder / "reference.png").convert("RGB")
            reference.crop(assets.roi_bbox).save(folder / "reference_roi.png")
            make_stage_grid(output, assets, slot, RHOS, folder / "grid.png")

    atomic_csv(output / "raw_results.csv", rows)
    per_seed = compute_per_seed(rows)
    atomic_csv(output / "per_seed_metrics.csv", per_seed)
    per_sample = compute_per_sample(per_seed, rows, statuses)
    atomic_csv(output / "per_sample_summary.csv", per_sample)
    summary, classification = build_summary(per_seed, per_sample, rows, statuses)
    atomic_text(output / "summary.md", summary)
    completion = {
        "status": "COMPLETE", "classification": classification,
        "raw_rows": len(rows), "per_seed_rows": len(per_seed), "per_sample_rows": len(per_sample),
        "references": 12, "edit_outputs": 60,
    }
    atomic_json(output / "completion_report.json", completion)
    print(json.dumps(completion))


def compute_per_seed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    keys = sorted({(row["sample_id"], int(row["seed_slot"])) for row in rows})
    for sample_id, slot in keys:
        group = sorted([row for row in rows if row["sample_id"] == sample_id and int(row["seed_slot"]) == slot],
                       key=lambda row: float(row["rho"]), reverse=True)
        rhos = [float(row["rho"]) for row in group]
        global_values = [float(row["dino_progress_global"]) for row in group]
        roi_values = [float(row["dino_progress_roi"]) for row in group]
        sg, sr = safe_spearman(rhos, global_values), safe_spearman(rhos, roi_values)
        leakage = [float(row["leakage_ratio_l1"]) for row in group if float(row["rho"]) != 1.0]
        result.append({
            "sample_id": sample_id, "seed_slot": slot,
            "spearman_global": sg, "progress_range_global": float(np.ptp(global_values)),
            "orientation_global": orientation(sg),
            "spearman_roi": sr, "progress_range_roi": float(np.ptp(roi_values)),
            "orientation_roi": orientation(sr),
            "global_qualifying_direction": "positive" if sg >= 0.6 else "negative" if sg <= -0.6 else "none",
            "median_leakage_l1": float(np.median(leakage)),
            "reference_preflight_status": group[0]["reference_preflight_status"],
        })
    if len(result) != 12:
        raise RuntimeError(f"Expected 12 per-seed rows, got {len(result)}")
    return result


def compute_per_sample(per_seed: list[dict[str, Any]], rows: list[dict[str, Any]],
                       statuses: dict[str, str]) -> list[dict[str, Any]]:
    result = []
    for sample_id in sorted(statuses):
        units = sorted([row for row in per_seed if row["sample_id"] == sample_id], key=lambda row: row["seed_slot"])
        positive = sum(float(row["spearman_global"]) >= 0.6 for row in units)
        negative = sum(float(row["spearman_global"]) <= -0.6 for row in units)
        dominant = "positive" if positive >= 2 else "negative" if negative >= 2 else "inconsistent/none"
        roi_positive = sum(float(row["spearman_roi"]) >= 0.6 for row in units)
        roi_negative = sum(float(row["spearman_roi"]) <= -0.6 for row in units)
        roi_summary = "positive" if roi_positive >= 2 else "negative" if roi_negative >= 2 else "inconsistent/none"
        sample_rows = [row for row in rows if row["sample_id"] == sample_id and float(row["rho"]) != 1.0]
        result.append({
            "sample_id": sample_id,
            **{f"spearman_global_seed_{i}": units[i]["spearman_global"] for i in range(3)},
            **{f"spearman_roi_seed_{i}": units[i]["spearman_roi"] for i in range(3)},
            "global_sample_dominant_orientation": dominant,
            "roi_sample_orientation_summary": roi_summary,
            "global_qualifying_seed_count": max(positive, negative),
            "median_progress_range_global": float(np.median([row["progress_range_global"] for row in units])),
            "median_progress_range_roi": float(np.median([row["progress_range_roi"] for row in units])),
            "median_leakage_l1": float(np.median([float(row["leakage_ratio_l1"]) for row in sample_rows])),
            "median_leakage_lpips": float(np.median([float(row["leakage_ratio_lpips"]) for row in sample_rows])),
            "reference_preflight_status": statuses[sample_id],
            "dino_degenerate_global": any(parse_bool(row["dino_degenerate_global"]) for row in sample_rows),
            "dino_degenerate_roi": any(parse_bool(row["dino_degenerate_roi"]) for row in sample_rows),
        })
    return result


def build_summary(per_seed: list[dict[str, Any]], per_sample: list[dict[str, Any]],
                  rows: list[dict[str, Any]], statuses: dict[str, str]) -> tuple[str, str]:
    orientations = {direction: [row for row in per_sample
                                if row["global_sample_dominant_orientation"] == direction
                                and row["reference_preflight_status"] == "PASS"]
                    for direction in ("positive", "negative")}
    dominant_direction = max(orientations, key=lambda key: len(orientations[key]))
    dominant_count = len(orientations[dominant_direction])
    median_range = float(np.median([float(row["progress_range_global"]) for row in per_seed]))
    leakages = [float(row["leakage_ratio_l1"]) for row in rows if float(row["rho"]) != 1.0]
    median_leakage = float(np.median(leakages))
    roi_agreement = sum(row["global_sample_dominant_orientation"] == row["roi_sample_orientation_summary"]
                        and row["global_sample_dominant_orientation"] in {"positive", "negative"}
                        for row in per_sample)
    numeric_go = dominant_count >= 3 and median_range >= 0.15 and median_leakage < 0.3
    # The locked Strong-GO definition also requires visual evidence. Numerical
    # success is therefore reported as pending qualitative confirmation.
    classification = "Case A candidate (qualitative grid confirmation required)" if numeric_go else (
        "Case B" if median_range < 0.15 and median_leakage < 0.3 else
        "Case C" if dominant_count < 3 else
        "Case D" if median_leakage >= 0.3 else "Case E"
    )
    lines = [
        "# Coupling-as-Strength micro-test summary", "",
        f"Reference preflight: {sum(v == 'PASS' for v in statuses.values())}/4 PASS.",
        f"Global dominant orientation: {dominant_direction}, {dominant_count}/4 PASS samples.",
        f"Median global progress range across 12 sample/seed units: {median_range:.6f}.",
        f"Median L1 leakage ratio across 48 non-rho1 branches: {median_leakage:.6f}.",
        f"ROI/global sample-orientation agreement: {roi_agreement}/4.",
        f"Fixed-rule classification: **{classification}**.", "",
        "Per-sample global and ROI statistics are recorded without p-values in `per_sample_summary.csv`.",
        "ROI-DINO is a localization-aware parallel diagnostic and does not replace the locked global-DINO threshold.", "",
        "RF-equivalent SDE 的第一个 `sigma=1` inference step 是 deterministic，`diffusion_coeff=0`。因此，本实验中的 variable Brownian coupling 实际作用于 scheduler 中 earliest three non-zero-diffusion steps，而不是第一个 inference step。", "",
        "若本实验未观察到稳定、方向一致且低泄漏的 strength-control 信号，结论只是否定当前 FLUX-Kontext + identity-reference + RF-equivalent SDE + early edit-region scalar-rho coupling parameterization 的可行性。该结果不能被扩大解释为所有 trajectory-coupling 方法、所有 coupling fields 或所有 stochastic editing formulations 都不可能实现连续强度控制。", "",
    ]
    return "\n".join(lines), classification


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--manifest", default="configs/coupling_strength_4_cases.json")
    root.add_argument("--output", default="outputs/coupling_strength_microtest")
    root.add_argument("--model-path", default="/root/autodl-tmp/FLUX.1-Kontext-dev")
    root.add_argument("--dino-model", default="/root/autodl-tmp/models--facebook--dinov2-base/snapshots/f9e44c814b77203eaa57a6bdbbd535f21ede1415")
    root.add_argument("--clip-cache-path", default="/root/autodl-tmp/models--openai--clip-vit-large-patch14")
    root.add_argument("--device", default="cuda")
    root.add_argument("--eval-device", default="cuda")
    root.add_argument("--force", action="store_true")
    root.add_argument("--skip-resource-gate", action="store_true")
    root.add_argument("--resource-wait-minutes", type=int, default=1440)
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("validate-noise")
    sub.add_parser("smoke")
    sub.add_parser("preflight")
    sub.add_parser("run")
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--preflight-status", required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "validate-noise":
        validate_noise(args)
    elif args.command in {"smoke", "preflight", "run"}:
        if args.command == "run":
            if not hasattr(args, "preflight_status"):
                status_file = Path(args.output) / "reference_preflight_review.json"
            else:
                status_file = Path(args.preflight_status)
            statuses = validate_preflight_file(status_file, load_cases(Path(args.manifest)))
            if sum(value == "SEVERE_FAIL" for value in statuses.values()) >= 2:
                review = {"samples": [{"sample_id": key, "status": value} for key, value in statuses.items()]}
                atomic_text(Path(args.output) / "reference_preflight_failed.md", preflight_markdown(review))
                atomic_json(Path(args.output) / "completion_report.json", {
                    "status": "REFERENCE CONSTRUCTION INCONCLUSIVE", "full_sweep_executed": False,
                })
                return
        generation_command(args, args.command)
    elif args.command == "analyze":
        analyze(args)


if __name__ == "__main__":
    main()

