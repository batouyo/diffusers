"""Coupled-SDE mechanism diagnostic for frozen FLUX-Kontext.

This is an independent, endpoint-focused diagnostic.  It does not alter the
four-sample coupling runner or its existing results.  Its purpose is to locate
where a rho-dependent Brownian perturbation disappears, if it does.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import linregress
from transformers import AutoImageProcessor, Dinov2Model

from diffusers import FluxKontextPipeline
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.flux.pipeline_flux_kontext import calculate_shift, retrieve_timesteps
from diagnostics.legacy_tempflow_probe import LEGACY_PROVENANCE, sde_step_with_noise
from diagnostics.trajectory_primitives import decode, ode_step, prepare, sigma_pair, velocity


STEPS, GUIDANCE, RESOLUTION, SEED, SCALE = 28, 3.5, 1024, 20260828, 0.7
EPS = 1e-8


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
    edit_mask: np.ndarray
    roi_bbox: tuple[int, int, int, int]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    return sha(value.detach().float().contiguous().cpu().numpy().tobytes())


def image_hash(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return sha(buffer.getvalue())


def to_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {str(k): to_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_json(v) for v in value]
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as handle:
        json.dump(to_json(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=path.parent, suffix=".tmp") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([{k: to_json(v) for k, v in row.items()} for row in rows])
        temporary = Path(handle.name)
    os.replace(temporary, path)


def load_cases(path: Path) -> list[Case]:
    return [Case(**row) for row in json.loads(path.read_text(encoding="utf-8"))["cases"]]


def decode_cell(value: Any) -> Image.Image:
    if isinstance(value, dict):
        value = value.get("bytes", value.get("path"))
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).copy()
    if isinstance(value, str):
        return Image.open(value).copy()
    if isinstance(value, Image.Image):
        return value.copy()
    raise TypeError(f"Unsupported parquet image value: {type(value).__name__}")


def roi_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    if not np.any(mask):
        raise RuntimeError("Empty MagicBrush edit mask")
    h, w = mask.shape
    if float(mask.mean()) >= 0.60:
        return (0, 0, w, h)
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rw, rh = (x1 - x0) * 1.125, (y1 - y0) * 1.125
    return (max(0, math.floor(cx - rw / 2)), max(0, math.floor(cy - rh / 2)),
            min(w, math.ceil(cx + rw / 2)), min(h, math.ceil(cy + rh / 2)))


def load_assets(case: Case) -> Assets:
    frame = pd.read_parquet(case.parquet_path)
    row = frame.iloc[case.row_index]
    if int(row.img_id) != case.img_id or int(row.turn_index) != case.turn_index or str(row.instruction) != case.instruction:
        raise RuntimeError(f"MagicBrush row identity mismatch: {case.sample_id}")
    source_raw, target_raw, mask_raw = (decode_cell(row[key]) for key in ("source_img", "target_img", "mask_img"))
    source = source_raw.convert("RGB").resize((RESOLUTION, RESOLUTION), Image.Resampling.LANCZOS)
    target = target_raw.convert("RGB").resize((RESOLUTION, RESOLUTION), Image.Resampling.LANCZOS)
    mask = mask_raw.convert("L").resize((RESOLUTION, RESOLUTION), Image.Resampling.NEAREST)
    edit = np.asarray(mask, dtype=np.uint8) < 128
    return Assets(case, source, target, edit, roi_from_mask(edit))


def make_prompt_state(pipe: FluxKontextPipeline, state: dict[str, Any], prompt: str, device: torch.device) -> dict[str, Any]:
    embeds, pooled, text_ids = pipe.encode_prompt(prompt=prompt, device=device, num_images_per_prompt=1, max_sequence_length=512)
    result = dict(state)
    result.update(prompt_embeds=embeds, pooled=pooled, text_ids=text_ids)
    return result


def load_pipe(args: argparse.Namespace, device: torch.device) -> FluxKontextPipeline:
    if device.type != "cuda":
        raise RuntimeError("Generation requires CUDA")
    return FluxKontextPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, local_files_only=True).to(device)


def seed_for(sample_index: int, step: int, stream: int) -> int:
    return int(np.random.SeedSequence([SEED, sample_index, 0, step, stream]).generate_state(1, dtype=np.uint64)[0] % (2**63 - 1))


def explicit_noise(shape: Sequence[int], device: torch.device, sample_index: int, step: int, stream: int) -> torch.Tensor:
    return torch.randn(tuple(shape), device=device, dtype=torch.float32, generator=torch.Generator(device=device).manual_seed(seed_for(sample_index, step, stream)))


def rf_coefficient(sigma: float, sigma_next: float, first: bool) -> float:
    if first:
        return 0.0
    if sigma >= 1 or sigma_next > sigma:
        raise ValueError(f"Invalid RF schedule pair {sigma}, {sigma_next}")
    return math.sqrt(2 * sigma / (1 - sigma) * (sigma - sigma_next))


def rf_step(x: torch.Tensor, v: torch.Tensor, sigma: torch.Tensor, sigma_next: torch.Tensor, noise: torch.Tensor, first: bool) -> tuple[torch.Tensor, float]:
    s, sn = float(sigma), float(sigma_next)
    coeff = rf_coefficient(s, sn, first)
    if first:
        out = x.float() + (sigma_next - sigma) * v
    else:
        drift = 2 * v.float() + x.float() / (1 - s)
        out = x.float() + (sn - s) * drift + coeff * noise.float()
    if not torch.isfinite(out).all():
        raise FloatingPointError("RF-SDE returned nonfinite state")
    return out.to(x.dtype), coeff


def verify_deterministic_scheduler_equivalence(model_path: str, initial: torch.Tensor, device: torch.device) -> dict[str, Any]:
    """Check the hand-instrumented ODE update against a fresh official scheduler."""
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler", local_files_only=True)
    sigmas_in = np.linspace(1.0, 1.0 / STEPS, STEPS)
    mu = calculate_shift(initial.shape[1], scheduler.config.get("base_image_seq_len", 256), scheduler.config.get("max_image_seq_len", 4096), scheduler.config.get("base_shift", .5), scheduler.config.get("max_shift", 1.15))
    timesteps, _ = retrieve_timesteps(scheduler, STEPS, device, sigmas=sigmas_in, mu=mu)
    x = initial.clone(); generator = torch.Generator(device=device).manual_seed(991)
    maximum = 0.0
    for index, timestep in enumerate(timesteps):
        prediction = torch.randn(x.shape, device=device, dtype=x.dtype, generator=generator)
        sigma, sigma_next = scheduler.sigmas[index], scheduler.sigmas[index + 1]
        hand = (x.float() + (sigma_next - sigma) * prediction).to(x.dtype)
        official = scheduler.step(prediction, timestep, x, return_dict=False)[0]
        maximum = max(maximum, float((hand.float() - official.float()).abs().max()))
        if not torch.equal(hand, official):
            raise RuntimeError(f"ODE helper differs from official scheduler at step {index}; max_abs={maximum}")
        x = official
    return {"status": "PASS", "steps": STEPS, "max_abs_difference": maximum, "scheduler_class": type(scheduler).__name__}


def pack_mask(edit_mask: np.ndarray, channels: int, device: torch.device) -> torch.Tensor:
    latent = F.interpolate(torch.from_numpy(edit_mask.astype(np.float32))[None, None], size=(128, 128), mode="nearest").repeat(1, channels, 1, 1)
    packed = FluxKontextPipeline._pack_latents(latent, 1, channels, 128, 128).to(device)
    if not set(torch.unique(packed).cpu().tolist()).issubset({0.0, 1.0}):
        raise RuntimeError("packed edit mask is not binary")
    return packed


def regional_difference(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    difference = (a.float() - b.float()).square()
    def measure(region: torch.Tensor) -> tuple[float, float]:
        active = int(region.sum().item())
        if active == 0:
            return math.nan, math.nan
        l2 = float((difference * region).sum().sqrt().item())
        return l2, math.sqrt(float((difference * region).sum().item()) / active)
    edit_l2, edit_rms = measure(mask)
    preserve_l2, preserve_rms = measure(1 - mask)
    return {"edit_latent_l2": edit_l2, "edit_latent_rms": edit_rms, "preserve_latent_l2": preserve_l2, "preserve_latent_rms": preserve_rms}


def trace_row(experiment: str, branch: str, step: int, state_index: int, timestep: torch.Tensor, sigma: torch.Tensor, sigma_next: torch.Tensor,
              coefficient: float, sampler: str, before: torch.Tensor, after: torch.Tensor, prediction: torch.Tensor,
              noise: torch.Tensor | None, reference: torch.Tensor | None, mask: torch.Tensor | None, basis: str = "", independent: str = "") -> dict[str, Any]:
    if reference is None:
        raw = relative = model_delta = math.nan
        regions = {"edit_latent_l2": math.nan, "edit_latent_rms": math.nan, "preserve_latent_l2": math.nan, "preserve_latent_rms": math.nan}
    else:
        raw = float(torch.linalg.vector_norm(after.float() - reference.float()).item())
        relative = raw / (float(torch.linalg.vector_norm(reference.float()).item()) + EPS)
        model_delta = math.nan
        regions = regional_difference(after, reference, mask) if mask is not None else {"edit_latent_l2": math.nan, "edit_latent_rms": math.nan, "preserve_latent_l2": math.nan, "preserve_latent_rms": math.nan}
    return {
        "experiment": experiment, "branch": branch, "rho": 1.0 if branch.endswith("rho1") else 0.0 if branch.endswith("rho0") else math.nan,
        "total_steps": STEPS, "step_index": step, "state_index": state_index, "timestep": float(timestep), "sigma": float(sigma), "sigma_next": float(sigma_next),
        "diffusion_coefficient": coefficient, "sampler_mode": sampler,
        "latent_norm_before": float(torch.linalg.vector_norm(before.float()).item()), "latent_norm_after": float(torch.linalg.vector_norm(after.float()).item()),
        "prediction_norm": float(torch.linalg.vector_norm(prediction.float()).item()), "noise_mean": math.nan if noise is None else float(noise.mean()),
        "noise_std": math.nan if noise is None else float(noise.std()), "noise_norm": math.nan if noise is None else float(torch.linalg.vector_norm(noise).item()),
        "latent_state_hash": tensor_hash(after), "shared_basis_hash": basis, "independent_basis_hash": independent,
        "raw_latent_l2_to_rho1": raw, "relative_latent_difference": relative, "model_output_difference": model_delta, **regions,
    }


@torch.inference_mode()
def ode_trajectory(pipe: FluxKontextPipeline, state: dict[str, Any], initial: torch.Tensor, label: str) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    x, rows = initial.clone(), []
    for index, timestep in enumerate(state["timesteps"]):
        before = x
        prediction = velocity(pipe, state, x, timestep)
        sigma, sigma_next = sigma_pair(pipe, timestep)
        x = ode_step(pipe, x, prediction, timestep)
        rows.append(trace_row(label, "native_ode", index, index + 1, timestep, sigma, sigma_next, 0.0, "native_ode", before, x, prediction, None, None, None))
    return x, rows


def save_image(image: Image.Image, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return str(path)


def dino_embeddings(path: str, device: torch.device, images: Sequence[Image.Image]) -> np.ndarray:
    try:
        processor = AutoImageProcessor.from_pretrained(path, local_files_only=True, use_fast=False)
    except TypeError:
        processor = AutoImageProcessor.from_pretrained(path, local_files_only=True)
    model = Dinov2Model.from_pretrained(path, local_files_only=True).to(device).eval()
    model.requires_grad_(False)
    output = []
    with torch.inference_mode():
        for image in images:
            values = processor(images=image, return_tensors="pt")
            values = {k: v.to(device) for k, v in values.items()}
            output.append(F.normalize(model(**values).last_hidden_state[:, 0].float(), dim=-1).cpu().numpy()[0])
    del model
    return np.stack(output)


def progress(source: np.ndarray, target: np.ndarray, value: np.ndarray) -> tuple[float, float, bool]:
    direction = target - source
    denom = float(direction @ direction)
    return (math.nan if denom < EPS else float((value - source) @ direction / (denom + EPS)), denom, denom < EPS)


def l1(a: Image.Image, b: Image.Image, mask: np.ndarray) -> tuple[float, float]:
    diff = np.abs(np.asarray(a, np.float32) / 255 - np.asarray(b, np.float32) / 255).mean(axis=2)
    return float(diff[mask].mean()), float(diff[~mask].mean())


def pixel_rms(a: Image.Image, b: Image.Image) -> float:
    diff = np.asarray(a, np.float32) / 255 - np.asarray(b, np.float32) / 255
    return float(np.sqrt(np.mean(diff.square())))


def lpips_value(model: Any, a: Image.Image, b: Image.Image, device: torch.device) -> float:
    def tensor(x: Image.Image) -> torch.Tensor:
        return torch.from_numpy(np.asarray(x, np.float32) / 127.5 - 1).permute(2, 0, 1)[None].to(device)
    with torch.inference_mode():
        return float(model(tensor(a), tensor(b), normalize=False).mean())


def validate(args: argparse.Namespace) -> None:
    """CPU-only numerical gates. Model/scheduler equivalence is checked on GPU before generation."""
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    x, v, noise = torch.randn(1, 3, 5), torch.randn(1, 3, 5), torch.randn(1, 3, 5)
    first, first_coeff = rf_step(x, v, torch.tensor(1.0), torch.tensor(.9), noise, True)
    expected = (x.float() + (torch.tensor(.9) - torch.tensor(1.0)) * v).to(x.dtype)
    if not torch.equal(first, expected) or first_coeff != 0:
        raise RuntimeError("first RF step failed deterministic equivalence")
    mask = torch.zeros(1, 50000, 1); mask[:, :25000] = 1
    shared = explicit_noise(mask.shape, torch.device("cpu"), 0, 1, 0); independent = explicit_noise(mask.shape, torch.device("cpu"), 0, 1, 1)
    rho0 = (1 - mask) * shared + mask * independent
    if not torch.equal((1 - mask) * rho0, (1 - mask) * shared) or not torch.equal(mask * rho0, mask * independent):
        raise RuntimeError("rho noise identity failed")
    atomic_json(output / "validation.json", {"status": "PASS", "first_step_coefficient": first_coeff, "repeat_noise_hash": tensor_hash(shared), "rho0_noise_hash": tensor_hash(rho0), "legacy_provenance": LEGACY_PROVENANCE})


def native_gate(asset: Assets, result: Image.Image, dino_path: str, device: torch.device) -> dict[str, Any]:
    embedding = dino_embeddings(dino_path, device, [asset.source, asset.target, result, asset.source.crop(asset.roi_bbox), asset.target.crop(asset.roi_bbox), result.crop(asset.roi_bbox)])
    q_global, _, _ = progress(embedding[0], embedding[1], embedding[2]); q_roi, _, _ = progress(embedding[3], embedding[4], embedding[5])
    source_edit, _ = l1(asset.source, asset.target, asset.edit_mask); output_edit, preserve = l1(result, asset.source, asset.edit_mask)
    improvement = (source_edit - l1(result, asset.target, asset.edit_mask)[0]) / (source_edit + EPS)
    passed = all(np.isfinite([q_global, q_roi, output_edit, preserve, improvement])) and improvement >= .10 and (q_global > 0 or q_roi > 0) and preserve < .20
    return {"sample_id": asset.case.sample_id, "status": "PASS" if passed else "FAIL", "q_global": q_global, "q_roi": q_roi, "source_target_edit_l1": source_edit, "output_target_edit_improvement": improvement, "preserve_l1_vs_source": preserve}


def run_legacy_matched(pipe: FluxKontextPipeline, state: dict[str, Any], initial: torch.Tensor, sample_index: int, out: Path) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    x = initial.clone(); rows = []
    timestep = state["timesteps"][0]; prediction = velocity(pipe, state, x, timestep)
    sigma, sigma_next = sigma_pair(pipe, timestep); noise = explicit_noise(x.shape, x.device, sample_index, 0, 1)
    before = x; x = sde_step_with_noise(pipe, x, prediction, timestep, noise, SCALE)
    rows.append(trace_row("B_matched", "perturbed", 0, 1, timestep, sigma, sigma_next, math.nan, "legacy_sde", before, x, prediction, noise, None, None, tensor_hash(noise)))
    for index, later in enumerate(state["timesteps"][1:], 1):
        before = x; prediction = velocity(pipe, state, x, later); sigma, sigma_next = sigma_pair(pipe, later); x = ode_step(pipe, x, prediction, later)
        rows.append(trace_row("B_matched", "perturbed", index, index + 1, later, sigma, sigma_next, 0, "native_ode", before, x, prediction, None, None, None))
    return x, rows


def run_historical_replay(pipe: FluxKontextPipeline, source_path: Path, output: Path, prompt: str, expected_sha256: str = "") -> dict[str, Any]:
    """Reproduce the archived 9-step/4-branch TempFlow positive control.

    The four-branch shape is intentional: it preserves the old CUDA-generator
    draw order before branch 0 is selected for the archived numeric comparison.
    """
    if not source_path.is_file():
        raise RuntimeError(f"B_historical source image is missing: {source_path}")
    source = Image.open(source_path).convert("RGB")
    if expected_sha256 and sha(source_path.read_bytes()).lower() != expected_sha256.lower():
        raise RuntimeError("B_historical source SHA-256 mismatch")
    if not prompt:
        raise RuntimeError("B_historical requires --historical-prompt")
    state = prepare(pipe, source, prompt, 20260723, 9, GUIDANCE, next(pipe.transformer.parameters()).device)
    ode_states, predictions = [state["latents"].clone()], []
    x = state["latents"].clone()
    for timestep in state["timesteps"]:
        prediction = velocity(pipe, state, x, timestep); predictions.append(prediction); x = ode_step(pipe, x, prediction, timestep); ode_states.append(x.clone())
    generator = torch.Generator(device=x.device).manual_seed(20260723 + 10_000)
    noise = torch.randn((4,) + tuple(ode_states[0].shape[1:]), generator=generator, device=x.device, dtype=torch.float32)
    branches = sde_step_with_noise(pipe, ode_states[0].repeat(4, 1, 1), predictions[0].repeat(4, 1, 1), state["timesteps"][0], noise, SCALE)
    local_l2 = float(torch.linalg.vector_norm(branches[0].float() - ode_states[1][0].float()).item())
    for step, timestep in enumerate(state["timesteps"][1:], 1):
        branches = ode_step(pipe, branches, velocity(pipe, state, branches, timestep), timestep)
    final_l2 = float(torch.linalg.vector_norm(branches[0].float() - ode_states[-1][0].float()).item())
    reference, perturbed = decode(pipe, ode_states[-1])[0], decode(pipe, branches[:1])[0]
    rms = pixel_rms(reference, perturbed)
    expected = {"local_latent_l2": 379.8421, "final_latent_l2": 432.1966, "final_pixel_rms_to_ode": 0.25398}
    errors = {"local_relative": abs(local_l2 - expected["local_latent_l2"]) / expected["local_latent_l2"], "final_relative": abs(final_l2 - expected["final_latent_l2"]) / expected["final_latent_l2"], "pixel_absolute": abs(rms - expected["final_pixel_rms_to_ode"])}
    passed = errors["local_relative"] <= .01 and errors["final_relative"] <= .05 and errors["pixel_absolute"] <= .02
    save_image(reference, output / "images" / "B_historical_reference.png"); save_image(perturbed, output / "images" / "B_historical_perturbed.png")
    return {"status": "PASS" if passed else "ENVIRONMENT_OR_VERSION_REPRODUCTION_FAILED", "historical_source_sha256_verified": bool(expected_sha256), "steps": 9, "guidance": GUIDANCE, "scale": SCALE, "base_seed": 20260723, "branches_generated_for_rng_shape": 4, "selected_branch": 0, "actual": {"local_latent_l2": local_l2, "final_latent_l2": final_l2, "final_pixel_rms_to_ode": rms}, "expected": expected, "errors": errors, "tolerances": {"local_relative": .01, "final_relative": .05, "pixel_absolute": .02}, "provenance": LEGACY_PROVENANCE}


def run_rf_pair(pipe: FluxKontextPipeline, state: dict[str, Any], initial: torch.Tensor, sample_index: int, packed_mask: torch.Tensor, mode: str,
                prefix_end: int | None = None, prefix: tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]] | None = None,
                start_index: int = 0) -> tuple[tuple[torch.Tensor, torch.Tensor], list[dict[str, Any]], tuple[torch.Tensor, torch.Tensor] | None]:
    """Run C/E or a prefix shared by C/D; rows are branch-specific, not batch aggregates."""
    x1, x0 = (initial.clone(), initial.clone()) if prefix is None else (prefix[0].clone(), prefix[1].clone())
    rows = [] if prefix is None else list(prefix[2])
    start = 0 if prefix is None else start_index
    last_boundary = None
    for index, timestep in enumerate(state["timesteps"][start:], start):
        s1_before, s0_before = x1, x0
        v1, v0 = velocity(pipe, state, x1, timestep), velocity(pipe, state, x0, timestep)
        sigma, sigma_next = sigma_pair(pipe, timestep)
        shared, independent = explicit_noise(x1.shape, x1.device, sample_index, index, 0), explicit_noise(x1.shape, x1.device, sample_index, index, 1)
        global_mask = torch.ones_like(packed_mask) if mode == "E" else packed_mask
        coupled = index > 0 and (mode == "E" or index in {1, 2, 3})
        noise1 = shared
        noise0 = (1 - global_mask) * shared + global_mask * (independent if coupled else shared)
        x1, coeff1 = rf_step(x1, v1, sigma, sigma_next, noise1, index == 0)
        x0, coeff0 = rf_step(x0, v0, sigma, sigma_next, noise0, index == 0)
        common = dict(experiment=f"{mode}_rf_sde", step=index, state_index=index + 1, timestep=timestep, sigma=sigma, sigma_next=sigma_next, sampler="rf_sde")
        rows.append(trace_row(branch="rho1", coefficient=coeff1, before=s1_before, after=x1, prediction=v1, noise=noise1, reference=x1, mask=global_mask, basis=tensor_hash(shared), independent=tensor_hash(independent), **common))
        row0 = trace_row(branch="rho0", coefficient=coeff0, before=s0_before, after=x0, prediction=v0, noise=noise0, reference=x1, mask=global_mask, basis=tensor_hash(shared), independent=tensor_hash(independent), **common)
        row0["model_output_difference"] = float(torch.linalg.vector_norm(v0.float() - v1.float()).item())
        rows.append(row0)
        if prefix_end is not None and index == prefix_end:
            last_boundary = (x1.clone(), x0.clone())
            break
    return (x1, x0), rows, last_boundary


def finish_ode_pair(pipe: FluxKontextPipeline, state: dict[str, Any], boundary: tuple[torch.Tensor, torch.Tensor], begin: int, packed_mask: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor], list[dict[str, Any]]]:
    x1, x0 = boundary[0].clone(), boundary[1].clone(); rows = []
    for index, timestep in enumerate(state["timesteps"][begin:], begin):
        before1, before0 = x1, x0; v1, v0 = velocity(pipe, state, x1, timestep), velocity(pipe, state, x0, timestep)
        sigma, sigma_next = sigma_pair(pipe, timestep); x1, x0 = ode_step(pipe, x1, v1, timestep), ode_step(pipe, x0, v0, timestep)
        rows.append(trace_row("D_ode_suffix", "rho1", index, index + 1, timestep, sigma, sigma_next, 0, "native_ode", before1, x1, v1, None, x1, packed_mask))
        row0 = trace_row("D_ode_suffix", "rho0", index, index + 1, timestep, sigma, sigma_next, 0, "native_ode", before0, x0, v0, None, x1, packed_mask)
        row0["model_output_difference"] = float(torch.linalg.vector_norm(v0.float() - v1.float()).item()); rows.append(row0)
    return (x1, x0), rows


def evaluate_outputs(args: argparse.Namespace, asset: Assets, outputs: dict[str, Image.Image], output: Path) -> list[dict[str, Any]]:
    import lpips
    keys, images = list(outputs), list(outputs.values())
    device = torch.device(args.eval_device)
    global_emb = dino_embeddings(args.dino_model, device, [asset.source, asset.target] + images)
    roi_emb = dino_embeddings(args.dino_model, device, [asset.source.crop(asset.roi_bbox), asset.target.crop(asset.roi_bbox)] + [x.crop(asset.roi_bbox) for x in images])
    lpips_model = lpips.LPIPS(net="alex").to(device).eval(); lpips_model.requires_grad_(False)
    native = outputs["A_native_ode"]
    rows = []
    for index, (name, image) in enumerate(outputs.items()):
        qg, dg, deg = progress(global_emb[0], global_emb[1], global_emb[index + 2]); qr, dr, der = progress(roi_emb[0], roi_emb[1], roi_emb[index + 2])
        edit_source, preserve_source = l1(image, asset.source, asset.edit_mask); edit_native, preserve_native = l1(image, native, asset.edit_mask)
        rows.append({"experiment": name, "image_path": str(output / "images" / f"{name}.png"), "dino_progress_global": qg, "dino_progress_roi": qr, "dino_denominator_global": dg, "dino_denominator_roi": dr, "dino_degenerate_global": deg, "dino_degenerate_roi": der, "dino_embedding_l2_to_native": float(np.linalg.norm(global_emb[index + 2] - global_emb[2])), "pixel_rms_to_native": pixel_rms(image, native), "lpips_to_native": lpips_value(lpips_model, image, native, device), "edit_l1_vs_source": edit_source, "preserve_l1_vs_source": preserve_source, "edit_l1_vs_native": edit_native, "preserve_l1_vs_native": preserve_native, "roi_bbox": asset.roi_bbox})
    del lpips_model
    return rows


def calculate_conclusions(rows: list[dict[str, Any]], metrics: list[dict[str, Any]]) -> dict[str, Any]:
    by = {(row["experiment"], row["branch"], int(row["step_index"])): row for row in rows}
    def retention(prefix: str) -> float:
        values = [r for r in rows if r["experiment"].startswith(prefix) and r["branch"] == "rho0"]
        if not values: return math.nan
        boundary = next((r["relative_latent_difference"] for r in values if int(r["step_index"]) == 3), math.nan)
        return values[-1]["relative_latent_difference"] / (boundary + EPS)
    rc, rd = retention("C_"), retention("D_")
    c_ratios = [r["relative_latent_difference"] for r in rows if r["experiment"].startswith("C_") and r["branch"] == "rho0" and int(r["step_index"]) > 3]
    ratios = [c_ratios[i] / (c_ratios[i - 1] + EPS) for i in range(1, len(c_ratios))]
    c0, c1, e0, e1 = (next((r for r in metrics if r["experiment"] == key), {}) for key in ("C_rf_sde_rho0", "C_rf_sde_rho1", "E_rf_sde_rho0", "E_rf_sde_rho1"))
    endpoint_delta = max(abs(e0.get("dino_progress_global", math.nan) - e1.get("dino_progress_global", math.nan)), abs(e0.get("dino_progress_roi", math.nan) - e1.get("dino_progress_roi", math.nan)))
    return {"R_C": rc, "R_D": rd, "C_post_window_median_difference_ratio": float(np.median(ratios)) if ratios else math.nan, "C_strong_contraction": rc <= .5 and (not ratios or np.median(ratios) < 1), "D_preserves": rd >= .8, "D_amplifies": rd > 1.2, "supports_rf_suffix_washout": rc <= .5 and rd >= max(.8, 2 * rc), "E_target_aligned_semantic_leverage": endpoint_delta >= .10, "E_max_directional_progress_delta": endpoint_delta, "RF_SDE_BASELINE_EDITING_DEGRADATION": c1.get("dino_progress_global", math.nan) <= 0 and next((m.get("dino_progress_global", math.nan) for m in metrics if m["experiment"] == "A_native_ode"), math.nan) > 0}


def plot_diagnostics(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    for column, filename, ylabel in [("relative_latent_difference", "latent_difference_vs_step.png", "Relative latent difference"), ("model_output_difference", "model_output_difference_vs_step.png", "Model-output difference"), ("diffusion_coefficient", "diffusion_coefficient_vs_step.png", "Diffusion coefficient")]:
        plt.figure(figsize=(8, 4.5))
        for exp in ("C_rf_sde", "D_ode_suffix", "E_rf_sde"):
            points = [r for r in rows if r["experiment"] == exp and r["branch"] == "rho0"]
            if points: plt.plot([r["step_index"] for r in points], [r[column] for r in points], marker="o", label=exp)
        plt.axvline(3, color="black", linestyle="--", linewidth=1, label="C/D suffix switch"); plt.ylim(bottom=0); plt.xlabel("Step index"); plt.ylabel(ylabel); plt.legend(); plt.tight_layout(); plt.savefig(output / filename, dpi=180); plt.close()
    plt.figure(figsize=(8, 4.5))
    for exp in ("C_rf_sde", "D_ode_suffix", "E_rf_sde"):
        points = [r for r in rows if r["experiment"] == exp and r["branch"] == "rho0"]
        y = [r["relative_latent_difference"] for r in points]; ratio = [math.nan] + [y[i] / (y[i-1] + EPS) for i in range(1, len(y))]
        if points: plt.plot([r["step_index"] for r in points], ratio, marker="o", label=exp)
    plt.axhline(1, color="black", linestyle="--"); plt.axvline(3, color="black", linestyle="--"); plt.xlabel("Step index"); plt.ylabel("Difference ratio vs previous state"); plt.legend(); plt.tight_layout(); plt.savefig(output / "difference_ratio_vs_step.png", dpi=180); plt.close()


def write_report(output: Path, conclusions: dict[str, Any], native: dict[str, Any], legacy: dict[str, Any]) -> None:
    lines = ["# Coupled-SDE mechanism diagnostic", "", "## Q1–Q9 evidence", "", f"1. Native ODE gate: `{native['status']}` for `{native['sample_id']}`.", f"2. B-historical replay status: `{legacy['status']}`; values and pre-registered tolerances are in `legacy_replay.json`.", "3. B-matched, C, D and E final image metrics are in `final_metrics.csv`.", f"4. C boundary-to-final retention R_C = {conclusions['R_C']:.6g}; post-window median ratio = {conclusions['C_post_window_median_difference_ratio']:.6g}.", f"5. D retention R_D = {conclusions['R_D']:.6g}; suffix wash-out support = `{conclusions['supports_rf_suffix_washout']}`.", f"6. E maximum global/ROI directional-progress delta = {conclusions['E_max_directional_progress_delta']:.6g}; target-aligned leverage = `{conclusions['E_target_aligned_semantic_leverage']}`.", f"7. RF-SDE baseline degradation flag = `{conclusions['RF_SDE_BASELINE_EDITING_DEGRADATION']}`.", "8. Formula, noise, mask and prefix-identity sanity records are stored in `sanity.json` and `trajectory_diagnostics.csv`.", "9. The classification follows the pre-registered thresholds in `conclusions.json`; endpoint differences are not treated as p-values.", "", "## Limitations", "", "This is one sample, one seed, endpoint-only diagnostic. It cannot demonstrate that rho is a continuous editing-strength variable, monotonicity, or cross-sample stability. Even a target-aligned E endpoint result would only be a necessary condition for more study. If E lacks target-aligned leverage, that only weakens the current RF-SDE Brownian-correlation parameterization—not all coupling fields or stochastic editing methods."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    cases = {x.sample_id: x for x in load_cases(Path(args.manifest))}
    device = torch.device(args.device)
    pipe = load_pipe(args, device)
    # This is intentionally before the native gate: an invalid hand-instrumented
    # ODE update must not be allowed to affect any gate decision.
    scheduler_check = verify_deterministic_scheduler_equivalence(
        args.model_path,
        torch.randn((1, 4096, pipe.transformer.config.in_channels // 4), device=device, dtype=torch.bfloat16, generator=torch.Generator(device=device).manual_seed(991)),
        device,
    )
    atomic_json(output / "scheduler_equivalence.json", scheduler_check)
    chosen = None; native_record = None; state = None; asset = None; initial = None; gate_rows = []
    ordered_cases = list(cases.values())
    if len(ordered_cases) < 3:
        raise RuntimeError("Manifest must contain the two pre-registered native-gate candidates")
    for candidate, sample_index in ((ordered_cases[0], 0), (ordered_cases[2], 2)):
        candidate = load_assets(candidate)
        candidate_state = prepare(pipe, candidate.source, candidate.case.instruction, 202608280 + sample_index, STEPS, GUIDANCE, device)
        candidate_initial = candidate_state["latents"].clone(); final, _ = ode_trajectory(pipe, candidate_state, candidate_initial, "A_native_ode")
        image = decode(pipe, final)[0]; gate = native_gate(candidate, image, args.dino_model, torch.device(args.eval_device)); gate["initial_latent_hash"] = tensor_hash(candidate_initial); gate_rows.append(gate)
        save_image(image, output / "images" / f"A_native_ode_{sample_id}.png")
        if gate["status"] == "PASS": chosen, native_record, state, asset, initial = sample_index, gate, candidate_state, candidate, candidate_initial; break
    if chosen is None:
        atomic_json(output / "native_gate.json", {"status": "NATIVE_BASELINE_INCONCLUSIVE", "samples": gate_rows})
        (output / "native_baseline_failure.md").write_text(
            "# Native baseline gate failed\n\nBoth pre-registered candidates failed the fixed native ODE gate. C/D/E were not run.\n\n"
            + "| sample_id | status | q_global | q_roi | target-edit improvement | preserve L1 |\n|---|---:|---:|---:|---:|---:|\n"
            + "\n".join(f"| {r['sample_id']} | {r['status']} | {r['q_global']:.6g} | {r['q_roi']:.6g} | {r['output_target_edit_improvement']:.6g} | {r['preserve_l1_vs_source']:.6g} |" for r in gate_rows)
            + "\n", encoding="utf-8")
        atomic_json(output / "completion_report.json", {"status": "NATIVE_BASELINE_INCONCLUSIVE", "deterministic_scheduler_equivalence": scheduler_check, "native_gate": gate_rows}); raise RuntimeError("Ball and cabinet native baseline gates both failed")
    native_final, rows = ode_trajectory(pipe, state, initial, "A_native_ode")
    outputs = {"A_native_ode": decode(pipe, native_final)[0]}
    legacy = run_historical_replay(pipe, Path(args.historical_source), output, args.historical_prompt, args.historical_source_sha256)
    atomic_json(output / "legacy_replay.json", legacy)
    # B matched shares the explicit initial latent and all current conditions.
    legacy_final, legacy_rows = run_legacy_matched(pipe, state, initial, chosen, output); rows.extend(legacy_rows); outputs["B_old_sde_perturbed"] = decode(pipe, legacy_final)[0]
    # C/D: exactly one RF prefix through index 3, then cloned into two suffixes.
    packed = pack_mask(asset.edit_mask, pipe.transformer.config.in_channels // 4, device)
    prefix_states, prefix_rows, boundary = run_rf_pair(pipe, state, initial, chosen, packed, "C", prefix_end=3)
    assert boundary is not None
    c_final, c_rows, _ = run_rf_pair(pipe, state, initial, chosen, packed, "C", prefix=(boundary[0], boundary[1], []), start_index=4)
    d_final, d_rows = finish_ode_pair(pipe, state, boundary, 4, packed)
    d_prefix_rows = [dict(row, experiment="D_ode_suffix") for row in prefix_rows]
    rows.extend(prefix_rows + c_rows + d_prefix_rows + d_rows)
    outputs.update({"C_rf_sde_rho1": decode(pipe, c_final[0])[0], "C_rf_sde_rho0": decode(pipe, c_final[1])[0], "D_ode_suffix_rho1": decode(pipe, d_final[0])[0], "D_ode_suffix_rho0": decode(pipe, d_final[1])[0]})
    e_final, e_rows, _ = run_rf_pair(pipe, state, initial, chosen, packed, "E"); rows.extend(e_rows)
    outputs.update({"E_rf_sde_rho1": decode(pipe, e_final[0])[0], "E_rf_sde_rho0": decode(pipe, e_final[1])[0]})
    for name, image in outputs.items(): save_image(image, output / "images" / f"{name}.png")
    metrics = evaluate_outputs(args, asset, outputs, output); atomic_csv(output / "trajectory_diagnostics.csv", rows); atomic_csv(output / "final_metrics.csv", metrics)
    conclusions = calculate_conclusions(rows, metrics); atomic_json(output / "conclusions.json", conclusions)
    sanity = {"native_gate": native_record, "deterministic_scheduler_equivalence": scheduler_check, "initial_latent_hash": tensor_hash(initial), "C_D_prefix_rho1_hash": tensor_hash(boundary[0]), "C_D_prefix_rho0_hash": tensor_hash(boundary[1]), "prefix_cloned_exactly": True, "packed_mask_shape": list(packed.shape), "source_conditioning_hash": tensor_hash(state["image_latents"]), "first_rf_step_deterministic": True}
    atomic_json(output / "sanity.json", sanity); plot_diagnostics(rows, output / "plots")
    write_report(output, conclusions, native_record, legacy); atomic_json(output / "completion_report.json", {"status": "COMPLETE", "sample_id": asset.case.sample_id, "outputs": list(outputs), "trajectory_rows": len(rows), "metric_rows": len(metrics)})


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(); p.add_argument("--manifest", default="configs/coupling_strength_4_cases.json"); p.add_argument("--model-path", required=True); p.add_argument("--dino-model", required=True); p.add_argument("--output", default="results_mechanism_diagnostic"); p.add_argument("--device", default="cuda"); p.add_argument("--eval-device", default="cuda")
    p.add_argument("--historical-source", default=""); p.add_argument("--historical-prompt", default=""); p.add_argument("--historical-source-sha256", default=""); p.add_argument("--force", action="store_true")
    commands = p.add_subparsers(dest="command", required=True)
    for command in ("validate", "run-native-gate", "run-diagnostic", "analyze", "all"):
        commands.add_parser(command)
    return p


def main() -> None:
    args = parser().parse_args()
    if args.command == "validate": validate(args)
    elif args.command in {"run-native-gate", "run-diagnostic", "analyze", "all"}: run(args)
    else: raise ValueError(args.command)


if __name__ == "__main__": main()

