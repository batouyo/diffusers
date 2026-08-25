"""TempFlow-style one-step SDE branching for frozen FLUX.1-Kontext.

For each fixed (source image, edit prompt, base seed), this program computes a
deterministic ODE trajectory.  At every requested timestep it samples a fixed
number of SDE actions from the same ODE state and completes each action with a
deterministic ODE suffix.  All terminal dispersion therefore has a localized
cause: that one SDE action.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from diffusers import FluxKontextPipeline
from diffusers.pipelines.flux.pipeline_flux_kontext import (
    PREFERRED_KONTEXT_RESOLUTIONS,
    calculate_shift,
    retrieve_timesteps,
)


@dataclass(frozen=True)
class Case:
    sample_id: str
    category: str
    source_image: str
    instruction: str
    base_seed: int


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_source_size(pipe: FluxKontextPipeline, image: Image.Image) -> tuple[int, int]:
    height, width = pipe.image_processor.get_default_height_width(image)
    ratio = width / height
    _, width, height = min((abs(ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS)
    multiple = pipe.vae_scale_factor * 2
    return width // multiple * multiple, height // multiple * multiple


@torch.inference_mode()
def prepare(
    pipe: FluxKontextPipeline,
    source_path: str,
    prompt: str,
    seed: int,
    steps: int,
    guidance_scale: float,
    device: torch.device,
) -> dict[str, Any]:
    source = Image.open(source_path).convert("RGB")
    source_width, source_height = choose_source_size(pipe, source)
    source_tensor = pipe.image_processor.preprocess(
        pipe.image_processor.resize(source, source_height, source_width), source_height, source_width
    )
    target_height = target_width = 1024
    generator = torch.Generator(device=device).manual_seed(seed)
    prompt_embeds, pooled, text_ids = pipe.encode_prompt(
        prompt=prompt, device=device, num_images_per_prompt=1, max_sequence_length=512
    )
    channels = pipe.transformer.config.in_channels // 4
    latents, image_latents, latent_ids, image_ids = pipe.prepare_latents(
        source_tensor, 1, channels, target_height, target_width, prompt_embeds.dtype, device, generator, None
    )
    if image_latents is None or image_ids is None:
        raise RuntimeError("Expected FLUX-Kontext source-image latents")
    all_image_ids = torch.cat([latent_ids, image_ids], dim=0)
    sigmas = np.linspace(1.0, 1.0 / steps, steps)
    mu = calculate_shift(
        latents.shape[1],
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, _ = retrieve_timesteps(pipe.scheduler, steps, device, sigmas=sigmas, mu=mu)
    guidance = torch.full((1,), guidance_scale, device=device, dtype=torch.float32)
    return {
        "source": source.resize((target_width, target_height), Image.Resampling.LANCZOS),
        "latents": latents,
        "image_latents": image_latents,
        "image_ids": all_image_ids,
        "prompt_embeds": prompt_embeds,
        "pooled": pooled,
        "text_ids": text_ids,
        "timesteps": timesteps,
        "guidance": guidance,
        "height": target_height,
        "width": target_width,
        "dtype": latents.dtype,
    }


@torch.inference_mode()
def velocity(pipe: FluxKontextPipeline, state: dict[str, Any], latents: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    batch = latents.shape[0]
    model_input = torch.cat([latents, state["image_latents"].repeat(batch, 1, 1)], dim=1)
    # Keep the Transformer call numerically identical to FluxKontextPipeline.__call__.
    # The pipeline receives bfloat16 weights/latents but does not enter an explicit
    # autocast context around this invocation.
    output = pipe.transformer(
        hidden_states=model_input,
        timestep=t.expand(batch).to(latents.dtype) / 1000,
        guidance=state["guidance"].expand(batch),
        pooled_projections=state["pooled"].repeat(batch, 1),
        encoder_hidden_states=state["prompt_embeds"].repeat(batch, 1, 1),
        txt_ids=state["text_ids"],
        img_ids=state["image_ids"],
        joint_attention_kwargs={},
        return_dict=False,
    )[0]
    return output[:, : latents.shape[1]]


def sigma_pair(pipe: FluxKontextPipeline, t: torch.Tensor, ndim: int) -> tuple[torch.Tensor, torch.Tensor]:
    index = int(pipe.scheduler.index_for_timestep(t))
    # Keep scheduler sigma values as CPU 0-D scalars. This is intentionally the
    # same representation used inside FlowMatchEulerDiscreteScheduler.step.
    sigma = pipe.scheduler.sigmas[index]
    next_sigma = pipe.scheduler.sigmas[index + 1]
    return sigma, next_sigma


def ode_step(pipe: FluxKontextPipeline, latents: torch.Tensor, prediction: torch.Tensor, t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    sigma, next_sigma = sigma_pair(pipe, t, latents.ndim)
    # Match FlowMatchEulerDiscreteScheduler.step exactly: the sample is upcast,
    # while model_output retains its bfloat16 precision through the multiplication.
    return (latents.float() + (next_sigma - sigma) * prediction).to(dtype)


def sde_step(
    pipe: FluxKontextPipeline,
    latents: torch.Tensor,
    prediction: torch.Tensor,
    t: torch.Tensor,
    branches: int,
    scale: float,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Exact public TempFlow per-step formula, except the batch is one causal state."""
    sigma, next_sigma = sigma_pair(pipe, t, latents.ndim)
    dt = next_sigma - sigma
    sigma_max = pipe.scheduler.sigmas[1]
    safe_sigma = torch.where(sigma == 1, sigma_max, sigma)
    std = torch.sqrt(sigma / (1 - safe_sigma)) * scale
    mean = latents.float() * (1 + std.square() / (2 * sigma) * dt) + prediction.float() * (
        1 + std.square() * (1 - sigma) / (2 * sigma)
    ) * dt
    mean = mean.repeat(branches, 1, 1)
    noise = torch.randn(mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)
    return (mean + std * torch.sqrt(-dt) * noise).to(dtype)


@torch.inference_mode()
def decode(pipe: FluxKontextPipeline, latents: torch.Tensor, height: int, width: int) -> list[Image.Image]:
    unpacked = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    unpacked = (unpacked / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        decoded = pipe.vae.decode(unpacked.to(pipe.vae.dtype), return_dict=False)[0]
    return pipe.image_processor.postprocess(decoded, output_type="pil")


def image_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def pairwise_l2(images: list[Image.Image]) -> float:
    values = [image_array(image) for image in images]
    pairs = [np.sqrt(np.mean((values[i] - values[j]) ** 2)) for i in range(len(values)) for j in range(i + 1, len(values))]
    return float(np.mean(pairs)) if pairs else 0.0


def save_sheet(images: list[Image.Image], title: str, path: Path) -> None:
    width, height = images[0].size
    canvas = Image.new("RGB", (len(images) * width, height + 28), "white")
    draw = ImageDraw.Draw(canvas)
    for index, image in enumerate(images):
        canvas.paste(image, (index * width, 28))
        draw.text((index * width + 4, 7), f"b{index}", fill="black")
    draw.text((4, 7), title, fill="black")
    canvas.save(path, quality=95, subsampling=0)


def load_cases(path: Path) -> list[Case]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        Case(
            sample_id=row["sample_id"],
            category=row["category"],
            source_image=str((path.parent / row["source_image"]).resolve())
            if not Path(row["source_image"]).is_absolute()
            else row["source_image"],
            instruction=row["instruction"], base_seed=int(row["base_seed"])
        )
        for row in payload["cases"]
    ]


@torch.inference_mode()
def run_unit(
    pipe: FluxKontextPipeline,
    case: Case,
    prompt: str,
    condition: str,
    seed: int,
    output: Path,
    steps: int,
    branches: int,
    scale: float,
    guidance: float,
    device: torch.device,
    branch_indices: set[int] | None = None,
    save_images: bool = False,
) -> list[dict[str, Any]]:
    state = prepare(pipe, case.source_image, prompt, seed, steps, guidance, device)
    current = state["latents"]
    ode_states = [current]
    predictions = []
    for t in state["timesteps"]:
        prediction = velocity(pipe, state, current, t)
        predictions.append(prediction)
        current = ode_step(pipe, current, prediction, t, state["dtype"])
        ode_states.append(current)
    ode_image = decode(pipe, ode_states[-1], state["height"], state["width"])[0]
    unit_dir = output / condition / case.sample_id / f"seed_{seed}"
    if save_images:
        unit_dir.mkdir(parents=True, exist_ok=True)
        state["source"].save(unit_dir / "source.png")
        ode_image.save(unit_dir / "ode_reference.png")
    rows = []
    for k, t in enumerate(state["timesteps"]):
        if branch_indices is not None and k not in branch_indices:
            continue
        generator = torch.Generator(device=device).manual_seed(seed * 1009 + k)
        branches_latent = sde_step(pipe, ode_states[k], predictions[k], t, branches, scale, generator, state["dtype"])
        local_l2 = torch.linalg.vector_norm((branches_latent.float() - ode_states[k + 1].float()).reshape(branches, -1), dim=1)
        for later_t in state["timesteps"][k + 1 :]:
            branches_latent = ode_step(pipe, branches_latent, velocity(pipe, state, branches_latent, later_t), later_t, state["dtype"])
        final_l2 = torch.linalg.vector_norm((branches_latent.float() - ode_states[-1].float()).reshape(branches, -1), dim=1)
        images = decode(pipe, branches_latent, state["height"], state["width"])
        step_dir = unit_dir / f"step_{k:02d}"
        image_paths = [""] * branches
        if save_images:
            step_dir.mkdir(exist_ok=True)
            for branch, image in enumerate(images):
                image_path = step_dir / f"branch_{branch:02d}.png"
                image.save(image_path)
                image_paths[branch] = str(image_path)
            save_sheet(images, f"{condition} | k={k}", step_dir / "sheet.jpg")
        terminal_pairwise = pairwise_l2(images)
        reference_l2 = [float(np.sqrt(np.mean((image_array(image) - image_array(ode_image)) ** 2))) for image in images]
        for branch in range(branches):
            rows.append({
                "sample_id": case.sample_id, "category": case.category, "condition": condition,
                "instruction": prompt, "base_seed": seed, "steps": steps, "branch_index": k,
                "branch": branch, "sde_scale": scale, "local_latent_l2": float(local_l2[branch].cpu()),
                "final_latent_l2": float(final_l2[branch].cpu()),
                "retention_ratio": float(final_l2[branch].cpu() / local_l2[branch].cpu().clamp_min(1e-8)),
                "terminal_pairwise_pixel_l2": terminal_pairwise,
                "terminal_to_ode_pixel_l2": reference_l2[branch],
                "image_path": image_paths[branch],
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--base-seeds", type=int, default=4)
    parser.add_argument("--branches", type=int, default=6)
    parser.add_argument("--sde-scale", type=float, default=0.7)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--condition", choices=["edit", "keep", "both"], default="edit")
    parser.add_argument(
        "--branch-indices",
        default=None,
        help="Comma-separated ODE indices to branch; default evaluates every index.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-id", default=None)
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save source, ODE reference, branch PNGs, and contact sheets.",
    )
    args = parser.parse_args()
    if args.steps != 10 or args.branches != 6:
        print("WARNING: using custom temporal-probe steps/branches.", flush=True)
    seed_all(20260722)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    pipe = FluxKontextPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16, local_files_only=True).to(device)
    pipe.set_progress_bar_config(disable=True)
    cases = load_cases(args.cases)
    if args.case_id:
        cases = [case for case in cases if case.sample_id == args.case_id]
        if not cases:
            raise ValueError(f"Unknown case id: {args.case_id}")
    if args.limit is not None:
        cases = cases[: args.limit]
    branch_indices = None
    if args.branch_indices:
        branch_indices = {int(item) for item in args.branch_indices.split(",")}
        if any(index < 0 or index >= args.steps for index in branch_indices):
            raise ValueError("branch indices must be in [0, steps)")
    all_rows: list[dict[str, Any]] = []
    for case in cases:
        conditions = [("edit", case.instruction)]
        if args.condition == "keep":
            conditions = [("keep", "Keep the image unchanged.")]
        elif args.condition == "both":
            conditions.append(("keep", "Keep the image unchanged."))
        for condition, prompt in conditions:
            for seed_offset in range(args.base_seeds):
                seed = case.base_seed + seed_offset
                print(f"{case.sample_id} {condition} seed={seed}", flush=True)
                all_rows.extend(
                    run_unit(
                        pipe, case, prompt, condition, seed, args.output, args.steps, args.branches,
                        args.sde_scale, args.guidance, device, branch_indices, args.save_images,
                    )
                )
    metrics_path = args.output / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    summary = {
        "cases": len(cases), "base_seeds": args.base_seeds, "branches": args.branches,
        "steps": args.steps, "sde_scale": args.sde_scale, "guidance": args.guidance,
        "condition": args.condition, "save_images": args.save_images, "metrics": str(metrics_path),
    }
    (args.output / "run_config.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pipe.maybe_free_model_hooks()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
