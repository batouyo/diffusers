#!/usr/bin/env python
"""Minimal shared-prefix/branching experiment for FLUX-Kontext H3."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO_ROOT / "examples" / "research_projects" / "flux_kontext_routing_causal_probe"))

from h3_metrics import compute_metrics
from pie_bench import CATEGORIES, load_samples
from plot_h3 import aggregate, make_plots, write_csv
from run_routing_probe import load_pipeline
from run_target_residual_oracle import capture_trajectory, direct_velocity, encode_conditioning
from vkeep_control import compute_v_keep


DEFAULT_BRANCHING = (0, 1, 2, 3, 4, 5, 8, 14)
DEFAULT_STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="/data15/hyp/dataset/PIE-Bench")
    parser.add_argument("--output-root", default="/data15/hyp/experiments/flux_kontext_h3_branch_probe/pilot50")
    parser.add_argument("--model-path", default="/data15/hyp/weight/FLUX.1-Kontext-dev")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--true-cfg-scale", type=float, default=1.0)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--per-category", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=20260822)
    parser.add_argument("--branching-steps", nargs="+", type=int, default=list(DEFAULT_BRANCHING))
    parser.add_argument("--strengths", nargs="+", type=float, default=list(DEFAULT_STRENGTHS))
    parser.add_argument("--skip-perceptual", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sample-id")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--trajectory-cache-root")
    return parser.parse_args()


def base_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_path": args.model_path,
        "dtype": args.dtype,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "true_cfg_scale": args.true_cfg_scale,
        "max_sequence_length": args.max_sequence_length,
    }


def _expand(value: torch.Tensor, batch: int, device: str, *, ids: bool = False) -> torch.Tensor:
    value = value.to(device)
    if ids and value.ndim == 2:
        value = value.unsqueeze(0)
    if value.shape[0] == batch:
        return value
    if value.shape[0] != 1:
        raise ValueError(f"cannot expand conditioning batch {value.shape[0]} to {batch}")
    return value.expand(batch, *value.shape[1:])


def batched_velocity(
    pipeline: Any,
    current: torch.Tensor,
    timestep: torch.Tensor,
    image_tail: torch.Tensor,
    img_ids: torch.Tensor,
    conditioning: dict[str, torch.Tensor],
    guidance_scale: float,
    device: str,
) -> torch.Tensor:
    batch = current.shape[0]
    expanded_conditioning = {
        key: value.to(device) if key == "txt_ids" else _expand(value, batch, device)
        for key, value in conditioning.items()
    }
    return direct_velocity(
        pipeline.transformer,
        current,
        timestep.to(device).flatten()[0].expand(batch),
        _expand(image_tail, batch, device),
        img_ids.to(device),
        expanded_conditioning,
        guidance_scale,
        device,
    )


def branch_rollout(
    pipeline: Any,
    trajectory: dict[str, Any],
    conditioning: dict[str, torch.Tensor],
    branching_step: int,
    strengths: list[float],
    sigmas: list[float],
    guidance_scale: float,
    device: str,
) -> torch.Tensor:
    target_inputs = trajectory["target_inputs"]
    if branching_step < 0 or branching_step >= len(target_inputs):
        raise ValueError(f"branching step {branching_step} is outside trajectory")
    current = target_inputs[branching_step].to(device)
    current = current.expand(len(strengths), *current.shape[1:]).clone()
    source = trajectory["image_tail"].to(device)
    source = source.expand(len(strengths), *source.shape[1:])
    strength_tensor = torch.tensor(strengths, device=device, dtype=torch.float32).view(-1, 1, 1)
    with torch.inference_mode():
        for step in range(branching_step, len(target_inputs)):
            sigma = float(sigmas[step])
            sigma_next = float(sigmas[step + 1])
            timestep = trajectory["timesteps"][step]
            v_edit = batched_velocity(
                pipeline,
                current,
                timestep,
                trajectory["image_tail"],
                trajectory["img_ids"],
                conditioning,
                guidance_scale,
                device,
            )
            # s=1 is the normal instruction endpoint by definition. Reusing
            # the captured full trajectory velocity prevents batch-kernel
            # rounding from changing the endpoint used in R_edit(k).
            if strengths[-1] == 1.0:
                v_edit[-1] = trajectory["velocities"][step].to(device=device, dtype=v_edit.dtype)
            v_keep = compute_v_keep(current, source, sigma).to(v_edit.dtype)
            velocity = (strength_tensor * v_edit.float() + (1.0 - strength_tensor) * v_keep.float()).to(current.dtype)
            current = (current.float() + (sigma_next - sigma) * velocity.float()).to(current.dtype)
            if not torch.isfinite(current.float()).all():
                raise FloatingPointError(f"non-finite latent at branching={branching_step}, step={step}")
    return current.detach().cpu()


def decode_target_image(pipeline: Any, packed: torch.Tensor, width: int, height: int) -> Image.Image:
    latents = pipeline._unpack_latents(
        packed.to(pipeline._execution_device), height, width, pipeline.vae_scale_factor
    )
    shift = float(getattr(pipeline.vae.config, "shift_factor", 0.0))
    latents = latents / float(pipeline.vae.config.scaling_factor) + shift
    with torch.inference_mode():
        decoded = pipeline.vae.decode(latents.to(dtype=pipeline.vae.dtype), return_dict=False)[0]
    return pipeline.image_processor.postprocess(decoded, output_type="pil")[0]


def _load_perceptual(args: argparse.Namespace) -> Any | None:
    if args.skip_perceptual:
        return None
    from strength_overfit_evaluation import PerceptualModels

    return PerceptualModels(device=args.device)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device == "cpu":
        raise ValueError("capture_trajectory uses the CUDA generator; run H3 on H20 with --device cuda")
    if args.num_inference_steps != 28:
        raise ValueError("H3 pilot is defined for the default 28-step FLUX-Kontext schedule")
    if args.limit <= 0 or args.per_category <= 0:
        raise ValueError("limit and per-category must be positive")
    if any(step < 0 or step >= args.num_inference_steps for step in args.branching_steps):
        raise ValueError("branching steps must be valid denoising indices")
    if sorted(args.strengths) != list(args.strengths) or args.strengths[0] < 0 or args.strengths[-1] > 1:
        raise ValueError("strengths must be sorted and lie in [0, 1]")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    samples = load_samples(
        args.dataset_root,
        output_root / "dataset" / "source_images",
        per_category=args.per_category,
        base_seed=args.base_seed,
    )
    samples = samples[: args.limit]
    if args.sample_id is not None:
        samples = [sample for sample in samples if sample.sample_id == args.sample_id]
    samples = samples[args.start_index : args.end_index]
    if not samples:
        raise ValueError("no samples selected")
    config = base_config(args)
    pipeline = load_pipeline(config, args.device)
    pipeline.set_progress_bar_config(disable=True)
    models = _load_perceptual(args)
    trajectory_cache_root = Path(args.trajectory_cache_root) if args.trajectory_cache_root else None
    manifest = {
        "model_path": args.model_path,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "true_cfg_scale": args.true_cfg_scale,
        "branching_steps": args.branching_steps,
        "strengths": args.strengths,
        "dataset_root": args.dataset_root,
        "samples": [sample.sample_id for sample in samples],
        "mask_encoding": "flat (start, length) intervals; edit region; preservation is complement",
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_root / "manifest.jsonl").write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for sample_index, sample in enumerate(samples, start=1):
        print(f"[h3] sample {sample_index}/{len(samples)} {sample.sample_id}", flush=True)
        row = {
            "sample_id": sample.sample_id,
            "source_image": sample.source_image,
            "seed": sample.seed,
        }
        source = Image.open(sample.source_image).convert("RGB")
        trajectory_path = output_root / "trajectories" / f"{sample.sample_id}.pt"
        cached_trajectory_path = (
            trajectory_cache_root / "trajectories" / f"{sample.sample_id}.pt"
            if trajectory_cache_root is not None
            else trajectory_path
        )
        normal_image_path = output_root / "images" / sample.sample_id / "normal_full_edit.png"
        cached_normal_image_path = (
            trajectory_cache_root / "images" / sample.sample_id / "normal_full_edit.png"
            if trajectory_cache_root is not None
            else normal_image_path
        )
        if args.resume and cached_trajectory_path.is_file() and cached_normal_image_path.is_file():
            trajectory = torch.load(cached_trajectory_path, map_location="cpu", weights_only=False)
            full_image = Image.open(cached_normal_image_path).convert("RGB")
        else:
            full_image, trajectory = capture_trajectory(pipeline, config, row, sample.target_prompt)
        working_size = full_image.size
        source_eval = source.resize(working_size, Image.Resampling.LANCZOS)
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        if not trajectory_path.is_file() or not args.resume:
            torch.save({**trajectory, "sigmas": pipeline.scheduler.sigmas.detach().cpu()}, trajectory_path)
        conditioning = encode_conditioning(pipeline, config, sample.target_prompt)
        if "sigmas" in trajectory:
            scheduler_sigmas = trajectory["sigmas"].detach().float().cpu().tolist()
        else:
            scheduler_sigmas = pipeline.scheduler.sigmas.detach().float().cpu().tolist()
        if len(scheduler_sigmas) < args.num_inference_steps + 1:
            scheduler_sigmas = [float(value.flatten()[0].item()) / 1000.0 for value in trajectory["timesteps"]] + [0.0]
        image_root = output_root / "images" / sample.sample_id
        image_root.mkdir(parents=True, exist_ok=True)
        full_image.save(image_root / "normal_full_edit.png")
        for branching_step in args.branching_steps:
            states = branch_rollout(
                pipeline,
                trajectory,
                conditioning,
                branching_step,
                args.strengths,
                scheduler_sigmas,
                args.guidance_scale,
                args.device,
            )
            outputs = []
            branch_root = image_root / f"k_{branching_step:02d}"
            branch_root.mkdir(parents=True, exist_ok=True)
            for index, strength in enumerate(args.strengths):
                image = decode_target_image(pipeline, states[index : index + 1], working_size[0], working_size[1])
                outputs.append(image)
                image.save(branch_root / f"s_{strength:.2f}.png")
            metric = compute_metrics(
                source=source_eval,
                outputs=outputs,
                strengths=args.strengths,
                edit_mask=sample.mask,
                models=models,
                target_text=sample.target_prompt,
            )
            final_step = args.num_inference_steps - 1
            expected_full_final = (
                trajectory["target_inputs"][final_step].float()
                + (float(scheduler_sigmas[final_step + 1]) - float(scheduler_sigmas[final_step]))
                * trajectory["velocities"][final_step].float()
            )
            normal_error = float((states[-1].float() - expected_full_final.float()).square().mean().sqrt().item())
            record = {
                "sample_id": sample.sample_id,
                "category": sample.category,
                "branching_step": branching_step,
                "seed": sample.seed,
                "source_prompt": sample.source_prompt,
                "target_prompt": sample.target_prompt,
                "strengths": args.strengths,
                "normal_suffix_rms_error": normal_error,
                **metric,
            }
            records.append(record)
            print(json.dumps({key: record.get(key) for key in ("sample_id", "branching_step", "edit_dynamic_range", "monotonicity", "normal_suffix_rms_error")}), flush=True)
    records_path = output_root / "records" / "per_sample.jsonl"
    _write_jsonl(records_path, records)
    aggregate_rows = aggregate(records)
    write_csv(output_root / "tables" / "aggregate_by_k.csv", aggregate_rows)
    write_csv(output_root / "tables" / "category_by_k.csv", aggregate_rows)
    write_csv(output_root / "tables" / "per_sample.csv", records)
    make_plots(aggregate_rows, output_root / "plots")
    summary = {
        "samples": len(samples),
        "branching_steps": args.branching_steps,
        "records": len(records),
        "elapsed_seconds": time.perf_counter() - started,
        "output_root": str(output_root),
    }
    (output_root / "reports").mkdir(parents=True, exist_ok=True)
    (output_root / "reports" / "h3_summary.md").write_text(
        "# FLUX-Kontext H3 pilot\n\n" + json.dumps(summary, indent=2) + "\n\nSee `tables/aggregate_by_k.csv` and `plots/` for results.\n",
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2), flush=True)
