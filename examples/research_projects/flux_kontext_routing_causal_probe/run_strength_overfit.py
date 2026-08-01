#!/usr/bin/env python
"""Unified CLI for the strength-conditioned FLUX-Kontext overfit experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from run_routing_probe import load_pipeline
from run_target_residual_oracle import capture_trajectory, direct_velocity, encode_conditioning, freeze_pipeline, pipeline_kwargs
from run_vkeep_validation import configure_attention_backend
from oracle_residual import TargetResidualIntervention
from strength_overfit_data import (
    append_jsonl,
    assert_diffusers_checkout,
    environment_fingerprint,
    input_contract,
    load_config,
    load_metadata,
    native_working_size,
    refuse_overwrite,
    sample_dict,
    save_preprocessed_image,
    sha256_file,
    write_json,
)
from strength_overfit_evaluation import (
    append_metrics, relative_rms, save_contact_sheet, write_summary_csv,
)
from strength_overfit_masks import TemporalMaskEMA, make_mask
from strength_overfit_training import (
    checked_teacher_pair,
    euler_step,
    finite_or_raise,
    loss_weights,
    sample_strength_pair,
    velocity_losses,
)
from strength_residual import StrengthResidualIntervention


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("preflight", "prepare", "train", "evaluate", "report"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--skip-gpu-smoke", action="store_true")
    return parser.parse_args()


def base_sampling_config(config: dict[str, Any]) -> dict[str, Any]:
    scheduler = config["scheduler"]
    return {
        "model_path": config["model_path"],
        "dtype": config["dtype"],
        "num_inference_steps": int(scheduler["num_inference_steps"]),
        "guidance_scale": float(scheduler["guidance_scale"]),
        "true_cfg_scale": float(scheduler["true_cfg_scale"]),
        "max_sequence_length": int(scheduler["max_sequence_length"]),
    }


def run_root(config: dict[str, Any]) -> Path:
    return Path(config["output_root"]) / "runs" / str(config["run_id"])


def config_fingerprint(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": config["run_id"],
        "stage": config["stage"],
        "model_path": config["model_path"],
        "adapter": config["adapter"],
        "scheduler": config["scheduler"],
        "attention_backend": config["attention_backend"],
        "seed": config["seed"],
    }


def initialize_run(config: dict[str, Any], resume: bool) -> Path:
    root = refuse_overwrite(run_root(config), resume=resume, fingerprint=config_fingerprint(config))
    for relative in (
        "checkpoints", "logs", "metrics", "images/source", "images/full_edit",
        "images/neutral", "contact_sheets", "masks/raw", "masks/normalized",
        "masks/hard", "masks/soft", "masks/temporal", "reports", "dataset/preprocessed",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    write_json(root / "resolved_config.json", config)
    write_json(root / "environment.json", environment_fingerprint(REPO_ROOT))
    return root


def configure_pipeline(pipeline: Any, config: dict[str, Any]) -> None:
    backend = configure_attention_backend(pipeline, {"attention_backend": config["attention_backend"]})
    if backend != "_native_flash":
        raise RuntimeError(f"expected _native_flash, found {backend}")
    freeze_pipeline(pipeline)
    if hasattr(pipeline.transformer, "enable_gradient_checkpointing"):
        pipeline.transformer.enable_gradient_checkpointing()


def cache_path(root: Path, sample_id: str, seed: int) -> Path:
    return root / "dataset" / "states" / sample_id / f"seed_{seed}.pt"


def sigma_from_timestep(timestep: torch.Tensor) -> float:
    value = float(timestep.detach().float().flatten()[0].item())
    return value / 1000.0 if value > 2.0 else value


def load_or_create_cache(
    pipeline: Any,
    base: dict[str, Any],
    sample: Any,
    seed: int,
    root: Path,
    device: str,
    resume: bool,
) -> dict[str, Any]:
    path = cache_path(root, sample.sample_id, seed)
    if path.exists():
        if not resume:
            raise FileExistsError(f"cache already exists: {path}; use --resume")
        return torch.load(path, map_location="cpu", weights_only=False)
    row = sample_dict(sample)
    row["seed"] = int(seed)
    full_image, full = capture_trajectory(pipeline, base, row, sample.full_prompt)
    neutral_image, _neutral = capture_trajectory(pipeline, base, row, sample.neutral_prompt)
    full_conditioning = encode_conditioning(pipeline, base, sample.full_prompt)
    neutral_conditioning = encode_conditioning(pipeline, base, sample.neutral_prompt)
    full_velocities, neutral_velocities = [], []
    contracts: list[dict[str, Any]] = []
    with torch.no_grad():
        for state, timestep in zip(full["target_inputs"], full["timesteps"]):
            v_edit = direct_velocity(
                pipeline.transformer, state, timestep, full["image_tail"], full["img_ids"],
                full_conditioning, base["guidance_scale"], device,
            )
            v_neutral = direct_velocity(
                pipeline.transformer, state, timestep, full["image_tail"], full["img_ids"],
                neutral_conditioning, base["guidance_scale"], device,
            )
            full_velocities.append(v_edit.detach().cpu())
            neutral_velocities.append(v_neutral.detach().cpu())
            contracts.append(
                input_contract(
                    target_latents=state,
                    source_latents=full["image_tail"],
                    timestep=timestep,
                    sigma=sigma_from_timestep(timestep),
                    text_ids=full_conditioning["txt_ids"],
                    image_ids=full["img_ids"],
                    seed=seed,
                )
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample_id": sample.sample_id,
        "seed": seed,
        "layout": full["layout"],
        "target_states": full["target_inputs"],
        "timesteps": full["timesteps"],
        "sigmas": torch.tensor([sigma_from_timestep(value) for value in full["timesteps"]]),
        "v_edit": torch.cat(full_velocities),
        "v_neutral": torch.cat(neutral_velocities),
        "image_tail": full["image_tail"],
        "img_ids": full["img_ids"],
        "contracts": contracts,
        "source_sha256": sha256_file(sample.source_image),
    }
    torch.save(payload, path)
    full_image.save(root / "images" / "full_edit" / f"{sample.sample_id}_seed{seed}.png")
    neutral_image.save(root / "images" / "neutral" / f"{sample.sample_id}_seed{seed}.png")
    return payload


def cpu_self_checks() -> dict[str, Any]:
    command = [
        "/home/gem/anaconda3/envs/SEAdapter/bin/python", "-m", "pytest", "-q",
        "tests/test_strength_residual.py", "tests/test_strength_masks.py",
        "tests/test_strength_training_contract.py", "tests/test_strength_scheduler.py",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stdout + "\n" + result.stderr)
    return {"pytest": result.stdout.strip()}


def gpu_smoke(config: dict[str, Any], device: str) -> dict[str, Any]:
    legacy = config.get("legacy")
    if legacy is None:
        return {"skipped": "no legacy sample configured"}
    with open(ROOT / legacy["config"], encoding="utf-8") as handle:
        legacy_config = json.load(handle)
    with open(legacy_config["base_run_config"], encoding="utf-8") as handle:
        base = json.load(handle)
    base["num_inference_steps"] = int(config.get("preflight", {}).get("smoke_steps", 4))
    manifest = Path(legacy_config["manifest_path"])
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    wanted = legacy["sample_ids"][0]
    row = next(item for item in rows if item["sample_id"] == wanted)
    pipeline = load_pipeline(base, device)
    configure_pipeline(pipeline, {**config, "attention_backend": "_native_flash"})
    started = time.time()
    with torch.no_grad():
        image = pipeline(**pipeline_kwargs(base, row, row["instruction"])).images[0]
    if image.width < 1 or image.height < 1:
        raise RuntimeError("GPU smoke returned an empty image")
    return {"sample_id": wanted, "seconds": time.time() - started, "image_size": list(image.size), "peak_gb": torch.cuda.max_memory_allocated() / 2**30}


def do_preflight(config: dict[str, Any], args: argparse.Namespace) -> None:
    assert_diffusers_checkout(REPO_ROOT)
    root = initialize_run(config, args.resume)
    checks = cpu_self_checks()
    if not args.skip_gpu_smoke and config.get("preflight", {}).get("require_gpu_smoke", False):
        checks["gpu_smoke"] = gpu_smoke(config, args.device)
    write_json(root / "logs" / "preflight.json", checks)
    print(json.dumps(checks, indent=2), flush=True)


def do_prepare(config: dict[str, Any], args: argparse.Namespace) -> None:
    assert_diffusers_checkout(REPO_ROOT)
    root = initialize_run(config, args.resume)
    samples = load_metadata(config["metadata_path"])
    if config.get("sample_limit") is not None:
        samples = samples[: int(config["sample_limit"])]
    base = base_sampling_config(config)
    pipeline = load_pipeline(base, args.device)
    configure_pipeline(pipeline, config)
    rows = []
    for sample in samples:
        source = Image.open(sample.source_image).convert("RGB")
        width, height = native_working_size(source)
        save_preprocessed_image(source, root / "dataset" / "preprocessed" / f"{sample.sample_id}.png", width, height)
        all_seeds = list(sample.train_noise_seeds) + list(sample.validation_noise_seeds) + list(sample.rollout_seeds)
        for seed in all_seeds:
            cache = load_or_create_cache(pipeline, base, sample, seed, root, args.device, args.resume)
            rows.append({"sample_id": sample.sample_id, "seed": seed, "target_tokens": cache["layout"]["target_tokens"], "states": int(cache["target_states"].shape[0])})
    write_json(root / "dataset" / "prepare_summary.json", rows)


def _build_intervention(config: dict[str, Any], cache: dict[str, Any], pipeline: Any, legacy: bool) -> Any:
    model_config = pipeline.transformer.config
    # This FLUX-Kontext checkout stores a FrozenDict without inner_dim.
    hidden_size = int(model_config["attention_head_dim"]) * int(model_config["num_attention_heads"])
    target_tokens = int(cache["layout"]["target_tokens"])
    layers = list(config["adapter"]["layers"])
    if legacy:
        return TargetResidualIntervention(pipeline.transformer, layers, target_tokens=target_tokens, hidden_size=hidden_size, rank=int(config["adapter"]["rank"]), alpha=float(config["adapter"]["alpha"]))
    intervention = StrengthResidualIntervention(
        pipeline.transformer, layers, target_tokens=target_tokens, hidden_size=hidden_size,
        rank=int(config["adapter"]["rank"]), alpha=float(config["adapter"]["alpha"]),
        gate_hidden_dim=int(config["adapter"]["gate_hidden_dim"]),
    )
    # The intervention deliberately does not own the frozen transformer, so
    # its trainable modules must be placed explicitly beside that transformer.
    adapter_device = pipeline._execution_device
    intervention.adapters.to(adapter_device)
    intervention.gate.to(adapter_device)
    return intervention


def _student_velocity(intervention: Any, legacy: bool, state: torch.Tensor, timestep: torch.Tensor, cache: dict[str, Any], conditioning: dict[str, torch.Tensor], base: dict[str, Any], device: str, strength: float, sigma: float, spatial_weight: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    if legacy:
        intervention.set_scale(1.0 - strength)
        intervention.reset_sequence()
        intervention.reset_metrics()
    else:
        intervention.set_context(strength=strength, sigma=sigma, spatial_weight=spatial_weight)
        intervention.reset_sequence()
        intervention.reset_metrics()
    intervention.collect_metrics = True
    with intervention.applied():
        velocity = direct_velocity(intervention.transformer, state, timestep, cache["image_tail"], cache["img_ids"], conditioning, base["guidance_scale"], device)
    return velocity, intervention.metric_regularizer()


def do_train(config: dict[str, Any], args: argparse.Namespace) -> None:
    assert_diffusers_checkout(REPO_ROOT)
    root = initialize_run(config, args.resume)
    samples = load_metadata(config["metadata_path"])
    if config.get("sample_limit") is not None:
        samples = samples[: int(config["sample_limit"])]
    base = base_sampling_config(config)
    caches = []
    for sample in samples:
        for seed in sample.train_noise_seeds:
            path = cache_path(root, sample.sample_id, seed)
            if not path.exists():
                raise FileNotFoundError(f"missing prepared cache {path}; run --mode prepare first")
            caches.append((sample, torch.load(path, map_location="cpu", weights_only=False)))
    pipeline = load_pipeline(base, args.device)
    configure_pipeline(pipeline, config)
    # Stateful residual hooks are not re-entrant under activation
    # checkpoint recomputation; use the validated native-flash path directly.
    if hasattr(pipeline.transformer, "disable_gradient_checkpointing"):
        pipeline.transformer.disable_gradient_checkpointing()
    legacy = config["stage"] == "previous_scaling"
    intervention = _build_intervention(config, caches[0][1], pipeline, legacy)
    optimizer = torch.optim.AdamW(list(intervention.parameters()), lr=float(config["training"]["learning_rate"]), betas=tuple(config["training"]["betas"]), weight_decay=float(config["training"]["weight_decay"]))
    conditioning = {sample.sample_id: {"edit": encode_conditioning(pipeline, base, sample.full_prompt), "neutral": encode_conditioning(pipeline, base, sample.neutral_prompt)} for sample in samples}
    generator = torch.Generator().manual_seed(int(config["seed"]))
    heldout = set(int(value) for value in config["scheduler"]["heldout_indices"])
    log_path = root / "logs" / "train.jsonl"
    best = float("inf")
    for step in range(1, int(config["training"]["max_updates"]) + 1):
        sample, cache = caches[int(torch.randint(len(caches), (), generator=generator).item())]
        valid_indices = [index for index in range(cache["target_states"].shape[0]) if index not in heldout and (config["online"]["rollout_steps"] < 2 or index + 1 < cache["target_states"].shape[0])]
        index = valid_indices[int(torch.randint(len(valid_indices), (), generator=generator).item())]
        state = cache["target_states"][index]
        timestep = cache["timesteps"][index]
        sigma = float(cache["sigmas"][index].item())
        pair = sample_strength_pair(generator, device=torch.device("cpu"), pair_probability=float(config["strength"]["pair_probability"]))
        if legacy:
            pair = type(pair)(first=0.0, second=None)
        # The configured warm-up has zero monotonic/progress weights, so a
        # second retained student graph is both unnecessary and prohibitively
        # expensive at 1024px.  Enable paired strengths when the ramp begins.
        if step <= int(config["loss"]["warmup_steps"]):
            pair = type(pair)(first=pair.first, second=None)
        if int(config["online"]["rollout_steps"]) == 2 and float(torch.rand((), generator=generator).item()) >= float(config["online"]["static_probability"]):
            current = state.detach()
            total = None
            mask_ema = TemporalMaskEMA(beta=float(config["mask"]["beta"])) if config["mask"]["type"] != "none" else None
            for offset in range(2):
                current_index = index + offset
                current_timestep = cache["timesteps"][current_index]
                current_sigma = float(cache["sigmas"][current_index].item())
                next_sigma = float(cache["sigmas"][current_index + 1].item())
                current_contract = input_contract(
                    target_latents=current,
                    source_latents=cache["image_tail"],
                    timestep=current_timestep,
                    sigma=current_sigma,
                    text_ids=conditioning[sample.sample_id]["edit"]["txt_ids"],
                    image_ids=cache["img_ids"],
                    seed=int(cache["seed"]),
                )
                edit_call = lambda: (
                    direct_velocity(pipeline.transformer, current, current_timestep, cache["image_tail"], cache["img_ids"], conditioning[sample.sample_id]["edit"], base["guidance_scale"], args.device),
                    current_contract,
                )
                neutral_call = lambda: (
                    direct_velocity(pipeline.transformer, current, current_timestep, cache["image_tail"], cache["img_ids"], conditioning[sample.sample_id]["neutral"], base["guidance_scale"], args.device),
                    current_contract,
                )
                v_edit, v_neutral, _contract = checked_teacher_pair(edit_call=edit_call, neutral_call=neutral_call)
                spatial = None
                if config["mask"]["type"] != "none":
                    result = make_mask(v_edit, v_neutral, mask_type=config["mask"]["type"], tau=float(config["mask"]["tau"]), temperature=float(config["mask"]["temperature"]), lambda_bg=float(config["mask"]["lambda_bg"]), ema=mask_ema)
                    spatial = result.weight
                student, regularizer = _student_velocity(intervention, legacy, current, current_timestep, cache, conditioning[sample.sample_id]["edit"], base, args.device, pair.first, current_sigma, spatial)
                mono_weight, progress_weight = loss_weights(step, warmup_steps=int(config["loss"]["warmup_steps"]), ramp_steps=int(config["loss"]["ramp_steps"]), lambda_mono=float(config["loss"]["lambda_mono"]), lambda_progress=float(config["loss"]["lambda_progress"]))
                losses = velocity_losses(v_student=student, v_edit=v_edit, v_neutral=v_neutral, strength=pair.first, lambda_mono=mono_weight, lambda_progress=progress_weight, lambda_reg=float(config["loss"]["lambda_reg"]), regularizer=regularizer, margin=float(config["loss"]["margin"]), eps=float(config["loss"]["epsilon"]))
                total = losses["total"] if total is None else total + losses["total"]
                current = euler_step(current, student.detach(), current_sigma, next_sigma).detach().cpu()
            losses["total"] = total / 2
        else:
            v_edit = cache["v_edit"][index:index + 1].to(args.device)
            v_neutral = cache["v_neutral"][index:index + 1].to(args.device)
            student, regularizer = _student_velocity(intervention, legacy, state, timestep, cache, conditioning[sample.sample_id]["edit"], base, args.device, pair.first, sigma, None)
            second_student = None
            if pair.second is not None:
                second_student, second_regularizer = _student_velocity(intervention, legacy, state, timestep, cache, conditioning[sample.sample_id]["edit"], base, args.device, pair.second, sigma, None)
                regularizer = 0.5 * (regularizer + second_regularizer)
            mono_weight, progress_weight = loss_weights(step, warmup_steps=int(config["loss"]["warmup_steps"]), ramp_steps=int(config["loss"]["ramp_steps"]), lambda_mono=float(config["loss"]["lambda_mono"]), lambda_progress=float(config["loss"]["lambda_progress"]))
            losses = velocity_losses(v_student=student, v_edit=v_edit, v_neutral=v_neutral, strength=pair.first, second_student=second_student, second_strength=pair.second, lambda_mono=mono_weight, lambda_progress=progress_weight, lambda_reg=float(config["loss"]["lambda_reg"]), regularizer=regularizer, margin=float(config["loss"]["margin"]), eps=float(config["loss"]["epsilon"]))
        finite_or_raise({name: value for name, value in losses.items() if torch.is_tensor(value)})
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(list(intervention.parameters()), float(config["training"]["gradient_clip_norm"]))
        optimizer.step()
        row = {"step": step, "sample_id": sample.sample_id, "state_index": index, "strength": pair.first, "second_strength": pair.second, "grad_norm": float(grad_norm), "peak_cuda_mem_gb": torch.cuda.max_memory_allocated() / 2**30, **{name: float(value.detach().item()) for name, value in losses.items()}, "layer_metrics": intervention.detached_metrics(), "parameter_norms": intervention.parameter_norms()}
        append_jsonl(log_path, row)
        if row["total"] < best:
            best = row["total"]
        if step % int(config["training"]["checkpoint_interval"]) == 0 or step == int(config["training"]["max_updates"]):
            adapter_state = intervention.state_dict() if not legacy else {name: value.detach().cpu() for name, value in intervention.named_parameters()}
            torch.save({"step": step, "config": config, "adapter": adapter_state, "optimizer": optimizer.state_dict(), "best_train_loss": best}, root / "checkpoints" / f"checkpoint_{step:06d}.pt")
        if step == 1 or step % 10 == 0:
            print(json.dumps({key: row[key] for key in ("step", "sample_id", "strength", "total", "velocity", "grad_norm")}), flush=True)


def _decode_target_image(pipeline: Any, packed: torch.Tensor, width: int, height: int) -> Image.Image:
    latents = pipeline._unpack_latents(
        packed.to(pipeline._execution_device),
        height,
        width,
        pipeline.vae_scale_factor,
    )
    shift = float(getattr(pipeline.vae.config, "shift_factor", 0.0))
    latents = latents / float(pipeline.vae.config.scaling_factor) + shift
    with torch.no_grad():
        decoded = pipeline.vae.decode(latents.to(dtype=pipeline.vae.dtype), return_dict=False)[0]
    return pipeline.image_processor.postprocess(decoded, output_type="pil")[0]


def _latest_checkpoint(root: Path) -> Path:
    candidates = sorted((root / "checkpoints").glob("checkpoint_*.pt"))
    if not candidates:
        raise FileNotFoundError("no checkpoint found; train before evaluating")
    return candidates[-1]


def _restore_intervention(intervention: Any, checkpoint_path: Path, legacy: bool) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload["adapter"]
    if legacy:
        for name, parameter in intervention.named_parameters():
            parameter.data.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))
    else:
        intervention.load_state_dict(state)


def _rollout_strength(
    *,
    pipeline: Any,
    intervention: Any,
    legacy: bool,
    cache: dict[str, Any],
    conditioning: dict[str, torch.Tensor],
    base: dict[str, Any],
    strength: float,
    mask_config: dict[str, Any],
    device: str,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    current = cache["target_states"][0].detach().cpu()
    ema = TemporalMaskEMA(beta=float(mask_config["beta"])) if mask_config["type"] != "none" else None
    trace: list[dict[str, Any]] = []
    for index, timestep in enumerate(cache["timesteps"]):
        sigma = float(cache["sigmas"][index].item())
        next_sigma = float(cache["sigmas"][index + 1].item()) if index + 1 < len(cache["sigmas"]) else 0.0
        spatial = None
        if mask_config["type"] != "none":
            with torch.no_grad():
                v_edit = direct_velocity(
                    pipeline.transformer, current, timestep, cache["image_tail"], cache["img_ids"],
                    conditioning["edit"], base["guidance_scale"], device,
                )
                v_neutral = direct_velocity(
                    pipeline.transformer, current, timestep, cache["image_tail"], cache["img_ids"],
                    conditioning["neutral"], base["guidance_scale"], device,
                )
            spatial = make_mask(
                v_edit, v_neutral, mask_type=mask_config["type"], tau=float(mask_config["tau"]),
                temperature=float(mask_config["temperature"]), lambda_bg=float(mask_config["lambda_bg"]), ema=ema,
            ).weight
        velocity, _regularizer = _student_velocity(
            intervention, legacy, current, timestep, cache, conditioning["edit"], base, device, strength, sigma, spatial,
        )
        current = euler_step(current, velocity.detach(), sigma, next_sigma).detach().cpu()
        trace.append({"step": index, "sigma": sigma, "latent_rms": float(current.float().square().mean().sqrt().item()), "layer_metrics": intervention.detached_metrics()})
    return current, trace


def do_evaluate(config: dict[str, Any], args: argparse.Namespace) -> None:
    assert_diffusers_checkout(REPO_ROOT)
    root = initialize_run(config, args.resume)
    samples = load_metadata(config["metadata_path"])
    if config.get("sample_limit") is not None:
        samples = samples[: int(config["sample_limit"])]
    base = base_sampling_config(config)
    pipeline = load_pipeline(base, args.device)
    configure_pipeline(pipeline, config)
    legacy = config["stage"] == "previous_scaling"
    checkpoint = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint(root)
    all_rows: list[dict[str, Any]] = []
    for sample in samples:
        conditioning = {
            "edit": encode_conditioning(pipeline, base, sample.full_prompt),
            "neutral": encode_conditioning(pipeline, base, sample.neutral_prompt),
        }
        for seed in sample.rollout_seeds:
            cache = torch.load(cache_path(root, sample.sample_id, seed), map_location="cpu", weights_only=False)
            intervention = _build_intervention(config, cache, pipeline, legacy)
            _restore_intervention(intervention, checkpoint, legacy)
            width = int(sample.resolved_width or 1024)
            height = int(sample.resolved_height or 1024)
            full_latent, _ = _rollout_strength(
                pipeline=pipeline, intervention=intervention, legacy=legacy, cache=cache, conditioning=conditioning,
                base=base, strength=1.0, mask_config={**config["mask"], "type": "none"}, device=args.device,
            )
            neutral_intervention = _build_intervention(config, cache, pipeline, legacy)
            if legacy:
                neutral_intervention.set_scale(1.0)
            else:
                neutral_intervention.set_context(strength=0.0, sigma=float(cache["sigmas"][0]), spatial_weight=None)
            images: list[Image.Image] = []
            labels: list[str] = []
            seed_rows: list[dict[str, Any]] = []
            for strength in [float(value) for value in config["evaluation"]["strengths"]]:
                latent, trace = _rollout_strength(
                    pipeline=pipeline, intervention=intervention, legacy=legacy, cache=cache, conditioning=conditioning,
                    base=base, strength=strength, mask_config=config["mask"], device=args.device,
                )
                image = _decode_target_image(pipeline, latent, width, height)
                image_path = root / "images" / config["stage"] / sample.sample_id / f"seed_{seed}_s_{strength:.1f}.png"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(image_path)
                images.append(image)
                labels.append(f"s={strength:.1f}")
                seed_rows.append({
                    "method": config["method"], "stage": config["stage"], "sample_id": sample.sample_id, "seed": seed,
                    "strength": strength, "image": str(image_path), "latent_to_full_relative_rms": relative_rms(latent, full_latent),
                    "trace_steps": len(trace), "peak_cuda_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
                })
            save_contact_sheet(images, labels, root / "contact_sheets" / f"{config['stage']}_{sample.sample_id}_seed{seed}.png")
            all_rows.extend(seed_rows)
    append_metrics(root / "metrics" / "raw_metrics.jsonl", all_rows)
    write_summary_csv(root / "metrics" / "summary_by_sample.csv", all_rows)
    write_json(root / "metrics" / "evaluation_complete.json", {"checkpoint": str(checkpoint), "rows": len(all_rows), "perceptual_metrics": "pending local DINO/LPIPS/CLIP pass; no metric is synthesized"})
    print(json.dumps({"checkpoint": str(checkpoint), "rows": len(all_rows)}, indent=2), flush=True)



def do_report(config: dict[str, Any], args: argparse.Namespace) -> None:
    root = initialize_run(config, args.resume)
    logs = root / "logs" / "train.jsonl"
    rows = [json.loads(line) for line in logs.read_text(encoding="utf-8").splitlines()] if logs.exists() else []
    conclusion = "No result has been inferred. Run evaluation and visually inspect contact sheets before choosing A, B, or C."
    (root / "reports" / f"{config['stage']}_report.md").write_text(f"# {config['stage']} report\n\n{conclusion}\n\nTraining rows: {len(rows)}\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if config.get("attention_backend") != "_native_flash":
        raise ValueError("only _native_flash is permitted")
    if args.mode == "preflight":
        do_preflight(config, args)
    elif args.mode == "prepare":
        do_prepare(config, args)
    elif args.mode == "train":
        do_train(config, args)
    elif args.mode == "evaluate":
        do_evaluate(config, args)
    else:
        do_report(config, args)


if __name__ == "__main__":
    main()

