#!/usr/bin/env python
"""Per-sample target-residual Oracle experiment for FLUX-Kontext."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from transformers import AutoImageProcessor, AutoModel

from oracle_residual import TargetResidualIntervention, freeze_module
from oracle_run_state import (
    code_fingerprint,
    model_fingerprint,
    optimizer_to_device,
    require_fingerprint,
    stable_hash,
    tensor_state_hash,
    write_status,
)
from prepare_oracle_manifest import select_layers
from probe_utils import (
    atomic_torch_save,
    atomic_write_json,
    ensure_run_config,
    file_sha256,
    load_json,
    load_jsonl,
)
from routing_attention import RoutingLayout
from run_routing_probe import load_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs" / "oracle_smoke.json"))
    parser.add_argument("--sample-id")
    parser.add_argument(
        "--combo",
        choices=("primary", "primary_secondary", "primary_secondary_middle", "negative", "early_triplet"),
    )
    parser.add_argument("--rank", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--eval-state-count", type=int)
    parser.add_argument("--run-root")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stage", choices=("gradient-check", "train-check", "full"), default="full")
    parser.add_argument("--state-index", type=int, default=0)
    parser.add_argument("--enforce-acceptance", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_base_config(config: dict[str, Any]) -> dict[str, Any]:
    return load_json(config["base_run_config"])


def find_sample(config: dict[str, Any], sample_id: str) -> dict[str, Any]:
    matches = [sample for sample in load_jsonl(config["manifest_path"]) if sample["sample_id"] == sample_id]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest entry for {sample_id}, found {len(matches)}")
    sample = dict(matches[0])
    base = load_base_config(config)
    sample["seed"] = int(sample.get("seed", base["seed"]))
    return sample


def layer_combinations(config: dict[str, Any]) -> dict[str, list[str]]:
    selected = select_layers(config)
    primary = selected["primary"]
    secondary = selected["secondary"]
    middle = selected["middle_single"]
    combinations = {
        "primary": [primary],
        "primary_secondary": [primary, secondary],
        "primary_secondary_middle": [primary, secondary, middle],
        "negative": [selected["negative_control"]],
    }
    quick = config.get("quick_validation", {})
    if quick.get("enabled"):
        fixed_layers = [str(value) for value in quick["fixed_layers"]]
        expected = ["dual.00", "dual.01", "dual.02"]
        if fixed_layers != expected:
            raise ValueError(f"quick fixed_layers must be exactly {expected}, found {fixed_layers}")
        combinations["early_triplet"] = fixed_layers
    return combinations


def combo_name(layer_ids: list[str]) -> str:
    return "__".join(layer_ids).replace(".", "_")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def atomic_save_png(image: Image.Image, path: Path) -> None:
    ensure_parent(path)
    temporary = path.with_name(f".{path.name}.tmp")
    image.save(temporary, format="PNG")
    temporary.replace(path)


def freeze_pipeline(pipeline: Any) -> list[str]:
    """Freeze every original module; only the separately-created Oracle may train."""
    frozen: list[str] = []
    for name in ("transformer", "text_encoder", "text_encoder_2", "image_encoder", "vae"):
        module = getattr(pipeline, name, None)
        if not isinstance(module, torch.nn.Module):
            continue
        freeze_module(module)
        module.eval()
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise RuntimeError(f"failed to freeze pipeline component {name}")
        frozen.append(name)
    return frozen


def assert_base_model_unmodified(pipeline: Any) -> None:
    for module_name in ("transformer", "text_encoder", "text_encoder_2", "image_encoder", "vae"):
        module = getattr(pipeline, module_name, None)
        if not isinstance(module, torch.nn.Module):
            continue
        trainable = [name for name, parameter in module.named_parameters() if parameter.requires_grad]
        gradients = [name for name, parameter in module.named_parameters() if parameter.grad is not None]
        if trainable or gradients:
            raise RuntimeError(
                f"original module {module_name} changed training state: "
                f"trainable={trainable[:3]} gradients={gradients[:3]}"
            )


def capture_rng_state(generator: torch.Generator) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "sample_generator": generator.get_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any], generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    generator.set_state(state["sample_generator"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def build_fingerprints(
    oracle_config: dict[str, Any],
    base_config: dict[str, Any],
    sample: dict[str, Any],
    selected_layers: list[str],
    combo_id: str,
    rank: int,
    run_root: Path,
) -> dict[str, Any]:
    code = code_fingerprint(
        [
            Path(__file__),
            Path(__file__).parent / "oracle_residual.py",
            Path(__file__).parent / "oracle_run_state.py",
            Path(__file__).parent / "prepare_oracle_manifest.py",
            Path(__file__).parent / "run_routing_probe.py",
            Path(__file__).parent / "routing_attention.py",
        ]
    )
    model = model_fingerprint(
        base_config["model_path"],
        run_root / "metadata" / "model_fingerprint.json",
    )
    source_path = Path(sample["source_image"])
    trajectory_spec = {
        "model_content_sha256": model["content_sha256"],
        "code_sha256": code["sha256"],
        "source_image": str(source_path.resolve()),
        "source_sha256": file_sha256(source_path),
        "sample_id": sample["sample_id"],
        "instruction": sample["instruction"],
        "teacher_mode": oracle_config["teacher_mode"],
        "teacher_prompt": oracle_config["teacher_prompt"],
        "seed": int(sample["seed"]),
        "sampling": {
            key: base_config[key]
            for key in ("num_inference_steps", "guidance_scale", "true_cfg_scale")
        },
        "max_sequence_length": int(base_config.get("max_sequence_length", 512)),
        "dtype": base_config.get("dtype", "bfloat16"),
        "attention_backend": oracle_config.get("training_attention_backend", "_native_flash"),
    }
    trajectory_hash = stable_hash(trajectory_spec)
    run_spec = {
        "trajectory_sha256": trajectory_hash,
        "combo_id": combo_id,
        "layers": selected_layers,
        "rank": rank,
        "training": oracle_config["training"],
        "rollout_strengths": oracle_config["rollout_strengths"],
    }
    return {
        "code": code,
        "model": model,
        "trajectory_spec": trajectory_spec,
        "trajectory_sha256": trajectory_hash,
        "run_spec": run_spec,
        "run_sha256": stable_hash(run_spec),
    }


class TrajectoryCapture:
    def __init__(self) -> None:
        self.layout: RoutingLayout | None = None
        self.target_inputs: list[torch.Tensor] = []
        self.timesteps: list[torch.Tensor] = []
        self.velocities: list[torch.Tensor] = []
        self.image_tail: torch.Tensor | None = None
        self.img_ids: torch.Tensor | None = None
        self.pending = False

    def pre_hook(self, _module: Any, _args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if self.layout is None:
            self.layout = RoutingLayout.from_runtime(
                kwargs["encoder_hidden_states"], kwargs["hidden_states"], kwargs["img_ids"]
            )
        layout = self.layout
        hidden_states = kwargs["hidden_states"]
        self.target_inputs.append(hidden_states[:, : layout.target_tokens].detach().cpu().clone())
        self.timesteps.append(kwargs["timestep"].detach().cpu().clone())
        image_tail = hidden_states[:, layout.target_tokens :].detach().cpu().clone()
        if self.image_tail is None:
            self.image_tail = image_tail
            self.img_ids = kwargs["img_ids"].detach().cpu().clone()
        elif not torch.equal(self.image_tail, image_tail):
            raise RuntimeError("source image latent tail changed across the trajectory")
        self.pending = True

    def post_hook(
        self,
        _module: Any,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        output: tuple[torch.Tensor, ...],
    ) -> None:
        if not self.pending or self.layout is None:
            raise RuntimeError("trajectory capture post-hook is out of sync")
        self.velocities.append(output[0][:, : self.layout.target_tokens].detach().cpu().clone())
        self.pending = False

    def finalize(self, expected_steps: int) -> dict[str, Any]:
        if self.layout is None or self.image_tail is None or self.img_ids is None:
            raise RuntimeError("trajectory capture did not run")
        if len(self.target_inputs) != expected_steps or len(self.velocities) != expected_steps:
            raise RuntimeError(
                f"captured states={len(self.target_inputs)} velocities={len(self.velocities)} expected={expected_steps}"
            )
        return {
            "layout": {
                "text_tokens": self.layout.text_tokens,
                "target_tokens": self.layout.target_tokens,
                "source_tokens": self.layout.source_tokens,
                "target_grid_height": self.layout.target_grid_height,
                "target_grid_width": self.layout.target_grid_width,
            },
            "target_inputs": torch.stack(self.target_inputs),
            "timesteps": torch.stack(self.timesteps),
            "velocities": torch.stack(self.velocities),
            "image_tail": self.image_tail,
            "img_ids": self.img_ids,
        }


def pipeline_kwargs(base_config: dict[str, Any], sample: dict[str, Any], prompt: str) -> dict[str, Any]:
    return {
        "image": Image.open(sample["source_image"]).convert("RGB"),
        "prompt": prompt,
        "num_inference_steps": int(base_config["num_inference_steps"]),
        "guidance_scale": float(base_config["guidance_scale"]),
        "true_cfg_scale": float(base_config["true_cfg_scale"]),
        "generator": torch.Generator(device="cuda").manual_seed(int(sample["seed"])),
        "max_sequence_length": int(base_config.get("max_sequence_length", 512)),
        "_auto_resize": True,
    }


def capture_trajectory(
    pipeline: Any,
    base_config: dict[str, Any],
    sample: dict[str, Any],
    prompt: str,
) -> tuple[Image.Image, dict[str, Any]]:
    capture = TrajectoryCapture()
    pre_handle = pipeline.transformer.register_forward_pre_hook(capture.pre_hook, with_kwargs=True)
    post_handle = pipeline.transformer.register_forward_hook(capture.post_hook, with_kwargs=True)
    try:
        with torch.inference_mode():
            image = pipeline(**pipeline_kwargs(base_config, sample, prompt)).images[0]
    finally:
        pre_handle.remove()
        post_handle.remove()
    return image, capture.finalize(int(base_config["num_inference_steps"]))


def trajectory_paths(sample_root: Path) -> dict[str, Path]:
    return {
        "full": sample_root / "trajectory" / "full_edit.pt",
        "keep": sample_root / "trajectory" / "keep_teacher.pt",
        "full_image": sample_root / "images" / "full_edit.png",
        "keep_image": sample_root / "images" / "keep_teacher.png",
        "metadata": sample_root / "trajectory" / "metadata.json",
    }


def load_or_capture_trajectories(
    pipeline: Any,
    base_config: dict[str, Any],
    oracle_config: dict[str, Any],
    sample: dict[str, Any],
    sample_root: Path,
    trajectory_fingerprint: str,
    resume: bool,
) -> tuple[Image.Image, dict[str, Any], Image.Image, dict[str, Any]]:
    paths = trajectory_paths(sample_root)
    complete = all(path.is_file() for path in paths.values())
    if resume and complete:
        require_fingerprint(paths["metadata"], trajectory_fingerprint, "captured trajectory")
        return (
            Image.open(paths["full_image"]).convert("RGB"),
            torch.load(paths["full"], map_location="cpu", weights_only=False),
            Image.open(paths["keep_image"]).convert("RGB"),
            torch.load(paths["keep"], map_location="cpu", weights_only=False),
        )
    full_image, full = capture_trajectory(pipeline, base_config, sample, sample["instruction"])
    keep_image, keep = capture_trajectory(pipeline, base_config, sample, str(oracle_config["teacher_prompt"]))
    for key in ("target_inputs", "image_tail", "img_ids"):
        if not torch.equal(full[key][0] if key == "target_inputs" else full[key], keep[key][0] if key == "target_inputs" else keep[key]):
            raise RuntimeError(f"full and keep initial sampling state differs for {key}")
    atomic_torch_save(paths["full"], full)
    atomic_torch_save(paths["keep"], keep)
    atomic_save_png(full_image, paths["full_image"])
    atomic_save_png(keep_image, paths["keep_image"])
    atomic_write_json(
        paths["metadata"],
        {
            "fingerprint": trajectory_fingerprint,
            "full_image_sha256": file_sha256(paths["full_image"]),
            "keep_image_sha256": file_sha256(paths["keep_image"]),
        },
    )
    return full_image, full, keep_image, keep


def encode_conditioning(pipeline: Any, base_config: dict[str, Any], prompt: str) -> dict[str, torch.Tensor]:
    with torch.inference_mode():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            prompt_embeds=None,
            pooled_prompt_embeds=None,
            device=pipeline._execution_device,
            num_images_per_prompt=1,
            max_sequence_length=int(base_config.get("max_sequence_length", 512)),
            lora_scale=None,
        )
    # Tensors created under inference_mode cannot safely participate in a later
    # autograd graph. Clone after leaving the context to create normal tensors.
    return {
        "encoder_hidden_states": prompt_embeds.detach().clone(),
        "pooled_projections": pooled_prompt_embeds.detach().clone(),
        "txt_ids": text_ids.detach().clone(),
    }


def direct_velocity(
    transformer: Any,
    target_state: torch.Tensor,
    timestep: torch.Tensor,
    image_tail: torch.Tensor,
    img_ids: torch.Tensor,
    conditioning: dict[str, torch.Tensor],
    guidance_scale: float,
    device: str,
) -> torch.Tensor:
    if target_state.ndim == 2 and image_tail.ndim == 3:
        target_state = target_state.unsqueeze(0)
    if target_state.ndim != image_tail.ndim:
        raise ValueError(
            f"target/image token ranks differ: target={tuple(target_state.shape)} "
            f"image_tail={tuple(image_tail.shape)}"
        )
    hidden_states = torch.cat([target_state, image_tail], dim=1).to(device)
    # Gradient checkpointing needs at least one differentiable input.  The
    # base Transformer remains frozen; this gradient is only the carrier for
    # the trainable residual hooks and is discarded after each update.
    if torch.is_grad_enabled():
        hidden_states.requires_grad_(True)
    timestep = timestep.to(device)
    guidance = torch.full([hidden_states.shape[0]], guidance_scale, device=device, dtype=torch.float32)
    output = transformer(
        hidden_states=hidden_states,
        timestep=timestep,
        guidance=guidance,
        pooled_projections=conditioning["pooled_projections"],
        encoder_hidden_states=conditioning["encoder_hidden_states"],
        txt_ids=conditioning["txt_ids"],
        img_ids=img_ids.to(device),
        joint_attention_kwargs=None,
        return_dict=False,
    )[0]
    return output[:, : target_state.shape[1]]


def training_cache_path(sample_root: Path) -> Path:
    return sample_root / "trajectory" / "matched_velocity_cache.pt"


def build_training_cache(
    transformer: Any,
    full: dict[str, Any],
    keep: dict[str, Any],
    full_conditioning: dict[str, torch.Tensor],
    keep_conditioning: dict[str, torch.Tensor],
    base_config: dict[str, Any],
    sample_root: Path,
    trajectory_fingerprint: str,
    device: str,
    resume: bool,
) -> dict[str, Any]:
    path = training_cache_path(sample_root)
    if resume and path.is_file():
        cache = torch.load(path, map_location="cpu", weights_only=False)
        if cache.get("fingerprint") != trajectory_fingerprint:
            raise RuntimeError(
                "refusing to reuse matched-velocity cache with a different model/sample/sampling/code fingerprint"
            )
        return cache
    target_states = torch.cat([full["target_inputs"], keep["target_inputs"]], dim=0)
    timesteps = torch.cat([full["timesteps"], keep["timesteps"]], dim=0)
    full_velocities: list[torch.Tensor] = []
    keep_velocities: list[torch.Tensor] = []
    image_tail = full["image_tail"]
    img_ids = full["img_ids"]
    with torch.inference_mode():
        for cache_index, (state, timestep) in enumerate(zip(target_states, timesteps), start=1):
            print(f"[cache] matched velocity {cache_index}/{len(target_states)}", flush=True)
            full_velocity = direct_velocity(
                transformer,
                state,
                timestep,
                image_tail,
                img_ids,
                full_conditioning,
                float(base_config["guidance_scale"]),
                device,
            )
            keep_velocity = direct_velocity(
                transformer,
                state,
                timestep,
                image_tail,
                img_ids,
                keep_conditioning,
                float(base_config["guidance_scale"]),
                device,
            )
            full_velocities.append(full_velocity.detach().cpu())
            keep_velocities.append(keep_velocity.detach().cpu())
    # The direct call must reproduce the pipeline's conditional velocity on
    # captured full-edit states. This catches mismatched timesteps, IDs, or
    # conditioning before any Oracle parameter is optimized.
    expected_steps = int(base_config["num_inference_steps"])
    if len(full_velocities) < expected_steps:
        raise RuntimeError(f"matched velocity cache has only {len(full_velocities)} states")
    direct_errors: list[float] = []
    for index in range(expected_steps):
        reference = full["velocities"][index]
        candidate = full_velocities[index]
        relative_error = float(
            (candidate.float() - reference.float()).square().mean().sqrt()
            / (reference.float().square().mean().sqrt() + 1e-12)
        )
        if relative_error > 1e-4:
            raise RuntimeError(
                f"direct full-edit velocity does not reproduce captured pipeline output at step {index}: "
                f"relative RMS error={relative_error:.3e}"
            )
        direct_errors.append(relative_error)
    cache = {
        "fingerprint": trajectory_fingerprint,
        "target_states": target_states,
        "timesteps": timesteps,
        "full_velocities": torch.cat(full_velocities),
        "keep_velocities": torch.cat(keep_velocities),
        "image_tail": image_tail,
        "img_ids": img_ids,
        "direct_full_edit_relative_rms_errors": direct_errors,
    }
    atomic_torch_save(path, cache)
    return cache


def build_single_state_cache(
    transformer: Any,
    full: dict[str, Any],
    full_conditioning: dict[str, torch.Tensor],
    keep_conditioning: dict[str, torch.Tensor],
    base_config: dict[str, Any],
    state_index: int,
    device: str,
) -> dict[str, Any]:
    expected_steps = int(base_config["num_inference_steps"])
    if not 0 <= state_index < expected_steps:
        raise ValueError(f"single-state index must be in [0, {expected_steps - 1}]")
    target_state = full["target_inputs"][state_index]
    timestep = full["timesteps"][state_index]
    with torch.inference_mode():
        full_velocity = direct_velocity(
            transformer,
            target_state,
            timestep,
            full["image_tail"],
            full["img_ids"],
            full_conditioning,
            float(base_config["guidance_scale"]),
            device,
        )
        keep_velocity = direct_velocity(
            transformer,
            target_state,
            timestep,
            full["image_tail"],
            full["img_ids"],
            keep_conditioning,
            float(base_config["guidance_scale"]),
            device,
        )
    reference = full["velocities"][state_index]
    relative_error = float(
        (full_velocity.detach().cpu().float() - reference.float()).square().mean().sqrt()
        / (reference.float().square().mean().sqrt() + 1e-12)
    )
    if relative_error > 1e-4:
        raise RuntimeError(
            f"single-state direct call does not reproduce pipeline at step {state_index}: "
            f"relative RMS error={relative_error:.3e}"
        )
    return {
        "target_states": full["target_inputs"][state_index : state_index + 1],
        "timesteps": full["timesteps"][state_index : state_index + 1],
        "full_velocities": full_velocity.detach().cpu(),
        "keep_velocities": keep_velocity.detach().cpu(),
        "image_tail": full["image_tail"],
        "img_ids": full["img_ids"],
        "direct_full_edit_relative_rms_errors": [relative_error],
    }


def mse_velocity(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction.float() - target.float()).square().mean()


def teacher_velocity(cache: dict[str, Any], index: int) -> torch.Tensor:
    """Return the configured velocity teacher while preserving legacy caches."""
    key = "teacher_velocities" if "teacher_velocities" in cache else "keep_velocities"
    value = cache[key][index]
    if cache.get("teacher_requires_batch_dim") and value.ndim == 2:
        return value.unsqueeze(0)
    return value


def assert_finite_tensor(name: str, value: torch.Tensor) -> None:
    if torch.isfinite(value).all():
        return
    finite = torch.isfinite(value)
    finite_max = float(value[finite].float().abs().max().item()) if bool(finite.any()) else float("nan")
    raise FloatingPointError(f"non-finite {name}: shape={tuple(value.shape)} finite_abs_max={finite_max}")


def assert_finite_parameters(intervention: TargetResidualIntervention, stage: str) -> None:
    for name, parameter in intervention.adapters.named_parameters():
        assert_finite_tensor(f"{stage} parameter {name}", parameter.detach())


def adapter_gradient_norms(intervention: TargetResidualIntervention) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for layer_id in intervention.layer_ids:
        adapter = intervention.adapter(layer_id)
        down = 0.0 if adapter.down.grad is None else float(adapter.down.grad.detach().float().norm().item())
        up = 0.0 if adapter.up.grad is None else float(adapter.up.grad.detach().float().norm().item())
        output[layer_id] = {
            "down_l2": down,
            "up_l2": up,
            "combined_l2": float((down * down + up * up) ** 0.5),
        }
    return output


def evaluate_oracle(
    transformer: Any,
    intervention: TargetResidualIntervention,
    cache: dict[str, Any],
    full_conditioning: dict[str, torch.Tensor],
    base_config: dict[str, Any],
    state_indices: list[int],
    device: str,
) -> dict[str, float]:
    errors: list[float] = []
    baselines: list[float] = []
    per_state: dict[str, float] = {}
    with torch.inference_mode():
        intervention.collect_metrics = False
        intervention.reset_metrics()
        for index in state_indices:
            prediction = direct_velocity(
                transformer,
                cache["target_states"][index],
                cache["timesteps"][index],
                cache["image_tail"],
                cache["img_ids"],
                full_conditioning,
                float(base_config["guidance_scale"]),
                device,
            )
            teacher = teacher_velocity(cache, index).to(device)
            error = float(mse_velocity(prediction, teacher).item())
            baseline = float(mse_velocity(cache["full_velocities"][index], teacher_velocity(cache, index)).item())
            errors.append(error)
            baselines.append(baseline)
            per_state[f"state_{index}_velocity_mse"] = error
            per_state[f"state_{index}_baseline_velocity_mse"] = baseline
            per_state[f"state_{index}_relative_to_baseline"] = error / (baseline + 1e-12)
    mean_error = float(np.mean(errors))
    baseline_error = float(np.mean(baselines))
    return {
        "velocity_mse": mean_error,
        "baseline_velocity_mse": baseline_error,
        "relative_to_baseline": mean_error / (baseline_error + 1e-12),
        **per_state,
    }


def checkpoint_path(oracle_root: Path) -> Path:
    return oracle_root / "checkpoints" / "best.pt"


def latest_checkpoint_path(oracle_root: Path) -> Path:
    return oracle_root / "checkpoints" / "latest.pt"


def adapter_state_cpu(intervention: TargetResidualIntervention) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in intervention.adapters.state_dict().items()
    }


def gradient_check(
    pipeline: Any,
    intervention: TargetResidualIntervention,
    cache: dict[str, Any],
    full_conditioning: dict[str, torch.Tensor],
    base_config: dict[str, Any],
    state_index: int,
    device: str,
) -> dict[str, Any]:
    if not 0 <= state_index < int(cache["target_states"].shape[0]):
        raise ValueError(f"state index {state_index} is outside the matched cache")
    intervention.set_scale(1.0)
    intervention.collect_metrics = True
    intervention.adapters.zero_grad(set_to_none=True)
    intervention.reset_metrics()
    with intervention.applied():
        uncorrected = direct_velocity(
            pipeline.transformer,
            cache["target_states"][state_index],
            cache["timesteps"][state_index],
            cache["image_tail"],
            cache["img_ids"],
            full_conditioning,
            float(base_config["guidance_scale"]),
            device,
        )
        # Zero initialization must be an exact no-op.
        cached_full = cache["full_velocities"][state_index].unsqueeze(0)
        if not torch.equal(uncorrected.detach().cpu(), cached_full):
            delta = float(
                (
                    uncorrected.detach().float().cpu()
                    - cached_full.float()
                )
                .abs()
                .max()
                .item()
            )
            raise RuntimeError(f"zero-initialized Oracle changed velocity; max_abs_delta={delta:.3e}")
        target = teacher_velocity(cache, state_index).to(device)
        velocity_loss = mse_velocity(uncorrected, target)
        residual_penalty = intervention.metric_regularizer()
        total_loss = velocity_loss + residual_penalty
        for name, value in (
            ("gradient-check prediction", uncorrected),
            ("gradient-check velocity loss", velocity_loss),
            ("gradient-check residual penalty", residual_penalty),
            ("gradient-check total loss", total_loss),
        ):
            assert_finite_tensor(name, value)
        total_loss.backward()

    gradient_summary: dict[str, dict[str, float | bool]] = {}
    for name, parameter in intervention.adapters.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"Oracle parameter has no gradient: {name}")
        assert_finite_tensor(f"gradient-check gradient {name}", parameter.grad)
        norm = float(parameter.grad.detach().float().norm().item())
        gradient_summary[name] = {"norm": norm, "nonzero": norm > 0.0}
        if name.endswith(".down") and norm != 0.0:
            raise RuntimeError(f"zero-initialized down projection should have zero first-step gradient: {name}={norm}")
        if name.endswith(".up") and norm == 0.0:
            raise RuntimeError(f"up projection should receive a nonzero first-step gradient: {name}")
    assert_base_model_unmodified(pipeline)
    result = {
        "state_index": state_index,
        "velocity_mse": float(velocity_loss.detach().item()),
        "residual_penalty": float(residual_penalty.detach().item()),
        "adapter_gradients": gradient_summary,
        "residual_metrics": intervention.detached_metrics(),
    }
    intervention.adapters.zero_grad(set_to_none=True)
    return result


def train_oracle(
    pipeline: Any,
    transformer: Any,
    intervention: TargetResidualIntervention,
    cache: dict[str, Any],
    full_conditioning: dict[str, torch.Tensor],
    base_config: dict[str, Any],
    oracle_config: dict[str, Any],
    oracle_root: Path,
    iterations: int,
    device: str,
    resume: bool,
    run_fingerprint: str,
    save_evaluation_checkpoints: bool = False,
) -> tuple[list[dict[str, Any]], Path]:
    best_checkpoint = checkpoint_path(oracle_root)
    latest_checkpoint = latest_checkpoint_path(oracle_root)
    training = oracle_config["training"]
    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    optimizer = torch.optim.AdamW(
        list(intervention.parameters()),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    total_states = int(cache["target_states"].shape[0])
    evaluation_settings = oracle_config.get("evaluation", oracle_config["smoke"])
    eval_count = min(int(evaluation_settings["eval_state_count"]), total_states)
    evaluation_indices = np.linspace(0, total_states - 1, num=eval_count, dtype=int).tolist()
    history: list[dict[str, Any]] = []
    best_score = math.inf
    best_adapters: dict[str, torch.Tensor] | None = None
    start_iteration = 1
    generator = torch.Generator(device="cpu").manual_seed(seed)

    intervention.set_scale(1.0)
    intervention.collect_metrics = True
    with intervention.applied():
        if resume and latest_checkpoint.is_file():
            payload = torch.load(latest_checkpoint, map_location="cpu", weights_only=False)
            if payload.get("run_fingerprint") != run_fingerprint:
                raise RuntimeError("refusing to resume Oracle optimizer from an incompatible run")
            intervention.adapters.load_state_dict(payload["adapters"])
            optimizer.load_state_dict(payload["optimizer"])
            optimizer_to_device(optimizer, device)
            history = payload["history"]
            best_score = float(payload["best_score"])
            best_adapters = payload["best_adapters"]
            restore_rng_state(payload["rng_state"], generator)
            start_iteration = int(payload["iteration"]) + 1
            print(
                f"[train] resumed from iteration {start_iteration - 1}; target={iterations}",
                flush=True,
            )
        else:
            intervention.reset_metrics()
            initial_evaluation = evaluate_oracle(
                transformer,
                intervention,
                cache,
                full_conditioning,
                base_config,
                evaluation_indices,
                device,
            )
            if not all(math.isfinite(value) for value in initial_evaluation.values()):
                raise FloatingPointError(f"non-finite iteration-0 evaluation: {initial_evaluation}")
            history.append(
                {
                    "iteration": 0,
                    "train_state_index": None,
                    "train_velocity_mse": None,
                    "train_residual_penalty": 0.0,
                    "train_total_loss": None,
                    "gradient_norm": 0.0,
                    "gradient_norms_by_layer": {
                        layer_id: {"down_l2": 0.0, "up_l2": 0.0, "combined_l2": 0.0}
                        for layer_id in intervention.layer_ids
                    },
                    "evaluation": initial_evaluation,
                    "parameter_norms": intervention.parameter_norms(),
                    "last_step_residual_metrics": [],
                    "evaluation_residual_metrics": aggregate_residual_metrics(intervention.detached_metrics()),
                }
            )
            best_score = initial_evaluation["velocity_mse"]
            best_adapters = adapter_state_cpu(intervention)
            atomic_torch_save(
                best_checkpoint,
                {
                    "run_fingerprint": run_fingerprint,
                    "iteration": 0,
                    "adapters": best_adapters,
                    "history": history,
                    "best_evaluation": initial_evaluation,
                    "parameter_norms": intervention.parameter_norms(),
                },
            )
            if save_evaluation_checkpoints:
                atomic_torch_save(
                    oracle_root / "checkpoints" / "evaluations" / "iter_0000.pt",
                    {
                        "run_fingerprint": run_fingerprint,
                        "iteration": 0,
                        "adapters": best_adapters,
                        "evaluation": initial_evaluation,
                    },
                )
            print(
                f"[train] iteration 0/{iterations} relative_velocity_mse="
                f"{initial_evaluation['relative_to_baseline']:.6f}",
                flush=True,
            )

        for iteration in range(start_iteration, iterations + 1):
            state_index = int(torch.randint(total_states, (1,), generator=generator).item())
            optimizer.zero_grad(set_to_none=True)
            intervention.reset_metrics()
            prediction = direct_velocity(
                transformer,
                cache["target_states"][state_index],
                cache["timesteps"][state_index],
                cache["image_tail"],
                cache["img_ids"],
                full_conditioning,
                float(base_config["guidance_scale"]),
                device,
            )
            target = teacher_velocity(cache, state_index).to(device)
            velocity_loss = mse_velocity(prediction, target)
            residual_penalty = intervention.metric_regularizer()
            loss = velocity_loss + float(training["residual_regularization"]) * residual_penalty
            assert_finite_tensor("training prediction", prediction)
            assert_finite_tensor("training velocity loss", velocity_loss)
            assert_finite_tensor("training residual penalty", residual_penalty)
            assert_finite_tensor("training total loss", loss)
            loss.backward()
            for name, parameter in intervention.adapters.named_parameters():
                if parameter.grad is not None:
                    assert_finite_tensor(f"training gradient {name}", parameter.grad)
            assert_base_model_unmodified(pipeline)
            per_layer_gradient_norms = adapter_gradient_norms(intervention)
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(list(intervention.parameters()), float(training["gradient_clip_norm"])).item()
            )
            if not math.isfinite(gradient_norm):
                raise FloatingPointError(f"non-finite clipped gradient norm at iteration {iteration}")
            optimizer.step()
            assert_finite_parameters(intervention, "after optimizer step")

            train_metrics = intervention.detached_metrics()
            if (
                iteration == 1
                or iteration % int(evaluation_settings["eval_interval"]) == 0
                or iteration == iterations
            ):
                intervention.collect_metrics = True
                intervention.reset_metrics()
                evaluation_result = evaluate_oracle(
                    transformer,
                    intervention,
                    cache,
                    full_conditioning,
                    base_config,
                    evaluation_indices,
                    device,
                )
                if not all(math.isfinite(value) for value in evaluation_result.values()):
                    raise FloatingPointError(f"non-finite Oracle evaluation: {evaluation_result}")
                evaluation_metrics = aggregate_residual_metrics(intervention.detached_metrics())
                intervention.collect_metrics = True
                row = {
                    "iteration": iteration,
                    "train_state_index": state_index,
                    "train_velocity_mse": float(velocity_loss.detach().item()),
                    "train_residual_penalty": float(residual_penalty.detach().item()),
                    "train_total_loss": float(loss.detach().item()),
                    "gradient_norm": gradient_norm,
                    "gradient_norms_by_layer": per_layer_gradient_norms,
                    "evaluation": evaluation_result,
                    "parameter_norms": intervention.parameter_norms(),
                    "last_step_residual_metrics": train_metrics,
                    "evaluation_residual_metrics": evaluation_metrics,
                }
                history.append(row)
                current_adapters = adapter_state_cpu(intervention)
                if evaluation_result["velocity_mse"] < best_score:
                    best_score = evaluation_result["velocity_mse"]
                    best_adapters = current_adapters
                    atomic_torch_save(
                        best_checkpoint,
                        {
                            "run_fingerprint": run_fingerprint,
                            "iteration": iteration,
                            "adapters": best_adapters,
                            "history": history,
                            "best_evaluation": evaluation_result,
                            "parameter_norms": intervention.parameter_norms(),
                        },
                    )
                if best_adapters is None:
                    raise RuntimeError("missing best Oracle adapter state")
                atomic_torch_save(
                    latest_checkpoint,
                    {
                        "run_fingerprint": run_fingerprint,
                        "iteration": iteration,
                        "adapters": current_adapters,
                        "optimizer": optimizer.state_dict(),
                        "rng_state": capture_rng_state(generator),
                        "history": history,
                        "best_score": best_score,
                        "best_adapters": best_adapters,
                    },
                )
                atomic_write_json(oracle_root / "training_history.json", history)
                print(
                    f"[train] iteration {iteration}/{iterations} "
                    f"train_mse={float(velocity_loss.detach().item()):.6e} "
                    f"relative_velocity_mse={evaluation_result['relative_to_baseline']:.6f}",
                    flush=True,
                )

    if best_adapters is None or not best_checkpoint.is_file():
        raise RuntimeError("Oracle optimization produced no valid checkpoint")
    intervention.adapters.load_state_dict(best_adapters)
    atomic_write_json(oracle_root / "training_history.json", history)
    write_loss_curve(oracle_root / "plots" / "velocity_loss.png", history)
    return history, best_checkpoint


def font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def write_loss_curve(path: Path, history: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ensure_parent(path)
    iterations = [row["iteration"] for row in history]
    values = [row["evaluation"]["relative_to_baseline"] for row in history]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(iterations, values, marker="o", label="Oracle / baseline velocity MSE")
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("optimization iteration")
    axis.set_ylabel("relative velocity MSE")
    axis.set_title("Oracle velocity fit")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def rollout_paths(oracle_root: Path, strength: float) -> tuple[Path, Path]:
    tag = f"edit_{strength:.2f}".replace(".", "p")
    return oracle_root / "rollouts" / f"{tag}.png", oracle_root / "rollouts" / f"{tag}.json"


def validate_rollout_record(
    record: dict[str, Any],
    image_path: Path,
    run_fingerprint: str,
    adapter_hash: str,
) -> None:
    if (
        record.get("run_fingerprint") != run_fingerprint
        or record.get("adapter_sha256") != adapter_hash
        or record.get("output_sha256") != file_sha256(image_path)
    ):
        raise RuntimeError(f"refusing to reuse stale rollout {image_path}")


def aggregate_residual_metrics(metrics: list[dict[str, float | str]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for metric in metrics:
        grouped[str(metric["layer_id"])].append(metric)
    output: dict[str, dict[str, float]] = {}
    for layer_id, rows in grouped.items():
        output[layer_id] = {
            key: float(np.mean([float(row[key]) for row in rows]))
            for key in (
                "regularizer",
                "correction_relative_hidden_rms",
                "correction_relative_update_rms",
                "correction_rms",
                "hidden_rms",
                "update_rms",
            )
        }
    return output


def run_scaled_rollouts(
    pipeline: Any,
    base_config: dict[str, Any],
    sample: dict[str, Any],
    full_image_path: Path,
    intervention: TargetResidualIntervention,
    oracle_root: Path,
    strengths: list[float],
    device: str,
    resume: bool,
    run_fingerprint: str,
    adapter_hash: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with torch.inference_mode(), intervention.applied():
        for edit_strength in strengths:
            image_path, record_path = rollout_paths(oracle_root, edit_strength)
            if resume and image_path.is_file() and record_path.is_file():
                record = load_json(record_path)
                validate_rollout_record(record, image_path, run_fingerprint, adapter_hash)
                records.append(record)
                continue
            correction_scale = 1.0 - float(edit_strength)
            intervention.set_scale(correction_scale)
            intervention.collect_metrics = correction_scale != 0.0
            intervention.reset_metrics()
            generator = torch.Generator(device=device).manual_seed(int(sample["seed"]))
            started = time.perf_counter()
            image = pipeline(
                image=Image.open(sample["source_image"]).convert("RGB"),
                prompt=sample["instruction"],
                num_inference_steps=int(base_config["num_inference_steps"]),
                guidance_scale=float(base_config["guidance_scale"]),
                true_cfg_scale=float(base_config["true_cfg_scale"]),
                generator=generator,
                max_sequence_length=int(base_config.get("max_sequence_length", 512)),
                _auto_resize=True,
            ).images[0]
            elapsed = time.perf_counter() - started
            atomic_save_png(image, image_path)
            record = {
                "run_fingerprint": run_fingerprint,
                "adapter_sha256": adapter_hash,
                "edit_strength": float(edit_strength),
                "correction_scale": correction_scale,
                "output_image": str(image_path),
                "output_sha256": file_sha256(image_path),
                "elapsed_seconds": elapsed,
                "residual_metrics": aggregate_residual_metrics(intervention.detached_metrics()),
            }
            if correction_scale == 0.0:
                record["exact_full_edit_hash_match"] = record["output_sha256"] == file_sha256(full_image_path)
                if not record["exact_full_edit_hash_match"]:
                    raise RuntimeError("correction-disabled rollout differs from full-edit baseline")
            atomic_write_json(record_path, record)
            records.append(record)
    return records


def image_embedding(paths: list[Path], dino_model: Path, device: str) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    processor = AutoImageProcessor.from_pretrained(dino_model)
    model = AutoModel.from_pretrained(dino_model, torch_dtype=torch.bfloat16).to(device).eval()
    images = [Image.open(path).convert("RGB") for path in paths]
    with torch.inference_mode():
        inputs = processor(images=images, return_tensors="pt")
        values = inputs["pixel_values"].to(device=device, dtype=torch.bfloat16)
        hidden = model(pixel_values=values).last_hidden_state.float()
    output = {}
    for index, path in enumerate(paths):
        output[str(path)] = (
            torch.nn.functional.normalize(hidden[index, 0], dim=-1).cpu(),
            torch.nn.functional.normalize(hidden[index, 1:], dim=-1).cpu(),
        )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def dino_distance(first: tuple[torch.Tensor, torch.Tensor], second: tuple[torch.Tensor, torch.Tensor]) -> dict[str, float]:
    first_cls, first_patch = first
    second_cls, second_patch = second
    if first_patch.shape != second_patch.shape:
        raise ValueError("DINO patch grids differ")
    cls = float(1.0 - torch.sum(first_cls * second_cls).item())
    patch = float(torch.mean(1.0 - torch.sum(first_patch * second_patch, dim=-1)).item())
    return {"cls_distance": cls, "patch_distance": patch, "mean_distance": 0.5 * (cls + patch)}


def grayscale_laplacian_variance(path: Path, target_size: tuple[int, int] | None = None) -> float:
    image = Image.open(path).convert("L")
    if target_size is not None:
        image = image.resize(target_size, Image.Resampling.BICUBIC)
    values = np.asarray(image, dtype=np.float32)
    laplacian = (
        -4 * values[1:-1, 1:-1]
        + values[:-2, 1:-1]
        + values[2:, 1:-1]
        + values[1:-1, :-2]
        + values[1:-1, 2:]
    )
    return float(laplacian.var())


def pseudo_non_edit_mae(source_path: Path, full_path: Path, candidate_path: Path, quantile: float) -> float:
    full = Image.open(full_path).convert("RGB")
    size = full.size
    source = np.asarray(Image.open(source_path).convert("RGB").resize(size, Image.Resampling.BICUBIC), dtype=np.float32)
    edited = np.asarray(full, dtype=np.float32)
    candidate = np.asarray(Image.open(candidate_path).convert("RGB").resize(size, Image.Resampling.BICUBIC), dtype=np.float32)
    full_difference = np.abs(edited - source).mean(axis=2)
    threshold = float(np.quantile(full_difference, quantile))
    non_edit = full_difference < threshold
    if not bool(non_edit.any()):
        return float("nan")
    return float(np.abs(candidate - source).mean(axis=2)[non_edit].mean())


def make_contact_sheet(
    oracle_root: Path,
    sample: dict[str, Any],
    source_path: Path,
    full_path: Path,
    keep_path: Path,
    rollout_records: list[dict[str, Any]],
) -> None:
    panels: list[tuple[str, Path]] = [
        ("source", source_path),
        ("full edit", full_path),
        ("keep teacher", keep_path),
    ]
    panels.extend((f"edit={row['edit_strength']:.2f}", Path(row["output_image"])) for row in rollout_records)
    cell, header, footer = 512, 82, 34
    canvas = Image.new("RGB", (cell * len(panels), header + cell + footer), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), f"{sample['sample_id']} | target residual Oracle", fill="black", font=font(22))
    draw.text((12, 42), sample["instruction"][:150], fill="black", font=font(16))
    for index, (label, path) in enumerate(panels):
        image = Image.open(path).convert("RGB")
        thumbnail = ImageOps.contain(image, (cell, cell))
        x = index * cell + (cell - thumbnail.width) // 2
        y = header + (cell - thumbnail.height) // 2
        canvas.paste(thumbnail, (x, y))
        draw.text((index * cell + 8, header + cell + 6), label, fill="black", font=font(17))
    atomic_save_png(canvas, oracle_root / "contact_sheet.png")


def score_rollouts(
    oracle_config: dict[str, Any],
    sample: dict[str, Any],
    source_path: Path,
    full_path: Path,
    keep_path: Path,
    rollout_records: list[dict[str, Any]],
    oracle_root: Path,
    device: str,
) -> dict[str, Any]:
    paths = [source_path, full_path, keep_path] + [Path(row["output_image"]) for row in rollout_records]
    embeddings = image_embedding(paths, Path(oracle_config["dino_model"]), device)
    source_embedding = embeddings[str(source_path)]
    full_embedding = embeddings[str(full_path)]
    keep_embedding = embeddings[str(keep_path)]
    results = []
    local = sample["category"] in set(oracle_config["quality"]["local_categories"])
    for row in rollout_records:
        image_path = Path(row["output_image"])
        entry = dict(row)
        entry["dino_to_source"] = dino_distance(embeddings[str(image_path)], source_embedding)
        entry["dino_to_full_edit"] = dino_distance(embeddings[str(image_path)], full_embedding)
        entry["dino_to_keep_teacher"] = dino_distance(embeddings[str(image_path)], keep_embedding)
        entry["laplacian_variance"] = grayscale_laplacian_variance(image_path)
        entry["non_edit_pixel_mae_proxy"] = (
            pseudo_non_edit_mae(source_path, full_path, image_path, float(oracle_config["quality"]["pseudo_edit_quantile"]))
            if local
            else None
        )
        results.append(entry)

    ordered = sorted(results, key=lambda row: row["edit_strength"], reverse=True)
    source_distances = [row["dino_to_source"]["mean_distance"] for row in ordered]
    keep_distances = [row["dino_to_keep_teacher"]["mean_distance"] for row in ordered]
    full_distances = [row["dino_to_full_edit"]["mean_distance"] for row in ordered]
    tolerance = 1e-5
    summary = {
        "teacher_to_source": dino_distance(keep_embedding, source_embedding),
        "full_edit_to_source": dino_distance(full_embedding, source_embedding),
        "baseline_laplacian_variance": {
            "source": grayscale_laplacian_variance(source_path),
            "full_edit": grayscale_laplacian_variance(full_path),
            "keep_teacher": grayscale_laplacian_variance(keep_path),
        },
        "strength_order": [row["edit_strength"] for row in ordered],
        "source_distance_nonincreasing_violations": int(
            sum(next_value > value + tolerance for value, next_value in zip(source_distances, source_distances[1:]))
        ),
        "keep_distance_nonincreasing_violations": int(
            sum(next_value > value + tolerance for value, next_value in zip(keep_distances, keep_distances[1:]))
        ),
        "full_distance_nondecreasing_violations": int(
            sum(next_value < value - tolerance for value, next_value in zip(full_distances, full_distances[1:]))
        ),
        "rollouts": results,
    }
    atomic_write_json(oracle_root / "rollout_scores.json", summary)
    return summary


def evaluate_acceptance(
    oracle_config: dict[str, Any],
    history: list[dict[str, Any]],
    rollout_scores: dict[str, Any],
) -> dict[str, Any]:
    thresholds = {
        "minimum_velocity_improvement": 0.05,
        "maximum_correction_relative_hidden_rms": 0.20,
        "minimum_sharpness_ratio": 0.70,
        "maximum_order_violations": 1,
        "minimum_dino_improvement": 0.0,
    }
    thresholds.update(oracle_config.get("acceptance", {}))
    quick_mode = bool(oracle_config.get("quick_validation", {}).get("enabled"))
    initial_ratio = float(history[0]["evaluation"]["relative_to_baseline"])
    best_ratio = min(float(row["evaluation"]["relative_to_baseline"]) for row in history)
    velocity_improvement = (initial_ratio - best_ratio) / max(initial_ratio, 1e-12)

    fully_corrected = min(
        rollout_scores["rollouts"],
        key=lambda row: abs(float(row["edit_strength"])),
    )
    layer_metrics = fully_corrected.get("residual_metrics", {})
    correction_ratios = [
        float(metrics["correction_relative_hidden_rms"])
        for metrics in layer_metrics.values()
    ]
    mean_correction_ratio = float(np.mean(correction_ratios)) if correction_ratios else 0.0
    baseline_sharpness = rollout_scores["baseline_laplacian_variance"]
    sharpness_reference = min(
        float(baseline_sharpness["full_edit"]),
        float(baseline_sharpness["keep_teacher"]),
    )
    sharpness_ratio = float(fully_corrected["laplacian_variance"]) / max(sharpness_reference, 1e-12)
    corrected_sharpness_ratios = [
        float(row["laplacian_variance"]) / max(sharpness_reference, 1e-12)
        for row in rollout_scores["rollouts"]
        if float(row["edit_strength"]) < 1.0
    ]
    minimum_corrected_sharpness_ratio = (
        min(corrected_sharpness_ratios) if corrected_sharpness_ratios else sharpness_ratio
    )
    edit_one = min(
        rollout_scores["rollouts"],
        key=lambda row: abs(float(row["edit_strength"]) - 1.0),
    )
    edit_one_source = float(edit_one["dino_to_source"]["mean_distance"])
    edit_one_keep = float(edit_one["dino_to_keep_teacher"]["mean_distance"])
    corrected_source = float(fully_corrected["dino_to_source"]["mean_distance"])
    corrected_keep = float(fully_corrected["dino_to_keep_teacher"]["mean_distance"])
    source_dino_improvement = (edit_one_source - corrected_source) / max(edit_one_source, 1e-12)
    keep_dino_improvement = (edit_one_keep - corrected_keep) / max(edit_one_keep, 1e-12)
    checks = {
        "teacher_is_closer_to_source": (
            rollout_scores["teacher_to_source"]["mean_distance"]
            < rollout_scores["full_edit_to_source"]["mean_distance"]
        ),
        "edit_1_exact_full_edit": bool(edit_one.get("exact_full_edit_hash_match", False)),
        "velocity_improvement": velocity_improvement >= float(thresholds["minimum_velocity_improvement"]),
        "correction_size": (
            mean_correction_ratio <= float(thresholds["maximum_correction_relative_hidden_rms"])
        ),
        "sharpness": sharpness_ratio >= float(thresholds["minimum_sharpness_ratio"]),
        "source_order": (
            int(rollout_scores["source_distance_nonincreasing_violations"])
            <= int(thresholds["maximum_order_violations"])
        ),
        "keep_order": (
            int(rollout_scores["keep_distance_nonincreasing_violations"])
            <= int(thresholds["maximum_order_violations"])
        ),
    }
    if quick_mode:
        expected_strengths = [1.0, 0.5, 0.0]
        actual_strengths = [float(value) for value in rollout_scores["strength_order"]]
        checks.update(
            {
                "quick_strengths": actual_strengths == expected_strengths,
                "full_order": (
                    int(rollout_scores["full_distance_nondecreasing_violations"])
                    <= int(thresholds["maximum_order_violations"])
                ),
                "source_dino_improvement": (
                    source_dino_improvement >= float(thresholds["minimum_dino_improvement"])
                ),
                "keep_dino_improvement": (
                    keep_dino_improvement >= float(thresholds["minimum_dino_improvement"])
                ),
                "all_corrected_sharpness": (
                    minimum_corrected_sharpness_ratio >= float(thresholds["minimum_sharpness_ratio"])
                ),
            }
        )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "measurements": {
            "initial_velocity_ratio": initial_ratio,
            "best_velocity_ratio": best_ratio,
            "velocity_improvement": velocity_improvement,
            "mean_correction_relative_hidden_rms": mean_correction_ratio,
            "fully_corrected_sharpness_ratio": sharpness_ratio,
            "minimum_corrected_sharpness_ratio": minimum_corrected_sharpness_ratio,
            "source_dino_improvement": source_dino_improvement,
            "keep_dino_improvement": keep_dino_improvement,
        },
        "thresholds": thresholds,
    }


def validate_teacher(
    oracle_config: dict[str, Any],
    source_path: Path,
    full_path: Path,
    keep_path: Path,
    device: str,
) -> dict[str, Any]:
    embeddings = image_embedding([source_path, full_path, keep_path], Path(oracle_config["dino_model"]), device)
    source = embeddings[str(source_path)]
    full_distance = dino_distance(embeddings[str(full_path)], source)
    keep_distance = dino_distance(embeddings[str(keep_path)], source)
    result = {
        "mode": oracle_config["teacher_mode"],
        "prompt": oracle_config["teacher_prompt"],
        "full_edit_to_source": full_distance,
        "keep_to_source": keep_distance,
        "keep_is_closer": keep_distance["mean_distance"] < full_distance["mean_distance"],
    }
    return result


def save_csv(path: Path, history: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    rows = []
    for row in history:
        rows.append(
            {
                "iteration": row["iteration"],
                "train_state_index": row["train_state_index"],
                "train_velocity_mse": row["train_velocity_mse"],
                "train_residual_penalty": row["train_residual_penalty"],
                "train_total_loss": row["train_total_loss"],
                "gradient_norm": row["gradient_norm"],
                "eval_velocity_mse": row["evaluation"]["velocity_mse"],
                "eval_baseline_velocity_mse": row["evaluation"]["baseline_velocity_mse"],
                "eval_relative_to_baseline": row["evaluation"]["relative_to_baseline"],
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    args = parse_args()
    oracle_config = load_json(args.config)
    if args.eval_interval is not None or args.eval_state_count is not None:
        oracle_config = dict(oracle_config)
        oracle_config["evaluation"] = dict(oracle_config["smoke"])
        if args.eval_interval is not None:
            oracle_config["evaluation"]["eval_interval"] = int(args.eval_interval)
        if args.eval_state_count is not None:
            oracle_config["evaluation"]["eval_state_count"] = int(args.eval_state_count)
    base_config = load_base_config(oracle_config)
    selection = select_layers(oracle_config)
    combinations = layer_combinations(oracle_config)
    sample_id = args.sample_id or oracle_config["smoke"]["sample_id"]
    combo_id = args.combo or "primary"
    rank = int(args.rank or oracle_config["smoke"]["rank"])
    iterations = int(args.iterations or oracle_config["smoke"]["iterations"])
    sample = find_sample(oracle_config, sample_id)
    selected_layers = combinations[combo_id]
    quick = oracle_config.get("quick_validation", {})
    if quick.get("enabled"):
        if combo_id != "early_triplet":
            raise ValueError("quick validation only permits combo=early_triplet")
        if rank not in {4, 16}:
            raise ValueError("quick validation only permits rank 4 or explicit rank 16 fallback")
        expected_strengths = [1.0, 0.5, 0.0]
        actual_strengths = [float(value) for value in oracle_config["rollout_strengths"]]
        if actual_strengths != expected_strengths:
            raise ValueError(
                f"quick rollout_strengths must be exactly {expected_strengths}, found {actual_strengths}"
            )
    run_root = Path(args.run_root or oracle_config["output_root"])
    sample_root = run_root / "samples" / sample_id
    oracle_root = sample_root / "oracles" / combo_name(selected_layers) / f"rank_{rank:02d}"
    execution_root = (
        oracle_root
        if args.stage == "full"
        else oracle_root / "validation" / args.stage.replace("-", "_")
    )

    run_config = {
        "stage": args.stage,
        "oracle_config_path": str(Path(args.config).resolve()),
        "oracle_config": oracle_config,
        "base_run_config": base_config,
        "sample": sample,
        "layer_selection": selection,
        "combo_id": combo_id,
        "layers": selected_layers,
        "rank": rank,
        "requested_iterations": iterations,
        "state_index": args.state_index,
        "device": args.device,
        "resume": args.resume,
    }
    if args.dry_run:
        print(json.dumps(run_config, ensure_ascii=False, indent=2))
        return
    write_status(
        execution_root,
        state="running",
        stage="fingerprinting",
        message="hashing code, model, sample, and run settings",
    )
    try:
        fingerprints = build_fingerprints(
            oracle_config,
            base_config,
            sample,
            selected_layers,
            combo_id,
            rank,
            run_root,
        )
        run_config["fingerprints"] = fingerprints
        ensure_run_config(
            execution_root / "run_identity.json",
            {
                "stage": args.stage,
                "sample_id": sample_id,
                "combo_id": combo_id,
                "layers": selected_layers,
                "rank": rank,
                "run_sha256": fingerprints["run_sha256"],
            },
        )
        atomic_write_json(execution_root / "resolved_config.json", run_config)

        write_status(
            execution_root,
            state="running",
            stage="loading_model",
            message="loading and freezing all original FLUX-Kontext components",
        )
        pipeline = load_pipeline(base_config, args.device)
        attention_backend = str(oracle_config.get("training_attention_backend", "_native_flash"))
        pipeline.transformer.set_attention_backend(attention_backend)
        frozen_components = freeze_pipeline(pipeline)
        if not pipeline.transformer.is_gradient_checkpointing:
            pipeline.transformer.enable_gradient_checkpointing()
        assert_base_model_unmodified(pipeline)

        write_status(
            execution_root,
            state="running",
            stage="trajectory",
            message="capturing or validating full-edit and keep trajectories",
            frozen_components=frozen_components,
        )
        _full_image, full, _keep_image, keep = load_or_capture_trajectories(
            pipeline,
            base_config,
            oracle_config,
            sample,
            sample_root,
            fingerprints["trajectory_sha256"],
            args.resume,
        )
        paths = trajectory_paths(sample_root)
        teacher_validation: dict[str, Any] | None = None
        if args.stage == "full":
            teacher_validation_path = sample_root / "teacher_validation.json"
            if args.resume and teacher_validation_path.is_file():
                teacher_validation = require_fingerprint(
                    teacher_validation_path,
                    fingerprints["trajectory_sha256"],
                    "teacher validation",
                )
            else:
                teacher_validation = validate_teacher(
                    oracle_config,
                    Path(sample["source_image"]),
                    paths["full_image"],
                    paths["keep_image"],
                    args.device,
                )
                teacher_validation["fingerprint"] = fingerprints["trajectory_sha256"]
                atomic_write_json(teacher_validation_path, teacher_validation)
            if (
                oracle_config["require_teacher_closer_to_source"]
                and not teacher_validation["keep_is_closer"]
            ):
                policy = str(quick.get("teacher_invalid_policy", "fail"))
                if quick.get("enabled") and policy == "skip":
                    atomic_write_json(execution_root / "teacher_invalid.json", teacher_validation)
                    write_status(
                        execution_root,
                        state="teacher_invalid",
                        stage="teacher_validation",
                        message="keep teacher is not closer to source; Oracle training was skipped",
                        teacher_validation=teacher_validation,
                    )
                if save_evaluation_checkpoints:
                    atomic_torch_save(
                        oracle_root / "checkpoints" / "evaluations" / f"iter_{iteration:04d}.pt",
                        {
                            "run_fingerprint": run_fingerprint,
                            "iteration": iteration,
                            "adapters": current_adapters,
                            "evaluation": evaluation_result,
                            "parameter_norms": intervention.parameter_norms(),
                            "gradient_norms_by_layer": per_layer_gradient_norms,
                        },
                    )
                    print(
                        json.dumps(
                            {
                                "sample_id": sample_id,
                                "state": "teacher_invalid",
                                "teacher_validation": teacher_validation,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    return
                raise RuntimeError(
                    f"empty-prompt keep teacher failed source-proximity gate: {teacher_validation}"
                )
        full_conditioning = encode_conditioning(pipeline, base_config, sample["instruction"])
        keep_conditioning = encode_conditioning(pipeline, base_config, str(oracle_config["teacher_prompt"]))

        if args.stage in {"gradient-check", "train-check"}:
            cache = build_single_state_cache(
                pipeline.transformer,
                full,
                full_conditioning,
                keep_conditioning,
                base_config,
                args.state_index,
                args.device,
            )
        else:
            write_status(
                execution_root,
                state="running",
                stage="matched_velocity_cache",
                message="building and validating all matched states",
            )
            cache = build_training_cache(
                pipeline.transformer,
                full,
                keep,
                full_conditioning,
                keep_conditioning,
                base_config,
                sample_root,
                fingerprints["trajectory_sha256"],
                args.device,
                args.resume,
            )

        # Initialize the trainable Oracle deterministically and only after all
        # base-model work has been frozen.
        training_seed = int(oracle_config["training"]["seed"])
        torch.manual_seed(training_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(training_seed)
        layout = RoutingLayout(**full["layout"])
        intervention = TargetResidualIntervention(
            pipeline.transformer,
            selected_layers,
            target_tokens=layout.target_tokens,
            hidden_size=int(pipeline.transformer.inner_dim),
            rank=rank,
        )
        intervention.adapters.to(args.device)

        if args.stage == "gradient-check":
            write_status(
                execution_root,
                state="running",
                stage="gradient_check",
                message="checking one real state without updating parameters",
            )
            check = gradient_check(
                pipeline,
                intervention,
                cache,
                full_conditioning,
                base_config,
                0,
                args.device,
            )
            atomic_write_json(execution_root / "gradient_check.json", check)
            write_status(
                execution_root,
                state="complete",
                stage="gradient_check",
                message="single-state forward/backward check passed without an optimizer update",
            )
            print(json.dumps(check, ensure_ascii=False), flush=True)
            return

        if args.stage == "train-check":
            oracle_config = dict(oracle_config)
            oracle_config["evaluation"] = {"eval_interval": 1, "eval_state_count": 1}

        write_status(
            execution_root,
            state="running",
            stage="training",
            message="optimizing target-only Oracle residual",
            requested_iterations=iterations,
        )
        history, checkpoint = train_oracle(
            pipeline,
            pipeline.transformer,
            intervention,
            cache,
            full_conditioning,
            base_config,
            oracle_config,
            execution_root,
            iterations,
            args.device,
            args.resume,
            fingerprints["run_sha256"],
        )
        assert_base_model_unmodified(pipeline)
        save_csv(execution_root / "training_history.csv", history)

        if args.stage == "train-check":
            initial = float(history[0]["evaluation"]["velocity_mse"])
            final = float(history[-1]["evaluation"]["velocity_mse"])
            result = {
                "requested_iterations": iterations,
                "last_saved_iteration": int(history[-1]["iteration"]),
                "resumed_from_checkpoint": bool(args.resume),
                "initial_velocity_mse": initial,
                "final_velocity_mse": final,
                "finite": math.isfinite(initial) and math.isfinite(final),
                "not_worse_after_two_updates": None if iterations < 2 else final <= initial,
                "checkpoint": str(checkpoint),
            }
            if not result["finite"] or result["last_saved_iteration"] != iterations:
                raise RuntimeError(f"two-update validation did not finish cleanly: {result}")
            if iterations >= 2 and not result["not_worse_after_two_updates"]:
                raise RuntimeError(f"two-update validation immediately worsened keep error: {result}")
            atomic_write_json(execution_root / "train_check.json", result)
            write_status(
                execution_root,
                state="complete",
                stage="train_check",
                message="short optimizer and resume check passed",
                result=result,
            )
            print(json.dumps(result, ensure_ascii=False), flush=True)
            return

        if teacher_validation is None:
            raise RuntimeError("full run reached rollout without a validated keep teacher")

        adapter_hash = tensor_state_hash(adapter_state_cpu(intervention))
        write_status(
            execution_root,
            state="running",
            stage="rollout",
            message=f"generating {len(oracle_config['rollout_strengths'])} complete sampling trajectories",
            adapter_sha256=adapter_hash,
        )
        rollout_records = run_scaled_rollouts(
            pipeline,
            base_config,
            sample,
            paths["full_image"],
            intervention,
            oracle_root,
            [float(value) for value in oracle_config["rollout_strengths"]],
            args.device,
            args.resume,
            fingerprints["run_sha256"],
            adapter_hash,
        )
        make_contact_sheet(
            oracle_root,
            sample,
            Path(sample["source_image"]),
            paths["full_image"],
            paths["keep_image"],
            rollout_records,
        )
        write_status(
            execution_root,
            state="running",
            stage="scoring",
            message="computing image similarity, order, sharpness, and non-edit proxies",
        )
        rollout_scores = score_rollouts(
            oracle_config,
            sample,
            Path(sample["source_image"]),
            paths["full_image"],
            paths["keep_image"],
            rollout_records,
            oracle_root,
            args.device,
        )
        acceptance = evaluate_acceptance(oracle_config, history, rollout_scores)
        atomic_write_json(oracle_root / "acceptance.json", acceptance)
        result = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "adapter_sha256": adapter_hash,
            "run_fingerprint": fingerprints["run_sha256"],
            "teacher_validation": teacher_validation,
            "best_velocity_fit": min(
                history,
                key=lambda row: row["evaluation"]["velocity_mse"],
            )["evaluation"],
            "rollout_scores": rollout_scores,
            "acceptance": acceptance,
        }
        atomic_write_json(oracle_root / "result.json", result)
        if args.enforce_acceptance and not acceptance["passed"]:
            raise RuntimeError(f"single-sample smoke acceptance gates failed: {acceptance}")
        atomic_write_json(oracle_root / "complete.json", result)
        write_status(
            execution_root,
            state="complete",
            stage="complete",
            message="Oracle run completed",
            acceptance_passed=acceptance["passed"],
        )
        print(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "layers": selected_layers,
                    "rank": rank,
                    "oracle_root": str(oracle_root),
                    "best_velocity_ratio": min(
                        row["evaluation"]["relative_to_baseline"] for row in history
                    ),
                    "acceptance_passed": acceptance["passed"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    except Exception as error:
        write_status(
            execution_root,
            state="failed",
            stage="failed",
            message=str(error),
            error_type=type(error).__name__,
        )
        raise


if __name__ == "__main__":
    main()
