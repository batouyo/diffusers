#!/usr/bin/env python3
"""Visualize image-space SigLIP reward gradients along a FLUX-Kontext run.

The sampler mirrors the current FluxKontextPipeline denoising loop. It only
observes the model output before scheduler.step(); reward gradients never
modify the generation trajectory.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

DEFAULT_MODEL_PATH = "/data15/hyp/weight/FLUX.1-Kontext-dev"
DEFAULT_SIGLIP_PATH = "/data15/hyp/weight/reward_models/siglip-so400m-patch14-384"
DEFAULT_OUTPUT_DIR = "/data15/hyp/project_storage/flux-kontext-block-probing/reward_gradient_visualization"
DEFAULT_EDIT_INSTRUCTION = "Change the cup to black."
DEFAULT_SOURCE_REWARD_TEXT = "a red cup"
DEFAULT_TARGET_REWARD_TEXT = "a black cup"
FIXED_NUM_INFERENCE_STEPS = 15
DEFAULT_STEPS = (1, 2, 3, 4, 8, 12, 15)


def select_visualization_steps(num_steps: int, requested: Iterable[int] = DEFAULT_STEPS) -> list[int]:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    result = []
    for step in requested:
        step = int(step)
        if 1 <= step <= num_steps and step not in result:
            result.append(step)
    if not result:
        raise ValueError("at least one visualization step must be inside the denoising run")
    return result


def clean_prediction_from_flow(sample: torch.Tensor, model_output: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
    """Recover x0 from x_t = (1-sigma)x0 + sigma*noise."""
    sigma_tensor = torch.as_tensor(sigma, device=sample.device, dtype=sample.dtype)
    while sigma_tensor.ndim < sample.ndim:
        sigma_tensor = sigma_tensor.unsqueeze(-1)
    return sample - sigma_tensor * model_output


def rgb_gradient_magnitude(image_grad: torch.Tensor) -> torch.Tensor:
    if image_grad.ndim == 3:
        image_grad = image_grad.unsqueeze(0)
    if image_grad.ndim != 4 or image_grad.shape[1] != 3:
        raise ValueError(f"expected gradient shape [B,3,H,W], got {tuple(image_grad.shape)}")
    return torch.sqrt(torch.clamp(image_grad.square().sum(dim=1), min=0.0))


def compute_image_gradient(image: torch.Tensor, reward_fn) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute d reward / d image after detaching the generator prediction."""
    image_for_reward = image.detach().requires_grad_(True)
    with torch.enable_grad():
        reward = reward_fn(image_for_reward)
        if reward.ndim:
            reward = reward.mean()
        if not reward.requires_grad:
            raise RuntimeError("reward is not differentiable with respect to the image")
        image_grad = torch.autograd.grad(reward, image_for_reward, retain_graph=False)[0]
    return reward.detach(), image_grad.detach()


def directional_reward(target_score: torch.Tensor, source_score: torch.Tensor) -> torch.Tensor:
    return target_score - source_score


def per_step_percentile_normalize(magnitudes: dict[int, np.ndarray], low_percentile: float = 1.0, high_percentile: float = 99.0):
    if not 0 <= low_percentile < high_percentile <= 100:
        raise ValueError("percentiles must satisfy 0 <= low < high <= 100")
    normalized, bounds = {}, {}
    for step, value in magnitudes.items():
        low = float(np.percentile(value, low_percentile))
        high = float(np.percentile(value, high_percentile))
        if not np.isfinite(low) or not np.isfinite(high):
            raise ValueError("gradient magnitudes contain non-finite values")
        if high <= low:
            high = low + 1e-12
        normalized[step] = np.clip((value.astype(np.float32) - low) / (high - low), 0.0, 1.0).astype(np.float32)
        bounds[step] = (low, high)
    return normalized, bounds


def smooth_heatmap(heatmap: np.ndarray, gaussian_sigma: float) -> np.ndarray:
    if gaussian_sigma <= 0:
        return np.clip(heatmap.astype(np.float32), 0.0, 1.0)
    from scipy.ndimage import gaussian_filter
    return np.clip(gaussian_filter(heatmap.astype(np.float32), sigma=float(gaussian_sigma), mode="nearest"), 0.0, 1.0)


def heatmap_rgb(heatmap: np.ndarray, gaussian_sigma: float = 0.0) -> np.ndarray:
    return plt.get_cmap("turbo")(smooth_heatmap(heatmap, gaussian_sigma))[..., :3].astype(np.float32)


def global_percentile_normalize(magnitudes: dict[int, np.ndarray], low_percentile: float = 1.0, high_percentile: float = 99.0):
    if not 0 <= low_percentile < high_percentile <= 100:
        raise ValueError("percentiles must satisfy 0 <= low < high <= 100")
    all_values = np.concatenate([value.reshape(-1) for value in magnitudes.values()])
    low = float(np.percentile(all_values, low_percentile))
    high = float(np.percentile(all_values, high_percentile))
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("gradient magnitudes contain non-finite values")
    if high <= low:
        high = low + 1e-12
    normalized = {
        step: np.clip((value.astype(np.float32) - low) / (high - low), 0.0, 1.0).astype(np.float32)
        for step, value in magnitudes.items()
    }
    return normalized, low, high


def _to_display_array(image: torch.Tensor) -> np.ndarray:
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"expected image shape [B,3,H,W] or [3,H,W], got {tuple(image.shape)}")
    return np.clip(image.detach().float().cpu().permute(1, 2, 0).numpy(), 0.0, 1.0)


def make_overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45, gaussian_sigma: float = 0.0) -> np.ndarray:
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    display_heatmap = heatmap.astype(np.float32)
    if gaussian_sigma > 0:
        from scipy.ndimage import gaussian_filter

        display_heatmap = gaussian_filter(display_heatmap, sigma=float(gaussian_sigma), mode="nearest")
        display_heatmap = np.clip(display_heatmap, 0.0, 1.0)
    colors = plt.get_cmap("turbo")(display_heatmap)[..., :3]
    effective_alpha = alpha * display_heatmap[..., None]
    return np.clip((1.0 - effective_alpha) * image + effective_alpha * colors, 0.0, 1.0)


def _save_rgb(path: Path, array: np.ndarray, dpi: int = 300) -> None:
    Image.fromarray(np.round(np.clip(array, 0, 1) * 255).astype(np.uint8), mode="RGB").save(path, dpi=(dpi, dpi))


def _decode_packed_latents(pipe, packed: torch.Tensor, height: int, width: int) -> torch.Tensor:
    unpacked = pipe._unpack_latents(packed, height, width, pipe.vae_scale_factor)
    unpacked = unpacked / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    decoded = pipe.vae.decode(unpacked, return_dict=False)[0]
    return pipe.image_processor.postprocess(decoded, output_type="pt").clamp(0, 1)


def _resolve_dimensions(pipe, source: Image.Image, max_area: int) -> tuple[int, int]:
    multiple_of = pipe.vae_scale_factor * 2
    image_height, image_width = pipe.image_processor.get_default_height_width(source)
    image_aspect = image_width / image_height
    from diffusers.pipelines.flux.pipeline_flux_kontext import PREFERRED_KONTEXT_RESOLUTIONS

    _, width, height = min((abs(image_aspect - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS)
    if height * width > max_area:
        scale = (max_area / (height * width)) ** 0.5
        height, width = int(height * scale), int(width * scale)
    width = max(multiple_of, width // multiple_of * multiple_of)
    height = max(multiple_of, height // multiple_of * multiple_of)
    return height, width


class TorchSigLIPReward:
    """SigLIP cosine scores with differentiable torch-only preprocessing."""

    def __init__(self, model_path: str, device: torch.device, dtype: torch.dtype = torch.float32):
        from transformers import SiglipModel, SiglipTokenizer
        self.model = SiglipModel.from_pretrained(model_path, local_files_only=True, torch_dtype=dtype).to(device)
        self.tokenizer = SiglipTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model.eval().requires_grad_(False)
        cfg = getattr(self.model, "config", None)
        vision_cfg = getattr(cfg, "vision_config", cfg)
        self.image_size = int(getattr(vision_cfg, "image_size", 384))
        self.image_mean = tuple(getattr(vision_cfg, "image_mean", (0.5, 0.5, 0.5)))
        self.image_std = tuple(getattr(vision_cfg, "image_std", (0.5, 0.5, 0.5)))
        self.device, self.dtype = device, dtype
        self._target_features = None
        self._source_features = None

    @staticmethod
    def _extract_features(output: object) -> torch.Tensor:
        if torch.is_tensor(output):
            return output
        for name in ("text_embeds", "image_embeds", "pooler_output", "last_hidden_state"):
            value = getattr(output, name, None)
            if value is not None:
                return value.mean(dim=1) if name == "last_hidden_state" and value.ndim >= 3 else value
        raise TypeError(f"unsupported SigLIP output type: {type(output)}")

    def prepare(self, source_text: str, target_text: str) -> None:
        inputs = self.tokenizer([source_text, target_text], padding="max_length", truncation=True, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            features = self._extract_features(self.model.get_text_features(**inputs))
        features = torch.nn.functional.normalize(features, dim=-1)
        self._source_features = features[0:1]
        self._target_features = features[1:2]

    def _image_features(self, image: torch.Tensor) -> torch.Tensor:
        image = image.clamp(0, 1)
        if image.shape[-2:] != (self.image_size, self.image_size):
            image = torch.nn.functional.interpolate(image, size=(self.image_size, self.image_size), mode="bicubic", align_corners=False, antialias=True)
        mean = torch.tensor(self.image_mean, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.image_std, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        image = (image - mean) / std
        features = self._extract_features(self.model.get_image_features(pixel_values=image.to(dtype=self.dtype)))
        return torch.nn.functional.normalize(features, dim=-1)

    def score_pair(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._target_features is None or self._source_features is None:
            raise RuntimeError("call prepare() before scoring")
        features = self._image_features(image)
        target = (features * self._target_features.to(features)).sum(dim=-1).mean()
        source = (features * self._source_features.to(features)).sum(dim=-1).mean()
        return target, source


def compute_siglip_reward_gradient(image: torch.Tensor, reward_model: TorchSigLIPReward, mode: str):
    image_for_reward = image.detach().float().requires_grad_(True)
    with torch.enable_grad():
        target_score, source_score = reward_model.score_pair(image_for_reward)
        reward = directional_reward(target_score, source_score) if mode == "directional" else target_score
        if not reward.requires_grad:
            raise RuntimeError("reward is not differentiable with respect to the image")
        image_grad = torch.autograd.grad(reward, image_for_reward, retain_graph=False)[0]
    return reward.detach(), target_score.detach(), source_score.detach(), image_grad.detach()


def sampler_parity_check(pipe, image, prompt, height, width, seed, num_inference_steps, guidance_scale, max_sequence_length, max_area, manual_latents, device):
    with torch.no_grad():
        generator = torch.Generator(device=device).manual_seed(seed)
        official = pipe(
            image=image, prompt=prompt, height=height, width=width,
            num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
            generator=generator, output_type="latent",
            max_sequence_length=max_sequence_length, max_area=max_area,
        ).images
    diff = (official.float() - manual_latents.float()).abs()
    result = {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "allclose_atol_rtol_2e-2": bool(torch.allclose(official.float(), manual_latents.float(), atol=2e-2, rtol=2e-2)),
    }
    print(f"[parity] max_abs={result['max_abs']:.6g} mean_abs={result['mean_abs']:.6g} allclose={result['allclose_atol_rtol_2e-2']}", flush=True)
    return result

def run_experiment(args: argparse.Namespace) -> Path:
    from diffusers import FluxKontextPipeline
    from diffusers.pipelines.flux.pipeline_flux_kontext import calculate_shift, retrieve_timesteps

    if args.num_inference_steps != 15:
        raise ValueError("num_inference_steps is fixed at 15")
    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    source = Image.open(args.input_image).convert("RGB")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pipe = FluxKontextPipeline.from_pretrained(args.model_path, torch_dtype=dtype, local_files_only=True).to(device)
    pipe.set_progress_bar_config(disable=True)
    for module in (pipe.vae, pipe.transformer, pipe.text_encoder, pipe.text_encoder_2):
        if module is not None:
            module.eval().requires_grad_(False)

    height, width = _resolve_dimensions(pipe, source, args.max_area)
    resized_source = pipe.image_processor.resize(source, height, width)
    source_tensor = pipe.image_processor.preprocess(resized_source, height=height, width=width).to(device=device, dtype=dtype)
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(prompt=args.edit_instruction, prompt_2=None, device=device, num_images_per_prompt=1, max_sequence_length=args.max_sequence_length)
        num_channels_latents = pipe.transformer.config.in_channels // 4
        generator = torch.Generator(device=device).manual_seed(args.seed)
        latents, image_latents, latent_ids, image_ids = pipe.prepare_latents(source_tensor, 1, num_channels_latents, height, width, prompt_embeds.dtype, device, generator)
        if image_ids is not None:
            latent_ids = torch.cat([latent_ids, image_ids], dim=0)
        sigmas = np.linspace(1.0, 1 / args.num_inference_steps, args.num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(image_seq_len, pipe.scheduler.config.get("base_image_seq_len", 256), pipe.scheduler.config.get("max_image_seq_len", 4096), pipe.scheduler.config.get("base_shift", 0.5), pipe.scheduler.config.get("max_shift", 1.15))
        timesteps, _ = retrieve_timesteps(pipe.scheduler, args.num_inference_steps, device, sigmas=sigmas, mu=mu)
        pipe.scheduler.set_begin_index(0)
        guidance = torch.full([1], args.guidance_scale, device=device, dtype=torch.float32) if pipe.transformer.config.guidance_embeds else None

    visual_steps = select_visualization_steps(args.num_inference_steps, args.visualization_steps)
    reward = TorchSigLIPReward(args.siglip_path, device, torch.float32)
    reward.prepare(args.source_reward_text, args.target_reward_text)
    records: dict[int, dict] = {}
    magnitudes: dict[int, np.ndarray] = {}
    for index, timestep in enumerate(timesteps):
        step_number = index + 1
        sigma = float(pipe.scheduler.sigmas[index].item())
        with torch.no_grad():
            model_input = latents if image_latents is None else torch.cat([latents, image_latents], dim=1)
            timestep_batch = timestep.expand(latents.shape[0]).to(latents.dtype)
            velocity = pipe.transformer(hidden_states=model_input, timestep=timestep_batch / 1000, guidance=guidance, pooled_projections=pooled_prompt_embeds, encoder_hidden_states=prompt_embeds, txt_ids=text_ids, img_ids=latent_ids, joint_attention_kwargs={}, return_dict=False)[0]
            velocity = velocity[:, : latents.size(1)]
            z_t = latents.detach().clone()
            v_t = velocity.detach().clone()
            if step_number in visual_steps:
                z0_pred = clean_prediction_from_flow(z_t, v_t, sigma)
                clean_prediction = _decode_packed_latents(pipe, z0_pred, height, width).detach()
        if step_number in visual_steps:
            # Keep FLUX/VAE in its configured dtype, but use float32 for
            # SigLIP's image branch and input gradient to avoid bfloat16
            # quantization and patch-grid artifacts.
            reward_value, target_score, source_score, image_grad = compute_siglip_reward_gradient(clean_prediction.float(), reward, args.reward_mode)
            magnitude = rgb_gradient_magnitude(image_grad)[0].float().cpu().numpy()
            magnitudes[step_number] = magnitude
            grad_flat = image_grad[0].float()
            records[step_number] = {
                "z_t": z_t.cpu(), "v_t": v_t.cpu(), "sigma": sigma,
                "timestep": float(timestep.item()), "clean_prediction": clean_prediction.cpu(),
                "raw_gradient": image_grad[0].float().cpu().numpy(), "gradient_magnitude": magnitude,
                "reward": float(reward_value.item()), "target_similarity": float(target_score.item()),
                "source_similarity": float(source_score.item()), "grad_mean": float(grad_flat.abs().mean().item()),
                "grad_max": float(grad_flat.abs().max().item()), "grad_l2": float(torch.linalg.vector_norm(grad_flat).item()),
            }
            print(f"step={step_number} sigma={sigma:.6f} target_similarity={records[step_number]['target_similarity']:.8f} source_similarity={records[step_number]['source_similarity']:.8f} directional_reward={records[step_number]['reward']:.8f} grad_mean={records[step_number]['grad_mean']:.6g} grad_max={records[step_number]['grad_max']:.6g} grad_l2={records[step_number]['grad_l2']:.6g}", flush=True)
        with torch.no_grad():
            latents = pipe.scheduler.step(velocity, timestep, latents, return_dict=False)[0]
    final_latents = latents.detach()
    parity = sampler_parity_check(
        pipe, resized_source, args.edit_instruction, height, width, args.seed,
        args.num_inference_steps, args.guidance_scale, args.max_sequence_length,
        args.max_area, final_latents, device,
    )
    per_step, per_step_bounds = per_step_percentile_normalize(
        magnitudes, args.low_percentile, args.high_percentile
    )
    shared, shared_low, shared_high = global_percentile_normalize(
        magnitudes, args.low_percentile, args.high_percentile
    )
    source_array = np.asarray(resized_source, dtype=np.float32) / 255.0
    _save_rgb(output_dir / "source.png", source_array, args.dpi)
    with torch.no_grad():
        edited_array = _to_display_array(_decode_packed_latents(pipe, final_latents, height, width)[0])
    _save_rgb(output_dir / "edited.png", edited_array, args.dpi)

    metadata = {
        "input_image": os.path.abspath(args.input_image),
        "model_path": os.path.abspath(args.model_path),
        "siglip_path": os.path.abspath(args.siglip_path),
        "edit_instruction": args.edit_instruction,
        "source_reward_text": args.source_reward_text,
        "target_reward_text": args.target_reward_text,
        "reward_mode": args.reward_mode,
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "visualization_steps": visual_steps,
        "height": height,
        "width": width,
        "percentile_low": args.low_percentile,
        "percentile_high": args.high_percentile,
        "per_step_clip_bounds": {str(k): list(v) for k, v in per_step_bounds.items()},
        "shared_clip_low": shared_low,
        "shared_clip_high": shared_high,
        "gaussian_sigma": args.gaussian_sigma,
        "overlay_alpha": args.overlay_alpha,
        "colormap": "turbo",
        "dtype": args.dtype,
        "reward_dtype": "float32",
        "sampler_parity": parity,
        "steps": {
            str(step): {
                key: records[step][key]
                for key in (
                    "timestep", "sigma", "reward", "target_similarity",
                    "source_similarity", "grad_mean", "grad_max", "grad_l2",
                )
            }
            for step in visual_steps
        },
    }

    for step_number in visual_steps:
        record = records[step_number]
        step_dir = output_dir / "steps" / f"step_{step_number:02d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        torch.save(record["z_t"], step_dir / "latent.pt")
        torch.save(record["v_t"], step_dir / "velocity.pt")
        torch.save(record["clean_prediction"], step_dir / "clean_prediction.pt")
        np.save(step_dir / "raw_gradient.npy", record["raw_gradient"].astype(np.float32))
        np.save(step_dir / "gradient_magnitude.npy", record["gradient_magnitude"].astype(np.float32))
        np.save(step_dir / "heatmap.npy", per_step[step_number].astype(np.float32))
        np.save(step_dir / "heatmap_shared.npy", shared[step_number].astype(np.float32))
        _save_rgb(step_dir / "clean_prediction.png", _to_display_array(record["clean_prediction"]), args.dpi)
        _save_rgb(step_dir / "heatmap_raw_or_normalized.png", heatmap_rgb(per_step[step_number], args.gaussian_sigma), args.dpi)
        _save_rgb(step_dir / "heatmap_shared.png", heatmap_rgb(shared[step_number], args.gaussian_sigma), args.dpi)
        _save_rgb(
            step_dir / "overlay.png",
            make_overlay(_to_display_array(record["clean_prediction"]), per_step[step_number], args.overlay_alpha, args.gaussian_sigma),
            args.dpi,
        )
        _save_rgb(
            step_dir / "overlay_shared.png",
            make_overlay(_to_display_array(record["clean_prediction"]), shared[step_number], args.overlay_alpha, args.gaussian_sigma),
            args.dpi,
        )
        (step_dir / "sigma.json").write_text(
            json.dumps({
                "step": step_number,
                "timestep": record["timestep"],
                "sigma": record["sigma"],
                "target_similarity": record["target_similarity"],
                "source_similarity": record["source_similarity"],
                "directional_reward": record["reward"],
                "grad_mean": record["grad_mean"],
                "grad_max": record["grad_max"],
                "grad_l2": record["grad_l2"],
            }, indent=2),
            encoding="utf-8",
        )

    labels = ["Source", *[f"Step {step}" for step in visual_steps], "Edited"]
    per_arrays = [source_array] + [
        make_overlay(_to_display_array(records[step]["clean_prediction"]), per_step[step], args.overlay_alpha, args.gaussian_sigma)
        for step in visual_steps
    ] + [edited_array]
    shared_arrays = [source_array] + [
        make_overlay(_to_display_array(records[step]["clean_prediction"]), shared[step], args.overlay_alpha, args.gaussian_sigma)
        for step in visual_steps
    ] + [edited_array]

    def save_evolution(arrays: list[np.ndarray], stem: str) -> None:
        fig, axes = plt.subplots(1, len(arrays), figsize=(2.6 * len(arrays), 4.4), squeeze=False)
        for axis, label, array in zip(axes[0], labels, arrays):
            axis.imshow(array)
            axis.set_title(label, fontsize=10)
            axis.axis("off")
        fig.tight_layout(pad=0.6)
        fig.savefig(output_dir / f"{stem}.png", dpi=args.dpi, bbox_inches="tight")
        fig.savefig(output_dir / f"{stem}.pdf", dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

    save_evolution(per_arrays, "reward_gradient_evolution_per_step")
    save_evolution(shared_arrays, "reward_gradient_evolution_shared")
    save_evolution(per_arrays, "reward_gradient_evolution")
    (output_dir / "trajectory_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(output_dir)
    return output_dir

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-image", required=True)
    parser.add_argument("--edit-instruction", default=DEFAULT_EDIT_INSTRUCTION)
    parser.add_argument("--source-reward-text", default=DEFAULT_SOURCE_REWARD_TEXT)
    parser.add_argument("--target-reward-text", default=DEFAULT_TARGET_REWARD_TEXT)
    parser.add_argument("--reward-mode", choices=("target", "directional"), default="directional")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--siglip-path", default=DEFAULT_SIGLIP_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=15)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--max-area", type=int, default=1024**2)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--visualization-steps", type=int, nargs="+", default=list(DEFAULT_STEPS))
    parser.add_argument("--low-percentile", type=float, default=1.0)
    parser.add_argument("--high-percentile", type=float, default=99.0)
    parser.add_argument("--gaussian-sigma", type=float, default=2.0)
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


if __name__ == "__main__":
    run_experiment(build_parser().parse_args())
