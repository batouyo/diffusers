"""Numerical primitives shared by the Coupled-SDE mechanism diagnostic.

The implementation is copied from the frozen early-response probe's Kontext
trajectory path, with only the source argument generalized from a path to a
PIL image.  Keeping this small module in the diagnostic tree prevents the
diagnostic from depending on an untracked ``run_temporal_probe.py`` file.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from diffusers import FluxKontextPipeline
from diffusers.pipelines.flux.pipeline_flux_kontext import calculate_shift, retrieve_timesteps


@torch.inference_mode()
def prepare(
    pipe: FluxKontextPipeline,
    source: Image.Image,
    prompt: str,
    seed: int,
    steps: int,
    guidance_scale: float,
    device: torch.device,
) -> dict[str, Any]:
    """Prepare the exact 1024-pixel Kontext state used by the prior probe."""
    source = source.convert("RGB")
    height = width = 1024
    source_tensor = pipe.image_processor.preprocess(
        pipe.image_processor.resize(source, height, width), height, width
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    prompt_embeds, pooled, text_ids = pipe.encode_prompt(
        prompt=prompt, device=device, num_images_per_prompt=1, max_sequence_length=512
    )
    channels = pipe.transformer.config.in_channels // 4
    latents, image_latents, latent_ids, image_ids = pipe.prepare_latents(
        source_tensor, 1, channels, height, width, prompt_embeds.dtype, device, generator, None
    )
    if image_latents is None or image_ids is None:
        raise RuntimeError("Expected FLUX-Kontext source-conditioning latents")
    image_ids = torch.cat([latent_ids, image_ids], dim=0)
    sigmas = np.linspace(1.0, 1.0 / steps, steps)
    mu = calculate_shift(
        latents.shape[1], pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5), pipe.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, _ = retrieve_timesteps(pipe.scheduler, steps, device, sigmas=sigmas, mu=mu)
    return {
        "latents": latents, "image_latents": image_latents, "image_ids": image_ids,
        "prompt_embeds": prompt_embeds, "pooled": pooled, "text_ids": text_ids,
        "timesteps": timesteps,
        "guidance": torch.full((1,), guidance_scale, device=device, dtype=torch.float32),
        "height": height, "width": width, "dtype": latents.dtype,
    }


@torch.inference_mode()
def velocity(pipe: FluxKontextPipeline, state: dict[str, Any], latents: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    batch = latents.shape[0]
    model_input = torch.cat([latents, state["image_latents"].repeat(batch, 1, 1)], dim=1)
    output = pipe.transformer(
        hidden_states=model_input,
        timestep=timestep.expand(batch).to(latents.dtype) / 1000,
        guidance=state["guidance"].expand(batch),
        pooled_projections=state["pooled"].repeat(batch, 1),
        encoder_hidden_states=state["prompt_embeds"].repeat(batch, 1, 1),
        txt_ids=state["text_ids"], img_ids=state["image_ids"],
        joint_attention_kwargs={}, return_dict=False,
    )[0]
    return output[:, :latents.shape[1]]


def sigma_pair(pipe: FluxKontextPipeline, timestep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    index = int(pipe.scheduler.index_for_timestep(timestep))
    return pipe.scheduler.sigmas[index], pipe.scheduler.sigmas[index + 1]


def ode_step(pipe: FluxKontextPipeline, latents: torch.Tensor, prediction: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    """Operation order matches ``FlowMatchEulerDiscreteScheduler.step``."""
    sigma, sigma_next = sigma_pair(pipe, timestep)
    return (latents.float() + (sigma_next - sigma) * prediction).to(latents.dtype)


@torch.inference_mode()
def decode(pipe: FluxKontextPipeline, latents: torch.Tensor, height: int = 1024, width: int = 1024) -> list[Image.Image]:
    unpacked = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    unpacked = unpacked / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        decoded = pipe.vae.decode(unpacked.to(pipe.vae.dtype), return_dict=False)[0]
    return pipe.image_processor.postprocess(decoded, output_type="pil")

