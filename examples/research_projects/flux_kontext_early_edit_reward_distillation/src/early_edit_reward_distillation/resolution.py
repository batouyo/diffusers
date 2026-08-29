"""FLUX-Kontext resolution and packed-token accounting."""
from __future__ import annotations
from typing import Any

def resolve_dimensions(height: int, width: int, vae_scale_factor: int = 8) -> dict[str, int]:
    if height <= 0 or width <= 0 or vae_scale_factor <= 0: raise ValueError("dimensions and VAE scale factor must be positive")
    multiple = vae_scale_factor * 2
    resolved_height, resolved_width = height // multiple * multiple, width // multiple * multiple
    if min(resolved_height, resolved_width) <= 0: raise ValueError("image is smaller than one packed latent cell")
    latent_height, latent_width = resolved_height // vae_scale_factor, resolved_width // vae_scale_factor
    grid_height, grid_width = latent_height // 2, latent_width // 2
    return {"input_height": height, "input_width": width, "resolved_height": resolved_height, "resolved_width": resolved_width, "vae_scale_factor": vae_scale_factor, "latent_height": latent_height, "latent_width": latent_width, "packed_grid_height": grid_height, "packed_grid_width": grid_width, "generated_image_tokens": grid_height * grid_width}

def resolution_audit(image_sizes: list[tuple[int, int]], requested: tuple[int, int] = (512, 512), vae_scale_factor: int = 8) -> dict[str, Any]:
    return {"requested_resolution": {"height": requested[0], "width": requested[1]}, "vae_scale_factor": vae_scale_factor, "requested": resolve_dimensions(requested[0], requested[1], vae_scale_factor), "inputs": [{"source_size": {"width": w, "height": h}, "inference": resolve_dimensions(requested[0], requested[1], vae_scale_factor)} for w, h in image_sizes]}
