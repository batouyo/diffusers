"""FLUX-Kontext resolution and packed-token accounting."""
from __future__ import annotations
from typing import Any, Iterable

PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568), (688, 1504), (720, 1456), (752, 1392), (800, 1328),
    (832, 1248), (880, 1184), (944, 1104), (1024, 1024), (1104, 944),
    (1184, 880), (1248, 832), (1328, 800), (1392, 752), (1456, 720),
    (1504, 688), (1568, 672),
]

def resolve_dimensions(height: int, width: int, vae_scale_factor: int = 8) -> dict[str, int]:
    if height <= 0 or width <= 0 or vae_scale_factor <= 0: raise ValueError("dimensions and VAE scale factor must be positive")
    multiple = vae_scale_factor * 2
    resolved_height, resolved_width = height // multiple * multiple, width // multiple * multiple
    if min(resolved_height, resolved_width) <= 0: raise ValueError("image is smaller than one packed latent cell")
    latent_height, latent_width = resolved_height // vae_scale_factor, resolved_width // vae_scale_factor
    grid_height, grid_width = latent_height // 2, latent_width // 2
    return {"input_height": height, "input_width": width, "resolved_height": resolved_height, "resolved_width": resolved_width, "vae_scale_factor": vae_scale_factor, "latent_height": latent_height, "latent_width": latent_width, "packed_grid_height": grid_height, "packed_grid_width": grid_width, "generated_image_tokens": grid_height * grid_width}

def choose_preferred_source_size(height: int, width: int, vae_scale_factor: int = 8) -> tuple[int, int]:
    """Mirror Kontext's aspect-ratio nearest preferred source resolution."""
    if height <= 0 or width <= 0:
        raise ValueError("source dimensions must be positive")
    ratio = width / height
    _, preferred_width, preferred_height = min(
        (abs(ratio - candidate_width / candidate_height), candidate_width, candidate_height)
        for candidate_width, candidate_height in PREFERRED_KONTEXT_RESOLUTIONS
    )
    multiple = vae_scale_factor * 2
    return preferred_height // multiple * multiple, preferred_width // multiple * multiple

def resolution_audit(image_sizes: Iterable[tuple[int, int]], requested: tuple[int, int] = (512, 512), vae_scale_factor: int = 8) -> dict[str, Any]:
    inputs = []
    for width, height in image_sizes:
        source_height, source_width = choose_preferred_source_size(height, width, vae_scale_factor)
        inputs.append({"source_size": {"width": width, "height": height}, "source_conditioning_auto_resize": resolve_dimensions(source_height, source_width, vae_scale_factor), "inference_target": resolve_dimensions(requested[0], requested[1], vae_scale_factor)})
    return {"requested_resolution": {"height": requested[0], "width": requested[1]}, "vae_scale_factor": vae_scale_factor, "requested": resolve_dimensions(requested[0], requested[1], vae_scale_factor), "inputs": inputs}
