"""Exact TempFlow-style one-step SDE primitive from the archived early probe.

Provenance: batouyo/diffusers, branch ``codex/early-response-results``, commit
``db7837a27ed9cfd35d721745996a13abe70a6532``, source blob
``96305728c948a661a2134e69208c2e87a4f13ebb``.  This is deliberately separate
from the RF-equivalent SDE used by the coupling micro-test.
"""

from __future__ import annotations

from typing import Any

import torch
from diffusers import FluxKontextPipeline

from diagnostics.trajectory_primitives import sigma_pair


LEGACY_PROVENANCE: dict[str, Any] = {
    "repository": "batouyo/diffusers",
    "branch": "codex/early-response-results",
    "commit": "db7837a27ed9cfd35d721745996a13abe70a6532",
    "source_blob": "96305728c948a661a2134e69208c2e87a4f13ebb",
    "function": "sde_step_with_noise",
    "formula": "TempFlow-style mean/std, one SDE step then deterministic Kontext ODE suffix",
}


def sde_step_with_noise(
    pipe: FluxKontextPipeline,
    latents: torch.Tensor,
    prediction: torch.Tensor,
    timestep: torch.Tensor,
    noise: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Verbatim numerical formula from the archived explicit-noise runner."""
    sigma, sigma_next = sigma_pair(pipe, timestep)
    delta_t = sigma_next - sigma
    sigma_max = pipe.scheduler.sigmas[1]
    safe_sigma = torch.where(sigma == 1, sigma_max, sigma)
    std = torch.sqrt(sigma / (1 - safe_sigma)) * scale
    mean = latents.float() * (1 + std.square() / (2 * sigma) * delta_t) + prediction.float() * (
        1 + std.square() * (1 - sigma) / (2 * sigma)
    ) * delta_t
    return (mean + std * torch.sqrt(-delta_t) * noise.to(mean.dtype)).to(latents.dtype)

