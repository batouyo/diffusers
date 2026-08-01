"""Velocity-difference masks for target-token residual injection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class MaskResult:
    raw: torch.Tensor
    normalized: torch.Tensor
    weight: torch.Tensor
    degenerate: torch.Tensor


def token_velocity_difference(v_edit: torch.Tensor, v_neutral: torch.Tensor) -> torch.Tensor:
    if v_edit.shape != v_neutral.shape or v_edit.ndim != 3:
        raise ValueError("velocities must share [batch, target_tokens, channels] shape")
    return torch.linalg.vector_norm(v_edit.float() - v_neutral.float(), dim=-1)


def robust_normalize_mask(raw: torch.Tensor, low_quantile: float = 0.05, high_quantile: float = 0.95, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    if raw.ndim != 2:
        raise ValueError("raw mask must have shape [batch, tokens]")
    if not 0 <= low_quantile < high_quantile <= 1:
        raise ValueError("invalid quantile bounds")
    low = torch.quantile(raw.float(), low_quantile, dim=1, keepdim=True)
    high = torch.quantile(raw.float(), high_quantile, dim=1, keepdim=True)
    span = high - low
    degenerate = span <= eps
    normalized = ((raw.float() - low) / span.clamp_min(eps)).clamp(0.0, 1.0)
    normalized = torch.where(degenerate.expand_as(normalized), torch.ones_like(normalized), normalized)
    return normalized.detach(), degenerate.squeeze(1).detach()


def mask_weight(
    normalized: torch.Tensor,
    *,
    mask_type: str,
    tau: float,
    temperature: float,
    lambda_bg: float,
) -> torch.Tensor:
    if normalized.ndim != 2:
        raise ValueError("normalized mask must have shape [batch, tokens]")
    if not 0 <= tau <= 1 or not 0 <= lambda_bg <= 1:
        raise ValueError("tau and lambda_bg must be in [0, 1]")
    if mask_type == "none":
        edit = torch.ones_like(normalized)
    elif mask_type == "hard":
        edit = (normalized >= tau).to(normalized.dtype)
    elif mask_type == "soft":
        if temperature <= 0:
            raise ValueError("soft mask temperature must be positive")
        edit = torch.sigmoid((normalized - tau) / temperature)
    else:
        raise ValueError(f"unknown mask_type {mask_type!r}")
    return (lambda_bg + (1.0 - lambda_bg) * edit).unsqueeze(-1).detach()


class TemporalMaskEMA:
    def __init__(self, beta: float = 0.7) -> None:
        if not 0 <= beta < 1:
            raise ValueError("beta must be in [0, 1)")
        self.beta = float(beta)
        self.previous: torch.Tensor | None = None

    def reset(self) -> None:
        self.previous = None

    def update(self, raw_mask: torch.Tensor) -> torch.Tensor:
        current = raw_mask.detach().float()
        if self.previous is None:
            self.previous = current
        else:
            if self.previous.shape != current.shape:
                raise ValueError("mask shape changed within one trajectory")
            self.previous = self.beta * self.previous + (1.0 - self.beta) * current
        return self.previous


def make_mask(
    v_edit: torch.Tensor,
    v_neutral: torch.Tensor,
    *,
    mask_type: str,
    tau: float,
    temperature: float,
    lambda_bg: float,
    ema: TemporalMaskEMA | None = None,
) -> MaskResult:
    raw = token_velocity_difference(v_edit, v_neutral)
    raw_for_norm = ema.update(raw) if ema is not None else raw.detach()
    normalized, degenerate = robust_normalize_mask(raw_for_norm)
    weight = mask_weight(
        normalized,
        mask_type=mask_type,
        tau=tau,
        temperature=temperature,
        lambda_bg=lambda_bg,
    )
    return MaskResult(raw=raw.detach(), normalized=normalized, weight=weight, degenerate=degenerate)


def infer_token_grid(tokens: int, height: int, width: int, vae_scale_factor: int) -> tuple[int, int]:
    latent_h, latent_w = max(1, height // vae_scale_factor), max(1, width // vae_scale_factor)
    for grid in ((latent_h, latent_w), (max(1, latent_h // 2), max(1, latent_w // 2))):
        if grid[0] * grid[1] == tokens:
            return grid
    target_ratio = height / max(width, 1)
    candidates = [(h, tokens // h) for h in range(1, int(tokens**0.5) + 1) if tokens % h == 0]
    if not candidates:
        raise ValueError(f"cannot infer grid for {tokens} tokens")
    return min(candidates, key=lambda item: abs(item[0] / item[1] - target_ratio))


def token_mask_to_image(mask: torch.Tensor, height: int, width: int, vae_scale_factor: int) -> torch.Tensor:
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise ValueError("mask must have shape [batch, tokens]")
    grid_h, grid_w = infer_token_grid(mask.shape[1], height, width, vae_scale_factor)
    image = mask.reshape(mask.shape[0], 1, grid_h, grid_w)
    return F.interpolate(image, size=(height, width), mode="bilinear", align_corners=False)

