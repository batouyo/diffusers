"""Pure tensor/numerical pieces used by the FLUX-Kontext experiment."""
from __future__ import annotations
import hashlib
import math
from dataclasses import dataclass
from typing import Callable, Sequence
import torch

def rf_diffusion_coefficient(sigma: float, sigma_next: float, first_step: bool) -> float:
    if first_step:
        return 0.0
    if not 0.0 < sigma < 1.0 or sigma_next > sigma:
        raise ValueError(f"invalid RF schedule pair: sigma={sigma}, sigma_next={sigma_next}")
    return math.sqrt(2.0 * sigma / (1.0 - sigma) * (sigma - sigma_next))

def rf_sde_step(sample: torch.Tensor, model_output: torch.Tensor, sigma: float, sigma_next: float, noise: torch.Tensor, first_step: bool = False) -> tuple[torch.Tensor, dict[str, float]]:
    coefficient = rf_diffusion_coefficient(sigma, sigma_next, first_step)
    x, prediction, brownian = sample.float(), model_output.float(), noise.float()
    drift = prediction if first_step else 2.0 * prediction + x / (1.0 - sigma)
    updated = x + (sigma_next - sigma) * drift + coefficient * brownian
    return updated.to(sample.dtype), {"sigma": float(sigma), "sigma_next": float(sigma_next), "diffusion_coeff": float(coefficient), "noise_mean": float(brownian.mean()), "noise_std": float(brownian.std())}

def native_euler_sde_step(sample: torch.Tensor, model_output: torch.Tensor, sigma: float, sigma_next: float, noise: torch.Tensor, alpha: float = 0.0, diffusion_scale: float = 1.0, first_step: bool = False) -> tuple[torch.Tensor, dict[str, float]]:
    """Kontext-native Euler mean plus calibrated stochastic perturbation.

    The transformer output is a native velocity.  It must not be interpreted
    as the score/drift expected by the official syncSDE inversion scheduler.
    """
    if alpha < 0.0 or diffusion_scale < 0.0:
        raise ValueError("alpha and diffusion_scale must be non-negative")
    coefficient = rf_diffusion_coefficient(sigma, sigma_next, first_step)
    x = sample.float()
    prediction = model_output.float()
    brownian = noise.float()
    mean = x + (float(sigma_next) - float(sigma)) * prediction
    perturbation = float(alpha) * float(diffusion_scale) * coefficient * brownian
    updated = (mean + perturbation).to(sample.dtype)
    return updated, {"sigma": float(sigma), "sigma_next": float(sigma_next), "diffusion_coeff": float(coefficient), "alpha": float(alpha), "diffusion_scale": float(diffusion_scale), "mean_norm": float(mean.norm().item()), "noise_norm": float(brownian.norm().item()), "perturbation_norm": float(perturbation.norm().item()), "finite": bool(torch.isfinite(updated).all().item())}

def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]

def regional_delta_norms(candidate: torch.Tensor, mean: torch.Tensor, edit_mask: torch.Tensor) -> dict[str, float]:
    mask = edit_mask.to(device=candidate.device, dtype=torch.bool).reshape(1, -1, 1)
    delta = (candidate.float() - mean.float())
    preserve = delta.masked_select(~mask.expand_as(delta))
    edit = delta.masked_select(mask.expand_as(delta))
    return {"delta_norm": float(delta.norm().item()), "preserve_delta_norm": float(preserve.norm().item()), "edit_delta_norm": float(edit.norm().item())}

def noise_correlations(shared: torch.Tensor, mixed: torch.Tensor, independent: torch.Tensor, edit_mask: torch.Tensor) -> dict[str, float]:
    mask = edit_mask.to(device=shared.device, dtype=torch.bool)
    if mask.shape != shared.shape:
        mask = mask.reshape(1, -1, 1)
    mask = mask.expand_as(shared)
    def corr(a: torch.Tensor, b: torch.Tensor) -> float:
        a, b = a.flatten().float(), b.flatten().float()
        if a.numel() < 2 or a.std() == 0 or b.std() == 0: return float("nan")
        return float(torch.corrcoef(torch.stack([a, b]))[0, 1].item())
    return {"preserve_shared_correlation": corr(shared.masked_select(~mask), mixed.masked_select(~mask)), "edit_independent_correlation": corr(independent.masked_select(mask), mixed.masked_select(mask))}

def coupled_noise(shared: torch.Tensor, independent: torch.Tensor, edit_mask: torch.Tensor, rho: float = 0.0) -> torch.Tensor:
    if shared.shape != independent.shape:
        raise ValueError("shared and independent noise must have identical shapes")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [0, 1]")
    mask = edit_mask.to(dtype=shared.dtype, device=shared.device)
    while mask.ndim < shared.ndim:
        mask = mask.unsqueeze(-1)
    edit_noise = rho * shared + math.sqrt(max(0.0, 1.0 - rho * rho)) * independent
    return shared * (1.0 - mask) + edit_noise * mask

def critical_nonzero_steps(sigmas: Sequence[float], tolerance: float = 1e-12) -> list[dict[str, float | int]]:
    if len(sigmas) < 3:
        raise ValueError("at least two scheduler transitions are required")
    result = []
    for index, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
        coefficient = rf_diffusion_coefficient(float(sigma), float(sigma_next), index == 0)
        if coefficient > tolerance:
            result.append({"index": index, "post_step_index": index + 1, "sigma": float(sigma), "sigma_next": float(sigma_next), "diffusion_coeff": float(coefficient)})
    return result

@dataclass(frozen=True)
class BranchRecord:
    stage: int
    branch_step_index: int
    post_branch_step_index: int
    winner_index: int
    rewards: tuple[float, ...]
    repeated_rewards: tuple[float, ...]

def greedy_two_stage_branch(initial_state: torch.Tensor, branch_steps: Sequence[int], make_candidates: Callable[[torch.Tensor, int], Sequence[torch.Tensor]], score_candidates: Callable[[Sequence[torch.Tensor], int], Sequence[float]], rollout_to_step: Callable[[torch.Tensor, int, int], torch.Tensor], repeat_top2: Callable[[Sequence[torch.Tensor], Sequence[int], int], Sequence[float]] | None = None) -> tuple[torch.Tensor, list[BranchRecord]]:
    if len(branch_steps) != 2:
        raise ValueError("exactly two branch steps are required")
    state, records = initial_state, []
    for stage, branch_step in enumerate(branch_steps, start=1):
        candidates = list(make_candidates(state, branch_step))
        if len(candidates) != 4:
            raise ValueError("the minimal search fixes K=4")
        final_candidates = [rollout_to_step(candidate, branch_step + 1, stage) for candidate in candidates]
        rewards = tuple(float(x) for x in score_candidates(final_candidates, stage))
        if len(rewards) != 4:
            raise ValueError("one reward is required per candidate")
        top2 = sorted(range(4), key=lambda i: (-rewards[i], i))[:2]
        repeated = tuple(float(x) for x in (repeat_top2(final_candidates, top2, stage) if repeat_top2 else ()))
        means = list(rewards)
        if repeated:
            if len(repeated) != 4:
                raise ValueError("repeat_top2 must return four scores")
            means[top2[0]] = (rewards[top2[0]] + repeated[0] + repeated[1]) / 3.0
            means[top2[1]] = (rewards[top2[1]] + repeated[2] + repeated[3]) / 3.0
        winner = max(range(4), key=lambda i: (means[i], -i))
        state = final_candidates[winner]
        records.append(BranchRecord(stage, branch_step, branch_step + 1, winner, rewards, repeated))
    return state, records

def align_image_token_mask(pixel_mask: torch.Tensor, image_token_shape: tuple[int, int]) -> torch.Tensor:
    if pixel_mask.ndim != 2:
        raise ValueError("pixel_mask must be HxW")
    pooled = torch.nn.functional.interpolate(pixel_mask.float()[None, None], size=image_token_shape, mode="area")[0, 0]
    return pooled.flatten() > 0.5
