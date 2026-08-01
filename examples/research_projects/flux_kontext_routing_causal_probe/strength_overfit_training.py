"""Losses and model-call helpers for continuous-strength velocity supervision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F

from strength_overfit_data import assert_same_contract


@dataclass(frozen=True)
class StrengthPair:
    first: float
    second: float | None


def sample_strength(generator: torch.Generator, *, device: torch.device) -> float:
    branch = float(torch.rand((), generator=generator).item())
    if branch < 0.2:
        return 0.0
    if branch < 0.4:
        return 1.0
    return float(torch.rand((), generator=generator).item())


def sample_strength_pair(generator: torch.Generator, *, device: torch.device, pair_probability: float = 0.5) -> StrengthPair:
    first = sample_strength(generator, device=device)
    if float(torch.rand((), generator=generator).item()) >= pair_probability:
        return StrengthPair(first=first, second=None)
    second = sample_strength(generator, device=device)
    if first == second:
        second = min(1.0, first + 0.5) if first < 1 else 0.5
    return StrengthPair(first=min(first, second), second=max(first, second))


def interpolated_teacher(v_edit: torch.Tensor, v_neutral: torch.Tensor, strength: float | torch.Tensor) -> torch.Tensor:
    s = torch.as_tensor(strength, device=v_edit.device, dtype=torch.float32).view(-1, 1, 1)
    if s.shape[0] == 1 and v_edit.shape[0] > 1:
        s = s.expand(v_edit.shape[0], -1, -1)
    if v_edit.shape != v_neutral.shape:
        raise ValueError("teacher velocity shapes differ")
    return (s * v_edit.float() + (1.0 - s) * v_neutral.float()).to(v_edit.dtype)


def progress_q(v_student: torch.Tensor, v_edit: torch.Tensor, v_neutral: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    direction = (v_edit.float() - v_neutral.float()).flatten(1)
    displacement = (v_student.float() - v_neutral.float()).flatten(1)
    denominator = direction.square().sum(dim=1)
    q = (displacement * direction).sum(dim=1) / denominator.clamp_min(eps)
    return q, denominator


def velocity_losses(
    *,
    v_student: torch.Tensor,
    v_edit: torch.Tensor,
    v_neutral: torch.Tensor,
    strength: float,
    second_student: torch.Tensor | None = None,
    second_strength: float | None = None,
    lambda_mono: float = 0.0,
    lambda_progress: float = 0.0,
    lambda_reg: float = 0.0,
    regularizer: torch.Tensor | None = None,
    margin: float = 0.01,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    target = interpolated_teacher(v_edit, v_neutral, strength)
    velocity = F.mse_loss(v_student.float(), target.float())
    q_first, direction_norm = progress_q(v_student, v_edit, v_neutral, eps)
    valid = direction_norm > eps
    progress = (q_first[valid] - float(strength)).abs().mean() if valid.any() else velocity.new_zeros(())
    mono = velocity.new_zeros(())
    second_velocity = velocity.new_zeros(())
    if second_student is not None and second_strength is not None:
        second_target = interpolated_teacher(v_edit, v_neutral, second_strength)
        second_velocity = F.mse_loss(second_student.float(), second_target.float())
        q_second, _ = progress_q(second_student, v_edit, v_neutral, eps)
        if valid.any():
            mono = F.relu(q_first[valid] - q_second[valid] + margin).mean()
            progress = 0.5 * (progress + (q_second[valid] - float(second_strength)).abs().mean())
    reg = velocity.new_zeros(()) if regularizer is None else regularizer.float()
    total = velocity + second_velocity + lambda_mono * mono + lambda_progress * progress + lambda_reg * reg
    return {
        "total": total,
        "velocity": velocity,
        "second_velocity": second_velocity,
        "monotonic": mono,
        "progress": progress,
        "regularizer": reg,
        "q_mean": q_first.detach().mean(),
        "direction_degenerate": (~valid).float().mean(),
    }


def loss_weights(step: int, *, warmup_steps: int = 300, ramp_steps: int = 200, lambda_mono: float = 0.05, lambda_progress: float = 0.10) -> tuple[float, float]:
    if step < warmup_steps:
        return 0.0, 0.0
    ratio = min(1.0, (step - warmup_steps + 1) / max(1, ramp_steps))
    return lambda_mono * ratio, lambda_progress * ratio


def checked_teacher_pair(
    *,
    edit_call: Callable[[], tuple[torch.Tensor, dict[str, Any]]],
    neutral_call: Callable[[], tuple[torch.Tensor, dict[str, Any]]],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    with torch.no_grad():
        v_edit, edit_contract = edit_call()
        v_neutral, neutral_contract = neutral_call()
    assert_same_contract(edit_contract, neutral_contract)
    return v_edit.detach(), v_neutral.detach(), edit_contract


def euler_step(sample: torch.Tensor, velocity: torch.Tensor, sigma: float | torch.Tensor, sigma_next: float | torch.Tensor) -> torch.Tensor:
    # Cached rollout states live on CPU while FLUX velocities are produced on CUDA.
    update_device = velocity.device
    sample_float = sample.to(device=update_device, dtype=torch.float32)
    delta = torch.as_tensor(sigma_next, device=update_device, dtype=torch.float32) - torch.as_tensor(sigma, device=update_device, dtype=torch.float32)
    return (sample_float + delta * velocity.float()).to(dtype=sample.dtype)


def finite_or_raise(named_values: dict[str, torch.Tensor]) -> None:
    for name, value in named_values.items():
        if not torch.isfinite(value.detach().float()).all():
            raise FloatingPointError(f"{name} contains NaN or Inf")

