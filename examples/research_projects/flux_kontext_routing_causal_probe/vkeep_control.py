"""Training-free v_keep intervention for the native FLUX-Kontext scheduler."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from routing_attention import RoutingLayout


@dataclass(frozen=True)
class VKeepCondition:
    key: str
    intervention_steps: int
    keep_weight: float


def condition_key(intervention_steps: int, keep_weight: float) -> str:
    weight = f"{float(keep_weight):.2f}".replace(".", "p")
    return f"first_{int(intervention_steps):02d}__keep_{weight}"


def build_conditions(
    intervention_steps: list[int],
    keep_weights: list[float],
) -> list[VKeepCondition]:
    conditions = [
        VKeepCondition(
            key=condition_key(steps, weight),
            intervention_steps=int(steps),
            keep_weight=float(weight),
        )
        for steps in intervention_steps
        for weight in keep_weights
    ]
    identities = {(row.intervention_steps, row.keep_weight) for row in conditions}
    if len(identities) != len(conditions):
        raise ValueError("v_keep condition grid contains duplicate conditions")
    for row in conditions:
        if row.intervention_steps < 0:
            raise ValueError("intervention_steps must be non-negative")
        if not 0.0 <= row.keep_weight <= 1.0:
            raise ValueError("keep_weight must be in [0, 1]")
    return conditions


def _assert_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value.float()).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")


def compute_v_keep(
    current_latent: torch.Tensor,
    source_latent: torch.Tensor,
    sigma: float,
    *,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    if current_latent.shape != source_latent.shape:
        raise ValueError(
            f"target/source packed latent shapes differ: {tuple(current_latent.shape)} "
            f"vs {tuple(source_latent.shape)}"
        )
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma <= epsilon:
        raise ValueError(f"sigma must be finite and greater than {epsilon}, found {sigma}")
    _assert_finite("current_latent", current_latent)
    _assert_finite("source_latent", source_latent)
    result = (current_latent.float() - source_latent.float()) / sigma
    _assert_finite("v_keep", result)
    return result


def blend_velocity(
    model_velocity: torch.Tensor,
    keep_velocity: torch.Tensor,
    keep_weight: float,
) -> torch.Tensor:
    weight = float(keep_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"keep_weight must be in [0, 1], found {weight}")
    if model_velocity.shape != keep_velocity.shape:
        raise ValueError("model and keep velocity shapes differ")
    _assert_finite("model_velocity", model_velocity)
    _assert_finite("keep_velocity", keep_velocity)
    if weight == 0.0:
        return model_velocity
    if weight == 1.0:
        return keep_velocity.to(model_velocity.dtype)
    result = (
        (1.0 - weight) * model_velocity.float()
        + weight * keep_velocity.float()
    ).to(model_velocity.dtype)
    _assert_finite("blended_velocity", result)
    return result


def euler_update(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigma: float,
    sigma_next: float,
) -> torch.Tensor:
    return sample.float() + (float(sigma_next) - float(sigma)) * velocity.float()


def relative_rms(first: torch.Tensor, second: torch.Tensor, epsilon: float = 1e-12) -> float:
    first_fp32 = first.float()
    second_fp32 = second.float()
    return float(
        (first_fp32 - second_fp32).square().mean().sqrt().item()
        / (second_fp32.square().mean().sqrt().item() + epsilon)
    )


def rms(value: torch.Tensor) -> float:
    return float(value.float().square().mean().sqrt().item())


class VKeepSchedulerController:
    """Capture the packed source latent and intervene at scheduler.step."""

    def __init__(
        self,
        pipeline: Any,
        *,
        intervention_steps: int,
        keep_weight: float,
        timestep_tolerance: float = 2e-3,
        scheduler_tolerance: float = 5e-3,
    ) -> None:
        if intervention_steps < 0:
            raise ValueError("intervention_steps must be non-negative")
        if not 0.0 <= float(keep_weight) <= 1.0:
            raise ValueError("keep_weight must be in [0, 1]")
        self.pipeline = pipeline
        self.intervention_steps = int(intervention_steps)
        self.keep_weight = float(keep_weight)
        self.timestep_tolerance = float(timestep_tolerance)
        self.scheduler_tolerance = float(scheduler_tolerance)
        self.source_latent: torch.Tensor | None = None
        self.target_tokens: int | None = None
        self.records: list[dict[str, Any]] = []
        self.call_index = 0
        self._hook: Any = None
        self._original_step: Any = None
        self._step_was_instance_attribute = False
        self._previous_instance_step: Any = None
        self.restored = False

    def _capture_source(
        self,
        _module: Any,
        _args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        layout = RoutingLayout.from_runtime(
            kwargs["encoder_hidden_states"],
            kwargs["hidden_states"],
            kwargs["img_ids"],
        )
        if layout.source_tokens != layout.target_tokens:
            raise ValueError(
                "v_keep requires matching source/target packed grids, found "
                f"target={layout.target_tokens} source={layout.source_tokens}"
            )
        hidden = kwargs["hidden_states"]
        source = hidden[:, layout.target_tokens :].detach()
        if source.shape[1] != layout.target_tokens:
            raise ValueError("captured source latent does not match target token count")
        if self.source_latent is None:
            self.source_latent = source.clone()
            self.target_tokens = layout.target_tokens
        elif self.source_latent.shape != source.shape:
            raise ValueError("source latent shape changed during rollout")

    def _sigma_values(self, timestep: Any) -> tuple[float, float]:
        scheduler = self.pipeline.scheduler
        scheduler_index = getattr(scheduler, "step_index", None)
        if scheduler_index is None and hasattr(scheduler, "_init_step_index"):
            scheduler._init_step_index(timestep)
            scheduler_index = scheduler.step_index
        sigma_index = self.call_index if scheduler_index is None else int(scheduler_index)
        if sigma_index != self.call_index:
            raise RuntimeError(
                f"scheduler/call indices diverged: scheduler={sigma_index} "
                f"controller={self.call_index}"
            )
        if sigma_index + 1 >= len(scheduler.sigmas):
            raise RuntimeError(
                f"scheduler has no sigma pair for step {self.call_index}: "
                f"len(sigmas)={len(scheduler.sigmas)}"
            )
        sigma = float(scheduler.sigmas[sigma_index].item())
        sigma_next = float(scheduler.sigmas[sigma_index + 1].item())
        timestep_value = float(
            timestep.detach().float().flatten()[0].item()
            if torch.is_tensor(timestep)
            else timestep
        )
        expected_timestep = sigma * float(scheduler.config.num_train_timesteps)
        if abs(timestep_value - expected_timestep) > self.timestep_tolerance * max(
            1.0, abs(expected_timestep)
        ):
            raise RuntimeError(
                f"scheduler timestep/sigma mismatch at step {self.call_index}: "
                f"timestep={timestep_value} sigma={sigma}"
            )
        if sigma_next >= sigma:
            raise RuntimeError(
                f"v_keep expects descending sigmas, found sigma={sigma} next={sigma_next}"
            )
        return sigma, sigma_next

    @staticmethod
    def _distance(sample: torch.Tensor, source: torch.Tensor) -> float:
        return float((sample.float() - source.float()).square().mean().sqrt().item())

    def _wrapped_step(
        self,
        model_output: torch.Tensor,
        timestep: Any,
        sample: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if self._original_step is None:
            raise RuntimeError("v_keep scheduler wrapper is not active")
        if self.source_latent is None:
            raise RuntimeError("source latent was not captured before scheduler.step")
        source = self.source_latent.to(device=sample.device)
        if source.shape != sample.shape:
            raise ValueError(
                f"scheduler target/source shapes differ: {tuple(sample.shape)} vs {tuple(source.shape)}"
            )
        sigma, sigma_next = self._sigma_values(timestep)
        active = self.call_index < self.intervention_steps and self.keep_weight > 0.0
        keep_velocity = compute_v_keep(sample, source, sigma)
        used_velocity = (
            blend_velocity(model_output, keep_velocity, self.keep_weight)
            if active
            else model_output
        )
        expected = euler_update(sample, used_velocity, sigma, sigma_next)
        native_expected = euler_update(sample, model_output, sigma, sigma_next)
        result = self._original_step(
            used_velocity,
            timestep,
            sample,
            *args,
            **kwargs,
        )
        actual = result[0]
        scheduler_error = relative_rms(actual, expected.to(actual.dtype))
        if scheduler_error > self.scheduler_tolerance:
            raise RuntimeError(
                f"native scheduler update disagrees with verified Euler update at step "
                f"{self.call_index}: relative_rms={scheduler_error:.3e}"
            )
        record = {
            "step_index": self.call_index,
            "active": active,
            "keep_weight": self.keep_weight if active else 0.0,
            "sigma": sigma,
            "sigma_next": sigma_next,
            "dt": sigma_next - sigma,
            "model_velocity_rms": rms(model_output),
            "keep_velocity_rms": rms(keep_velocity),
            "model_to_keep_mse": float(
                (model_output.float() - keep_velocity.float()).square().mean().item()
            ),
            "model_to_keep_relative_rms": relative_rms(model_output, keep_velocity),
            "used_velocity_rms": rms(used_velocity),
            "used_to_model_relative_rms": relative_rms(used_velocity, model_output),
            "distance_to_source_before": self._distance(sample, source),
            "distance_to_source_after": self._distance(actual, source),
            "native_distance_to_source_after": self._distance(native_expected, source),
            "scheduler_relative_rms_error": scheduler_error,
        }
        self.records.append(record)
        self.call_index += 1
        return result

    def __enter__(self) -> "VKeepSchedulerController":
        scheduler = self.pipeline.scheduler
        self._step_was_instance_attribute = "step" in scheduler.__dict__
        if self._step_was_instance_attribute:
            self._previous_instance_step = scheduler.__dict__["step"]
        self._original_step = scheduler.step
        scheduler.step = self._wrapped_step
        self._hook = self.pipeline.transformer.register_forward_pre_hook(
            self._capture_source,
            with_kwargs=True,
        )
        self.restored = False
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
        scheduler = self.pipeline.scheduler
        if self._step_was_instance_attribute:
            scheduler.step = self._previous_instance_step
        elif "step" in scheduler.__dict__:
            del scheduler.__dict__["step"]
        self.restored = True

    def summary(self) -> dict[str, Any]:
        active = [row for row in self.records if row["active"]]
        return {
            "intervention_steps": self.intervention_steps,
            "keep_weight": self.keep_weight,
            "captured_target_tokens": self.target_tokens,
            "scheduler_calls": len(self.records),
            "active_calls": len(active),
            "restored": self.restored,
            "records": self.records,
        }
