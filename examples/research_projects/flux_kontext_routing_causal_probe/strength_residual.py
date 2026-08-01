"""Strength- and time-conditioned target-token residual adapters for FLUX-Kontext."""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from oracle_residual import adapter_key, resolve_block


def _as_batch_scalar(value: float | torch.Tensor, batch_size: int, *, device: torch.device, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=torch.float32).flatten()
    if tensor.numel() == 1:
        tensor = tensor.expand(batch_size)
    if tensor.numel() != batch_size:
        raise ValueError(f"{name} has {tensor.numel()} values for batch size {batch_size}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return tensor


class TargetStrengthResidual(nn.Module):
    """Zero-initialized low-rank target residual with a non-linear bottleneck."""

    def __init__(self, hidden_size: int, rank: int, alpha: float | None = None) -> None:
        super().__init__()
        if not 1 <= rank <= hidden_size:
            raise ValueError(f"rank must be in [1, {hidden_size}], got {rank}")
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.alpha = float(rank if alpha is None else alpha)
        self.down = nn.Parameter(torch.empty(rank, hidden_size, dtype=torch.float32))
        self.up = nn.Parameter(torch.zeros(hidden_size, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))

    def forward(self, target_hidden: torch.Tensor) -> torch.Tensor:
        normalized = F.rms_norm(target_hidden.float(), (self.hidden_size,))
        low_rank = F.linear(normalized, self.down)
        correction = F.linear(F.silu(low_rank), self.up) * (self.alpha / self.rank)
        return correction.to(dtype=target_hidden.dtype)

    def parameter_norms(self) -> dict[str, float]:
        return {
            "down_l2": float(torch.linalg.vector_norm(self.down.detach().float()).item()),
            "up_l2": float(torch.linalg.vector_norm(self.up.detach().float()).item()),
            "combined_l2": float((self.down.detach().float().square().sum() + self.up.detach().float().square().sum()).sqrt().item()),
        }


class TimeStrengthGate(nn.Module):
    """One bounded scalar gate per controlled block."""

    def __init__(self, num_layers: int, hidden_dim: int = 32) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        self.in_proj = nn.Linear(3, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, num_layers)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, sigma: float | torch.Tensor, strength: float | torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        sigma_batch = _as_batch_scalar(sigma, batch_size, device=device, name="sigma")
        strength_batch = _as_batch_scalar(strength, batch_size, device=device, name="strength")
        if torch.any(strength_batch < 0) or torch.any(strength_batch > 1):
            raise ValueError("strength must be in [0, 1]")
        features = torch.stack([sigma_batch, strength_batch, sigma_batch * strength_batch], dim=-1)
        raw = self.out_proj(F.silu(self.in_proj(features)))
        return 1.0 + torch.tanh(raw)


@dataclass(frozen=True)
class StrengthContext:
    strength: torch.Tensor
    sigma: torch.Tensor
    spatial_weight: torch.Tensor | None
    active_call_indices: tuple[int, ...] | None


@dataclass
class StrengthResidualMetric:
    layer_id: str
    regularizer: torch.Tensor
    residual_rms: torch.Tensor
    hidden_rms: torch.Tensor
    residual_relative_hidden_rms: torch.Tensor
    gate_mean: torch.Tensor
    gate_variance: torch.Tensor
    gate_max: torch.Tensor

    def detached_dict(self) -> dict[str, float | str]:
        return {
            "layer_id": self.layer_id,
            "regularizer": float(self.regularizer.detach().item()),
            "residual_rms": float(self.residual_rms.detach().item()),
            "hidden_rms": float(self.hidden_rms.detach().item()),
            "residual_relative_hidden_rms": float(self.residual_relative_hidden_rms.detach().item()),
            "gate_mean": float(self.gate_mean.detach().item()),
            "gate_variance": float(self.gate_variance.detach().item()),
            "gate_max": float(self.gate_max.detach().item()),
        }


class StrengthResidualIntervention:
    """Post-block target-only residual hooks with an explicit strength endpoint."""

    def __init__(
        self,
        transformer: Any,
        layer_ids: list[str],
        *,
        target_tokens: int,
        hidden_size: int,
        rank: int,
        alpha: float | None = None,
        gate_hidden_dim: int = 32,
    ) -> None:
        if not layer_ids:
            raise ValueError("at least one layer is required")
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError("layer ids must be unique")
        if target_tokens < 1:
            raise ValueError("target_tokens must be positive")
        self.transformer = transformer
        self.layer_ids = tuple(layer_ids)
        self.target_tokens = int(target_tokens)
        self.adapters = nn.ModuleDict({adapter_key(layer_id): TargetStrengthResidual(hidden_size, rank, alpha) for layer_id in self.layer_ids})
        self.gate = TimeStrengthGate(len(layer_ids), gate_hidden_dim)
        self.context: StrengthContext | None = None
        self.call_index = 0
        self.collect_metrics = False
        self.metrics: list[StrengthResidualMetric] = []
        self.activation_records: list[dict[str, Any]] = []
        self._current_active = False
        self._current_gates: torch.Tensor | None = None
        self._pre_target: dict[str, torch.Tensor] = {}
        self._handles: list[Any] = []

    def parameters(self) -> Iterator[nn.Parameter]:
        yield from self.adapters.parameters()
        yield from self.gate.parameters()

    def adapter(self, layer_id: str) -> TargetStrengthResidual:
        return self.adapters[adapter_key(layer_id)]

    def set_context(
        self,
        *,
        strength: float | torch.Tensor,
        sigma: float | torch.Tensor,
        spatial_weight: torch.Tensor | None = None,
        active_call_indices: Iterable[int] | None = None,
    ) -> None:
        strength_tensor = torch.as_tensor(strength, dtype=torch.float32).flatten()
        if strength_tensor.numel() < 1 or not torch.isfinite(strength_tensor).all():
            raise ValueError("strength must be finite")
        if torch.any(strength_tensor < 0) or torch.any(strength_tensor > 1):
            raise ValueError("strength must be in [0, 1]")
        sigma_tensor = torch.as_tensor(sigma, dtype=torch.float32).flatten()
        if sigma_tensor.numel() not in {1, strength_tensor.numel()}:
            raise ValueError("sigma must be scalar or have the same batch size as strength")
        active = None
        if active_call_indices is not None:
            active = tuple(int(index) for index in active_call_indices)
            if len(set(active)) != len(active) or any(index < 0 for index in active):
                raise ValueError("active_call_indices must be unique non-negative integers")
        self.context = StrengthContext(strength_tensor, sigma_tensor, None if spatial_weight is None else spatial_weight.detach(), active)

    def reset_sequence(self) -> None:
        self.call_index = 0
        self.activation_records.clear()
        self._current_active = False
        self._current_gates = None

    def reset_metrics(self) -> None:
        self.metrics.clear()
        self._pre_target.clear()

    def metric_regularizer(self) -> torch.Tensor:
        if not self.metrics:
            return next(self.parameters()).new_zeros((), dtype=torch.float32)
        return torch.stack([metric.regularizer for metric in self.metrics]).mean()

    def detached_metrics(self) -> list[dict[str, float | str]]:
        return [metric.detached_dict() for metric in self.metrics]

    def parameter_norms(self) -> dict[str, dict[str, float]]:
        result = {layer_id: self.adapter(layer_id).parameter_norms() for layer_id in self.layer_ids}
        result["_gate"] = {
            "in_proj_l2": float(torch.linalg.vector_norm(self.gate.in_proj.weight.detach().float()).item()),
            "out_proj_l2": float(torch.linalg.vector_norm(self.gate.out_proj.weight.detach().float()).item()),
        }
        return result

    def gradient_norms(self) -> dict[str, float]:
        values: dict[str, float] = {}
        for name, parameter in self.named_parameters():
            if parameter.grad is not None:
                values[name] = float(torch.linalg.vector_norm(parameter.grad.detach().float()).item())
        return values

    def named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        for name, parameter in self.adapters.named_parameters():
            yield f"adapters.{name}", parameter
        for name, parameter in self.gate.named_parameters():
            yield f"gate.{name}", parameter

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Portable state without registering the frozen transformer as a child."""
        return {name: parameter.detach().clone() for name, parameter in self.named_parameters()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        expected = dict(self.named_parameters())
        if set(state) != set(expected):
            raise ValueError(f"checkpoint parameter mismatch: missing={set(expected) - set(state)}, extra={set(state) - set(expected)}")
        for name, parameter in expected.items():
            parameter.data.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))
    def _resolved_context(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, bool]:
        if self.context is None:
            raise RuntimeError("strength residual context was not set")
        strength = _as_batch_scalar(self.context.strength, batch_size, device=device, name="strength")
        sigma = _as_batch_scalar(self.context.sigma, batch_size, device=device, name="sigma")
        allowed = self.context.active_call_indices is None or self.call_index in self.context.active_call_indices
        spatial = self.context.spatial_weight
        if spatial is not None:
            spatial = spatial.to(device=device, dtype=torch.float32)
            if spatial.ndim == 2:
                spatial = spatial.unsqueeze(-1)
            if spatial.ndim != 3 or spatial.shape[1] != self.target_tokens or spatial.shape[-1] != 1:
                raise ValueError(f"spatial_weight must have shape [B,{self.target_tokens}] or [B,{self.target_tokens},1], got {tuple(spatial.shape)}")
            if spatial.shape[0] not in {1, batch_size}:
                raise ValueError("spatial_weight batch size mismatch")
            if spatial.shape[0] == 1 and batch_size > 1:
                spatial = spatial.expand(batch_size, -1, -1)
            if not torch.isfinite(spatial).all():
                raise ValueError("spatial_weight contains NaN or Inf")
        return strength, sigma, spatial, allowed

    def _pre_hook(self, layer_id: str):
        def hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            if hidden_states is None:
                raise RuntimeError(f"{layer_id}: missing image hidden states")
            if hidden_states.shape[1] < self.target_tokens:
                raise RuntimeError(f"{layer_id}: image sequence shorter than target token count")
            if layer_id == self.layer_ids[0]:
                strength, sigma, _spatial, allowed = self._resolved_context(hidden_states.shape[0], hidden_states.device)
                self._current_active = bool(allowed and not torch.all(strength == 1.0))
                self._current_gates = self.gate(sigma, strength, hidden_states.shape[0], hidden_states.device)
                self.activation_records.append({"call_index": self.call_index, "active": self._current_active, "strength_min": float(strength.min().item()), "strength_max": float(strength.max().item())})
            if self.collect_metrics and self._current_active:
                self._pre_target[layer_id] = hidden_states[:, : self.target_tokens]

        return hook

    def _post_hook(self, layer_id: str):
        def hook(_module: Any, _args: tuple[Any, ...], _kwargs: dict[str, Any], output: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            if not self._current_active:
                if layer_id == self.layer_ids[-1]:
                    self.call_index += 1
                return output
            if self._current_gates is None:
                raise RuntimeError("strength gates were not computed")
            text_states, image_states = output
            if image_states.shape[1] < self.target_tokens:
                raise RuntimeError(f"{layer_id}: output image sequence shorter than target tokens")
            strength, _sigma, spatial, _allowed = self._resolved_context(image_states.shape[0], image_states.device)
            target = image_states[:, : self.target_tokens]
            layer_index = self.layer_ids.index(layer_id)
            gate_value = self._current_gates[:, layer_index].to(device=target.device, dtype=target.dtype).view(-1, 1, 1)
            correction = self.adapter(layer_id)(target)
            correction = correction * (1.0 - strength).to(target.dtype).view(-1, 1, 1) * gate_value
            if spatial is not None:
                correction = correction * spatial.to(dtype=target.dtype)
            modified_image = image_states.clone()
            modified_image[:, : self.target_tokens] = target + correction
            if self.collect_metrics:
                before = self._pre_target.pop(layer_id, None)
                if before is None:
                    raise RuntimeError(f"{layer_id}: missing pre-hook metric state")
                correction_mse = correction.float().square().mean()
                hidden_mse = target.float().square().mean()
                self.metrics.append(
                    StrengthResidualMetric(
                        layer_id=layer_id,
                        regularizer=correction_mse / hidden_mse.detach().clamp_min(1e-12),
                        residual_rms=correction_mse.detach().sqrt(),
                        hidden_rms=hidden_mse.detach().sqrt(),
                        residual_relative_hidden_rms=correction_mse.detach().sqrt() / (hidden_mse.detach().sqrt() + 1e-12),
                        gate_mean=gate_value.detach().float().mean(),
                        gate_variance=gate_value.detach().float().var(unbiased=False),
                        gate_max=gate_value.detach().float().amax(),
                    )
                )
            if layer_id == self.layer_ids[-1]:
                self.call_index += 1
            return text_states, modified_image

        return hook

    def attach(self) -> "StrengthResidualIntervention":
        if self._handles:
            raise RuntimeError("strength residual intervention is already attached")
        for layer_id in self.layer_ids:
            block = resolve_block(self.transformer, layer_id)
            self._handles.append(block.register_forward_pre_hook(self._pre_hook(layer_id), with_kwargs=True))
            self._handles.append(block.register_forward_hook(self._post_hook(layer_id), with_kwargs=True))
        return self

    def remove(self) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        self._pre_target.clear()

    @contextlib.contextmanager
    def applied(self) -> Iterator["StrengthResidualIntervention"]:
        self.attach()
        try:
            yield self
        finally:
            self.remove()

