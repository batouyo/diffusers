"""Target-token, per-sample low-rank Oracle residuals for FLUX-Kontext."""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


def resolve_block(transformer: Any, layer_id: str) -> Any:
    stream, separator, raw_index = layer_id.partition(".")
    if separator != "." or not raw_index.isdigit():
        raise ValueError(f"invalid layer id {layer_id!r}")
    index = int(raw_index)
    if stream == "dual":
        blocks = transformer.transformer_blocks
    elif stream == "single":
        blocks = transformer.single_transformer_blocks
    else:
        raise ValueError(f"unsupported stream {stream!r}")
    if not 0 <= index < len(blocks):
        raise IndexError(f"layer {layer_id} is out of range")
    return blocks[index]


def adapter_key(layer_id: str) -> str:
    return layer_id.replace(".", "_")


class TargetLowRankResidual(nn.Module):
    """LoRA-shaped residual with exactly zero initial output.

    The down factor is randomized while the up factor is zero. This preserves a
    zero initial correction without the dead-gradient problem caused by
    initializing both factors to zero.
    """

    def __init__(self, hidden_size: int, rank: int, alpha: float | None = None) -> None:
        super().__init__()
        if rank < 1 or rank > hidden_size:
            raise ValueError(f"rank must be in [1, {hidden_size}], got {rank}")
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.alpha = float(rank if alpha is None else alpha)
        self.down = nn.Parameter(torch.empty(rank, hidden_size, dtype=torch.float32))
        self.up = nn.Parameter(torch.zeros(hidden_size, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        source_dtype = hidden_states.dtype
        normalized = F.rms_norm(hidden_states.float(), (self.hidden_size,))
        low_rank = F.linear(normalized, self.down)
        correction = F.linear(low_rank, self.up) * (self.alpha / self.rank)
        return correction.to(dtype=source_dtype)

    def parameter_norms(self) -> dict[str, float]:
        return {
            "down_l2": float(torch.linalg.vector_norm(self.down.detach().float()).item()),
            "up_l2": float(torch.linalg.vector_norm(self.up.detach().float()).item()),
            "combined_l2": float(
                (self.down.detach().float().square().sum() + self.up.detach().float().square().sum())
                .sqrt()
                .item()
            ),
        }


@dataclass
class ResidualMetric:
    layer_id: str
    regularizer: torch.Tensor
    correction_relative_hidden_rms: torch.Tensor
    correction_relative_update_rms: torch.Tensor
    correction_rms: torch.Tensor
    hidden_rms: torch.Tensor
    update_rms: torch.Tensor

    def detached_dict(self) -> dict[str, float | str]:
        return {
            "layer_id": self.layer_id,
            "regularizer": float(self.regularizer.detach().item()),
            "correction_relative_hidden_rms": float(self.correction_relative_hidden_rms.detach().item()),
            "correction_relative_update_rms": float(self.correction_relative_update_rms.detach().item()),
            "correction_rms": float(self.correction_rms.detach().item()),
            "hidden_rms": float(self.hidden_rms.detach().item()),
            "update_rms": float(self.update_rms.detach().item()),
        }


class TargetResidualIntervention:
    """Attach target-only residuals after selected FLUX blocks."""

    def __init__(
        self,
        transformer: Any,
        layer_ids: list[str],
        *,
        target_tokens: int,
        hidden_size: int,
        rank: int,
        alpha: float | None = None,
    ) -> None:
        if not layer_ids:
            raise ValueError("at least one layer is required")
        if target_tokens < 1:
            raise ValueError("target_tokens must be positive")
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError("layer ids must be unique")
        self.transformer = transformer
        self.layer_ids = tuple(layer_ids)
        self.target_tokens = int(target_tokens)
        self.adapters = nn.ModuleDict(
            {
                adapter_key(layer_id): TargetLowRankResidual(hidden_size, rank, alpha)
                for layer_id in self.layer_ids
            }
        )
        self.scale = 1.0
        # ``None`` means every Transformer call is eligible, which is the
        # training behaviour used by the original Oracle.  Rollout-only
        # experiments may restrict the residual to selected denoising calls.
        self.active_call_indices: tuple[int, ...] | None = None
        self.call_index = 0
        self.activation_records: list[dict[str, Any]] = []
        self._current_call_index: int | None = None
        self._current_active = False
        self.collect_metrics = False
        self.metrics: list[ResidualMetric] = []
        self._pre_target: dict[str, torch.Tensor] = {}
        self._handles: list[Any] = []

    def parameters(self) -> Iterator[nn.Parameter]:
        return self.adapters.parameters()

    def adapter(self, layer_id: str) -> TargetLowRankResidual:
        return self.adapters[adapter_key(layer_id)]

    def set_scale(self, scale: float) -> None:
        if scale < 0:
            raise ValueError("residual scale must be non-negative")
        self.scale = float(scale)

    def set_active_call_indices(self, indices: Iterable[int] | None) -> None:
        """Restrict residual application to Transformer-call indices.

        A value of ``None`` keeps the historical behaviour (all calls are
        eligible).  The sequence counter is deliberately reset separately so
        callers can configure a gate before each complete pipeline rollout.
        """
        if indices is None:
            self.active_call_indices = None
            return
        values = tuple(int(value) for value in indices)
        if any(value < 0 for value in values) or len(set(values)) != len(values):
            raise ValueError("active call indices must be unique non-negative integers")
        self.active_call_indices = values

    def reset_sequence(self) -> None:
        self.call_index = 0
        self.activation_records.clear()
        self._current_call_index = None
        self._current_active = False

    def activation_summary(self) -> dict[str, Any]:
        active = [row["call_index"] for row in self.activation_records if row["active"]]
        skipped = [row["call_index"] for row in self.activation_records if not row["active"]]
        return {
            "active_call_indices": None if self.active_call_indices is None else list(self.active_call_indices),
            "transformer_calls": len(self.activation_records),
            "active_calls": active,
            "skipped_calls": skipped,
        }

    def reset_metrics(self) -> None:
        self.metrics.clear()
        self._pre_target.clear()

    def metric_regularizer(self) -> torch.Tensor:
        if not self.metrics:
            return next(self.parameters()).new_zeros((), dtype=torch.float32)
        # Do not form this as square(sqrt(mean(correction**2))). At exactly
        # zero correction, autograd encounters 0 * inf through sqrt'(0) and
        # produces NaN even though the simplified expression is well-defined.
        return torch.stack([item.regularizer for item in self.metrics]).mean()

    def detached_metrics(self) -> list[dict[str, float | str]]:
        return [metric.detached_dict() for metric in self.metrics]

    def parameter_norms(self) -> dict[str, dict[str, float]]:
        return {layer_id: self.adapter(layer_id).parameter_norms() for layer_id in self.layer_ids}

    def _pre_hook(self, layer_id: str):
        def hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            if layer_id == self.layer_ids[0]:
                current = self.call_index
                allowed = self.active_call_indices is None or current in self.active_call_indices
                self._current_call_index = current
                self._current_active = self.scale != 0.0 and allowed
                self.activation_records.append(
                    {
                        "call_index": current,
                        "active": self._current_active,
                        "scale": self.scale if self._current_active else 0.0,
                    }
                )
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            if hidden_states is None:
                raise RuntimeError(f"{layer_id}: missing image hidden states")
            if hidden_states.shape[1] < self.target_tokens:
                raise RuntimeError(f"{layer_id}: image sequence shorter than target token count")
            if self.collect_metrics and self._current_active:
                self._pre_target[layer_id] = hidden_states[:, : self.target_tokens]

        return hook

    def _post_hook(self, layer_id: str):
        def hook(
            _module: Any,
            _args: tuple[Any, ...],
            _kwargs: dict[str, Any],
            output: tuple[torch.Tensor, torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if self._current_call_index is None:
                raise RuntimeError(f"{layer_id}: missing Transformer-call activation state")
            if not self._current_active:
                if layer_id == self.layer_ids[-1]:
                    self.call_index += 1
                return output
            text_states, image_states = output
            target = image_states[:, : self.target_tokens]
            correction = self.adapter(layer_id)(target) * self.scale
            modified_image = image_states.clone()
            modified_image[:, : self.target_tokens] = target + correction
            if self.collect_metrics:
                before = self._pre_target.pop(layer_id, None)
                if before is None:
                    raise RuntimeError(f"{layer_id}: missing pre-hook metric state")
                correction_mean_square = correction.float().square().mean()
                hidden_mean_square = target.float().square().mean()
                update_mean_square = (target.float() - before.float()).square().mean()
                regularizer = correction_mean_square / hidden_mean_square.detach().clamp_min(1e-12)
                # These values are reporting-only. Detaching them prevents
                # metrics from retaining a second, unused autograd graph.
                correction_rms = correction_mean_square.detach().sqrt()
                hidden_rms = hidden_mean_square.detach().sqrt()
                update_rms = update_mean_square.detach().sqrt()
                self.metrics.append(
                    ResidualMetric(
                        layer_id=layer_id,
                        regularizer=regularizer,
                        correction_relative_hidden_rms=correction_rms / (hidden_rms + 1e-12),
                        correction_relative_update_rms=correction_rms / (update_rms + 1e-12),
                        correction_rms=correction_rms,
                        hidden_rms=hidden_rms,
                        update_rms=update_rms,
                    )
                )
            if layer_id == self.layer_ids[-1]:
                self.call_index += 1
            return text_states, modified_image

        return hook

    def attach(self) -> "TargetResidualIntervention":
        if self._handles:
            raise RuntimeError("residual intervention is already attached")
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
    def applied(self) -> Iterator["TargetResidualIntervention"]:
        self.attach()
        try:
            yield self
        finally:
            self.remove()


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)
