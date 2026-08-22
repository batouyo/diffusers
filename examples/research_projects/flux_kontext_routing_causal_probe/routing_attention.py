"""Temporary attention-routing interventions for FLUX-Kontext.

The implementation deliberately delegates the full attention call to the
installed processor first.  It then recomputes only target-query attention
with explicit FP32 logits and replaces only the target-query outputs.
"""

from __future__ import annotations

import contextlib
import gc
import math
from dataclasses import dataclass
from typing import Any, Iterator

import torch
import torch.nn.functional as F

from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_flux import _get_qkv_projections


@dataclass(frozen=True)
class RoutingLayout:
    text_tokens: int
    target_tokens: int
    source_tokens: int
    target_grid_height: int
    target_grid_width: int

    @property
    def image_tokens(self) -> int:
        return self.target_tokens + self.source_tokens

    @property
    def joint_tokens(self) -> int:
        return self.text_tokens + self.image_tokens

    @property
    def text_slice(self) -> slice:
        return slice(0, self.text_tokens)

    @property
    def target_slice(self) -> slice:
        return slice(self.text_tokens, self.text_tokens + self.target_tokens)

    @property
    def source_slice(self) -> slice:
        return slice(self.text_tokens + self.target_tokens, self.joint_tokens)

    def validate_lengths(self, image_length: int, text_length: int) -> None:
        if text_length != self.text_tokens:
            raise ValueError(f"text length {text_length} != expected {self.text_tokens}")
        if image_length != self.image_tokens:
            raise ValueError(f"image length {image_length} != expected {self.image_tokens}")
        if self.target_grid_height * self.target_grid_width != self.target_tokens:
            raise ValueError(
                "target grid does not match target token count: "
                f"{self.target_grid_height}x{self.target_grid_width} != {self.target_tokens}"
            )

    def validate_img_ids(self, img_ids: torch.Tensor) -> None:
        ids = img_ids[0] if img_ids.ndim == 3 else img_ids
        if ids.ndim != 2 or ids.shape[-1] < 3:
            raise ValueError(f"expected img_ids [tokens,3], got {tuple(ids.shape)}")
        if ids.shape[0] != self.image_tokens:
            raise ValueError(f"img_ids length {ids.shape[0]} != expected {self.image_tokens}")
        stream_ids = ids[:, 0]
        target_ids = stream_ids[: self.target_tokens]
        source_ids = stream_ids[self.target_tokens :]
        if not torch.all(target_ids == 0):
            raise ValueError("target token img_ids[...,0] are not all zero")
        if not torch.all(source_ids == 1):
            raise ValueError("source token img_ids[...,0] are not all one")

        target_xy = ids[: self.target_tokens, 1:3].to(torch.long)
        y_unique = torch.unique(target_xy[:, 0])
        x_unique = torch.unique(target_xy[:, 1])
        if len(y_unique) != self.target_grid_height or len(x_unique) != self.target_grid_width:
            raise ValueError(
                "target token grid inferred from img_ids does not match layout: "
                f"{len(y_unique)}x{len(x_unique)} versus "
                f"{self.target_grid_height}x{self.target_grid_width}"
            )

    @classmethod
    def from_runtime(
        cls,
        encoder_hidden_states: torch.Tensor,
        hidden_states: torch.Tensor,
        img_ids: torch.Tensor,
    ) -> "RoutingLayout":
        ids = img_ids[0] if img_ids.ndim == 3 else img_ids
        if ids.ndim != 2 or ids.shape[-1] < 3:
            raise ValueError(f"expected img_ids [tokens,3], got {tuple(ids.shape)}")
        if ids.shape[0] != hidden_states.shape[1]:
            raise ValueError("img_ids and image hidden-state lengths differ")
        stream_ids = ids[:, 0]
        source_positions = torch.nonzero(stream_ids == 1, as_tuple=False).flatten()
        if source_positions.numel() == 0:
            raise ValueError("no source image tokens were found")
        target_tokens = int(source_positions[0].item())
        source_tokens = int(ids.shape[0] - target_tokens)
        target_xy = ids[:target_tokens, 1:3].to(torch.long)
        layout = cls(
            text_tokens=int(encoder_hidden_states.shape[1]),
            target_tokens=target_tokens,
            source_tokens=source_tokens,
            target_grid_height=int(torch.unique(target_xy[:, 0]).numel()),
            target_grid_width=int(torch.unique(target_xy[:, 1]).numel()),
        )
        layout.validate_lengths(hidden_states.shape[1], encoder_hidden_states.shape[1])
        layout.validate_img_ids(ids)
        return layout


@dataclass
class RoutingAttentionStats:
    group_mean: dict[str, float]
    group_std: dict[str, float]
    head_mean: torch.Tensor
    head_std: torch.Tensor
    token_mean: torch.Tensor
    mass_sum_max_error: float
    effective_query_chunk_size: int
    finite: bool
    q_shape: tuple[int, ...]
    k_shape: tuple[int, ...]
    v_shape: tuple[int, ...]
    target_logits_shape: tuple[int, ...]
    native_zero_output_relative_l2: float
    controlled_output_delta_relative_l2: float
    controlled_output_delta_rms: float
    native_target_rms: float

    def summary_dict(self) -> dict[str, Any]:
        return {
            "attention_source_mean": self.group_mean["source"],
            "attention_text_mean": self.group_mean["text"],
            "attention_target_mean": self.group_mean["target"],
            "attention_source_std": self.group_std["source"],
            "attention_text_std": self.group_std["text"],
            "attention_target_std": self.group_std["target"],
            "attention_mass_sum_max_error": self.mass_sum_max_error,
            "attention_finite": self.finite,
            "effective_query_chunk_size": self.effective_query_chunk_size,
            "q_shape": list(self.q_shape),
            "k_shape": list(self.k_shape),
            "v_shape": list(self.v_shape),
            "target_logits_shape": list(self.target_logits_shape),
            "native_zero_output_relative_l2": self.native_zero_output_relative_l2,
            "controlled_output_delta_relative_l2": self.controlled_output_delta_relative_l2,
            "controlled_output_delta_rms": self.controlled_output_delta_rms,
            "native_target_rms": self.native_target_rms,
        }


class RoutingAttnProcessor:
    """Wrap a FLUX processor and replace target-query outputs only."""

    _GROUP_NAMES = ("text", "target", "source")

    def __init__(
        self,
        original_processor: Any,
        layout: RoutingLayout,
        *,
        b_source: float = 0.0,
        b_text: float = 0.0,
        b_target: float = 0.0,
        query_chunk_size: int = 64,
        active_call_indices: set[int] | None = None,
        hard_mask_groups: set[str] | frozenset[str] | None = None,
    ) -> None:
        if b_target != 0.0:
            raise ValueError("b_target is fixed to zero in phase 1")
        if query_chunk_size < 1:
            raise ValueError("query_chunk_size must be positive")
        self.original_processor = original_processor
        self.layout = layout
        self.b_source = float(b_source)
        self.b_text = float(b_text)
        self.b_target = float(b_target)
        self.query_chunk_size = int(query_chunk_size)
        self.active_call_indices = active_call_indices
        self.hard_mask_groups = frozenset(hard_mask_groups or ())
        unsupported_masks = self.hard_mask_groups.difference({"text", "source"})
        if unsupported_masks:
            raise ValueError(f"unsupported hard-mask groups: {sorted(unsupported_masks)}")
        self.call_index = 0
        self.stats: RoutingAttentionStats | None = None
        self.history: list[dict[str, Any]] = []

    def _joint_qkv(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        image_rotary_emb: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )
        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        value = value.unflatten(-1, (attn.heads, -1))

        if encoder_hidden_states is not None:
            encoder_query = attn.norm_added_q(encoder_query.unflatten(-1, (attn.heads, -1)))
            encoder_key = attn.norm_added_k(encoder_key.unflatten(-1, (attn.heads, -1)))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        return query, key, value

    def _explicit_once(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_target = query[:, self.layout.target_slice]
        scale = 1.0 / math.sqrt(query.shape[-1])
        output_chunks: list[torch.Tensor] = []
        zero_output_chunks: list[torch.Tensor] = []
        mass_chunks: list[torch.Tensor] = []
        key_fp32 = key.float()
        value_fp32 = value.float()

        for start in range(0, self.layout.target_tokens, chunk_size):
            stop = min(start + chunk_size, self.layout.target_tokens)
            base_logits = torch.einsum("bqhd,bkhd->bhqk", q_target[:, start:stop].float(), key_fp32) * scale
            if self.b_source == 0.0 and self.b_text == 0.0 and not self.hard_mask_groups:
                logits = base_logits
                zero_weights = None
            else:
                zero_weights = torch.softmax(base_logits, dim=-1)
                logits = base_logits.clone()
                logits[..., self.layout.text_slice] += self.b_text
                logits[..., self.layout.source_slice] += self.b_source
                if "text" in self.hard_mask_groups:
                    logits[..., self.layout.text_slice] = -torch.inf
                if "source" in self.hard_mask_groups:
                    logits[..., self.layout.source_slice] = -torch.inf
            weights = torch.softmax(logits, dim=-1)
            controlled = torch.einsum("bhqk,bkhd->bqhd", weights, value_fp32)
            zero_controlled = (
                controlled
                if zero_weights is None
                else torch.einsum("bhqk,bkhd->bqhd", zero_weights, value_fp32)
            )
            masses = torch.stack(
                [
                    weights[..., self.layout.text_slice].sum(dim=-1),
                    weights[..., self.layout.target_slice].sum(dim=-1),
                    weights[..., self.layout.source_slice].sum(dim=-1),
                ],
                dim=-1,
            )
            output_chunks.append(controlled)
            zero_output_chunks.append(zero_controlled)
            mass_chunks.append(masses)
        return (
            torch.cat(output_chunks, dim=1),
            torch.cat(zero_output_chunks, dim=1),
            torch.cat(mass_chunks, dim=2),
        )

    def _explicit_with_fallback(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        chunk_size = min(self.query_chunk_size, self.layout.target_tokens)
        while True:
            try:
                output, zero_output, masses = self._explicit_once(query, key, value, chunk_size)
                return output, zero_output, masses, chunk_size
            except torch.cuda.OutOfMemoryError:
                if chunk_size <= 8:
                    raise
                gc.collect()
                torch.cuda.empty_cache()
                chunk_size = max(8, chunk_size // 2)

    def _make_stats(
        self,
        masses: torch.Tensor,
        chunk_size: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> RoutingAttentionStats:
        # masses: [batch, heads, target_query, 3]
        finite = bool(torch.isfinite(masses).all().item())
        group_mean_tensor = masses.mean(dim=(0, 1, 2))
        group_std_tensor = masses.std(dim=(0, 1, 2), unbiased=False)
        head_mean = masses.mean(dim=(0, 2)).detach().float().cpu()
        head_std = masses.std(dim=(0, 2), unbiased=False).detach().float().cpu()
        token_mean = masses.mean(dim=(0, 1)).detach().float().cpu()
        mass_sum_max_error = float((masses.sum(dim=-1) - 1.0).abs().max().item())
        return RoutingAttentionStats(
            group_mean={n: float(group_mean_tensor[i].item()) for i, n in enumerate(self._GROUP_NAMES)},
            group_std={n: float(group_std_tensor[i].item()) for i, n in enumerate(self._GROUP_NAMES)},
            head_mean=head_mean,
            head_std=head_std,
            token_mean=token_mean,
            mass_sum_max_error=mass_sum_max_error,
            effective_query_chunk_size=chunk_size,
            finite=finite,
            q_shape=tuple(query.shape),
            k_shape=tuple(key.shape),
            v_shape=tuple(value.shape),
            target_logits_shape=(
                query.shape[0],
                query.shape[2],
                self.layout.target_tokens,
                key.shape[1],
            ),
            native_zero_output_relative_l2=float("nan"),
            controlled_output_delta_relative_l2=float("nan"),
            controlled_output_delta_rms=float("nan"),
            native_target_rms=float("nan"),
        )

    def _record_output_delta(
        self,
        current_call: int,
        controlled_delta: torch.Tensor,
        native_target: torch.Tensor,
    ) -> None:
        if self.stats is None:
            raise RuntimeError("attention statistics are missing")
        delta_fp32 = controlled_delta.float()
        native_fp32 = native_target.float()
        self.stats.controlled_output_delta_relative_l2 = float(
            torch.linalg.vector_norm(delta_fp32).item()
            / (torch.linalg.vector_norm(native_fp32).item() + 1e-12)
        )
        self.stats.controlled_output_delta_rms = float(delta_fp32.square().mean().sqrt().item())
        self.stats.native_target_rms = float(native_fp32.square().mean().sqrt().item())
        self.history.append(
            {
                "call_index": current_call,
                "b_source": self.b_source,
                "b_text": self.b_text,
                "hard_mask_groups": sorted(self.hard_mask_groups),
                **self.stats.summary_dict(),
            }
        )

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if attention_mask is not None:
            raise NotImplementedError("phase-1 routing probe does not support attention masks")
        if kwargs:
            raise ValueError(f"unsupported attention kwargs: {sorted(kwargs)}")

        native = self.original_processor(
            attn,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            image_rotary_emb,
        )
        current_call = self.call_index
        self.call_index += 1
        if self.active_call_indices is not None and current_call not in self.active_call_indices:
            return native
        if encoder_hidden_states is None:
            self.layout.validate_lengths(
                image_length=hidden_states.shape[1] - self.layout.text_tokens,
                text_length=self.layout.text_tokens,
            )
        else:
            self.layout.validate_lengths(hidden_states.shape[1], encoder_hidden_states.shape[1])

        query, key, value = self._joint_qkv(attn, hidden_states, encoder_hidden_states, image_rotary_emb)
        if query.shape[1] != self.layout.joint_tokens:
            raise ValueError(f"joint Q length {query.shape[1]} != expected {self.layout.joint_tokens}")
        controlled, zero_controlled, masses, chunk_size = self._explicit_with_fallback(query, key, value)
        self.stats = self._make_stats(masses, chunk_size, query, key, value)
        controlled = controlled.flatten(2, 3)
        zero_controlled = zero_controlled.flatten(2, 3)

        if encoder_hidden_states is not None:
            native_image, native_text = native
            output_projection = attn.to_out[0]
            zero_projected = output_projection(zero_controlled.to(output_projection.weight.dtype).contiguous())
            zero_projected = attn.to_out[1](zero_projected)
            native_target = native_image[:, : self.layout.target_tokens]
            calibration_delta = zero_projected.float() - native_target.float()
            self.stats.native_zero_output_relative_l2 = float(
                torch.linalg.vector_norm(calibration_delta).item()
                / (torch.linalg.vector_norm(native_target.float()).item() + 1e-12)
            )
            # Subtract in FP32 before dtype conversion. Applying the original
            # projection without its bias is exactly the projected attention
            # increment and avoids subtracting two large BF16 outputs.
            controlled_delta = (controlled - zero_controlled).to(output_projection.weight.dtype)
            projected_delta = F.linear(
                controlled_delta.contiguous(),
                output_projection.weight,
                bias=None,
            )
            projected_delta = attn.to_out[1](projected_delta)
            self._record_output_delta(current_call, projected_delta, native_target)
            image_output = native_image.clone()
            image_output[:, : self.layout.target_tokens] = native_target + projected_delta
            return image_output, native_text

        joint_output = native.clone()
        native_target = native[:, self.layout.target_slice]
        calibration_delta = zero_controlled - native_target.float()
        self.stats.native_zero_output_relative_l2 = float(
            torch.linalg.vector_norm(calibration_delta).item()
            / (torch.linalg.vector_norm(native_target.float()).item() + 1e-12)
        )
        controlled_delta = (controlled - zero_controlled).to(native_target.dtype)
        self._record_output_delta(current_call, controlled_delta, native_target)
        joint_output[:, self.layout.target_slice] = native_target + controlled_delta
        return joint_output


def resolve_layer_attention(transformer: Any, layer_id: str) -> tuple[Any, str]:
    stream, sep, index_text = layer_id.partition(".")
    if sep != "." or not index_text.isdigit():
        raise ValueError(f"invalid layer id {layer_id!r}; expected dual.00 or single.00")
    index = int(index_text)
    if stream == "dual":
        blocks = transformer.transformer_blocks
    elif stream == "single":
        blocks = transformer.single_transformer_blocks
    else:
        raise ValueError(f"unsupported layer stream {stream!r}")
    if index >= len(blocks):
        raise IndexError(f"{layer_id} out of range; stream has {len(blocks)} blocks")
    return blocks[index].attn, f"{stream}.{index:02d}"


@contextlib.contextmanager
def temporary_routing_processor(
    transformer: Any,
    layer_id: str,
    layout: RoutingLayout,
    *,
    b_source: float,
    b_text: float,
    b_target: float = 0.0,
    query_chunk_size: int = 64,
    active_call_indices: set[int] | None = None,
    hard_mask_groups: set[str] | frozenset[str] | None = None,
) -> Iterator[RoutingAttnProcessor]:
    attention, _ = resolve_layer_attention(transformer, layer_id)
    original = attention.processor
    processor = RoutingAttnProcessor(
        original,
        layout,
        b_source=b_source,
        b_text=b_text,
        b_target=b_target,
        query_chunk_size=query_chunk_size,
        active_call_indices=active_call_indices,
        hard_mask_groups=hard_mask_groups,
    )
    attention.set_processor(processor)
    try:
        yield processor
    finally:
        attention.set_processor(original)
        if attention.processor is not original:
            raise RuntimeError(f"failed to restore original processor for {layer_id}")
