"""Safe, reversible text hidden-state interventions for Diffusers FLUX blocks."""

from __future__ import annotations

import inspect
import logging
import types
from dataclasses import dataclass
from typing import Literal

import torch

Mode = Literal["enhance_text", "disable_text", "remove_block"]
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BlockAddress:
    global_index: int
    local_index: int
    block_type: Literal["double", "single"]


def block_count(transformer: torch.nn.Module) -> int:
    return len(transformer.transformer_blocks) + len(transformer.single_transformer_blocks)


def resolve_block(transformer: torch.nn.Module, global_index: int) -> tuple[BlockAddress, torch.nn.Module]:
    n_double = len(transformer.transformer_blocks)
    n_single = len(transformer.single_transformer_blocks)
    if not 0 <= global_index < n_double + n_single:
        raise IndexError(f"global block index {global_index} outside [0, {n_double + n_single})")
    if global_index < n_double:
        return BlockAddress(global_index, global_index, "double"), transformer.transformer_blocks[global_index]
    local = global_index - n_double
    return BlockAddress(global_index, local, "single"), transformer.single_transformer_blocks[local]


def _argument(signature: inspect.Signature, args: tuple, kwargs: dict, name: str):
    bound = signature.bind_partial(*args, **kwargs)
    if name not in bound.arguments:
        raise RuntimeError(f"Block forward does not expose required argument {name!r}: {signature}")
    return bound.arguments[name]


class TextBlockIntervention:
    """Modify only ``encoder_hidden_states`` at one FLUX block input.

    The runtime Block contract is validated before registration. Enhance and disable use a
    kwargs-aware forward pre-hook. Remove patches only the selected module instance's forward
    and restores the exact prior instance state on exit, including exceptional exits.
    """

    _ACTIVE_ATTR = "_text_block_interventions_active"

    def __init__(
        self,
        transformer: torch.nn.Module,
        global_block_index: int,
        mode: Mode,
        alpha: float = 1.5,
        token_mask: torch.Tensor | None = None,
        *,
        allow_multi: bool = False,
    ) -> None:
        if mode not in {"enhance_text", "disable_text", "remove_block"}:
            raise ValueError(f"unsupported mode: {mode}")
        if mode == "enhance_text" and alpha <= 0:
            raise ValueError("alpha must be positive")
        self.transformer = transformer
        self.address, self.block = resolve_block(transformer, global_block_index)
        self.mode = mode
        self.alpha = float(alpha)
        self.token_mask = token_mask
        self.allow_multi = allow_multi
        self.call_count = 0
        self.last_input_shape: tuple[int, ...] | None = None
        self._handle = None
        self._signature = inspect.signature(self.block.forward)
        self._had_instance_forward = "forward" in self.block.__dict__
        self._instance_forward = self.block.__dict__.get("forward")
        self._entered = False

        required = {"hidden_states", "encoder_hidden_states"}
        if not required.issubset(self._signature.parameters):
            raise RuntimeError(
                f"Unsupported {self.address.block_type} block signature at global "
                f"{self.address.global_index}: {self._signature}"
            )

    def _pre_hook(self, module, args, kwargs):
        del module
        kwargs = dict(kwargs)
        bound = self._signature.bind_partial(*args, **kwargs)
        if "encoder_hidden_states" not in bound.arguments:
            raise RuntimeError(f"encoder_hidden_states missing at block {self.address.global_index}")
        encoder = bound.arguments["encoder_hidden_states"]
        self.call_count += 1
        self.last_input_shape = tuple(encoder.shape)

        if self.mode == "enhance_text":
            if self.alpha == 1.0:
                modified = encoder
            elif self.token_mask is None:
                modified = encoder * self.alpha
            else:
                mask = self.token_mask.to(device=encoder.device, dtype=encoder.dtype)
                while mask.ndim < encoder.ndim:
                    mask = mask.unsqueeze(-1)
                try:
                    torch.broadcast_shapes(mask.shape, encoder.shape)
                except RuntimeError as exc:
                    raise ValueError(
                        f"token_mask {tuple(mask.shape)} is not broadcastable to {tuple(encoder.shape)}"
                    ) from exc
                modified = encoder * (1 + (self.alpha - 1) * mask)
        elif self.mode == "disable_text":
            modified = torch.zeros_like(encoder)
        else:  # pragma: no cover - remove_block uses a forward replacement
            raise AssertionError("remove_block must not install a pre-hook")

        if "encoder_hidden_states" in kwargs:
            kwargs["encoder_hidden_states"] = modified
            return args, kwargs

        positional_names = list(self._signature.parameters)
        index = positional_names.index("encoder_hidden_states")
        mutable_args = list(args)
        if index >= len(mutable_args):
            raise RuntimeError("encoder_hidden_states could not be replaced safely")
        mutable_args[index] = modified
        return tuple(mutable_args), kwargs

    def _install_remove(self) -> None:
        signature = self._signature
        owner = self

        def skip_forward(module_self, *args, **kwargs):
            del module_self
            encoder = _argument(signature, args, kwargs, "encoder_hidden_states")
            hidden = _argument(signature, args, kwargs, "hidden_states")
            owner.call_count += 1
            owner.last_input_shape = tuple(encoder.shape)
            return encoder, hidden

        self.block.forward = types.MethodType(skip_forward, self.block)

    def __enter__(self) -> "TextBlockIntervention":
        if self._entered:
            raise RuntimeError("intervention context cannot be entered twice")
        active = getattr(self.transformer, self._ACTIVE_ATTR, None)
        if active is None:
            active = {}
            setattr(self.transformer, self._ACTIVE_ATTR, active)
        if active and not self.allow_multi:
            raise RuntimeError(f"another intervention is active: {sorted(active)}")
        if self.address.global_index in active:
            raise RuntimeError(f"block {self.address.global_index} already has an active intervention")
        active[self.address.global_index] = self
        self._entered = True
        try:
            if self.mode == "remove_block":
                self._install_remove()
            else:
                self._handle = self.block.register_forward_pre_hook(self._pre_hook, with_kwargs=True)
        except BaseException:
            active.pop(self.address.global_index, None)
            self._entered = False
            raise
        LOGGER.info(
            "installed %s intervention: type=%s local=%d global=%d alpha=%s",
            self.mode,
            self.address.block_type,
            self.address.local_index,
            self.address.global_index,
            self.alpha,
        )
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        try:
            if self._handle is not None:
                self._handle.remove()
                self._handle = None
            if self.mode == "remove_block":
                if self._had_instance_forward:
                    self.block.forward = self._instance_forward
                elif "forward" in self.block.__dict__:
                    delattr(self.block, "forward")
        finally:
            active = getattr(self.transformer, self._ACTIVE_ATTR, {})
            active.pop(self.address.global_index, None)
            if not active and hasattr(self.transformer, self._ACTIVE_ATTR):
                delattr(self.transformer, self._ACTIVE_ATTR)
            self._entered = False
        return False


def assert_no_active_interventions(transformer: torch.nn.Module) -> None:
    active = getattr(transformer, TextBlockIntervention._ACTIVE_ATTR, {})
    if active:
        raise RuntimeError(f"residual interventions detected: {sorted(active)}")
