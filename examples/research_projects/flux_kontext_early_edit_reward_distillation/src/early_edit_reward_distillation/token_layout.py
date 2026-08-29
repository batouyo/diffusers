"""Explicit token layout audit; no sequence-length guessing."""
from __future__ import annotations
from typing import Any

def audit_token_layout(*, generated_tokens: int, source_tokens: int, text_tokens: int, module_shapes: dict[str, tuple[int, ...]]) -> dict[str, Any]:
    if min(generated_tokens, source_tokens, text_tokens) < 0: raise ValueError("token counts must be non-negative")
    return {"generated_image_tokens": int(generated_tokens), "source_conditioning_tokens": int(source_tokens), "text_tokens": int(text_tokens), "module_shapes": {str(k): list(v) for k, v in module_shapes.items()}, "mask_target": "generated_image_tokens_only"}

class ForwardShapeAudit:
    def __init__(self) -> None: self.shapes: dict[str, tuple[int, ...]] = {}
    def hook(self, name: str):
        def capture(_module, inputs, _output):
            tensor = inputs[0] if inputs else None
            if tensor is not None and hasattr(tensor, "shape"): self.shapes[name] = tuple(int(x) for x in tensor.shape)
        return capture
