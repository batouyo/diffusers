"""Model-agnostic temporal and token gating helpers for selective LoRA."""
import torch
def timestep_gate(timestep_index: int, teacher_indices: set[int], scale: float = 1.0) -> float: return float(scale) if timestep_index in teacher_indices else 0.0
def masked_residual(residual: torch.Tensor, token_mask: torch.Tensor, scale: float) -> torch.Tensor:
    if residual.ndim < 2 or token_mask.ndim != 2 or residual.shape[:2] != token_mask.shape: raise ValueError("residual must be BxSxD and token_mask must be BxS")
    return residual * token_mask.to(residual.dtype).unsqueeze(-1) * float(scale)
def velocity_diagnostics(predicted: torch.Tensor, teacher: torch.Tensor, token_mask: torch.Tensor) -> dict[str, float]:
    mask = token_mask.to(dtype=predicted.dtype).unsqueeze(-1); diff = (predicted - teacher) * mask; denom = mask.sum().clamp_min(1.0)
    cosine = torch.nn.functional.cosine_similarity(predicted * mask, teacher * mask, dim=-1).mean()
    return {"mse": float((diff.square().sum() / denom).item()), "cosine": float(cosine.item()), "teacher_norm": float(((teacher * mask).square().sum().sqrt()).item()), "predicted_norm": float(((predicted * mask).square().sum().sqrt()).item())}
