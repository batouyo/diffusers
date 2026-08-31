"""Training-free coupled continuous-strength editing for FLUX-Kontext.

The neutral preservation prompt is an engineering approximation: Kontext does
not expose an oracle source-preservation velocity. The code keeps a real
preservation/edit coupled rollout, while SDE residuals are reused for every
strength without any new random sampling.
"""
from __future__ import annotations

import importlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence
import torch
import torch.nn.functional as F
import torch.nn.functional as F
from PIL import Image

from .core import coupled_noise, critical_nonzero_steps, native_euler_sde_step, noise_correlations, tensor_hash
from .trajectory import KontextState, _sigmas, prepare_state, velocity


class RewardScorer(Protocol):
    def score(self, source: Image.Image, candidate: Image.Image, instruction: str) -> float: ...


class CallableRewardScorer:
    def __init__(self, fn: Callable[..., float]):
        self.fn = fn

    def score(self, source: Image.Image, candidate: Image.Image, instruction: str) -> float:
        return float(self.fn(source=source, candidate=candidate, instruction=instruction))


class RewardUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class ContinuousStrengthConfig:
    neutral_prompt: str = "preserve the source image without any edit"
    height: int = 512
    width: int = 512
    steps: int = 28
    guidance_scale: float = 3.5
    critical_steps: int = 2
    critical_step_indices: tuple[int, ...] | None = None
    intervention_step_count: int = 4
    search_step_indices: tuple[int, ...] | None = None
    first_step_align_steps: int = 4
    similarity_threshold: float = 0.8
    similarity_mode: str = "elementwise"
    num_candidates: int = 4
    alpha: float = 0.05
    diffusion_scale: float = 1.0
    coupling_strength: float = 0.0
    mask_quantile: float = 0.75
    min_edit_ratio: float = 0.02
    max_edit_ratio: float = 0.40
    strengths: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

    def __post_init__(self):
        if self.num_candidates != 4:
            raise ValueError("the minimal prototype fixes num_candidates=4")
        if self.critical_steps < 1 or self.intervention_step_count < 0:
            raise ValueError("critical_steps must be positive")
        if self.critical_step_indices is not None and not self.critical_step_indices:
            raise ValueError("critical_step_indices cannot be empty")
        if not 0 <= self.coupling_strength <= 1:
            raise ValueError("coupling_strength must lie in [0, 1]")
        if not 0 <= self.min_edit_ratio <= self.max_edit_ratio <= 1:
            raise ValueError("invalid edit ratio bounds")
        if any(not 0 <= float(s) <= 1 for s in self.strengths):
            raise ValueError("strengths must lie in [0, 1]")


@dataclass
class TrajectoryTrace:
    prompt: str
    states: list[torch.Tensor]
    velocities: list[torch.Tensor]
    timesteps: list[float]
    sigmas: list[tuple[float, float]]
    terminal: torch.Tensor
    residuals: list[torch.Tensor] = field(default_factory=list)

    def metadata(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "num_steps": len(self.velocities),
            "state_hashes": [tensor_hash(x) for x in self.states],
            "terminal_hash": tensor_hash(self.terminal),
            "timesteps": self.timesteps,
            "sigmas": [list(x) for x in self.sigmas],
            "residual_norms": [float(x.float().norm()) for x in self.residuals],
        }


@dataclass
class TrajectoryBundle:
    preservation: TrajectoryTrace
    edited: TrajectoryTrace
    winner: TrajectoryTrace
    token_mask: torch.Tensor
    mask_scores: torch.Tensor
    branch_records: list[dict[str, Any]]
    winner_index: int
    rewards: list[float]
    metadata: dict[str, Any]
    branch_images: list[list[Image.Image]]
    preservation_state: Any = None
    edited_state: Any = None


def _critical_indices(pipe: Any, cfg: ContinuousStrengthConfig) -> list[int]:
    valid = critical_nonzero_steps(pipe.scheduler.sigmas.detach().cpu().flatten().tolist())
    valid_indices = {int(x["index"]) for x in valid}
    requested = cfg.search_step_indices if cfg.search_step_indices is not None else cfg.critical_step_indices
    indices = list(requested) if requested is not None else [int(x["index"]) for x in valid[: cfg.critical_steps]]
    if len(indices) != len(set(indices)) or any(i not in valid_indices for i in indices):
        raise ValueError("critical_step_indices must be unique non-zero scheduler transitions")
    return sorted(indices)


def _step(x, v, sigma, sigma_next):
    return (x.float() + (float(sigma_next) - float(sigma)) * v.float()).to(x.dtype)


def strength_step(x, preservation_velocity, edited_velocity, sigma, sigma_next, preservation_residual, reward_residual, strength):
    """One deterministic strength update with linearly scaled SDE residuals."""
    s = float(strength)
    if not 0.0 <= s <= 1.0:
        raise ValueError("strength must lie in [0, 1]")
    blended = preservation_velocity.float() + s * (edited_velocity.float() - preservation_velocity.float())
    residual = (1.0 - s) * preservation_residual.float() + s * reward_residual.float()
    return (_step(x, blended, sigma, sigma_next).float() + residual).to(x.dtype)


@torch.inference_mode()
def deterministic_trace(pipe: Any, state: KontextState) -> TrajectoryTrace:
    x = state.latents
    states = [x.clone()]
    velocities, timesteps, sigmas, residuals = [], [], [], []
    for timestep in state.timesteps:
        v = velocity(pipe, state, x, timestep)
        sigma, sigma_next = _sigmas(pipe, timestep, state)
        x = _step(x, v, sigma, sigma_next)
        velocities.append(v.clone()); timesteps.append(float(timestep)); sigmas.append((sigma, sigma_next)); residuals.append(torch.zeros_like(x)); states.append(x.clone())
    return TrajectoryTrace(str(state.metadata.get("prompt", "")), states, velocities, timesteps, sigmas, x.clone(), residuals)


def estimate_edit_token_mask(preservation: TrajectoryTrace, edited: TrajectoryTrace, critical_indices: Sequence[int], *, quantile=.75, min_ratio=.02, max_ratio=.40):
    if len(preservation.states) != len(edited.states):
        raise ValueError("paired traces must have equal lengths")
    if not critical_indices:
        raise ValueError("at least one critical index is required")
    scores = []
    for i in critical_indices:
        if int(i) + 1 >= len(preservation.states):
            raise IndexError(f"critical index {i} is outside trace")
        delta = (edited.states[int(i) + 1].float() - preservation.states[int(i) + 1].float()).squeeze(0)
        scores.append(delta.norm(dim=-1))
    raw = torch.stack(scores).amax(0)
    lo, hi = raw.min(), raw.max()
    normalized = torch.zeros_like(raw) if float(hi - lo) <= 1e-12 else (raw - lo) / (hi - lo)
    n = normalized.numel(); low = max(1, math.ceil(n * min_ratio)) if min_ratio else 0; high = max(low, min(n, math.floor(n * max_ratio)))
    mask = normalized >= torch.quantile(normalized, float(quantile))
    if int(mask.sum()) < low:
        mask = torch.zeros_like(mask, dtype=torch.bool); mask[torch.topk(normalized, low).indices] = True
    if int(mask.sum()) > high:
        mask = torch.zeros_like(mask, dtype=torch.bool); mask[torch.topk(normalized, high).indices] = True
    return mask, normalized


def select_winner(source, candidates, instruction, scorer, *, candidate_index=None):
    if len(candidates) != 4:
        raise ValueError("the minimal prototype fixes four candidates")
    if candidate_index is not None:
        if not 0 <= int(candidate_index) < 4:
            raise ValueError("candidate_index is outside candidate range")
        return int(candidate_index), [float("nan")] * 4, False
    if scorer is None:
        raise RewardUnavailable("a RewardScorer or explicit candidate_index is required")
    rewards = [float(scorer.score(source, x, instruction)) for x in candidates]
    if not all(math.isfinite(x) for x in rewards):
        raise ValueError("Reward returned a non-finite value")
    return max(range(4), key=lambda i: (rewards[i], -i)), rewards, True


@torch.inference_mode()
def _terminal(pipe, state, x, start, *, source_latent=None, intervention_step_count=0, similarity_threshold=0.8, similarity_mode="elementwise"):
    source_latent = state.image_latents if source_latent is None else source_latent
    for i in range(start, len(state.timesteps)):
        t = state.timesteps[i]; v_edit = velocity(pipe, state, x, t); a, b = _sigmas(pipe, t, state)
        if i < int(intervention_step_count):
            v = velo_edit_velocity(x, source_latent, v_edit, a, 1.0, threshold=similarity_threshold, mode=similarity_mode)[0]
        else:
            v = v_edit
        x = _step(x, v, a, b)
    return x


@torch.inference_mode()
def generate_coupled_branches(pipe, preservation_state, edited_state, token_mask, source, instruction, decode, scorer, *, seed, cfg, candidate_index=None):
    critical = set(_critical_indices(pipe, cfg)); p, e = preservation_state.latents, edited_state.latents
    p_states, e_states = [p.clone()], [e.clone()]; pvs, evs, ts, ss, p_residuals, e_residuals = [], [], [], [], [], []
    records, branch_images = [], []; final_rewards = []; winner_index = 0; used_reward = False
    mask = token_mask.to(e.device, dtype=torch.bool).reshape(1, -1, 1)
    for i, t in enumerate(edited_state.timesteps):
        a, b = _sigmas(pipe, t, edited_state)
        vp = reference_velocity(p, preservation_state.image_latents, a)
        ve_raw = velocity(pipe, edited_state, e, t)
        ve = velo_edit_velocity(e, edited_state.image_latents, ve_raw, a, 1.0, threshold=cfg.similarity_threshold, mode=cfg.similarity_mode)[0] if i < int(cfg.intervention_step_count) else ve_raw
        p_mean, e_mean = _step(p, vp, a, b), _step(e, ve, a, b)
        p_residual = torch.zeros_like(p); e_residual = torch.zeros_like(e)
        if i in critical:
            gen = torch.Generator(device=e.device).manual_seed(int(seed + i))
            shared = torch.randn(e.shape, generator=gen, device=e.device, dtype=torch.float32)
            independent = torch.randn((4,) + tuple(e.shape[1:]), generator=gen, device=e.device, dtype=torch.float32)
            p_next, _ = native_euler_sde_step(p, vp, a, b, shared, alpha=cfg.alpha, diffusion_scale=cfg.diffusion_scale, first_step=i == 0)
            mixed = coupled_noise(shared.expand_as(independent), independent, mask, rho=cfg.coupling_strength)
            correlation = noise_correlations(shared, mixed[:1], independent[:1], mask)
            candidates, terminals, diagnostics = [], [], []
            for j in range(4):
                candidate, diag = native_euler_sde_step(e, ve, a, b, mixed[j], alpha=cfg.alpha, diffusion_scale=cfg.diffusion_scale, first_step=i == 0)
                candidates.append(candidate); terminals.append(_terminal(pipe, edited_state, candidate, i + 1, source_latent=edited_state.image_latents, intervention_step_count=cfg.intervention_step_count, similarity_threshold=cfg.similarity_threshold, similarity_mode=cfg.similarity_mode))
                diagnostics.append({**diag, "candidate_index": j, "candidate_seed": int(seed + i), "state_hash": tensor_hash(candidate), "finite": bool(torch.isfinite(candidate).all())})
            stage_images = [decode(edited_state, x)[0] for x in terminals]; branch_images.append(stage_images)
            winner_index, rewards, used = select_winner(source, stage_images, instruction, scorer, candidate_index=candidate_index)
            final_rewards, used_reward = rewards, used_reward or used; e_next = candidates[winner_index]
            p_residual, e_residual = p_next - p_mean, e_next - e_mean
            records.append({"stage": len(records) + 1, "branch_step_index": i, "post_branch_step_index": i + 1, "winner_index": winner_index, "rewards": rewards, "used_reward": used, "seed": int(seed + i), "sigma": a, "sigma_next": b, "preservation_residual_norm": float(p_residual.float().norm()), "reward_residual_norm": float(e_residual.float().norm()), **correlation, "candidate_diagnostics": diagnostics})
        else:
            p_next, e_next = p_mean, e_mean
        pvs.append(vp.clone()); evs.append(ve.clone()); ts.append(float(t)); ss.append((a, b)); p_residuals.append(p_residual.clone()); e_residuals.append(e_residual.clone())
        p, e = p_next, e_next; p_states.append(p.clone()); e_states.append(e.clone())
    preservation = TrajectoryTrace(str(preservation_state.metadata.get("prompt", "")), p_states, pvs, ts, ss, p.clone(), p_residuals)
    winner = TrajectoryTrace(str(edited_state.metadata.get("prompt", "")), e_states, evs, ts, ss, e.clone(), e_residuals)
    return preservation, winner, records, winner_index, final_rewards, used_reward, branch_images


@torch.inference_mode()
def rollout_strengths(pipe, preservation, winner, strengths, *, preservation_state=None, edited_state=None, source_latent=None, intervention_step_count=4, search_step_indices=None, similarity_threshold=0.8, similarity_mode=None):
    if edited_state is None or winner is None: raise ValueError('rollout requires edited_state and winner')
    if source_latent is None: source_latent = edited_state.image_latents
    if similarity_mode is None: similarity_mode = 'elementwise'
    steps = set(int(i) for i in (search_step_indices or ()))
    output = {}
    for value in strengths:
        s_value = float(value)
        x = edited_state.latents.clone()
        for i, t in enumerate(edited_state.timesteps):
            v_edit = velocity(pipe, edited_state, x, t); sigma, sigma_next = _sigmas(pipe, t, edited_state)
            if i < int(intervention_step_count): v_out, _, _, _ = velo_edit_velocity(x, source_latent, v_edit, sigma, s_value, threshold=similarity_threshold, mode=similarity_mode)
            else: v_out = v_edit
            x = _step(x, v_out, sigma, sigma_next)
            if i in steps and i < len(winner.residuals) and i < len(preservation.residuals):
                x = (x.float() + s_value * (winner.residuals[i].float() - preservation.residuals[i].float())).to(x.dtype)
        output[s_value] = x.clone()
    return output


@torch.inference_mode()
def build_bundle(pipe, source, instruction, decode, scorer, *, seed, config=None, candidate_index=None):
    cfg = config or ContinuousStrengthConfig()
    edited_state = prepare_state(pipe, source, instruction, seed, height=cfg.height, width=cfg.width, steps=cfg.steps, guidance_scale=cfg.guidance_scale, first_step_align_steps=cfg.first_step_align_steps, device=pipe._execution_device)
    preservation_state = prepare_state(pipe, source, cfg.neutral_prompt, seed, height=cfg.height, width=cfg.width, steps=cfg.steps, guidance_scale=cfg.guidance_scale, first_step_align_steps=cfg.first_step_align_steps, device=pipe._execution_device)
    preservation_state.latents = edited_state.latents.clone()
    pilot_p, pilot_e = deterministic_trace(pipe, preservation_state), deterministic_trace(pipe, edited_state)
    critical = _critical_indices(pipe, cfg); mask, scores = estimate_edit_token_mask(pilot_p, pilot_e, critical, quantile=cfg.mask_quantile, min_ratio=cfg.min_edit_ratio, max_ratio=cfg.max_edit_ratio)
    preservation, winner, records, index, rewards, used, images = generate_coupled_branches(pipe, preservation_state, edited_state, mask, source, instruction, decode, scorer, seed=seed + 10000, cfg=cfg, candidate_index=candidate_index)
    metadata = {"seed": int(seed), "instruction": instruction, "neutral_prompt": cfg.neutral_prompt, "config": asdict(cfg), "critical_indices": critical, "search_step_indices": critical, "intervention_step_count": cfg.intervention_step_count, "intervention_steps_applied": list(range(min(cfg.intervention_step_count, len(edited_state.timesteps)))), "generated_tokens": int(edited_state.metadata["generated_tokens"]), "source_conditioning_tokens": int(edited_state.metadata["source_conditioning_tokens"]), "token_mask": mask.cpu().tolist(), "mask_scores": scores.cpu().tolist(), "reward_selected": bool(used and candidate_index is None), "coupled_sde": "preservation/edit share noise outside edit mask; edited candidates use independent noise inside edit mask", "first_step_alignment": edited_state.metadata.get("first_step_alignment"), "preservation": preservation.metadata(), "winner": winner.metadata()}
    return TrajectoryBundle(preservation, pilot_e, winner, mask, scores, records, index, rewards, metadata, images, preservation_state, edited_state)


def load_reward_factory(spec):
    if ":" not in spec: raise ValueError("reward factory must use module:attribute syntax")
    module, name = spec.split(":", 1); obj = getattr(importlib.import_module(module), name); scorer = obj() if callable(obj) else obj
    return scorer if hasattr(scorer, "score") else CallableRewardScorer(scorer)


def save_bundle_metadata(bundle, path):
    payload = {k: v for k, v in bundle.metadata.items()}; payload.update({"winner_index": bundle.winner_index, "rewards": bundle.rewards, "branch_records": bundle.branch_records})
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def save_bundle_tensors(bundle, path):
    torch.save({"preservation_states": bundle.preservation.states, "winner_states": bundle.winner.states, "preservation_velocities": bundle.preservation.velocities, "winner_velocities": bundle.winner.velocities, "preservation_residuals": bundle.preservation.residuals, "reward_residuals": bundle.winner.residuals, "token_mask": bundle.token_mask.cpu(), "mask_scores": bundle.mask_scores.cpu()}, path)


__all__ = ["CallableRewardScorer", "ContinuousStrengthConfig", "RewardScorer", "RewardUnavailable", "TrajectoryBundle", "TrajectoryTrace", "build_bundle", "deterministic_trace", "estimate_edit_token_mask", "generate_coupled_branches", "load_reward_factory", "rollout_strengths", "save_bundle_metadata", "save_bundle_tensors", "select_winner", "strength_step"]


def reference_velocity(z_t, z_source, sigma, eps=1e-8):
    if z_t.shape != z_source.shape: raise ValueError('source latent shape mismatch')
    return (z_t.float() - z_source.float()) / (float(sigma) + eps)

def velocity_similarity(v_edit, v_ref, mode='elementwise', eps=1e-8):
    if mode == 'elementwise': return (v_ref.float().abs() + eps) / (v_ref.float().abs() + eps + (v_edit.float() - v_ref.float()).abs())
    if mode == 'cosine': return (F.cosine_similarity(v_edit.float(), v_ref.float(), dim=-1, eps=eps) + 1.0) * 0.5
    raise ValueError('similarity mode must be elementwise or cosine')

def regional_velocity(v_edit, v_ref, strength, similarity, threshold=0.8):
    if not 0.0 <= float(strength) <= 1.0: raise ValueError('strength must lie in [0, 1]')
    preserve = similarity >= float(threshold)
    edit = ~preserve
    blended = (1.0 - float(strength)) * v_ref.float() + float(strength) * v_edit.float()
    return torch.where(preserve, v_ref.float(), blended).to(v_edit.dtype), edit, preserve

def velo_edit_velocity(z_t, z_source, v_edit, sigma, strength, threshold=0.8, mode='elementwise'):
    v_ref = reference_velocity(z_t, z_source, sigma)
    similarity = velocity_similarity(v_edit, v_ref, mode)
    output, edit, preserve = regional_velocity(v_edit, v_ref, strength, similarity, threshold)
    return output, edit, preserve, v_ref
