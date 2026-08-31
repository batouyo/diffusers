"""Training-free coupled continuous-strength editing for FLUX-Kontext.

The neutral preservation prompt is an engineering approximation: Kontext does
not expose an oracle source-preservation velocity. Early SDE uses preservation-aware
coupling during branch search to select a single edit-direction correction;
continuous strength is controlled only by the short VeloEdit-style velocity window.
"""
from __future__ import annotations

import importlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import torch
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
    steps: int = 30
    guidance_scale: float = 2.5
    critical_steps: int = 1
    critical_step_indices: tuple[int, ...] | None = None
    preserve_step_count: int = 4
    edit_strength_step_count: int = 2
    search_step_indices: tuple[int, ...] | None = None
    enable_search: bool = True
    enable_reward: bool = True
    enable_coupling: bool = True
    independent_sde: bool = False
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
    generator_device: str = "cpu"

    def __post_init__(self):
        if self.num_candidates != 4:
            raise ValueError("the minimal prototype fixes num_candidates=4")
        if self.critical_steps != 1:
            raise ValueError("the continuous-strength prototype uses one search stage")
        if self.preserve_step_count < 0 or self.edit_strength_step_count < 0:
            raise ValueError("step windows must be non-negative")
        if self.edit_strength_step_count > self.preserve_step_count:
            raise ValueError("edit strength window must be inside preserve window")
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
    selected_delta_velocity: torch.Tensor | None = None
    selected_search_edit_mask: torch.Tensor | None = None
    selected_search_step: int | None = None
    preservation_state: Any = None
    edited_state: Any = None


@dataclass
class PreparedSampleContext:
    edited_state: KontextState
    source: Image.Image
    instruction: str
    seed: int


@torch.inference_mode()
def prepare_sample_context(pipe, source, instruction, *, seed, config):
    state = prepare_state(
        pipe,
        source,
        instruction,
        seed,
        height=config.height,
        width=config.width,
        steps=config.steps,
        guidance_scale=config.guidance_scale,
        first_step_align_steps=config.first_step_align_steps,
        generator_device=config.generator_device,
        device=pipe._execution_device,
    )
    return PreparedSampleContext(state, source, instruction, int(seed))


def _critical_indices(state_or_pipe: Any, cfg: ContinuousStrengthConfig) -> list[int]:
    """Select stochastic transitions from the prepared state's effective schedule."""
    values = None
    if hasattr(state_or_pipe, "metadata"):
        values = state_or_pipe.metadata.get("effective_schedule", {}).get("effective_sigmas")
    if values is None:
        values = state_or_pipe.scheduler.sigmas.detach().cpu().flatten().tolist()
    valid = critical_nonzero_steps([float(x) for x in values])
    valid_indices = {int(x["index"]) for x in valid}
    requested = cfg.search_step_indices if cfg.search_step_indices is not None else cfg.critical_step_indices
    indices = list(requested) if requested is not None else [int(x["index"]) for x in valid[: cfg.critical_steps]]
    if len(indices) != 1 or len(indices) != len(set(indices)) or any(i not in valid_indices for i in indices):
        raise ValueError("continuous-strength search requires one unique non-zero scheduler transition")
    if cfg.enable_search:
        search_step = int(indices[0])
        if search_step >= int(cfg.edit_strength_step_count):
            raise ValueError("search step must be inside the edit-strength window")
        if search_step >= int(cfg.preserve_step_count):
            raise ValueError("search step must be inside the preserve window")
    return sorted(indices)


def _step(x, v, sigma, sigma_next):
    return (x.float() + (float(sigma_next) - float(sigma)) * v.float()).to(x.dtype)


def strength_step(x, preservation_velocity, edited_velocity, sigma, sigma_next, strength):
    """One deterministic VeloEdit velocity-interpolation update."""
    s = float(strength)
    if not 0.0 <= s <= 1.0:
        raise ValueError("strength must lie in [0, 1]")
    blended = preservation_velocity.float() + s * (edited_velocity.float() - preservation_velocity.float())
    return _step(x, blended, sigma, sigma_next)


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
    rewards = [float(x) for x in (scorer.score_many(source, candidates, instruction) if hasattr(scorer, "score_many") else [scorer.score(source, x, instruction) for x in candidates])]
    if not all(math.isfinite(x) for x in rewards):
        raise ValueError("Reward returned a non-finite value")
    return max(range(4), key=lambda i: (rewards[i], -i)), rewards, True


@torch.inference_mode()
def _terminal(pipe, state, x, start, *, source_latent=None, preserve_step_count=0, similarity_threshold=0.8, similarity_mode="elementwise"):
    source_latent = state.image_latents if source_latent is None else source_latent
    for i in range(start, len(state.timesteps)):
        t = state.timesteps[i]; v_edit = velocity(pipe, state, x, t); a, b = _sigmas(pipe, t, state)
        v = velo_edit_velocity(
            x, source_latent, v_edit, a, 1.0, threshold=similarity_threshold,
            mode=similarity_mode, preserve_active=i < int(preserve_step_count),
            edit_strength_active=False,
        )[0]
        x = _step(x, v, a, b)
    return x


@torch.inference_mode()
def _terminal_batch(pipe, state, x, start, *, source_latent=None, preserve_step_count=0, similarity_threshold=0.8, similarity_mode="elementwise"):
    """Roll out candidate batch with one transformer call per timestep."""
    source_latent = state.image_latents if source_latent is None else source_latent
    for i in range(start, len(state.timesteps)):
        t = state.timesteps[i]
        v_edit = velocity(pipe, state, x, t)
        a, b = _sigmas(pipe, t, state)
        v = velo_edit_velocity(
            x, source_latent, v_edit, a, 1.0, threshold=similarity_threshold,
            mode=similarity_mode, preserve_active=i < int(preserve_step_count),
            edit_strength_active=False,
        )[0]
        x = _step(x, v, a, b)
    return x


@torch.inference_mode()
def generate_coupled_branches(pipe, preservation_state, edited_state, token_mask, source, instruction, decode, scorer, *, seed, cfg, candidate_index=None):
    selected_steps = _critical_indices(edited_state, cfg)
    if len(selected_steps) != 1:
        raise ValueError("continuous-strength search requires exactly one transition")
    critical = set(selected_steps) if cfg.enable_search else set()
    p, e = preservation_state.latents, edited_state.latents
    p_states, e_states = [p.clone()], [e.clone()]; pvs, evs, ts, ss, p_residuals, e_residuals = [], [], [], [], [], []
    records, branch_images = [], []; final_rewards = []; winner_index = 0; used_reward = False
    selected_delta_velocity = None
    selected_search_edit_mask = None
    selected_search_step = next(iter(critical), None)
    full_edit_online_mask_trace = []
    mask = token_mask.to(e.device, dtype=torch.bool).reshape(1, -1, 1)
    for i, t in enumerate(edited_state.timesteps):
        a, b = _sigmas(pipe, t, edited_state)
        vp = reference_velocity(p, preservation_state.image_latents, a)
        ve_raw = velocity(pipe, edited_state, e, t)
        ve_info = velo_edit_velocity(
            e,
            edited_state.image_latents,
            ve_raw,
            a,
            1.0,
            threshold=cfg.similarity_threshold,
            mode=cfg.similarity_mode,
            preserve_active=i < int(cfg.preserve_step_count),
            edit_strength_active=i < int(cfg.edit_strength_step_count),
        )
        ve = ve_info[0]
        dynamic_edit_mask = (~ve_info[2]).to(e.device)
        if i < int(cfg.preserve_step_count):
            full_edit_online_mask_trace.append(
                {
                    "step_index": int(i),
                    "online_edit_ratio": float(dynamic_edit_mask.float().mean()),
                    "online_preserve_ratio": float(ve_info[2].float().mean()),
                }
            )
        p_mean, e_mean = _step(p, vp, a, b), _step(e, ve, a, b)
        p_residual = torch.zeros_like(p); e_residual = torch.zeros_like(e)
        p_next = p_mean
        if i in critical:
            gen = torch.Generator(device=e.device).manual_seed(int(seed + i))
            shared = torch.randn(e.shape, generator=gen, device=e.device, dtype=torch.float32)
            independent = torch.randn((4,) + tuple(e.shape[1:]), generator=gen, device=e.device, dtype=torch.float32)
            if cfg.independent_sde:
                preserve_noise = shared
                mixed = independent
            elif cfg.enable_coupling:
                preserve_noise = shared
                mixed = coupled_noise(shared.expand_as(independent), independent, dynamic_edit_mask.expand_as(independent), rho=cfg.coupling_strength)
            else:
                preserve_noise = torch.zeros_like(shared)
                mixed = independent * dynamic_edit_mask.float().expand_as(independent)
            correlation = noise_correlations(preserve_noise, mixed[:1], independent[:1], dynamic_edit_mask)
            p_next, _ = native_euler_sde_step(p, vp, a, b, preserve_noise, alpha=cfg.alpha, diffusion_scale=cfg.diffusion_scale, first_step=i == 0)
            p_residual = p_next - p_mean
            candidates, terminals, diagnostics = [], [], []
            for j in range(4):
                raw_candidate, diag = native_euler_sde_step(e, ve, a, b, mixed[j], alpha=cfg.alpha, diffusion_scale=cfg.diffusion_scale, first_step=i == 0)
                raw_residual = raw_candidate.float() - e_mean.float()
                relative_residual = raw_residual - p_residual.float()
                candidate = (e_mean.float() + relative_residual).to(e.dtype)
                candidates.append(candidate)
                diagnostics.append({**diag, "candidate_index": j, "candidate_seed": int(seed + i), "raw_state_hash": tensor_hash(raw_candidate), "corrected_state_hash": tensor_hash(candidate), "raw_residual_norm": float(raw_residual.norm().item()), "relative_residual_norm": float(relative_residual.norm().item()), "finite": bool(torch.isfinite(candidate).all().item())})
            if cfg.enable_reward:
                terminal_batch = _terminal_batch(
                    pipe,
                    edited_state,
                    torch.cat(candidates, dim=0),
                    i + 1,
                    source_latent=edited_state.image_latents,
                    preserve_step_count=cfg.preserve_step_count,
                    similarity_threshold=cfg.similarity_threshold,
                    similarity_mode=cfg.similarity_mode,
                )
                terminals = [terminal_batch[j:j + 1] for j in range(terminal_batch.shape[0])]
            else:
                terminals = [
                    _terminal(
                        pipe,
                        edited_state,
                        candidates[0],
                        i + 1,
                        source_latent=edited_state.image_latents,
                        preserve_step_count=cfg.preserve_step_count,
                        similarity_threshold=cfg.similarity_threshold,
                        similarity_mode=cfg.similarity_mode,
                    )
                ]
            stage_images = [decode(edited_state, x)[0] for x in terminals]; branch_images.append(stage_images)
            if cfg.enable_reward:
                winner_index, rewards, used = select_winner(source, stage_images, instruction, scorer, candidate_index=None)
            else:
                winner_index, rewards, used = 0, [float("nan")] * 4, False
            final_rewards, used_reward = rewards, used_reward or used
            selected_branch_candidate = candidates[winner_index]
            selected_residual = selected_branch_candidate.float() - e_mean.float()
            selected_search_edit_mask = dynamic_edit_mask.detach().clone()
            selected_delta_velocity = (
                selected_residual * selected_search_edit_mask.float()
            ) / (float(b) - float(a))
            if not torch.isfinite(selected_delta_velocity).all():
                raise RuntimeError("selected delta velocity is non-finite")
            delta = selected_delta_velocity.to(device=e.device, dtype=ve.dtype)
            mask = selected_search_edit_mask.to(device=e.device, dtype=torch.bool)
            while mask.ndim < ve.ndim:
                mask = mask.unsqueeze(-1)
            deployed_velocity = ve + delta * mask.to(ve.dtype)
            e_next = _step(e, deployed_velocity, a, b)
            reconstruction = e_next
            reconstruction_target = e_mean + selected_residual * selected_search_edit_mask.float()
            reconstruction_error = (reconstruction.float() - reconstruction_target.float()).abs()
            p_residual = p_next - p_mean
            e_residual = e_next.float() - e_mean.float()
            relative_residual = e_residual.float() - p_residual.float()
            records.append({
                "stage": 1,
                "branch_step_index": i,
                "post_branch_step_index": i + 1,
                "winner_index": winner_index,
                "rewards": rewards,
                "used_reward": used,
                "reward_candidate_policy": "reward_scores_full_corrected_branch; deploys_masked_edit_direction",
                "reward_selected_branch": int(winner_index),
                "deployed_edit_direction": "selected_edit_region_delta_velocity",
                "seed": int(seed + i),
                "sigma": a,
                "sigma_next": b,
                "selected_search_edit_ratio": float(dynamic_edit_mask.float().mean()),
                "selected_delta_velocity_norm": float(selected_delta_velocity.float().norm()),
                "winner_reconstruction_max_abs_error": float(reconstruction_error.max()),
                "winner_reconstruction_relative_l2": float(
                    reconstruction_error.norm() / reconstruction_target.float().norm().clamp_min(1e-8)
                ),
                "preservation_noise_norm": float(preserve_noise.float().norm()),
                "edited_noise_norm_mean": float(mixed.float().norm(dim=tuple(range(1, mixed.ndim))).mean()),
                "preservation_residual_norm": float(p_residual.float().norm()),
                "edited_residual_norm": float(e_residual.float().norm()),
                "relative_residual_norm": float(relative_residual.float().norm()),
                **correlation,
                "candidate_diagnostics": diagnostics,
            })
        else:
            p_next, e_next = p_mean, e_mean
        if i in critical:
            ve = deployed_velocity
        pvs.append(vp.clone()); evs.append(ve.clone()); ts.append(float(t)); ss.append((a, b)); p_residuals.append(p_residual.clone()); e_residuals.append(e_residual.clone())
        p, e = p_next, e_next; p_states.append(p.clone()); e_states.append(e.clone())
    preservation = TrajectoryTrace(str(preservation_state.metadata.get("prompt", "")), p_states, pvs, ts, ss, p.clone(), p_residuals)
    winner = TrajectoryTrace(str(edited_state.metadata.get("prompt", "")), e_states, evs, ts, ss, e.clone(), e_residuals)
    if cfg.enable_search and selected_delta_velocity is None:
        raise RuntimeError("enabled search did not produce a selected velocity")
    return (
        preservation,
        winner,
        records,
        winner_index,
        final_rewards,
        used_reward,
        branch_images,
        selected_delta_velocity,
        selected_search_edit_mask,
        selected_search_step,
        full_edit_online_mask_trace,
    )


@torch.inference_mode()
def rollout_strengths(
    pipe,
    preservation,
    winner,
    strengths,
    *,
    preservation_state=None,
    edited_state=None,
    source_latent=None,
    selected_delta_velocity=None,
    selected_search_edit_mask=None,
    selected_search_step=None,
    preserve_step_count=4,
    edit_strength_step_count=2,
    similarity_threshold=0.8,
    similarity_mode=None,
    strength_batch_size=None,
    online_mask_trace=None,
    cached_full_edit_online_mask_trace=None,
    reuse_full_edit_endpoint=False,
):
    if edited_state is None or winner is None: raise ValueError('rollout requires edited_state and winner')
    if strength_batch_size is not None and int(strength_batch_size) < 1:
        raise ValueError('strength_batch_size must be positive')
    if strength_batch_size is not None and len(strengths) > int(strength_batch_size):
        merged = {}
        values = list(strengths)
        for start in range(0, len(values), int(strength_batch_size)):
            merged.update(
                rollout_strengths(
                    pipe,
                    preservation,
                    winner,
                    values[start : start + int(strength_batch_size)],
                    preservation_state=preservation_state,
                    edited_state=edited_state,
                    source_latent=source_latent,
                    selected_delta_velocity=selected_delta_velocity,
                    selected_search_edit_mask=selected_search_edit_mask,
                    selected_search_step=selected_search_step,
                    preserve_step_count=preserve_step_count,
                    edit_strength_step_count=edit_strength_step_count,
                    similarity_threshold=similarity_threshold,
                    similarity_mode=similarity_mode,
                    online_mask_trace=online_mask_trace,
                    reuse_full_edit_endpoint=reuse_full_edit_endpoint,
                    cached_full_edit_online_mask_trace=cached_full_edit_online_mask_trace,
                )
            )
        return merged
    if source_latent is None: source_latent = edited_state.image_latents
    if similarity_mode is None: similarity_mode = 'elementwise'
    if selected_search_step is not None and selected_delta_velocity is None:
        raise ValueError("selected_search_step requires selected_delta_velocity")
    if selected_delta_velocity is not None and selected_search_edit_mask is None:
        raise ValueError("selected_delta_velocity requires selected_search_edit_mask")
    if selected_delta_velocity is not None and selected_search_step is None:
        raise ValueError("selected_delta_velocity requires selected_search_step")
    if selected_search_step is not None:
        if int(selected_search_step) < 0 or int(selected_search_step) >= int(edit_strength_step_count):
            raise ValueError("selected search step must be inside the edit-strength window")
        if int(selected_search_step) >= int(preserve_step_count):
            raise ValueError("selected search step must be inside the preserve window")
    values = [float(v) for v in strengths]
    reuse_winner_for_one = bool(reuse_full_edit_endpoint and any(value == 1.0 for value in values))
    model_values = [value for value in values if not (reuse_winner_for_one and value == 1.0)]
    output = {}
    if reuse_winner_for_one:
        output[1.0] = winner.terminal.clone()
        if online_mask_trace is not None:
            for item in (cached_full_edit_online_mask_trace or []):
                online_mask_trace.append({
                    "strength": 1.0,
                    "step_index": int(item["step_index"]),
                    "online_edit_ratio": float(item["online_edit_ratio"]),
                    "online_preserve_ratio": float(item["online_preserve_ratio"]),
                })
    if not model_values:
        return {value: output[value] for value in values}
    x = edited_state.latents.repeat(len(model_values), 1, 1)
    s_tensor = torch.tensor(model_values, device=x.device, dtype=torch.float32).view(-1, 1, 1)
    for i, t in enumerate(edited_state.timesteps):
        v_edit = velocity(pipe, edited_state, x, t)
        sigma, sigma_next = _sigmas(pipe, t, edited_state)
        src = source_latent if source_latent is not None else edited_state.image_latents
        v_ref = reference_velocity(x, src, sigma)
        similarity = velocity_similarity(v_edit, v_ref, similarity_mode)
        v_edit_target = v_edit
        if i == selected_search_step:
            delta = selected_delta_velocity.to(device=x.device, dtype=v_edit.dtype)
            if delta.shape[0] == 1 and x.shape[0] != 1:
                delta = delta.expand(x.shape[0], -1, -1)
            mask = selected_search_edit_mask.to(device=x.device, dtype=torch.bool)
            while mask.ndim < v_edit.ndim:
                mask = mask.unsqueeze(-1)
            v_edit_target = v_edit + delta * mask.to(v_edit.dtype)
        v_out, edit_mask, preserve_mask = regional_velocity(
            v_edit_target,
            v_ref,
            s_tensor,
            similarity,
            similarity_threshold,
            preserve_active=i < int(preserve_step_count),
            edit_strength_active=i < int(edit_strength_step_count),
        )
        if online_mask_trace is not None and i < int(preserve_step_count):
            for index, value in enumerate(model_values):
                online_mask_trace.append(
                    {
                        "strength": float(value),
                        "step_index": int(i),
                        "online_edit_ratio": float(edit_mask[index].float().mean()),
                        "online_preserve_ratio": float(preserve_mask[index].float().mean()),
                    }
                )
        x = _step(x, v_out, sigma, sigma_next)
    for index, value in enumerate(model_values):
        output[value] = x[index:index + 1].clone()
    return {value: output[value] for value in values}


@torch.inference_mode()
def build_bundle(pipe, source, instruction, decode, scorer, *, seed, config=None, candidate_index=None, prepared_context=None):
    cfg = config or ContinuousStrengthConfig()
    edited_state = prepared_context.edited_state if prepared_context is not None else prepare_state(
        pipe,
        source,
        instruction,
        seed,
        height=cfg.height,
        width=cfg.width,
        steps=cfg.steps,
        guidance_scale=cfg.guidance_scale,
        first_step_align_steps=cfg.first_step_align_steps,
        generator_device=cfg.generator_device,
        device=pipe._execution_device,
    )
    if cfg.enable_search:
        preservation_state = prepare_state(
            pipe, source, cfg.neutral_prompt, seed,
            height=cfg.height, width=cfg.width, steps=cfg.steps,
            guidance_scale=cfg.guidance_scale,
            first_step_align_steps=cfg.first_step_align_steps,
            generator_device=cfg.generator_device,
            device=pipe._execution_device,
        )
    else:
        preservation_state = replace(
            edited_state,
            latents=edited_state.latents.clone(),
            metadata={**edited_state.metadata, "prompt": cfg.neutral_prompt, "role": "preservation_reference"},
        )
    preservation_state.latents = edited_state.latents.clone()
    critical = _critical_indices(edited_state, cfg)
    if cfg.enable_search:
        pilot_p, pilot_e = deterministic_trace(pipe, preservation_state), deterministic_trace(pipe, edited_state)
        mask, scores = estimate_edit_token_mask(
            pilot_p, pilot_e, critical,
            quantile=cfg.mask_quantile, min_ratio=cfg.min_edit_ratio, max_ratio=cfg.max_edit_ratio,
        )
    else:
        mask = torch.zeros(edited_state.latents.shape[1], device=edited_state.latents.device, dtype=torch.bool)
        scores = torch.zeros_like(mask, dtype=torch.float32)
    (
        preservation,
        winner,
        records,
        index,
        rewards,
        used,
        images,
        selected_delta_velocity,
        selected_search_edit_mask,
        selected_search_step,
        full_edit_online_mask_trace,
    ) = generate_coupled_branches(
        pipe,
        preservation_state,
        edited_state,
        mask,
        source,
        instruction,
        decode,
        scorer,
        seed=seed + 10000,
        cfg=cfg,
        candidate_index=candidate_index,
    )
    actual_search = sorted({int(item["branch_step_index"]) for item in records})
    if cfg.enable_search and actual_search != critical: raise RuntimeError(f"search step mismatch: actual={actual_search}, metadata={critical}")
    if selected_search_step is not None:
        strength_control_regime = "one_step" if selected_search_step == 0 else "two_step_for_search_alignment"
    else:
        strength_control_regime = "one_step" if cfg.edit_strength_step_count == 1 else "two_step_for_search_alignment"
    metadata = {
        "seed": int(seed),
        "instruction": instruction,
        "neutral_prompt": cfg.neutral_prompt,
        "config": asdict(cfg),
        "critical_indices": critical,
        "search_step_indices": critical if cfg.enable_search else [],
        "preserve_step_count": int(cfg.preserve_step_count),
        "edit_strength_step_count": int(cfg.edit_strength_step_count),
        "preserve_active_steps": list(range(min(cfg.preserve_step_count, len(edited_state.timesteps)))),
        "edit_strength_active_steps": list(range(min(cfg.edit_strength_step_count, len(edited_state.timesteps)))),
        "strength_control_regime": strength_control_regime,
        "state_residual_strength_replay": False,
        "generated_tokens": int(edited_state.metadata["generated_tokens"]),
        "source_conditioning_tokens": int(edited_state.metadata["source_conditioning_tokens"]),
        "token_mask": mask.cpu().tolist(),
        "mask_scores": scores.cpu().tolist(),
        "reward_selected": bool(used and candidate_index is None),
        "search_stage_count": len(records),
        "selected_correction_applied_once": bool(selected_delta_velocity is not None and selected_search_step is not None),
        "s0_selected_correction_scale": 0.0,
        "reward_selected_branch": None if not records else int(records[0]["reward_selected_branch"]),
        "deployed_edit_direction": None if selected_delta_velocity is None else "selected_edit_region_delta_velocity",
        "coupled_sde": "search-time preservation-aware coupling; shared preserve-region noise and independent edit-region exploration only during early branch search",
        "first_step_alignment": edited_state.metadata.get("first_step_alignment"),
        "selected_search_step": selected_search_step,
        "selected_search_edit_mask": None if selected_search_edit_mask is None else selected_search_edit_mask.cpu().tolist(),
        "selected_delta_velocity": None if selected_delta_velocity is None else selected_delta_velocity.cpu().tolist(),
        "selected_delta_velocity_norm": None if selected_delta_velocity is None else float(selected_delta_velocity.float().norm()),
        "winner_reconstruction_max_abs_error": None if not records else records[0].get("winner_reconstruction_max_abs_error"),
        "winner_reconstruction_relative_l2": None if not records else records[0].get("winner_reconstruction_relative_l2"),
        "preservation": preservation.metadata(),
        "winner": winner.metadata(),
        "full_edit_online_mask_trace": full_edit_online_mask_trace,
    }
    metadata["schedule_trace"] = [
        {
            "step_index": int(i),
            "model_timestep": float(t),
            "sigma": float(_sigmas(pipe, t, edited_state)[0]),
            "sigma_next": float(_sigmas(pipe, t, edited_state)[1]),
            "preserve_active": bool(i < int(cfg.preserve_step_count)),
            "edit_strength_active": bool(i < int(cfg.edit_strength_step_count)),
        }
        for i, t in enumerate(edited_state.timesteps)
    ]
    return TrajectoryBundle(
        preservation,
        winner if not cfg.enable_search else pilot_e,
        winner,
        mask,
        scores,
        records,
        index,
        rewards,
        metadata,
        images,
        selected_delta_velocity,
        selected_search_edit_mask,
        selected_search_step,
        preservation_state,
        edited_state,
    )


def load_reward_factory(spec):
    if ":" not in spec: raise ValueError("reward factory must use module:attribute syntax")
    module, name = spec.split(":", 1); obj = getattr(importlib.import_module(module), name); scorer = obj() if callable(obj) else obj
    return scorer if hasattr(scorer, "score") else CallableRewardScorer(scorer)


def save_bundle_metadata(bundle, path):
    payload = {k: v for k, v in bundle.metadata.items()}; payload.update({"winner_index": bundle.winner_index, "rewards": bundle.rewards, "branch_records": bundle.branch_records})
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def save_bundle_tensors(bundle, path):
    torch.save(
        {
            "preservation_states": bundle.preservation.states,
            "winner_states": bundle.winner.states,
            "preservation_velocities": bundle.preservation.velocities,
            "winner_velocities": bundle.winner.velocities,
            "preservation_residuals": bundle.preservation.residuals,
            "edited_residuals": bundle.winner.residuals,
            "selected_delta_velocity": bundle.selected_delta_velocity,
            "selected_search_edit_mask": bundle.selected_search_edit_mask,
            "token_mask": bundle.token_mask.cpu(),
            "mask_scores": bundle.mask_scores.cpu(),
        },
        path,
    )


__all__ = ["CallableRewardScorer", "ContinuousStrengthConfig", "RewardScorer", "RewardUnavailable", "TrajectoryBundle", "TrajectoryTrace", "build_bundle", "deterministic_trace", "estimate_edit_token_mask", "generate_coupled_branches", "load_reward_factory", "rollout_strengths", "save_bundle_metadata", "save_bundle_tensors", "select_winner", "strength_step"]


def reference_velocity(z_t, z_source, sigma, eps=1e-8):
    if z_source.shape[0] == 1 and z_t.shape[0] != 1:
        z_source = z_source.expand(z_t.shape[0], -1, -1)
    if z_t.shape != z_source.shape: raise ValueError('source latent shape mismatch')
    return (z_t.float() - z_source.float()) / (float(sigma) + eps)

def velocity_similarity(v_edit, v_ref, mode='elementwise', eps=1e-8):
    if mode == 'elementwise': return (v_ref.float().abs() + eps) / (v_ref.float().abs() + eps + (v_edit.float() - v_ref.float()).abs())
    if mode == 'cosine': return ((F.cosine_similarity(v_edit.float(), v_ref.float(), dim=-1, eps=eps) + 1.0) * 0.5).unsqueeze(-1)
    raise ValueError('similarity mode must be elementwise or cosine')

def regional_velocity(
    v_edit,
    v_ref,
    strength,
    similarity,
    threshold=0.8,
    *,
    preserve_active=True,
    edit_strength_active=True,
):
    strength_tensor = torch.as_tensor(strength, device=v_edit.device, dtype=torch.float32)
    if bool(torch.any((strength_tensor < 0) | (strength_tensor > 1))): raise ValueError('strength must lie in [0, 1]')
    while strength_tensor.ndim < v_edit.ndim:
        strength_tensor = strength_tensor.unsqueeze(-1)
    preserve = similarity >= float(threshold)
    edit = ~preserve
    output = v_edit.float().clone()
    if preserve_active:
        output = torch.where(preserve, v_ref.float(), output)
    if edit_strength_active:
        blended = (1.0 - strength_tensor) * v_ref.float() + strength_tensor * v_edit.float()
        output = torch.where(edit, blended, output)
    return output.to(v_edit.dtype), edit, preserve

def velo_edit_velocity(
    z_t,
    z_source,
    v_edit,
    sigma,
    strength,
    threshold=0.8,
    mode='elementwise',
    *,
    preserve_active=True,
    edit_strength_active=True,
):
    v_ref = reference_velocity(z_t, z_source, sigma)
    similarity = velocity_similarity(v_edit, v_ref, mode)
    output, edit, preserve = regional_velocity(
        v_edit,
        v_ref,
        strength,
        similarity,
        threshold,
        preserve_active=preserve_active,
        edit_strength_active=edit_strength_active,
    )
    return output, edit, preserve, v_ref
