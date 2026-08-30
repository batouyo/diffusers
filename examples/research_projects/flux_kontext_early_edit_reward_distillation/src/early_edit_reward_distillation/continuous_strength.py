"""Training-free continuous-strength editing primitives for FLUX-Kontext.

The preservation prompt and paired-velocity interpolation are deliberate
engineering approximations: Kontext exposes an edit-conditioned velocity, not
an independent oracle preservation velocity.  The module keeps those
approximations explicit and makes all stochastic work happen before strength
rollout.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np
import torch
from PIL import Image

from .core import coupled_noise, critical_nonzero_steps, native_euler_sde_step, tensor_hash
from .metrics import region_l1
from .trajectory import KontextState, _sigmas, ode_step, prepare_state, velocity


class RewardScorer(Protocol):
    """Minimal pluggable image-edit reward contract."""

    def score(self, source: Image.Image, candidate: Image.Image, instruction: str) -> float:
        ...


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
    num_candidates: int = 4
    alpha: float = 0.05
    diffusion_scale: float = 1.0
    coupling_strength: float = 0.0
    mask_quantile: float = 0.75
    min_edit_ratio: float = 0.02
    max_edit_ratio: float = 0.40
    strengths: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

    def __post_init__(self) -> None:
        if self.num_candidates != 4:
            raise ValueError("the minimal prototype fixes num_candidates=4")
        if self.critical_steps < 1:
            raise ValueError("critical_steps must be positive")
        if not 0.0 <= self.coupling_strength <= 1.0:
            raise ValueError("coupling_strength must lie in [0, 1]")
        if not 0.0 <= self.mask_quantile <= 1.0:
            raise ValueError("mask_quantile must lie in [0, 1]")
        if not 0.0 <= self.min_edit_ratio <= self.max_edit_ratio <= 1.0:
            raise ValueError("edit ratio bounds must satisfy 0 <= min <= max <= 1")
        if any(not 0.0 <= float(s) <= 1.0 for s in self.strengths):
            raise ValueError("all strengths must lie in [0, 1]")


@dataclass
class TrajectoryTrace:
    """State/velocity cache; states has one more element than velocities."""

    prompt: str
    states: list[torch.Tensor]
    velocities: list[torch.Tensor]
    timesteps: list[float]
    sigmas: list[tuple[float, float]]
    terminal: torch.Tensor

    def metadata(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "num_steps": len(self.velocities),
            "state_hashes": [tensor_hash(x) for x in self.states],
            "terminal_hash": tensor_hash(self.terminal),
            "state_norms": [float(x.float().norm().item()) for x in self.states],
            "velocity_norms": [float(x.float().norm().item()) for x in self.velocities],
            "timesteps": self.timesteps,
            "sigmas": [list(x) for x in self.sigmas],
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


def _same_initial_latents(edit_state: KontextState, preserve_state: KontextState) -> None:
    if edit_state.latents.shape != preserve_state.latents.shape:
        raise ValueError("preservation and edited latent shapes differ")
    preserve_state.latents = edit_state.latents.detach().clone()


@torch.inference_mode()
def deterministic_trace(pipe: Any, state: KontextState, *, start: torch.Tensor | None = None) -> TrajectoryTrace:
    current = state.latents if start is None else start
    states = [current.detach().clone()]
    velocities: list[torch.Tensor] = []
    timesteps: list[float] = []
    sigmas: list[tuple[float, float]] = []
    for timestep in state.timesteps:
        pred = velocity(pipe, state, current, timestep)
        sigma, sigma_next = _sigmas(pipe, timestep)
        velocities.append(pred.detach().clone())
        timesteps.append(float(timestep))
        sigmas.append((sigma, sigma_next))
        current = (current.float() + (sigma_next - sigma) * pred.float()).to(current.dtype)
        states.append(current.detach().clone())
    return TrajectoryTrace(str(state.metadata.get("prompt", "")), states, velocities, timesteps, sigmas, current.detach().clone())


def estimate_edit_token_mask(
    preservation: TrajectoryTrace,
    edited: TrajectoryTrace,
    critical_indices: Sequence[int],
    *,
    quantile: float = 0.75,
    min_ratio: float = 0.02,
    max_ratio: float = 0.40,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(preservation.states) != len(edited.states):
        raise ValueError("paired traces must have equal lengths")
    if not critical_indices:
        raise ValueError("at least one critical index is required")
    scores = []
    for index in critical_indices:
        post = int(index) + 1
        if post >= len(preservation.states):
            raise IndexError(f"critical index {index} is outside trace")
        delta = (edited.states[post].float() - preservation.states[post].float()).squeeze(0)
        scores.append(delta.norm(dim=-1))
    raw = torch.stack(scores, dim=0).amax(dim=0)
    lo, hi = raw.min(), raw.max()
    normalized = torch.zeros_like(raw) if float(hi - lo) <= 1e-12 else (raw - lo) / (hi - lo)
    n = normalized.numel()
    min_tokens = max(1, int(math.ceil(n * min_ratio))) if min_ratio > 0 else 0
    max_tokens = max(min_tokens, min(n, int(math.floor(n * max_ratio))))
    selected = normalized >= torch.quantile(normalized, float(quantile))
    count = int(selected.sum().item())
    if count < min_tokens:
        selected = torch.zeros_like(selected, dtype=torch.bool)
        selected[torch.topk(normalized, k=min_tokens).indices] = True
    elif count > max_tokens:
        selected = torch.zeros_like(selected, dtype=torch.bool)
        selected[torch.topk(normalized, k=max_tokens).indices] = True
    return selected, normalized


def select_winner(
    source: Image.Image,
    candidates: Sequence[Image.Image],
    instruction: str,
    scorer: RewardScorer | None,
    *,
    candidate_index: int | None = None,
) -> tuple[int, list[float], bool]:
    if len(candidates) != 4:
        raise ValueError("the minimal prototype fixes four candidates")
    if candidate_index is not None:
        if not 0 <= int(candidate_index) < len(candidates):
            raise ValueError("candidate_index is outside candidate range")
        return int(candidate_index), [float("nan")] * len(candidates), False
    if scorer is None:
        raise RewardUnavailable("a RewardScorer or explicit candidate_index is required")
    rewards = [float(scorer.score(source, image, instruction)) for image in candidates]
    if not all(math.isfinite(x) for x in rewards):
        raise ValueError("Reward returned a non-finite value")
    winner = max(range(len(rewards)), key=lambda i: (rewards[i], -i))
    return winner, rewards, True


@torch.inference_mode()
def _rollout_terminal(pipe: Any, state: KontextState, current: torch.Tensor, start: int) -> torch.Tensor:
    return deterministic_trace(pipe, state, start=current).terminal if start == 0 else _trace_from_step(pipe, state, current, start).terminal


@torch.inference_mode()
def _trace_from_step(pipe: Any, state: KontextState, current: torch.Tensor, start: int) -> TrajectoryTrace:
    states = [current.detach().clone()]
    velocities: list[torch.Tensor] = []
    timesteps: list[float] = []
    sigmas: list[tuple[float, float]] = []
    for timestep in state.timesteps[start:]:
        pred = velocity(pipe, state, current, timestep)
        sigma, sigma_next = _sigmas(pipe, timestep)
        velocities.append(pred.detach().clone())
        timesteps.append(float(timestep))
        sigmas.append((sigma, sigma_next))
        current = (current.float() + (sigma_next - sigma) * pred.float()).to(current.dtype)
        states.append(current.detach().clone())
    return TrajectoryTrace(str(state.metadata.get("prompt", "")), states, velocities, timesteps, sigmas, current.detach().clone())


@torch.inference_mode()
def generate_early_branches(
    pipe: Any,
    state: KontextState,
    token_mask: torch.Tensor,
    source: Image.Image,
    instruction: str,
    decode: Callable[[KontextState, torch.Tensor], Sequence[Image.Image]],
    scorer: RewardScorer | None,
    *,
    seed: int,
    critical_count: int = 2,
    alpha: float = 0.05,
    diffusion_scale: float = 1.0,
    coupling_strength: float = 0.0,
    candidate_index: int | None = None,
) -> tuple[TrajectoryTrace, list[dict[str, Any]], int, list[float], bool, list[list[Image.Image]]]:
    selected = critical_nonzero_steps(pipe.scheduler.sigmas.detach().cpu().flatten().tolist())[:critical_count]
    if len(selected) < critical_count:
        raise RuntimeError("scheduler exposed fewer critical non-zero transitions than requested")
    current = state.latents
    current_step = 0
    trace_states = [current.detach().clone()]
    trace_velocities: list[torch.Tensor] = []
    trace_timesteps: list[float] = []
    trace_sigmas: list[tuple[float, float]] = []
    records: list[dict[str, Any]] = []
    final_rewards: list[float] = []
    used_reward = False
    winner_index = 0
    all_candidate_images: list[list[Image.Image]] = []
    for stage, item in enumerate(selected, start=1):
        step_index = int(item["index"])
        while current_step < step_index:
            timestep = state.timesteps[current_step]
            pred = velocity(pipe, state, current, timestep)
            sigma, sigma_next = _sigmas(pipe, timestep)
            trace_velocities.append(pred.detach().clone()); trace_timesteps.append(float(timestep)); trace_sigmas.append((sigma, sigma_next))
            current = (current.float() + (sigma_next - sigma) * pred.float()).to(current.dtype)
            trace_states.append(current.detach().clone()); current_step += 1
        timestep = state.timesteps[step_index]
        pred = velocity(pipe, state, current, timestep)
        sigma, sigma_next = _sigmas(pipe, timestep)
        gen = torch.Generator(device=current.device).manual_seed(int(seed + stage))
        shared = torch.randn(current.shape, generator=gen, device=current.device, dtype=torch.float32)
        independent = torch.randn((4,) + tuple(current.shape[1:]), generator=gen, device=current.device, dtype=torch.float32)
        mask = token_mask.to(device=current.device, dtype=torch.bool).reshape(1, -1, 1)
        mixed = coupled_noise(shared.expand_as(independent), independent, mask, rho=float(coupling_strength))
        candidates = []
        terminals = []
        diagnostics = []
        for candidate_id in range(4):
            candidate, diag = native_euler_sde_step(current, pred, sigma, sigma_next, mixed[candidate_id], alpha=alpha, diffusion_scale=diffusion_scale, first_step=step_index == 0)
            candidates.append(candidate)
            terminal = _rollout_terminal(pipe, state, candidate, step_index + 1)
            terminals.append(terminal)
            diagnostics.append({**diag, "candidate_index": candidate_id, "candidate_seed": int(seed + stage), "state_hash": tensor_hash(candidate), "finite": bool(torch.isfinite(candidate).all().item())})
        candidate_images = [decode(state, x)[0] for x in terminals]
        all_candidate_images.append(candidate_images)
        winner_index, rewards, used = select_winner(source, candidate_images, instruction, scorer, candidate_index=candidate_index)
        used_reward = used_reward or used
        final_rewards = rewards
        winner_state = candidates[winner_index]
        records.append({"stage": stage, "branch_step_index": step_index, "post_branch_step_index": step_index + 1, "winner_index": winner_index, "rewards": rewards, "used_reward": used, "branch_state_hash": tensor_hash(current), "candidate_diagnostics": diagnostics, "sigma": sigma, "sigma_next": sigma_next})
        # Cache the edited velocity at the branch step as the paired velocity
        # for later strength interpolation; the actual state transition here
        # is the selected SDE candidate rather than an Euler step.
        trace_velocities.append(pred.detach().clone())
        trace_timesteps.append(float(timestep))
        trace_sigmas.append((sigma, sigma_next))
        current = winner_state
        trace_states.append(current.detach().clone())
        current_step = step_index + 1
    while current_step < len(state.timesteps):
        timestep = state.timesteps[current_step]
        pred = velocity(pipe, state, current, timestep)
        sigma, sigma_next = _sigmas(pipe, timestep)
        trace_velocities.append(pred.detach().clone()); trace_timesteps.append(float(timestep)); trace_sigmas.append((sigma, sigma_next))
        current = (current.float() + (sigma_next - sigma) * pred.float()).to(current.dtype)
        trace_states.append(current.detach().clone()); current_step += 1
    trace = TrajectoryTrace(str(state.metadata.get("prompt", "")), trace_states, trace_velocities, trace_timesteps, trace_sigmas, current.detach().clone())
    return trace, records, winner_index, final_rewards, used_reward, all_candidate_images


@torch.inference_mode()
def rollout_strengths(
    pipe: Any,
    preservation: TrajectoryTrace,
    winner: TrajectoryTrace,
    strengths: Sequence[float],
) -> dict[float, torch.Tensor]:
    if len(preservation.velocities) != len(winner.velocities):
        raise ValueError("preservation and winner traces must have equal length")
    output: dict[float, torch.Tensor] = {}
    for strength in strengths:
        s = float(strength)
        if not 0.0 <= s <= 1.0:
            raise ValueError("strength must lie in [0, 1]")
        if s == 0.0:
            output[s] = preservation.terminal.detach().clone()
            continue
        if s == 1.0:
            output[s] = winner.terminal.detach().clone()
            continue
        current = preservation.states[0].detach().clone()
        for i, (v_preserve, v_full) in enumerate(zip(preservation.velocities, winner.velocities)):
            blended = v_preserve.float() + s * (v_full.float() - v_preserve.float())
            sigma, sigma_next = preservation.sigmas[i]
            current = (current.float() + (sigma_next - sigma) * blended).to(current.dtype)
        output[s] = current.detach().clone()
    return output


@torch.inference_mode()
def build_bundle(
    pipe: Any,
    source: Image.Image,
    instruction: str,
    decode: Callable[[KontextState, torch.Tensor], Sequence[Image.Image]],
    scorer: RewardScorer | None,
    *,
    seed: int,
    config: ContinuousStrengthConfig | None = None,
    candidate_index: int | None = None,
) -> TrajectoryBundle:
    cfg = config or ContinuousStrengthConfig()
    edited_state = prepare_state(pipe, source, instruction, seed, height=cfg.height, width=cfg.width, steps=cfg.steps, guidance_scale=cfg.guidance_scale, device=pipe._execution_device)
    preservation_state = prepare_state(pipe, source, cfg.neutral_prompt, seed, height=cfg.height, width=cfg.width, steps=cfg.steps, guidance_scale=cfg.guidance_scale, device=pipe._execution_device)
    _same_initial_latents(edited_state, preservation_state)
    preservation = deterministic_trace(pipe, preservation_state)
    edited = deterministic_trace(pipe, edited_state)
    critical = [int(x["index"]) for x in critical_nonzero_steps(pipe.scheduler.sigmas.detach().cpu().flatten().tolist())[: cfg.critical_steps]]
    token_mask, mask_scores = estimate_edit_token_mask(preservation, edited, critical, quantile=cfg.mask_quantile, min_ratio=cfg.min_edit_ratio, max_ratio=cfg.max_edit_ratio)
    winner, records, winner_index, rewards, used_reward, branch_images = generate_early_branches(pipe, edited_state, token_mask, source, instruction, decode, scorer, seed=seed + 10000, critical_count=cfg.critical_steps, alpha=cfg.alpha, diffusion_scale=cfg.diffusion_scale, coupling_strength=cfg.coupling_strength, candidate_index=candidate_index)
    metadata = {"seed": int(seed), "instruction": instruction, "neutral_prompt": cfg.neutral_prompt, "config": asdict(cfg), "critical_indices": critical, "generated_tokens": int(edited_state.metadata["generated_tokens"]), "source_conditioning_tokens": int(edited_state.metadata["source_conditioning_tokens"]), "token_mask_length": int(token_mask.numel()), "token_mask": token_mask.cpu().tolist(), "mask_scores": mask_scores.cpu().tolist(), "used_reward": used_reward, "reward_selected": used_reward and candidate_index is None, "initial_state_hash": tensor_hash(edited_state.latents), "preservation": preservation.metadata(), "edited": edited.metadata(), "winner": winner.metadata()}
    return TrajectoryBundle(preservation, edited, winner, token_mask, mask_scores, records, winner_index, rewards, metadata, branch_images)


def load_reward_factory(spec: str) -> RewardScorer:
    if ":" not in spec:
        raise ValueError("reward factory must use module:attribute syntax")
    module_name, attribute = spec.split(":", 1)
    obj = getattr(importlib.import_module(module_name), attribute)
    scorer = obj() if callable(obj) else obj
    if not hasattr(scorer, "score"):
        scorer = CallableRewardScorer(scorer)
    return scorer


def save_bundle_metadata(bundle: TrajectoryBundle, path: str | Path) -> None:
    payload = dict(bundle.metadata)
    payload.update({"winner_index": bundle.winner_index, "rewards": bundle.rewards, "branch_records": bundle.branch_records})
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def save_bundle_tensors(bundle: TrajectoryBundle, path: str | Path) -> None:
    torch.save({"preservation_states": bundle.preservation.states, "preservation_velocities": bundle.preservation.velocities, "winner_states": bundle.winner.states, "winner_velocities": bundle.winner.velocities, "token_mask": bundle.token_mask.cpu(), "mask_scores": bundle.mask_scores.cpu()}, path)


__all__ = ["CallableRewardScorer", "ContinuousStrengthConfig", "RewardScorer", "RewardUnavailable", "TrajectoryBundle", "TrajectoryTrace", "build_bundle", "deterministic_trace", "estimate_edit_token_mask", "generate_early_branches", "load_reward_factory", "rollout_strengths", "save_bundle_metadata", "save_bundle_tensors", "select_winner"]
