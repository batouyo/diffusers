"""Model-backed FLUX-Kontext trajectory capture and regional SDE search.

The module deliberately keeps the pipeline's native source-image conditioning:
the generated tokens are the only tokens branched or masked.  The target
resolution is explicit because Kontext's default source auto-resize is not the
same operation as the generated image resolution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image

from .core import coupled_noise, critical_nonzero_steps, native_euler_sde_step, noise_correlations, regional_delta_norms, rf_sde_step, tensor_hash
from .resolution import resolve_dimensions


@dataclass
class KontextState:
    latents: torch.Tensor
    image_latents: torch.Tensor
    image_ids: torch.Tensor
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    text_ids: torch.Tensor
    timesteps: torch.Tensor
    height: int
    width: int
    dtype: torch.dtype
    metadata: dict[str, Any]


def _schedule(pipe: Any, steps: int, device: torch.device, image_tokens: int) -> tuple[torch.Tensor, float]:
    from diffusers.pipelines.flux.pipeline_flux_kontext import calculate_shift, retrieve_timesteps

    sigmas = np.linspace(1.0, 1.0 / steps, steps)
    config = pipe.scheduler.config
    mu = calculate_shift(
        image_tokens,
        config.get("base_image_seq_len", 256),
        config.get("max_image_seq_len", 4096),
        config.get("base_shift", 0.5),
        config.get("max_shift", 1.15),
    )
    timesteps, _ = retrieve_timesteps(pipe.scheduler, steps, device, sigmas=sigmas, mu=mu)
    return timesteps, float(mu)


@torch.inference_mode()
def prepare_state(
    pipe: Any,
    source: Image.Image,
    instruction: str,
    seed: int,
    *,
    height: int = 512,
    width: int = 512,
    steps: int = 28,
    guidance_scale: float = 3.5,
    first_step_align_steps: int = 4,
    device: torch.device | str = "cuda",
) -> KontextState:
    device = torch.device(device)
    geometry = resolve_dimensions(height, width, int(pipe.vae_scale_factor))
    height, width = geometry["resolved_height"], geometry["resolved_width"]
    source = source.convert("RGB")
    source_tensor = pipe.image_processor.preprocess(
        pipe.image_processor.resize(source, height, width), height, width
    )
    prompt_embeds, pooled, text_ids = pipe.encode_prompt(
        prompt=instruction, device=device, num_images_per_prompt=1, max_sequence_length=512
    )
    channels = pipe.transformer.config.in_channels // 4
    generator = torch.Generator(device=device).manual_seed(int(seed))
    latents, image_latents, latent_ids, image_ids = pipe.prepare_latents(
        source_tensor, 1, channels, height, width, prompt_embeds.dtype, device, generator, None
    )
    if image_latents is None or image_ids is None:
        raise RuntimeError("FLUX-Kontext did not return source image conditioning latents")
    all_image_ids = torch.cat([latent_ids, image_ids], dim=0)
    timesteps, scheduler_mu = _schedule(pipe, steps, device, latents.shape[1])
    scheduler_sigmas = pipe.scheduler.sigmas.detach().cpu().flatten().tolist()
    raw_delta_sigma = float(scheduler_sigmas[1] - scheduler_sigmas[0])
    align_steps = max(1, int(first_step_align_steps))
    reference_delta_sigma = float(-1.0 / align_steps) if align_steps > 1 else raw_delta_sigma
    aligned_sigmas = list(scheduler_sigmas)
    if align_steps > 1 and len(aligned_sigmas) > 2:
        aligned_sigmas = [aligned_sigmas[0], aligned_sigmas[0] + reference_delta_sigma] + [float(x) for x in np.linspace(aligned_sigmas[0] + reference_delta_sigma, scheduler_sigmas[-1], len(scheduler_sigmas) - 1)[1:]]
    guidance = torch.full((latents.shape[0],), guidance_scale, device=device, dtype=torch.float32)
    return KontextState(
        latents=latents, image_latents=image_latents, image_ids=all_image_ids,
        prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled, text_ids=text_ids,
        timesteps=timesteps, height=height, width=width, dtype=latents.dtype,
        metadata={"seed": int(seed), "guidance_scale": float(guidance_scale), "steps": int(steps),
                  "resolution": geometry, "source_original_size": [source.width, source.height],
                  "scheduler_mu": scheduler_mu,
                  "first_step_alignment": {
                      "enabled": bool(align_steps > 1),
                      "reference_steps": align_steps,
                      "raw_delta_sigma": raw_delta_sigma,
                      "aligned_delta_sigma": reference_delta_sigma,
                      "reference_delta_sigma": reference_delta_sigma, "aligned_sigmas": aligned_sigmas,
                  },
                  "generated_tokens": int(latents.shape[1]),
                  "source_conditioning_tokens": int(image_latents.shape[1]),
                  "text_tokens": int(prompt_embeds.shape[1])},
    )


@torch.inference_mode()
def velocity(pipe: Any, state: KontextState, latents: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    batch = latents.shape[0]
    image_latents = state.image_latents.repeat(batch, 1, 1)
    image_ids = state.image_ids
    if image_ids.shape[0] != state.latents.shape[1] + state.image_latents.shape[1]:
        raise RuntimeError("image token ID layout is inconsistent with generated/source tokens")
    output = pipe.transformer(
        hidden_states=torch.cat([latents, image_latents], dim=1),
        timestep=timestep.expand(batch).to(latents.dtype) / 1000,
        guidance=torch.full((batch,), float(state.metadata["guidance_scale"]), device=latents.device),
        pooled_projections=state.pooled_prompt_embeds.repeat(batch, 1),
        encoder_hidden_states=state.prompt_embeds.repeat(batch, 1, 1),
        txt_ids=state.text_ids,
        img_ids=image_ids,
        joint_attention_kwargs={},
        return_dict=False,
    )[0]
    return output[:, : latents.shape[1]]


def _sigmas(pipe: Any, timestep: torch.Tensor, state: KontextState | None = None) -> tuple[float, float]:
    index = int(pipe.scheduler.index_for_timestep(timestep))
    alignment = state.metadata.get("first_step_alignment") if state is not None else None
    values = torch.tensor(alignment["aligned_sigmas"]) if alignment and alignment.get("enabled") and alignment.get("aligned_sigmas") else pipe.scheduler.sigmas.detach().cpu().flatten()
    sigma, sigma_next = float(values[index]), float(values[index + 1])
    return sigma, sigma_next


def ode_step(pipe: Any, latents: torch.Tensor, prediction: torch.Tensor, timestep: torch.Tensor, state: KontextState | None = None) -> torch.Tensor:
    sigma, next_sigma = _sigmas(pipe, timestep, state)
    return (latents.float() + (next_sigma - sigma) * prediction.float()).to(latents.dtype)


def branch_step(
    pipe: Any,
    state: KontextState,
    latents: torch.Tensor,
    step_index: int,
    token_mask: torch.Tensor,
    seed: int,
    *,
    candidates: int = 4,
    mode: str = "native_euler_sde",
    alpha: float = 0.05,
    diffusion_scale: float = 1.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if candidates != 4:
        raise ValueError("the minimal validation fixes K=4")
    timestep = state.timesteps[step_index]
    prediction = velocity(pipe, state, latents, timestep)
    sigma, sigma_next = _sigmas(pipe, timestep)
    generator = torch.Generator(device=latents.device).manual_seed(int(seed))
    shared = torch.randn(latents.shape, generator=generator, device=latents.device, dtype=torch.float32)
    independent = torch.randn((candidates,) + tuple(latents.shape[1:]), generator=generator, device=latents.device, dtype=torch.float32)
    token_mask = token_mask.reshape(1, -1)
    if token_mask.shape[1] != latents.shape[1]:
        raise ValueError(f"token mask length {token_mask.shape[1]} does not match generated tokens {latents.shape[1]}")
    noise = coupled_noise(shared.expand_as(independent), independent, token_mask, rho=0.0)
    if mode not in {"native_euler_sde", "official_syncsde_reference"}:
        raise ValueError(f"unknown branch mode: {mode}")
    outputs = []
    diagnostics = []
    mean = (latents.float() + (sigma_next - sigma) * prediction.float()).to(latents.dtype)
    for index in range(candidates):
        if mode == "native_euler_sde":
            candidate, diagnostic = native_euler_sde_step(latents, prediction, sigma, sigma_next, noise[index], alpha=alpha, diffusion_scale=diffusion_scale, first_step=step_index == 0)
        else:
            candidate, diagnostic = rf_sde_step(latents, prediction, sigma, sigma_next, noise[index], first_step=step_index == 0)
            diagnostic.update({"alpha": 1.0, "diffusion_scale": 1.0, "mean_norm": float(mean.float().norm().item()), "noise_norm": float(noise[index].norm().item()), "perturbation_norm": float((candidate.float() - mean.float()).norm().item()), "finite": bool(torch.isfinite(candidate).all().item())})
        outputs.append(candidate)
        diagnostics.append({**diagnostic, **regional_delta_norms(candidate, mean, token_mask[0]), **noise_correlations(shared, noise[index:index + 1], independent[index:index + 1], token_mask)})
    coeff = 0.0 if step_index == 0 else float((2 * sigma / (1 - sigma) * (sigma - sigma_next)) ** 0.5)
    return torch.cat(outputs, dim=0), {"step_index": int(step_index), "timestep": float(timestep), "sigma": sigma, "sigma_next": sigma_next, "diffusion_coeff": coeff, "seed": int(seed), "mode": mode, "alpha": float(alpha), "diffusion_scale": float(diffusion_scale), "branch_state_hash": tensor_hash(latents), "shared_noise_shape": list(shared.shape), "candidate_noise_shape": list(independent.shape), "candidate_diagnostics": diagnostics, "all_finite": all(item["finite"] for item in diagnostics)}


@torch.inference_mode()
def deterministic_rollout(pipe: Any, state: KontextState, candidates: torch.Tensor, start_step: int) -> torch.Tensor:
    current = candidates
    for index in range(start_step, len(state.timesteps)):
        current = ode_step(pipe, current, velocity(pipe, state, current, state.timesteps[index]), state.timesteps[index])
    return current


@torch.inference_mode()
def rollout_until(pipe: Any, state: KontextState, current: torch.Tensor, start_step: int, target_step: int) -> torch.Tensor:
    """Advance to the requested branch state without stepping past it."""
    for index in range(start_step, target_step):
        current = ode_step(pipe, current, velocity(pipe, state, current, state.timesteps[index]), state.timesteps[index])
    return current


@torch.inference_mode()
def two_stage_search(
    pipe: Any,
    state: KontextState,
    token_mask: torch.Tensor,
    score: Callable[[torch.Tensor], list[float]],
    *,
    seed: int,
    repeat_score: Callable[[torch.Tensor], list[float]] | None = None,
    mode: str = "native_euler_sde",
    alpha: float = 0.05,
    diffusion_scale: float = 1.0,
    stage_callback: Callable[[int, int, torch.Tensor, list[float]], None] | None = None,
    baseline_terminal: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    selected = critical_nonzero_steps([float(x) for x in pipe.scheduler.sigmas.detach().cpu().flatten().tolist()])[:2]
    if len(selected) < 2:
        raise RuntimeError("scheduler exposed fewer than two non-zero diffusion transitions")
    current = state.latents
    current_step = 0
    final_terminal = None
    records = []
    for stage, item in enumerate(selected, start=1):
        step_index = int(item["index"])
        current = rollout_until(pipe, state, current, current_step, step_index)
        if mode == "native_euler_sde" and alpha == 0.0 and baseline_terminal is not None:
            # Explicit identity path: do not re-run fused kernels for equivalent
            # zero-noise candidates, so their images are bitwise identical.
            candidates = current.repeat(4, 1, 1)
            terminal = baseline_terminal.repeat(4, 1, 1)
            sigma, sigma_next = _sigmas(pipe, state.timesteps[step_index])
            branch_meta = {"step_index": step_index, "timestep": float(state.timesteps[step_index]), "sigma": sigma, "sigma_next": sigma_next, "diffusion_coeff": 0.0 if step_index == 0 else float((2 * sigma / (1 - sigma) * (sigma - sigma_next)) ** 0.5), "seed": int(seed + stage), "mode": mode, "alpha": 0.0, "diffusion_scale": float(diffusion_scale), "branch_state_hash": tensor_hash(current), "zero_alpha_identity": True, "all_finite": True}
        else:
            candidates, branch_meta = branch_step(pipe, state, current, step_index, token_mask, seed + stage, mode=mode, alpha=alpha, diffusion_scale=diffusion_scale)
            terminal = deterministic_rollout(pipe, state, candidates, step_index + 1)
        rewards = [float(x) for x in score(terminal)]
        if len(rewards) != 4:
            raise ValueError("score must return one value per candidate")
        if stage_callback is not None:
            stage_callback(stage, step_index, terminal, rewards)
        top2 = sorted(range(4), key=lambda index: (-rewards[index], index))[:2]
        repeated = [float(x) for x in repeat_score(terminal[top2]) ] if repeat_score is not None else []
        means = rewards[:]
        if repeated:
            if len(repeated) != 4:
                raise ValueError("repeat_score must return two repeated scores per top-2 candidate")
            for rank, index in enumerate(top2):
                pair = repeated[2 * rank : 2 * rank + 2]
                means[index] = (rewards[index] + pair[0] + pair[1]) / 3.0
        winner = max(range(4), key=lambda index: (means[index], -index))
        current = candidates[winner:winner + 1]
        current_step = step_index + 1
        final_terminal = terminal[winner:winner + 1]
        records.append({"stage": stage, "branch_step_index": step_index, "post_branch_step_index": step_index + 1, "winner_index": winner, "rewards": rewards, "top2": top2, "repeated_rewards": repeated, "mean_rewards": means, **branch_meta})
    if final_terminal is None:
        raise RuntimeError("two-stage search did not produce a terminal state")
    return final_terminal, records
