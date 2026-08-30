#!/usr/bin/env python3
"""Build teacher cache from official EditScore-selected coupled trajectories."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from diffusers import FluxKontextPipeline
from early_edit_reward_distillation.cache import save_teacher_record
from early_edit_reward_distillation.core import critical_nonzero_steps
from early_edit_reward_distillation.trajectory import branch_step, prepare_state, rollout_until, velocity, deterministic_rollout

@torch.inference_mode()
def decode(pipe, state, latents):
    unpacked = pipe._unpack_latents(latents, state.height, state.width, pipe.vae_scale_factor)
    unpacked = unpacked / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    decoded = pipe.vae.decode(unpacked.to(pipe.vae.dtype), return_dict=False)[0]
    return pipe.image_processor.postprocess(decoded, output_type="pil")

def score_value(scorer, source, image, instruction, seed):
    old = getattr(scorer, "seed", None)
    if old is not None:
        scorer.seed = int(seed)
    result = scorer.evaluate([source, image], instruction)
    if old is not None:
        scorer.seed = old
    value = result["overall"] if isinstance(result, dict) else result
    return float(value.item() if hasattr(value, "item") else value)

@torch.inference_mode()
def select_winner(pipe, scorer, state, source, instruction, token_mask, seed, alpha):
    indices = [int(x["index"]) for x in critical_nonzero_steps(pipe.scheduler.sigmas.detach().cpu().flatten().tolist())[:2]]
    current, cur_step, records = state.latents, 0, []
    saved_states, saved_velocities = [], []
    for stage, idx in enumerate(indices):
        current = rollout_until(pipe, state, current, cur_step, idx)
        saved_states.append(current.detach().cpu())
        saved_velocities.append(velocity(pipe, state, current, state.timesteps[idx]).detach().cpu())
        candidates, branch_meta = branch_step(pipe, state, current, idx, token_mask, seed + stage, mode="native_euler_sde", alpha=alpha)
        terminals = [deterministic_rollout(pipe, state, candidates[i:i+1], idx + 1) for i in range(4)]
        images = [decode(pipe, state, x)[0] for x in terminals]
        rewards = [score_value(scorer, source, image, instruction, seed + stage * 1000 + i) for i, image in enumerate(images)]
        top2 = sorted(range(4), key=lambda i: (-rewards[i], i))[:2]
        repeated = {str(i): [score_value(scorer, source, images[i], instruction, seed + stage * 1000 + 100 + i * 10 + j) for j in range(2)] for i in top2}
        means = list(rewards)
        for i in top2:
            means[i] = float((rewards[i] + sum(repeated[str(i)])) / 3.0)
        winner = max(range(4), key=lambda i: (means[i], -i))
        records.append({"stage": stage, "branch_step_index": idx, "post_branch_step_index": idx + 1, "winner_index": winner, "rewards": rewards, "top2": top2, "repeated_rewards": repeated, "mean_rewards": means, "branch_metadata": branch_meta})
        current, cur_step = candidates[winner:winner+1], idx + 1
    return current, indices, saved_states, saved_velocities, records

@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--manifest", required=True); p.add_argument("--samples-root", required=True); p.add_argument("--output", required=True)
    p.add_argument("--editscore-model", required=True); p.add_argument("--editscore-lora", required=True)
    p.add_argument("--seed", type=int, default=20260830); p.add_argument("--height", type=int, default=512); p.add_argument("--width", type=int, default=512); p.add_argument("--steps", type=int, default=28); p.add_argument("--guidance", type=float, default=3.5); p.add_argument("--alpha", type=float, default=0.2); p.add_argument("--device", default="cuda")
    args = p.parse_args(); device = torch.device(args.device); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    pipe = FluxKontextPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16, local_files_only=True).to(device); pipe.set_progress_bar_config(disable=True)
    sys.path.insert(0, "/home/hyp/Code/EditScore")
    try:
        from editscore import EditScore
        scorer = EditScore(backbone="qwen3vl", model_name_or_path=args.editscore_model, lora_path=args.editscore_lora, score_range=25, num_pass=1)
    except Exception as exc:
        raise RuntimeError(f"official EditScore failed to load; cache generation stopped: {type(exc).__name__}: {exc}") from exc
    for ri, rec in enumerate(manifest):
        sid = str(rec["sample_id"]); folder = Path(args.samples_root) / sid; source = Image.open(folder / "source.png").convert("RGB"); mask = Image.open(folder / "edit_mask.png").convert("L"); seed = int(args.seed + ri * 100)
        state = prepare_state(pipe, source, str(rec["instruction"]), seed, height=args.height, width=args.width, steps=args.steps, guidance_scale=args.guidance, device=device)
        th, tw = state.height // (pipe.vae_scale_factor * 2), state.width // (pipe.vae_scale_factor * 2)
        token_mask = torch.nn.functional.interpolate(torch.from_numpy(np.asarray(mask, dtype="float32"))[None, None], size=(th, tw), mode="area")[0, 0].flatten().to(device) > 0.5
        winner, indices, base_states, base_vel, records = select_winner(pipe, scorer, state, source, str(rec["instruction"]), token_mask, seed + 10000, args.alpha)
        # Reconstruct winner states/velocities using the selected trajectory.
        win_states, win_vel = [], []; current, cur_step = state.latents, 0
        for stage, idx in enumerate(indices):
            current = rollout_until(pipe, state, current, cur_step, idx); win_states.append(current.detach().cpu()); win_vel.append(velocity(pipe, state, current, state.timesteps[idx]).detach().cpu())
            if stage < len(indices):
                candidates, _ = branch_step(pipe, state, current, idx, token_mask, seed + 10000 + stage, mode="native_euler_sde", alpha=args.alpha)
                current = candidates[records[stage]["winner_index"]:records[stage]["winner_index"]+1]; cur_step = idx + 1
        tensors = {}
        for i in range(2):
            tensors[f"baseline_state_t{i}"] = base_states[i][0]; tensors[f"winner_state_t{i}"] = win_states[i][0]; tensors[f"baseline_velocity_t{i}"] = base_vel[i][0]; tensors[f"teacher_velocity_t{i}"] = win_vel[i][0]; tensors[f"delta_velocity_t{i}"] = win_vel[i][0] - base_vel[i][0]
        tensors["token_mask"] = token_mask.cpu(); tensors.update({"image_latents": state.image_latents[0].cpu(), "image_ids": state.image_ids.cpu(), "prompt_embeds": state.prompt_embeds[0].cpu(), "pooled_prompt_embeds": state.pooled_prompt_embeds[0].cpu(), "text_ids": state.text_ids.cpu()})
        meta = {"instruction": str(rec["instruction"]), "seed": seed, "teacher_step_indices": indices, "branch_mode": "native_euler_sde", "alpha": args.alpha, "winner_selection": "official_editscore_two_stage_k4_top2_repeat", "reward_search_in_cache": True, "search_records": records, "state_metadata": state.metadata, "mask_area": float(rec.get("mask_area", 0.0))}
        save_teacher_record(out, sid, tensors, meta); print(json.dumps({"sample_id": sid, "indices": indices, "winner_indices": [r["winner_index"] for r in records]}), flush=True)

if __name__ == "__main__": main()
