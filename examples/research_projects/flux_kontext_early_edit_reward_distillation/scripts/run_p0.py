#!/usr/bin/env python3
"""Run one real FLUX-Kontext P0 trajectory at an explicit target resolution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from diffusers import FluxKontextPipeline

from early_edit_reward_distillation.trajectory import deterministic_rollout, prepare_state, two_stage_search, velocity


@torch.inference_mode()
def decode(pipe, state, latents):
    unpacked = pipe._unpack_latents(latents, state.height, state.width, pipe.vae_scale_factor)
    unpacked = unpacked / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    decoded = pipe.vae.decode(unpacked.to(pipe.vae.dtype), return_dict=False)[0]
    return pipe.image_processor.postprocess(decoded, output_type="pil")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--branch-mode", choices=("native_euler_sde", "official_syncsde_reference"), default="native_euler_sde")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--diffusion-scale", type=float, default=1.0)
    parser.add_argument("--alpha-probe", default=None, help="Comma-separated native alpha values; runs independent searches")
    parser.add_argument("--editscore-backbone", default="qwen3vl")
    parser.add_argument("--editscore-model", default="/data15/hyp/weight/Qwen3-VL-4B-Instruct")
    parser.add_argument("--editscore-lora", default="/data15/hyp/weight/EditScore-Qwen3-VL-4B-Instruct")
    args = parser.parse_args()
    device = torch.device(args.device)
    source = Image.open(args.sample_dir / "source.png").convert("RGB")
    mask = Image.open(args.sample_dir / "edit_mask.png").convert("L")
    pipe = FluxKontextPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16, local_files_only=True).to(device)
    pipe.set_progress_bar_config(disable=True)
    state = prepare_state(pipe, source, args.instruction, args.seed, height=args.height, width=args.width, steps=args.steps, guidance_scale=args.guidance, device=device)
    token_h = state.height // (pipe.vae_scale_factor * 2)
    token_w = state.width // (pipe.vae_scale_factor * 2)
    token_mask = torch.nn.functional.interpolate(torch.from_numpy(__import__("numpy").asarray(mask, dtype="float32"))[None, None], size=(token_h, token_w), mode="area")[0, 0].flatten().to(device) > 0.5
    baseline = deterministic_rollout(pipe, state, state.latents, 0)
    baseline_image = decode(pipe, state, baseline)[0]

    try:
        import sys
        sys.path.insert(0, "/home/hyp/Code/EditScore")
        from editscore import EditScore
        scorer = EditScore(backbone=args.editscore_backbone, model_name_or_path=args.editscore_model, lora_path=args.editscore_lora, score_range=25, num_pass=1)
    except Exception as exc:
        raise RuntimeError(f"official EditScore failed to load; P0 reward run is stopped: {type(exc).__name__}: {exc}") from exc

    def score(candidates):
        return [float(scorer.evaluate([source, image], args.instruction)["overall"]) for image in decode(pipe, state, candidates)]

    def repeat_score(top2):
        images = decode(pipe, state, top2)
        values = []
        for rank, image in enumerate(images):
            for repeat in range(2):
                old_seed = scorer.seed
                scorer.seed = int(args.seed + 1000 + rank * 10 + repeat)
                values.append(float(scorer.evaluate([source, image], args.instruction)["overall"]))
                scorer.seed = old_seed
        return values

    probe_values = [float(item) for item in args.alpha_probe.split(",")] if args.alpha_probe else [args.alpha]
    probe_results = []
    winner = None
    records = None
    selected_alpha = None
    best_reward = float("-inf")
    for alpha in probe_values:
        alpha_dir = args.output / f"alpha_{alpha:g}"
        def save_stage(stage, step_index, terminal, rewards, alpha_dir=alpha_dir):
            stage_dir = alpha_dir / f"stage_{stage:02d}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            for index, image in enumerate(decode(pipe, state, terminal)):
                image.save(stage_dir / f"candidate_{index:02d}_reward_{rewards[index]:.4f}.png")
        candidate_winner, candidate_records = two_stage_search(pipe, state, token_mask, score, seed=args.seed, repeat_score=repeat_score, mode=args.branch_mode, alpha=alpha, diffusion_scale=args.diffusion_scale, stage_callback=save_stage)
        final_reward = float(candidate_records[-1]["mean_rewards"][candidate_records[-1]["winner_index"]])
        probe_results.append({"alpha": alpha, "final_reward": final_reward, "records": candidate_records})
        if final_reward > best_reward:
            winner, records, selected_alpha, best_reward = candidate_winner, candidate_records, alpha, final_reward
    winner_image = decode(pipe, state, winner)[0]
    args.output.mkdir(parents=True, exist_ok=True)
    source.save(args.output / "source.png")
    mask.save(args.output / "edit_mask.png")
    baseline_image.save(args.output / "baseline.png")
    winner_image.save(args.output / "winner.png")
    payload = {"state": state.metadata, "token_mask_area": float(token_mask.float().mean().item()), "reward_model": "official_qwen3_vl_4b_editscore", "branch_mode": args.branch_mode, "alpha": selected_alpha, "diffusion_scale": args.diffusion_scale, "alpha_probe": probe_values, "records": records, "probe_results": probe_results, "output": str(args.output)}
    (args.output / "trajectory.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
