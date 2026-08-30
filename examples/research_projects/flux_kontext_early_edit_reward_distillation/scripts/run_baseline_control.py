#!/usr/bin/env python3
"""Native FLUX-Kontext baseline control at a requested resolution."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import torch
from PIL import Image
from diffusers import FluxKontextPipeline

from early_edit_reward_distillation.trajectory import deterministic_rollout, prepare_state


@torch.inference_mode()
def decode(pipe, state, latents):
    unpacked = pipe._unpack_latents(latents, state.height, state.width, pipe.vae_scale_factor)
    unpacked = unpacked / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    decoded = pipe.vae.decode(unpacked.to(pipe.vae.dtype), return_dict=False)[0]
    return pipe.image_processor.postprocess(decoded, output_type="pil")[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--sample-dir", type=Path, required=True)
    p.add_argument("--instruction", required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--height", type=int, default=1024); p.add_argument("--width", type=int, default=1024)
    p.add_argument("--steps", type=int, default=28); p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--seed", type=int, default=20260830); p.add_argument("--device", default="cuda")
    p.add_argument("--editscore-model", default="/data15/hyp/weight/Qwen3-VL-4B-Instruct")
    p.add_argument("--editscore-lora", default="/data15/hyp/weight/EditScore-Qwen3-VL-4B-Instruct")
    args = p.parse_args(); device = torch.device(args.device)
    source = Image.open(args.sample_dir / "source.png").convert("RGB")
    pipe = FluxKontextPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16, local_files_only=True).to(device)
    pipe.set_progress_bar_config(disable=True)
    state = prepare_state(pipe, source, args.instruction, args.seed, height=args.height, width=args.width, steps=args.steps, guidance_scale=args.guidance, device=device)
    terminal = deterministic_rollout(pipe, state, state.latents, 0)
    image = decode(pipe, state, terminal)
    sys.path.insert(0, "/home/hyp/Code/EditScore")
    from editscore import EditScore
    scorer = EditScore(backbone="qwen3vl", model_name_or_path=args.editscore_model, lora_path=args.editscore_lora, score_range=25, num_pass=1)
    result = scorer.evaluate([source, image], args.instruction)
    args.output.mkdir(parents=True, exist_ok=True); source.save(args.output / "source.png"); image.save(args.output / "baseline.png")
    payload = {"state": state.metadata, "resolution_mode": "explicit_target_and_source", "editscore": {key: (float(value) if hasattr(value, "item") else value) for key, value in result.items()}, "output": str(args.output)}
    (args.output / "baseline_control.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
