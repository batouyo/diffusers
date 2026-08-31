#!/usr/bin/env python3
"""Run the single-sample mask x Reward preservation diagnostic.

This diagnostic keeps the sampler unchanged.  The GT branch replaces only the
similarity map at runtime, and the preservation-aware scorer adds an outside-mask
pixel L1 penalty to the official EditScore.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from diffusers import FluxKontextPipeline

import early_edit_reward_distillation.continuous_strength as cs
from early_edit_reward_distillation.continuous_strength import (
    ContinuousStrengthConfig,
    build_bundle,
    save_bundle_metadata,
)
from early_edit_reward_distillation.rewards import build_official_editscore


def decode_fn(pipe, state, latents):
    unpacked = pipe._unpack_latents(
        latents, state.height, state.width, pipe.vae_scale_factor
    )
    unpacked = unpacked / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    image = pipe.vae.decode(unpacked.to(pipe.vae.dtype), return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")


class PreservationAwareReward:
    def __init__(self, base, source, edit_mask, penalty_weight):
        self.base = base
        self.source = source
        self.edit_mask = edit_mask
        self.penalty_weight = float(penalty_weight)

    def score(self, source, candidate, instruction):
        return self.score_many(source, [candidate], instruction)[0]

    def score_many(self, source, candidates, instruction):
        source_np = np.asarray(source.convert("RGB"), dtype=np.float32)
        outside = np.asarray(
            self.edit_mask.resize(source.size, Image.Resampling.NEAREST),
            dtype=np.float32,
        ) < 128
        scores = []
        for candidate in candidates:
            raw = float(self.base.score(source, candidate, instruction))
            candidate_np = np.asarray(candidate.convert("RGB"), dtype=np.float32)
            outside_l1 = (
                float(np.abs(candidate_np - source_np).mean(axis=2)[outside].mean() / 255.0)
                if outside.any()
                else 0.0
            )
            scores.append(raw - self.penalty_weight * outside_l1)
        return scores


def install_gt_similarity(pixel_mask):
    original = cs.velocity_similarity
    grid_mask = np.asarray(
        pixel_mask.resize((32, 32), Image.Resampling.NEAREST), dtype=np.uint8
    ) > 127

    def fixed_similarity(v_edit, v_ref, mode="elementwise"):
        mask = torch.as_tensor(
            grid_mask, device=v_edit.device, dtype=torch.bool
        ).reshape(1, -1, 1)
        mask = mask.expand(v_edit.shape[0], v_edit.shape[1], v_edit.shape[2])
        return torch.where(
            mask,
            torch.zeros_like(v_edit, dtype=torch.float32),
            torch.ones_like(v_edit, dtype=torch.float32),
        )

    cs.velocity_similarity = fixed_similarity
    return original


def summarize_records(bundle):
    stages = []
    for record in bundle.branch_records:
        diagnostics = record.get("candidate_diagnostics", [])
        stages.append(
            {
                "branch_step_index": record.get("branch_step_index"),
                "winner_index": record.get("winner_index"),
                "unique_corrected_state_hashes": len(
                    {item.get("corrected_state_hash") for item in diagnostics}
                ),
                "delta_residual_norms": [
                    item.get("delta_residual_norm") for item in diagnostics
                ],
                "preserve_shared_correlation": record.get(
                    "preserve_shared_correlation"
                ),
                "edit_independent_correlation": record.get(
                    "edit_independent_correlation"
                ),
            }
        )
    return stages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data15/hyp/weight/FLUX.1-Kontext-dev")
    parser.add_argument("--source", required=True)
    parser.add_argument("--edit-mask", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--editscore-model", default="/data15/hyp/weight/Qwen3-VL-4B-Instruct"
    )
    parser.add_argument(
        "--editscore-lora",
        default="/data15/hyp/weight/EditScore-Qwen3-VL-4B-Instruct",
    )
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=2.5)
    parser.add_argument("--penalty-weight", type=float, default=5.0)
    args = parser.parse_args()

    source = Image.open(args.source).convert("RGB")
    pixel_mask = Image.open(args.edit_mask).convert("L")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    pipe = FluxKontextPipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    def decode(state, latents):
        return decode_fn(pipe, state, latents)

    base_reward = build_official_editscore(
        args.editscore_model, args.editscore_lora, num_pass=1
    )
    aware_reward = PreservationAwareReward(
        base_reward, source, pixel_mask, args.penalty_weight
    )
    original_similarity = cs.velocity_similarity
    results = []

    try:
        for mask_name in ("dynamic", "gt"):
            if mask_name == "gt":
                install_gt_similarity(pixel_mask)
            else:
                cs.velocity_similarity = original_similarity

            for reward_name, scorer in (
                ("original", base_reward),
                ("preservation_aware", aware_reward),
            ):
                start = time.perf_counter()
                config = ContinuousStrengthConfig(
                    height=args.height,
                    width=args.width,
                    steps=args.steps,
                    guidance_scale=args.guidance,
                    first_step_align_steps=4,
                    enable_search=True,
                    enable_reward=True,
                    enable_coupling=True,
                    strengths=(1.0,),
                )
                bundle = build_bundle(
                    pipe,
                    source,
                    args.instruction,
                    decode,
                    scorer,
                    seed=args.seed,
                    config=config,
                )
                case = output / f"{mask_name}_{reward_name}"
                case.mkdir(parents=True, exist_ok=True)
                source.save(case / "source.png")
                winner = decode(bundle.edited_state, bundle.winner.terminal)[0]
                winner.save(case / "winner.png")
                save_bundle_metadata(bundle, case / "trajectory.json")
                (case / "branch_scores.json").write_text(
                    json.dumps(bundle.branch_records, indent=2, default=str) + "\n",
                    encoding="utf-8",
                )
                results.append(
                    {
                        "mask": mask_name,
                        "reward": reward_name,
                        "winner_index": bundle.winner_index,
                        "rewards": bundle.rewards,
                        "reward_selected": bundle.metadata["reward_selected"],
                        "seconds": time.perf_counter() - start,
                        "stages": summarize_records(bundle),
                    }
                )
                print(json.dumps(results[-1]), flush=True)
    finally:
        cs.velocity_similarity = original_similarity

    (output / "config.json").write_text(
        json.dumps(
            {
                "source": args.source,
                "edit_mask": args.edit_mask,
                "instruction": args.instruction,
                "seed": args.seed,
                "height": args.height,
                "width": args.width,
                "steps": args.steps,
                "guidance": args.guidance,
                "first_step_align_steps": 4,
                "penalty_weight": args.penalty_weight,
                "mask_modes": ["dynamic", "gt"],
                "reward_modes": ["original", "preservation_aware"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps({"results": results}, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
