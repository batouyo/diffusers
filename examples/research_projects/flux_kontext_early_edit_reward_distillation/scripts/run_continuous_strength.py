#!/usr/bin/env python3
"""Run the minimal Training-Free FLUX-Kontext continuous-strength prototype."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from diffusers import FluxKontextPipeline

from early_edit_reward_distillation.continuous_strength import (
    ContinuousStrengthConfig,
    build_bundle,
    load_reward_factory,
    rollout_strengths,
    save_bundle_metadata,
    save_bundle_tensors,
)
from early_edit_reward_distillation.metrics import region_l1


@torch.inference_mode()
def decode(pipe, state, latents):
    unpacked = pipe._unpack_latents(latents, state.height, state.width, pipe.vae_scale_factor)
    unpacked = unpacked / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    image = pipe.vae.decode(unpacked.to(pipe.vae.dtype), return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")


def load_records(args):
    if args.manifest:
        records = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        records = records[: args.count]
        if not args.samples_root:
            raise ValueError("--samples-root is required with --manifest")
        return [(str(r["sample_id"]), Path(args.samples_root) / str(r["sample_id"]), str(r["instruction"])) for r in records]
    if not args.source or not args.instruction:
        raise ValueError("single-sample mode requires --source and --instruction")
    return [(Path(args.source).stem, Path(args.source), args.instruction)]


def save_contact_sheet(images: dict[str, Image.Image], path: Path) -> None:
    width, height = 256, 256
    sheet = Image.new("RGB", (width * len(images), height + 28), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(images.items()):
        sheet.paste(image.convert("RGB").resize((width, height)), (index * width, 0))
        draw.text((index * width + 5, height + 7), label, fill="black")
    sheet.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--samples-root")
    parser.add_argument("--source")
    parser.add_argument("--instruction")
    parser.add_argument("--reward-factory", help="module:attribute returning a RewardScorer")
    parser.add_argument("--candidate-index", type=int)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--coupling-strength", type=float, default=0.0)
    parser.add_argument("--critical-step-indices", default=None, help="comma-separated scheduler transition indices; unset uses first non-zero transitions")
    parser.add_argument("--search-step-indices", default=None, help="comma-separated early SDE search transition indices")
    parser.add_argument("--intervention-step-count", type=int, default=4)
    parser.add_argument("--strengths", default="0,0.25,0.5,0.75,1")
    args = parser.parse_args()
    if args.reward_factory and args.candidate_index is not None:
        raise ValueError("choose --reward-factory or --candidate-index, not both")
    scorer = load_reward_factory(args.reward_factory) if args.reward_factory else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = FluxKontextPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16, local_files_only=True).to(device)
    pipe.set_progress_bar_config(disable=True)
    critical_indices = None if args.critical_step_indices is None else tuple(int(x) for x in args.critical_step_indices.split(",") if x.strip())
    search_indices = None if args.search_step_indices is None else tuple(int(x) for x in args.search_step_indices.split(",") if x.strip())
    config = ContinuousStrengthConfig(height=args.height, width=args.width, steps=args.steps, guidance_scale=args.guidance, alpha=args.alpha, coupling_strength=args.coupling_strength, critical_step_indices=critical_indices, search_step_indices=search_indices, intervention_step_count=args.intervention_step_count, strengths=tuple(float(x) for x in args.strengths.split(",")))
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    rows = []
    for record_index, (sample_id, sample_dir, instruction) in enumerate(load_records(args)):
        source_path = sample_dir / "source.png" if sample_dir.is_dir() and (sample_dir / "source.png").exists() else sample_dir
        mask_path = sample_dir / "edit_mask.png" if sample_dir.is_dir() else None
        source = Image.open(source_path).convert("RGB")
        pixel_mask = Image.open(mask_path).convert("L") if mask_path is not None and mask_path.exists() else Image.new("L", source.size, 255)
        sample_output = output / sample_id
        sample_output.mkdir(parents=True, exist_ok=True)
        decode_fn = lambda state, latents: decode(pipe, state, latents)
        bundle = build_bundle(pipe, source, instruction, decode_fn, scorer, seed=args.seed + record_index * 100, config=config, candidate_index=args.candidate_index)
        strengths = rollout_strengths(pipe, bundle.preservation, bundle.winner, config.strengths, preservation_state=bundle.preservation_state, edited_state=bundle.edited_state, intervention_step_count=config.intervention_step_count, search_step_indices=bundle.metadata.get("search_step_indices"), similarity_threshold=config.similarity_threshold, similarity_mode=config.similarity_mode)
        source.save(sample_output / "source.png")
        pixel_mask.save(sample_output / "edit_mask.png")
        mask = Image.fromarray((bundle.token_mask.cpu().numpy().astype(np.uint8) * 255).reshape(1, -1), mode="L")
        mask.save(sample_output / "edit_mask_tokens.png")
        branch_root = sample_output / "branch_candidates"
        for stage_index, stage_images in enumerate(bundle.branch_images, start=1):
            stage_dir = branch_root / f"stage_{stage_index}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            for candidate_index, image in enumerate(stage_images):
                image.save(stage_dir / f"branch_{candidate_index}.png")
        preservation_image = decode(pipe, prepare_state_for_decode(bundle, config), bundle.preservation.terminal)[0]
        winner_image = decode(pipe, prepare_state_for_decode(bundle, config), bundle.winner.terminal)[0]
        preservation_image.save(sample_output / "preservation.png")
        winner_image.save(sample_output / "winner_full_edit.png")
        contact = {"source": source, "preserve": preservation_image, "winner": winner_image}
        for strength, latent in strengths.items():
            image = decode(pipe, prepare_state_for_decode(bundle, config), latent)[0]
            image.save(sample_output / f"strength_{strength:.2f}.png")
            contact[f"s={strength:.2f}"] = image
            rows.append({"sample_id": sample_id, "strength": strength, "reward_selected": bundle.metadata["reward_selected"], "winner_index": bundle.winner_index, "preserve_l1": region_l1(source, image, pixel_mask, preserve=True), "edit_l1": region_l1(source, image, pixel_mask, preserve=False), "instruction": instruction})
        save_bundle_metadata(bundle, sample_output / "trajectory.json")
        save_bundle_tensors(bundle, sample_output / "trajectory.pt")
        (sample_output / "branch_scores.json").write_text(json.dumps(bundle.branch_records, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        save_contact_sheet(contact, sample_output / "contact_sheet.png")
    with (output / "strength_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"output": str(output), "samples": len(load_records(args)), "rows": len(rows)}, ensure_ascii=False, indent=2))


def prepare_state_for_decode(bundle, config):
    """Decode helper: the trace terminal already carries the required geometry."""
    return type("DecodeState", (), {"height": config.height, "width": config.width})()


if __name__ == "__main__":
    main()
