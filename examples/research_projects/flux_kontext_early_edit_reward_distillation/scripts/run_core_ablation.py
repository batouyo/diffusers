#!/usr/bin/env python3
"""Orchestrate the paired ablation through the shared continuous-strength sampler."""
from __future__ import annotations
import argparse, csv, json, sys
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from diffusers import FluxKontextPipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from early_edit_reward_distillation.continuous_strength import ContinuousStrengthConfig, build_bundle, rollout_strengths, save_bundle_metadata, save_bundle_tensors
from early_edit_reward_distillation.metrics import region_l1
from early_edit_reward_distillation.rewards import build_official_editscore

ARMS = {
    "velo_baseline": dict(enable_search=False, enable_reward=False, enable_coupling=False, independent_sde=False),
    "early_search": dict(enable_search=True, enable_reward=False, enable_coupling=False, independent_sde=False),
    "search_reward": dict(enable_search=True, enable_reward=True, enable_coupling=False, independent_sde=False),
    "independent_sde": dict(enable_search=True, enable_reward=True, enable_coupling=False, independent_sde=True),
    "full": dict(enable_search=True, enable_reward=True, enable_coupling=True, independent_sde=False),
}

@torch.inference_mode()
def decode(pipe, state, latents):
    unpacked = pipe._unpack_latents(latents, state.height, state.width, pipe.vae_scale_factor)
    unpacked = unpacked / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    image = pipe.vae.decode(unpacked.to(pipe.vae.dtype), return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]

@torch.inference_mode()
def decode_many(pipe, state, latents, batch_size=5):
    images = []
    for start in range(0, len(latents), max(1, int(batch_size))):
        batch = latents[start:start + max(1, int(batch_size))]
        unpacked = pipe._unpack_latents(batch, state.height, state.width, pipe.vae_scale_factor)
        unpacked = unpacked / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
        decoded = pipe.vae.decode(unpacked.to(pipe.vae.dtype), return_dict=False)[0]
        images.extend(pipe.image_processor.postprocess(decoded, output_type="pil"))

def sheet(images, path):
    cell = 192
    out = Image.new("RGB", (cell * len(images), cell + 24), "white")
    draw = ImageDraw.Draw(out)
    for i, (image, label) in enumerate(images):
        out.paste(image.convert("RGB").resize((cell, cell)), (i * cell, 0))
        draw.text((i * cell + 4, cell + 4), label, fill="black")
    out.save(path)

def load_records(paths, count):
    rows = []
    for path in paths:
        rows.extend(json.loads(Path(path).read_text(encoding="utf-8")))
    return rows[:count]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True, nargs="+")
    parser.add_argument("--samples-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--search-step-indices", default=None)
    parser.add_argument("--editscore-model", default="/data15/hyp/weight/Qwen3-VL-4B-Instruct")
    parser.add_argument("--editscore-lora", default="/data15/hyp/weight/EditScore-Qwen3-VL-4B-Instruct")
    parser.add_argument("--strength-batch-size", type=int, default=10)
    parser.add_argument("--vae-batch-size", type=int, default=5)
    parser.add_argument("--save-debug-tensors", action="store_true")
    parser.add_argument("--save-branch-diagnostics", action="store_true")
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-end", type=int, default=None)
    parser.add_argument("--methods", default=None, help="comma-separated arm names; unset runs all arms")
    args = parser.parse_args()
    records = load_records(args.manifest, args.count)[args.sample_start:args.sample_end]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = FluxKontextPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16, local_files_only=True).to(device)
    pipe.set_progress_bar_config(disable=True)
    selected_methods = list(ARMS) if args.methods is None else [x.strip() for x in args.methods.split(",") if x.strip()]
    unknown = [x for x in selected_methods if x not in ARMS]
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")
    scorer = build_official_editscore(args.editscore_model, args.editscore_lora, num_pass=1) if any(ARMS[x]["enable_reward"] for x in selected_methods) else None
    search_indices = None if args.search_step_indices is None else tuple(int(x) for x in args.search_step_indices.split(",") if x.strip())
    strengths = tuple(float(x) for x in np.linspace(0.0, 1.0, 10))
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True); timings = []
    all_rows = []
    for row_index, record in enumerate(records):
        sample_id = str(record["sample_id"]); sample_dir = Path(args.samples_root) / sample_id
        source = Image.open(sample_dir / "source.png").convert("RGB")
        pixel_mask = Image.open(sample_dir / "edit_mask.png").convert("L")
        instruction = str(record["instruction"]); seed = int(args.seed + row_index * 100)
        for method in selected_methods:
            method_t0 = time.perf_counter()
            switches = ARMS[method]
            cfg = ContinuousStrengthConfig(steps=30, guidance_scale=2.5, alpha=args.alpha, intervention_step_count=4, search_step_indices=search_indices, strengths=strengths, **switches)
            bundle = build_bundle(pipe, source, instruction, lambda state, x: [decode(pipe, state, x)], scorer, seed=seed, config=cfg, candidate_index=0 if not switches["enable_reward"] else None)
            values = rollout_strengths(pipe, bundle.preservation, bundle.winner, strengths, preservation_state=bundle.preservation_state, edited_state=bundle.edited_state, intervention_step_count=cfg.intervention_step_count, search_step_indices=bundle.metadata.get("search_step_indices"), similarity_threshold=cfg.similarity_threshold, similarity_mode=cfg.similarity_mode, strength_batch_size=args.strength_batch_size)
            method_dir = out / method / sample_id; method_dir.mkdir(parents=True, exist_ok=True)
            source.save(method_dir / "source.png"); pixel_mask.save(method_dir / "edit_mask.png")
            rendered = []
            for strength, latent in values.items():
                image = decode(pipe, bundle.edited_state, latent); image.save(method_dir / f"strength_{strength:.2f}.png"); rendered.append((image, f"s={strength:.2f}"))
                all_rows.append({"sample_id": sample_id, "method": method, "strength": strength, "edit_l1": region_l1(source, image, pixel_mask, False), "preserve_l1": region_l1(source, image, pixel_mask, True), "seed": seed, "winner_index": bundle.winner_index, "instruction": instruction})
            sheet(rendered, method_dir / "contact_sheet.png")
            save_bundle_metadata(bundle, method_dir / "trajectory.json"); (save_bundle_tensors(bundle, method_dir / "trajectory.pt") if args.save_debug_tensors else None); timings.append({"sample_id": sample_id, "method": method, "total_seconds": time.perf_counter() - method_t0})
            (method_dir / "branch_scores.json").write_text(json.dumps(bundle.branch_records, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
            for stage, images in (enumerate(bundle.branch_images, 1) if args.save_branch_diagnostics else []):
                stage_dir = method_dir / "branch_candidates" / f"stage_{stage}"; stage_dir.mkdir(parents=True, exist_ok=True)
                for candidate, image in enumerate(images): image.save(stage_dir / f"branch_{candidate}.png")
    with (out / "paired_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    (out / "config.json").write_text(json.dumps({"arms": ARMS, "seed": args.seed, "strengths": list(strengths), "search_step_indices": search_indices, "intervention_step_count": 4}, indent=2) + "\n", encoding="utf-8")
    Path(out / "timing.json").write_text(json.dumps(timings, indent=2) + "\n")
    print(json.dumps({"output": str(out), "samples": len(records), "rows": len(all_rows), "timing": timings}, indent=2))

if __name__ == "__main__": main()
