#!/usr/bin/env python3
"""Run the gated P1 A/B/C/D comparison on the fixed PIE-Bench holdout."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from diffusers import FluxKontextPipeline
from early_edit_reward_distillation.core import critical_nonzero_steps
from early_edit_reward_distillation.metrics import region_l1
from early_edit_reward_distillation.trajectory import branch_step, deterministic_rollout, prepare_state, rollout_until, two_stage_search

@torch.inference_mode()
def decode(pipe, state, latents):
    unpacked = pipe._unpack_latents(latents, state.height, state.width, pipe.vae_scale_factor)
    unpacked = unpacked / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    decoded = pipe.vae.decode(unpacked.to(pipe.vae.dtype), return_dict=False)[0]
    return pipe.image_processor.postprocess(decoded, output_type="pil")

def _result_value(result, key="overall"):
    value = result[key]
    return float(value.item() if hasattr(value, "item") else value)

@torch.inference_mode()
def _unselected_rollout(pipe, state, token_mask, seed, *, coupled, alpha):
    selected = [int(item["index"]) for item in critical_nonzero_steps(pipe.scheduler.sigmas.detach().cpu().flatten().tolist())[:2]]
    current, current_step, records = state.latents, 0, []
    branch_mask = token_mask if coupled else torch.ones_like(token_mask, dtype=torch.bool)
    for stage, step_index in enumerate(selected, start=1):
        current = rollout_until(pipe, state, current, current_step, step_index)
        candidates, metadata = branch_step(pipe, state, current, step_index, branch_mask, seed + stage, mode="native_euler_sde", alpha=alpha)
        current, current_step = candidates[:1], step_index + 1
        records.append({"stage": stage, "chosen_index": 0, **metadata})
    return deterministic_rollout(pipe, state, current, current_step), records

def _repeat_scores(scorer, source, images, instruction, seed):
    values, old_seed = [], getattr(scorer, "seed", None)
    for index, image in enumerate(images):
        for repeat in range(2):
            if old_seed is not None: scorer.seed = int(seed + 50000 + index * 10 + repeat)
            values.append(_result_value(scorer.evaluate([Image.fromarray(np.asarray(source).copy()).convert('RGB'), Image.fromarray(np.asarray(image).copy()).convert('RGB')], instruction)))
    if old_seed is not None: scorer.seed = old_seed
    return values

def main():
    p = argparse.ArgumentParser(); p.add_argument("--model", required=True); p.add_argument("--manifest", required=True); p.add_argument("--samples-root", required=True); p.add_argument("--output", required=True)
    p.add_argument("--editscore-model", default="/data15/hyp/weight/Qwen3-VL-4B-Instruct"); p.add_argument("--editscore-lora", default="/data15/hyp/weight/EditScore-Qwen3-VL-4B-Instruct")
    p.add_argument("--seed", type=int, default=20260830); p.add_argument("--generation-seeds", type=int, default=2); p.add_argument("--height", type=int, default=512); p.add_argument("--width", type=int, default=512); p.add_argument("--steps", type=int, default=28); p.add_argument("--guidance", type=float, default=3.5); p.add_argument("--alpha", type=float, default=0.2); p.add_argument("--device", default="cuda")
    args = p.parse_args(); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8")); device = torch.device(args.device)
    if not manifest: raise RuntimeError("P1 manifest is empty")
    pipe = FluxKontextPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16, local_files_only=True).to(device); pipe.set_progress_bar_config(disable=True)
    sys.path.insert(0, "/home/hyp/Code/EditScore")
    try:
        from editscore import EditScore
        scorer = EditScore(backbone="qwen3vl", model_name_or_path=args.editscore_model, lora_path=args.editscore_lora, score_range=25, num_pass=1)
    except Exception as exc:
        raise RuntimeError(f"official EditScore failed to load; P1 is stopped without fallback: {type(exc).__name__}: {exc}") from exc
    methods = ("A_native_baseline", "B_random_independent_sde", "C_random_coupled_sde", "D_editscore_selected_coupled_sde"); rows = []
    for record_index, record in enumerate(manifest):
        sample_id, sample_dir = str(record["sample_id"]), Path(args.samples_root) / str(record["sample_id"])
        source, mask = Image.open(sample_dir / "source.png").convert("RGB"), Image.open(sample_dir / "edit_mask.png").convert("L")
        for seed_offset in range(args.generation_seeds):
            generation_seed = int(args.seed + record_index * 100 + seed_offset)
            state = prepare_state(pipe, source, str(record["instruction"]), generation_seed, height=args.height, width=args.width, steps=args.steps, guidance_scale=args.guidance, device=device)
            token_h, token_w = state.height // (pipe.vae_scale_factor * 2), state.width // (pipe.vae_scale_factor * 2)
            token_mask = torch.nn.functional.interpolate(torch.from_numpy(np.asarray(mask, dtype="float32"))[None, None], size=(token_h, token_w), mode="area")[0, 0].flatten().to(device) > 0.5
            instruction = str(record["instruction"])
            def score_pil(images):
                return [_result_value(scorer.evaluate([source, image], instruction)) for image in images]
            def score_images(latents):
                return score_pil(decode(pipe, state, latents))
            baseline_latent = deterministic_rollout(pipe, state, state.latents, 0); baseline_image = decode(pipe, state, baseline_latent)[0]
            d_terminal, d_records = two_stage_search(pipe, state, token_mask, score_images, seed=generation_seed + 10000, repeat_score=lambda latents: _repeat_scores(scorer, source, decode(pipe, state, latents), instruction, generation_seed), alpha=args.alpha, baseline_terminal=baseline_latent)
            b_latent, b_records = _unselected_rollout(pipe, state, token_mask, generation_seed + 20000, coupled=False, alpha=args.alpha)
            c_latent, c_records = _unselected_rollout(pipe, state, token_mask, generation_seed + 30000, coupled=True, alpha=args.alpha)
            terminals = {methods[0]: (baseline_latent, baseline_image, []), methods[1]: (b_latent, decode(pipe, state, b_latent)[0], b_records), methods[2]: (c_latent, decode(pipe, state, c_latent)[0], c_records), methods[3]: (d_terminal, decode(pipe, state, d_terminal)[0], d_records)}
            sample_output = output / "samples" / sample_id / f"seed_{generation_seed}"; sample_output.mkdir(parents=True, exist_ok=True); source.save(sample_output / "source.png"); mask.save(sample_output / "edit_mask.png")
            for method, (latent, image, method_records) in terminals.items():
                image.save(sample_output / f"{method}.png"); reward = score_pil([image])[0]
                rows.append({"sample_id": sample_id, "category": record.get("category", ""), "generation_seed": generation_seed, "method": method, "reward": reward, "preserve_l1": region_l1(source, image, mask, preserve=True), "edit_l1": region_l1(source, image, mask, preserve=False), "mask_area": float(record.get("mask_area", 0.0)), "resolution": f"{state.height}x{state.width}", "generated_tokens": state.metadata["generated_tokens"], "source_conditioning_tokens": state.metadata["source_conditioning_tokens"], "scheduler_mu": state.metadata["scheduler_mu"], "branch_records": json.dumps(method_records, default=str)})
            (sample_output / "d_search.json").write_text(json.dumps(d_records, indent=2, default=str) + "\n")
    with (output / "p1_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = []
    for method in methods:
        values = [row for row in rows if row["method"] == method]
        summary.append({"method": method, "count": len(values), "reward_mean": float(np.mean([row["reward"] for row in values])), "reward_std": float(np.std([row["reward"] for row in values])), "preserve_l1_mean": float(np.mean([row["preserve_l1"] for row in values])), "edit_l1_mean": float(np.mean([row["edit_l1"] for row in values]))})
    (output / "p1_summary.json").write_text(json.dumps(summary, indent=2) + "\n"); print(json.dumps({"output": str(output), "rows": len(rows), "summary": summary}, indent=2))

if __name__ == "__main__": main()
