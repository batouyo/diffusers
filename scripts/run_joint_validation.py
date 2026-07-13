"""Held-out joint validation for selected, random, all-block, and TexTailor controls."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from PIL import Image

from interventions import TextBlockIntervention, assert_no_active_interventions
from probe_flux_kontext_blocks import (
    atomic_json,
    atomic_png,
    experiment_hash,
    file_sha256,
    load_config,
    load_dataset,
    load_pipeline,
    packed_noise_latents,
    resize_source,
    tensor_sha256,
)


@dataclass(frozen=True)
class JointJob:
    sample_id: str
    image: str
    instruction: str
    target_description: str
    category: str
    split: str
    seed: int
    arm: str
    intervention_mode: str
    block_indices: tuple[int, ...]
    alpha: float
    resolution: int


def random_controls(
    total_blocks: int,
    double_blocks: int,
    candidates: list[int],
    count: int,
    seed: int,
) -> list[tuple[int, ...]]:
    rng = random.Random(seed)
    double_count = sum(index < double_blocks for index in candidates)
    single_count = len(candidates) - double_count
    double_pool = [index for index in range(double_blocks) if index not in candidates]
    single_pool = [index for index in range(double_blocks, total_blocks) if index not in candidates]
    if len(double_pool) < double_count or len(single_pool) < single_count:
        raise RuntimeError("not enough blocks to build stream-matched random controls")
    result = set()
    attempts = 0
    while len(result) < count and attempts < 10000:
        selection = tuple(sorted(rng.sample(double_pool, double_count) + rng.sample(single_pool, single_count)))
        result.add(selection)
        attempts += 1
    if len(result) != count:
        raise RuntimeError(f"could only construct {len(result)} unique random controls")
    return sorted(result)


def arms(transformer, candidates: list[int], alpha: float, random_sets: int, seed: int):
    n_double = len(transformer.transformer_blocks)
    total = n_double + len(transformer.single_transformer_blocks)
    result = [("baseline", "none", tuple(), 1.0)]
    result.extend((f"candidate_single_g{index:03d}", "enhance_text", (index,), alpha) for index in candidates)
    result.append((f"candidate_disable_g{candidates[0]:03d}", "disable_text", (candidates[0],), 0.0))
    result.append(("candidate_combo", "enhance_text", tuple(sorted(candidates)), alpha))
    for index, blocks in enumerate(random_controls(total, n_double, candidates, random_sets, seed)):
        result.append((f"random_{index:02d}", "enhance_text", blocks, alpha))
    result.append(("all_blocks", "enhance_text", tuple(range(total)), alpha))
    budget_alpha = 1.0 + len(candidates) / total * (alpha - 1.0)
    result.append(("all_blocks_budget_matched", "enhance_text", tuple(range(total)), budget_alpha))
    textailor = tuple(index for index in [2, 7, 12, 17, 22] if index < total)
    result.append(("textailor_flux1dev_control", "enhance_text", textailor, alpha))
    return result


def output_paths(config: dict, job: JointJob) -> tuple[Path, Path]:
    run_root = Path(config["project"]["output_root"]) / config["project"]["run_id"]
    alpha = str(job.alpha).replace(".", "p")
    stem = f"seed{job.seed}_{job.arm}_a{alpha}_{job.resolution}"
    folder = run_root / "joint" / job.split / job.category / job.sample_id
    return folder / f"{stem}.png", folder / f"{stem}.json"


def joint_hash(config: dict, candidates: list[int], alpha: float) -> str:
    value = {
        "experiment_hash": experiment_hash(config),
        "script_sha256": file_sha256(__file__),
        "candidates": candidates,
        "alpha": alpha,
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def run(args) -> None:
    config = load_config(args.config)
    candidates = [int(value) for value in args.candidates.split(",") if value]
    if not candidates:
        raise ValueError("joint validation requires at least one independently selected candidate")
    pipe = load_pipeline(config, args.device)
    arm_list = arms(
        pipe.transformer,
        candidates,
        args.alpha,
        args.random_sets,
        config["statistics"]["random_seed"],
    )
    rows = [row for row in load_dataset(config["project"]["dataset_manifest"]) if row["split"] == args.split]
    if args.max_samples_per_category is not None:
        limited = []
        for category in config["dataset"]["categories"]:
            limited.extend(
                sorted((row for row in rows if row["category"] == category), key=lambda row: row["id"])[
                    : args.max_samples_per_category
                ]
            )
        rows = limited
    jobs = []
    resolution = args.resolution or config["inference"]["resolution"]
    for row in sorted(rows, key=lambda value: value["id"]):
        for seed in config["inference"]["seeds"]:
            for arm, intervention_mode, blocks, arm_alpha in arm_list:
                jobs.append(
                    JointJob(
                        sample_id=row["id"],
                        image=row["image"],
                        instruction=row["instruction"],
                        target_description=row["target_description"],
                        category=row["category"],
                        split=row["split"],
                        seed=int(seed),
                        arm=arm,
                        intervention_mode=intervention_mode,
                        block_indices=blocks,
                        alpha=float(arm_alpha),
                        resolution=int(resolution),
                    )
                )
    jobs = [job for index, job in enumerate(jobs) if index % args.num_shards == args.shard_id]
    fingerprint = joint_hash(config, candidates, args.alpha)
    dtype = getattr(torch, config["model"]["dtype"])
    for ordinal, job in enumerate(jobs, 1):
        image_path, meta_path = output_paths(config, job)
        if image_path.exists() and meta_path.exists():
            try:
                prior = json.loads(meta_path.read_text(encoding="utf-8"))
                if prior.get("joint_hash") == fingerprint and prior.get("status") == "complete":
                    continue
            except Exception:
                pass
        source = resize_source(job.image, job.resolution)
        latents = packed_noise_latents(pipe, job.resolution, job.seed, dtype, args.device)
        generator = torch.Generator(device=args.device).manual_seed(job.seed)
        contexts = contextlib.ExitStack()
        active = []
        started = time.perf_counter()
        try:
            with contexts, torch.inference_mode():
                for global_index in job.block_indices:
                    active.append(
                        contexts.enter_context(
                            TextBlockIntervention(
                                pipe.transformer,
                                global_index,
                                job.intervention_mode,
                                alpha=job.alpha,
                                allow_multi=True,
                            )
                        )
                    )
                result = pipe(
                    image=source,
                    prompt=job.instruction,
                    height=job.resolution,
                    width=job.resolution,
                    num_inference_steps=config["inference"]["num_inference_steps"],
                    guidance_scale=config["inference"]["guidance_scale"],
                    true_cfg_scale=config["inference"]["true_cfg_scale"],
                    generator=generator,
                    latents=latents.clone(),
                    output_type="pil",
                    max_sequence_length=config["inference"]["max_sequence_length"],
                    max_area=job.resolution * job.resolution,
                    _auto_resize=False,
                )
            assert_no_active_interventions(pipe.transformer)
            atomic_png(image_path, result.images[0])
            atomic_json(
                meta_path,
                {
                    **asdict(job),
                    "block_indices": list(job.block_indices),
                    "mode": "joint_validation",
                    "global_block_index": None,
                    "status": "complete",
                    "joint_hash": fingerprint,
                    "latent_hash": tensor_sha256(latents),
                    "source_sha256": file_sha256(job.image),
                    "output_sha256": file_sha256(image_path),
                    "output_path": str(image_path),
                    "elapsed_seconds": time.perf_counter() - started,
                    "intervention_calls": {str(item.address.global_index): item.call_count for item in active},
                },
            )
            print(f"[{ordinal}/{len(jobs)}] {job.arm} {job.sample_id} seed={job.seed}", flush=True)
        finally:
            assert_no_active_interventions(pipe.transformer)
            del latents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "probe_config.yaml"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--split", default="heldout")
    parser.add_argument("--random-sets", type=int, default=20)
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--max-samples-per-category", type=int)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
