"""CLI for FLUX.1-Kontext-dev block structure inspection and resumable probing."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml
from PIL import Image, ImageOps

from interventions import TextBlockIntervention, assert_no_active_interventions, resolve_block

LOGGER = logging.getLogger("flux_probe")


@dataclass(frozen=True)
class ProbeJob:
    sample_id: str
    image: str
    instruction: str
    target_description: str
    category: str
    split: str
    seed: int
    mode: str
    global_block_index: int | None
    alpha: float
    resolution: int


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def stable_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def experiment_hash(config: dict) -> str:
    """Fingerprint every input that can change generated pixels or task membership."""
    root = Path(__file__).resolve().parent
    sources = {}
    for name in ["probe_flux_kontext_blocks.py", "interventions.py", "probe_config.yaml"]:
        path = root / name
        sources[name] = file_sha256(path) if path.exists() else None
    manifest = Path(config["project"]["dataset_manifest"])
    return stable_hash(
        {
            "config": config,
            "source_sha256": sources,
            "dataset_sha256": file_sha256(manifest) if manifest.exists() else None,
        }
    )


def file_sha256(path: str | Path, chunk_size: int = 4 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_png(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.stem, suffix=".tmp.png", dir=path.parent)
    os.close(fd)
    try:
        image.save(temp_name, format="PNG", optimize=False)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def environment_report(config: dict) -> dict:
    import diffusers
    import transformers

    def git_head(path: str) -> str | None:
        try:
            return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            return None

    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "diffusers": diffusers.__version__,
        "diffusers_file": diffusers.__file__,
        "diffusers_git_head": git_head(str(Path(diffusers.__file__).resolve().parents[2])),
        "transformers": transformers.__version__,
        "transformers_file": transformers.__file__,
        "model_revision": config["model"]["revision"],
        "config_hash": experiment_hash(config),
    }


def load_pipeline(config: dict, device: str):
    from diffusers import FluxKontextPipeline

    dtype_name = config["model"].get("dtype", "bfloat16")
    dtype = getattr(torch, dtype_name)
    pipe = FluxKontextPipeline.from_pretrained(
        config["model"]["path"],
        torch_dtype=dtype,
        local_files_only=bool(config["model"].get("local_files_only", True)),
    )
    if config["model"].get("cpu_offload", False):
        pipe.enable_model_cpu_offload(device=int(device.split(":")[-1]))
    else:
        pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _shape_record(value):
    if torch.is_tensor(value):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, (list, tuple)):
        return [_shape_record(item) for item in value]
    return str(type(value).__name__)


def inspect_model(config: dict, device: str) -> Path:
    pipe = load_pipeline(config, device)
    transformer = pipe.transformer
    double_blocks = list(transformer.transformer_blocks)
    single_blocks = list(transformer.single_transformer_blocks)
    report = environment_report(config)
    report.update(
        {
            "transformer_type": f"{type(transformer).__module__}.{type(transformer).__name__}",
            "transformer_signature": str(inspect.signature(transformer.forward)),
            "double_block_count": len(double_blocks),
            "single_block_count": len(single_blocks),
            "total_block_count": len(double_blocks) + len(single_blocks),
            "blocks": [],
        }
    )
    records: dict[int, dict] = {}
    handles = []

    def pre_hook(global_index, block_type, local_index, signature):
        def hook(module, args, kwargs):
            del module
            bound = signature.bind_partial(*args, **kwargs)
            record = records.setdefault(global_index, {})
            record.update(
                {
                    "global_index": global_index,
                    "local_index": local_index,
                    "block_type": block_type,
                    "forward_signature": str(signature),
                    "inputs": {key: _shape_record(value) for key, value in bound.arguments.items()},
                }
            )
        return hook

    def post_hook(global_index):
        def hook(module, args, kwargs, output):
            del module, args, kwargs
            records.setdefault(global_index, {})["output"] = _shape_record(output)
        return hook

    for index, block in enumerate(double_blocks + single_blocks):
        block_type = "double" if index < len(double_blocks) else "single"
        local_index = index if block_type == "double" else index - len(double_blocks)
        signature = inspect.signature(block.forward)
        handles.append(block.register_forward_pre_hook(pre_hook(index, block_type, local_index, signature), with_kwargs=True))
        handles.append(block.register_forward_hook(post_hook(index), with_kwargs=True))

    source = Image.new("RGB", (512, 512), color=(128, 128, 128))
    generator = torch.Generator(device=device).manual_seed(42)
    try:
        with torch.inference_mode():
            pipe(
                image=source,
                prompt="Change the center of the image to metallic blue.",
                height=512,
                width=512,
                num_inference_steps=1,
                guidance_scale=config["inference"]["guidance_scale"],
                true_cfg_scale=config["inference"].get("true_cfg_scale", 1.0),
                generator=generator,
                output_type="latent",
                max_sequence_length=config["inference"].get("max_sequence_length", 512),
                max_area=512 * 512,
                _auto_resize=False,
            )
    finally:
        for handle in handles:
            handle.remove()
    if len(records) != report["total_block_count"]:
        raise RuntimeError(f"traced {len(records)} of {report['total_block_count']} blocks")
    report["blocks"] = [records[index] for index in sorted(records)]
    output_root = Path(config["project"]["output_root"]) / "preflight"
    path = output_root / "structure_report.json"
    atomic_json(path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return path


def load_dataset(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = {"id", "image", "instruction", "category", "target_description", "split"} - set(row)
            if missing:
                raise ValueError(f"dataset line {line_number} missing {sorted(missing)}")
            if not Path(row["image"]).exists():
                raise FileNotFoundError(row["image"])
            rows.append(row)
    return rows


def resize_source(path: str, resolution: int) -> Image.Image:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return ImageOps.fit(image, (resolution, resolution), method=Image.Resampling.LANCZOS)


def packed_noise_latents(pipe, resolution: int, seed: int, dtype: torch.dtype, device: str) -> torch.Tensor:
    channels = pipe.transformer.config.in_channels // 4
    latent_height = 2 * (resolution // (pipe.vae_scale_factor * 2))
    latent_width = 2 * (resolution // (pipe.vae_scale_factor * 2))
    generator = torch.Generator(device=device).manual_seed(seed)
    unpacked = torch.randn(
        (1, channels, latent_height, latent_width), generator=generator, device=device, dtype=dtype
    )
    return pipe._pack_latents(unpacked, 1, channels, latent_height, latent_width)


def job_output_paths(config: dict, job: ProbeJob) -> tuple[Path, Path]:
    run_root = Path(config["project"]["output_root"]) / config["project"]["run_id"]
    block = "none" if job.global_block_index is None else f"g{job.global_block_index:03d}"
    alpha = str(job.alpha).replace(".", "p")
    stem = f"seed{job.seed}_{job.mode}_{block}_a{alpha}_{job.resolution}"
    folder = run_root / "images" / job.split / job.category / job.sample_id
    return folder / f"{stem}.png", folder / f"{stem}.json"


def generate_jobs(config: dict, rows: list[dict], stage: str, blocks: list[int] | None, split: str) -> list[ProbeJob]:
    inference = config["inference"]
    dataset_cfg = config["dataset"]
    rows = [row for row in rows if row["split"] == split]
    if stage == "pilot":
        per_category = dataset_cfg["pilot_per_category"]
        selected = []
        for category in dataset_cfg["categories"]:
            selected.extend(sorted((row for row in rows if row["category"] == category), key=lambda row: row["id"])[:per_category])
        rows = selected
        seeds = [inference["pilot_seed"]]
        modes = ["baseline", "enhance_text"]
    else:
        seeds = list(inference["seeds"])
        modes = [stage]
    resolution = int(inference["resolution"])
    alpha = float(inference["alpha"])
    if blocks is None:
        blocks = []
    jobs = []
    for row in sorted(rows, key=lambda value: value["id"]):
        for seed in seeds:
            for mode in modes:
                target_blocks: Iterable[int | None] = [None] if mode == "baseline" else blocks
                for block in target_blocks:
                    jobs.append(
                        ProbeJob(
                            sample_id=row["id"],
                            image=row["image"],
                            instruction=row["instruction"],
                            target_description=row["target_description"],
                            category=row["category"],
                            split=row["split"],
                            seed=int(seed),
                            mode=mode,
                            global_block_index=block,
                            alpha=alpha,
                            resolution=resolution,
                        )
                    )
    return jobs


def run_jobs(
    config: dict,
    stage: str,
    device: str,
    shard_id: int,
    num_shards: int,
    blocks: list[int] | None,
    split: str,
    max_jobs: int | None,
) -> None:
    rows = load_dataset(config["project"]["dataset_manifest"])
    pipe = load_pipeline(config, device)
    total_blocks = len(pipe.transformer.transformer_blocks) + len(pipe.transformer.single_transformer_blocks)
    if stage != "baseline" and blocks is None:
        blocks = list(range(total_blocks))
    jobs = generate_jobs(config, rows, stage, blocks, split)
    jobs = [job for index, job in enumerate(jobs) if index % num_shards == shard_id]
    if max_jobs is not None:
        jobs = jobs[:max_jobs]
    config_hash = experiment_hash(config)
    LOGGER.info("running %d jobs on %s shard %d/%d", len(jobs), device, shard_id, num_shards)

    for ordinal, job in enumerate(jobs, 1):
        image_path, meta_path = job_output_paths(config, job)
        if image_path.exists() and meta_path.exists():
            try:
                existing = json.loads(meta_path.read_text(encoding="utf-8"))
                if existing.get("config_hash") == config_hash and existing.get("status") == "complete":
                    continue
            except Exception:
                pass
        source = resize_source(job.image, job.resolution)
        dtype = getattr(torch, config["model"].get("dtype", "bfloat16"))
        initial_latents = packed_noise_latents(pipe, job.resolution, job.seed, dtype, device)
        latent_hash = tensor_sha256(initial_latents)
        generator = torch.Generator(device=device).manual_seed(job.seed)
        started = time.perf_counter()
        intervention = contextlib.nullcontext()
        address = None
        if job.mode != "baseline":
            if job.global_block_index is None:
                raise ValueError(f"mode {job.mode} requires a block")
            address, _ = resolve_block(pipe.transformer, job.global_block_index)
            intervention = TextBlockIntervention(
                pipe.transformer,
                job.global_block_index,
                job.mode,
                alpha=job.alpha,
            )
        try:
            with intervention as active, torch.inference_mode():
                result = pipe(
                    image=source,
                    prompt=job.instruction,
                    height=job.resolution,
                    width=job.resolution,
                    num_inference_steps=config["inference"]["num_inference_steps"],
                    guidance_scale=config["inference"]["guidance_scale"],
                    true_cfg_scale=config["inference"].get("true_cfg_scale", 1.0),
                    generator=generator,
                    latents=initial_latents.clone(),
                    output_type="pil",
                    max_sequence_length=config["inference"].get("max_sequence_length", 512),
                    max_area=job.resolution * job.resolution,
                    _auto_resize=False,
                )
            output = result.images[0]
            assert_no_active_interventions(pipe.transformer)
            elapsed = time.perf_counter() - started
            atomic_png(image_path, output)
            metadata = {
                **asdict(job),
                "status": "complete",
                "config_hash": config_hash,
                "latent_hash": latent_hash,
                "source_sha256": file_sha256(job.image),
                "output_sha256": file_sha256(image_path),
                "output_path": str(image_path),
                "elapsed_seconds": elapsed,
                "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
                "block_address": asdict(address) if address else None,
                "intervention_calls": active.call_count if job.mode != "baseline" else 0,
            }
            atomic_json(meta_path, metadata)
            LOGGER.info("[%d/%d] %s", ordinal, len(jobs), image_path.name)
        except BaseException:
            assert_no_active_interventions(pipe.transformer)
            raise
        finally:
            del initial_latents
            if ordinal % 10 == 0:
                torch.cuda.empty_cache()


def identity_test(config: dict, device: str) -> Path:
    pipe = load_pipeline(config, device)
    source = Image.new("RGB", (512, 512), color=(96, 128, 160))
    latents = packed_noise_latents(pipe, 512, 42, torch.bfloat16, device)

    def invoke(intervention):
        generator = torch.Generator(device=device).manual_seed(42)
        with intervention, torch.inference_mode():
            return pipe(
                image=source,
                prompt="Change the center to metallic blue.",
                height=512,
                width=512,
                num_inference_steps=2,
                guidance_scale=config["inference"]["guidance_scale"],
                true_cfg_scale=1.0,
                generator=generator,
                latents=latents.clone(),
                output_type="latent",
                max_area=512 * 512,
                _auto_resize=False,
            ).images.detach().cpu()

    baseline = invoke(contextlib.nullcontext())
    checks = []
    for global_index in [0, len(pipe.transformer.transformer_blocks)]:
        context = TextBlockIntervention(pipe.transformer, global_index, "enhance_text", alpha=1.0)
        observed = invoke(context)
        checks.append(
            {
                "global_index": global_index,
                "max_absolute_error": float((observed - baseline).abs().max()),
                "exact_equal": bool(torch.equal(observed, baseline)),
                "call_count": context.call_count,
            }
        )
    assert_no_active_interventions(pipe.transformer)
    if any(check["max_absolute_error"] > 1e-6 for check in checks):
        raise AssertionError(checks)
    report = {"status": "pass", "checks": checks}
    path = Path(config["project"]["output_root"]) / "preflight" / "identity_report.json"
    atomic_json(path, report)
    print(json.dumps(report, indent=2))
    return path


def parse_blocks(text: str | None) -> list[int] | None:
    if text is None:
        return None
    if not text.strip():
        return []
    return [int(value) for value in text.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="probe_config.yaml")
    parser.add_argument("--device", default="cuda:0")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("inspect")
    subparsers.add_parser("identity-test")
    run = subparsers.add_parser("run")
    run.add_argument("--stage", required=True, choices=["pilot", "baseline", "enhance_text", "disable_text", "remove_block"])
    run.add_argument("--blocks")
    run.add_argument("--split", default="discovery")
    run.add_argument("--shard-id", type=int, default=0)
    run.add_argument("--num-shards", type=int, default=1)
    run.add_argument("--max-jobs", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    if args.command == "inspect":
        inspect_model(config, args.device)
    elif args.command == "identity-test":
        identity_test(config, args.device)
    else:
        run_jobs(
            config,
            args.stage,
            args.device,
            args.shard_id,
            args.num_shards,
            parse_blocks(args.blocks),
            args.split,
            args.max_jobs,
        )


if __name__ == "__main__":
    main()
