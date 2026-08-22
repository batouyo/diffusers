#!/usr/bin/env python
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from diffusers import FluxKontextPipeline

from probe_utils import (
    atomic_torch_save,
    atomic_write_json,
    configured_layers,
    cross_bias_conditions,
    directory_sha256,
    ensure_run_config,
    file_sha256,
    git_revision,
    load_json,
    load_jsonl,
    move_tensors,
    selected_step_indices,
    token_norm_map,
    velocity_metrics,
)
from routing_attention import RoutingLayout, temporary_routing_processor


SNAPSHOT_KEYS = (
    "hidden_states",
    "timestep",
    "guidance",
    "pooled_projections",
    "encoder_hidden_states",
    "txt_ids",
    "img_ids",
    "joint_attention_kwargs",
)


class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-step FLUX-Kontext routing causal probe")
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs" / "phase1.json"))
    parser.add_argument("--run-dir")
    parser.add_argument("--sample-ids", nargs="*")
    parser.add_argument("--layers", nargs="*")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


def clone_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(clone_cpu(item) for item in value)
    if isinstance(value, list):
        return [clone_cpu(item) for item in value]
    return value


class BaselineCapture:
    def __init__(self, selected_steps: list[int]) -> None:
        self.selected_steps = set(selected_steps)
        self.step_index = 0
        self.pending: dict[str, Any] | None = None
        self.snapshots: dict[int, dict[str, Any]] = {}
        self.sigmas: list[float] = []

    def pre_hook(self, _module: Any, _args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        sigma = float(kwargs["timestep"].flatten()[0].float().item())
        self.sigmas.append(sigma)
        if self.step_index not in self.selected_steps:
            self.pending = None
            return
        layout = RoutingLayout.from_runtime(
            kwargs["encoder_hidden_states"], kwargs["hidden_states"], kwargs["img_ids"]
        )
        self.pending = {
            "step_index": self.step_index,
            "layout": dataclasses.asdict(layout),
            "transformer_kwargs": {
                key: clone_cpu(kwargs[key]) for key in SNAPSHOT_KEYS if key in kwargs
            },
        }

    def post_hook(
        self,
        _module: Any,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        output: tuple[torch.Tensor, ...],
    ) -> None:
        if self.pending is not None:
            layout = RoutingLayout(**self.pending["layout"])
            self.pending["native_velocity"] = clone_cpu(output[0][:, : layout.target_tokens])
            self.snapshots[self.step_index] = self.pending
        self.pending = None
        self.step_index += 1


def load_pipeline(config: dict[str, Any], device: str) -> FluxKontextPipeline:
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[config["dtype"]]
    pipeline = FluxKontextPipeline.from_pretrained(config["model_path"], torch_dtype=dtype)
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=False)
    return pipeline


def baseline_paths(run_dir: Path, sample_id: str, steps: list[int]) -> tuple[Path, list[Path], Path]:
    image_path = run_dir / "baseline" / "images" / f"{sample_id}.png"
    snapshots = [run_dir / "baseline" / "snapshots" / sample_id / f"step_{step:03d}.pt" for step in steps]
    metadata = run_dir / "baseline" / "metadata" / f"{sample_id}.json"
    return image_path, snapshots, metadata


def capture_baseline(
    pipeline: FluxKontextPipeline,
    sample: dict[str, Any],
    config: dict[str, Any],
    run_dir: Path,
    step_indices: list[int],
    device: str,
    resume: bool,
) -> None:
    image_path, snapshot_paths, metadata_path = baseline_paths(run_dir, sample["sample_id"], step_indices)
    final_latent_path = run_dir / "baseline" / "final_latents" / f"{sample['sample_id']}.pt"
    if (
        resume
        and image_path.is_file()
        and metadata_path.is_file()
        and final_latent_path.is_file()
        and all(path.is_file() for path in snapshot_paths)
    ):
        print(f"[baseline] resume skip {sample['sample_id']}", flush=True)
        return

    source = Image.open(sample["source_image"]).convert("RGB")
    generator = torch.Generator(device=device).manual_seed(int(sample["seed"]))
    capture = BaselineCapture(step_indices)
    pre_handle = pipeline.transformer.register_forward_pre_hook(capture.pre_hook, with_kwargs=True)
    post_handle = pipeline.transformer.register_forward_hook(capture.post_hook, with_kwargs=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    final_latents: dict[str, torch.Tensor] = {}

    def capture_final_latents(_pipe: Any, step: int, _timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
        if step == int(config["num_inference_steps"]) - 1:
            final_latents["value"] = callback_kwargs["latents"].detach().cpu().clone()
        return callback_kwargs

    try:
        output = pipeline(
            image=source,
            prompt=sample["instruction"],
            num_inference_steps=int(config["num_inference_steps"]),
            guidance_scale=float(config["guidance_scale"]),
            true_cfg_scale=float(config["true_cfg_scale"]),
            generator=generator,
            max_sequence_length=int(config.get("max_sequence_length", 512)),
            _auto_resize=True,
            callback_on_step_end=capture_final_latents,
            callback_on_step_end_tensor_inputs=["latents"],
        ).images[0]
    finally:
        pre_handle.remove()
        post_handle.remove()
    elapsed = time.perf_counter() - started
    if set(capture.snapshots) != set(step_indices):
        raise RuntimeError(
            f"captured steps {sorted(capture.snapshots)} do not match requested {sorted(step_indices)}"
        )
    if len(capture.sigmas) != int(config["num_inference_steps"]):
        raise RuntimeError(f"captured {len(capture.sigmas)} Transformer calls, expected {config['num_inference_steps']}")
    if "value" not in final_latents:
        raise RuntimeError("baseline callback did not capture final latents")

    image_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(image_path)
    atomic_torch_save(final_latent_path, final_latents["value"])
    for step, path in zip(step_indices, snapshot_paths):
        snapshot = capture.snapshots[step]
        snapshot["scheduler_timestep"] = capture.sigmas[step] * 1000.0
        snapshot["sigma"] = capture.sigmas[step]
        snapshot["next_sigma"] = capture.sigmas[step + 1] if step + 1 < len(capture.sigmas) else 0.0
        snapshot["sample"] = sample
        atomic_torch_save(path, snapshot)
        if step == step_indices[0]:
            layout = RoutingLayout(**snapshot["layout"])
            print(
                json.dumps(
                    {
                        "sample_id": sample["sample_id"],
                        "text_tokens": layout.text_tokens,
                        "target_tokens": layout.target_tokens,
                        "source_tokens": layout.source_tokens,
                        "image_stream_target": [0, layout.target_tokens],
                        "image_stream_source": [layout.target_tokens, layout.image_tokens],
                        "joint_text": [0, layout.text_tokens],
                        "joint_target": [layout.target_slice.start, layout.target_slice.stop],
                        "joint_source": [layout.source_slice.start, layout.source_slice.stop],
                        "target_grid": [layout.target_grid_height, layout.target_grid_width],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    atomic_write_json(
        metadata_path,
        {
            "sample": sample,
            "step_indices": step_indices,
            "sigmas": capture.sigmas,
            "elapsed_seconds": elapsed,
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if torch.cuda.is_available() else 0
            ),
            "baseline_image": str(image_path),
            "final_latents": str(final_latent_path),
        },
    )
    print(f"[baseline] {sample['sample_id']} {elapsed:.2f}s", flush=True)


def load_snapshot(path: Path, device: str) -> dict[str, Any]:
    snapshot = torch.load(path, map_location="cpu", weights_only=False)
    snapshot["transformer_kwargs"] = move_tensors(snapshot["transformer_kwargs"], device)
    snapshot["native_velocity"] = snapshot["native_velocity"].to(device)
    return snapshot


def condition_id(b_source: float, b_text: float, repeat: int) -> str:
    def encode(value: float) -> str:
        return f"{value:+.2f}".replace("+", "p").replace("-", "m").replace(".", "p")

    return f"bs_{encode(b_source)}__bt_{encode(b_text)}__r{repeat}"


def record_path(
    run_dir: Path, sample_id: str, layer_id: str, step_index: int, b_source: float, b_text: float, repeat: int
) -> Path:
    return (
        run_dir
        / "records"
        / sample_id
        / layer_id
        / f"step_{step_index:03d}"
        / f"{condition_id(b_source, b_text, repeat)}.json"
    )


def run_condition(
    transformer: Any,
    snapshot: dict[str, Any],
    layer_id: str,
    config: dict[str, Any],
    b_source: float,
    b_text: float,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    layout = RoutingLayout(**snapshot["layout"])
    kwargs = dict(snapshot["transformer_kwargs"])
    kwargs["return_dict"] = False
    native_velocity = snapshot["native_velocity"]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(native_velocity.device)
    started = time.perf_counter()
    with torch.inference_mode(), temporary_routing_processor(
        transformer,
        layer_id,
        layout,
        b_source=b_source,
        b_text=b_text,
        query_chunk_size=int(config["attention_chunk_size"]),
    ) as processor:
        controlled_full = transformer(**kwargs)[0]
    elapsed = time.perf_counter() - started
    controlled = controlled_full[:, : layout.target_tokens]
    if processor.stats is None:
        raise RuntimeError("routing processor produced no attention statistics")
    metrics_native = velocity_metrics(controlled, native_velocity)
    result = {
        **metrics_native,
        **processor.stats.summary_dict(),
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(native_velocity.device)) if torch.cuda.is_available() else 0
        ),
    }
    velocity_map = token_norm_map(controlled - native_velocity, layout.target_grid_height, layout.target_grid_width)
    attention_map = processor.stats.token_mean.reshape(
        layout.target_grid_height, layout.target_grid_width, 3
    ).numpy()
    return result, controlled.detach().cpu(), velocity_map, attention_map, processor.stats.head_mean.numpy()


def execute_layer_step(
    pipeline: FluxKontextPipeline,
    sample: dict[str, Any],
    layer_id: str,
    step_index: int,
    snapshot_path: Path,
    config: dict[str, Any],
    run_dir: Path,
    resume: bool,
    smoke: bool,
) -> None:
    snapshot = load_snapshot(snapshot_path, str(pipeline._execution_device))
    native_velocity_cpu = snapshot["native_velocity"].detach().cpu()
    baseline_velocity_rms = float(native_velocity_cpu.float().square().mean().sqrt().item())
    base_conditions = (
        [
            {"b_source": 0.0, "b_text": 0.0, "b_target": 0.0},
            {"b_source": 0.1, "b_text": 0.0, "b_target": 0.0},
            {"b_source": 0.25, "b_text": 0.0, "b_target": 0.0},
            {"b_source": 0.0, "b_text": 0.1, "b_target": 0.0},
            {"b_source": 0.0, "b_text": 0.25, "b_target": 0.0},
        ]
        if smoke
        else cross_bias_conditions(config)
    )
    repeats = [] if smoke else config.get("repeat_conditions", [])
    jobs = [(condition, 0) for condition in base_conditions] + [(condition, 1) for condition in repeats]
    outputs: dict[str, np.ndarray] = {}
    map_path = run_dir / "velocity_maps" / sample["sample_id"] / layer_id / f"step_{step_index:03d}.npz"
    if resume and map_path.is_file():
        with np.load(map_path) as existing:
            outputs.update({key: existing[key] for key in existing.files})
    controlled_cache: dict[tuple[float, float], torch.Tensor] = {}
    zero_native_error: float | None = None
    completed_records: list[dict[str, Any]] = []

    def get_controlled(b_source: float, b_text: float) -> torch.Tensor:
        key = (b_source, b_text)
        if key not in controlled_cache:
            _, value, _, _, _ = run_condition(
                pipeline.transformer, snapshot, layer_id, config, b_source, b_text
            )
            controlled_cache[key] = value
        return controlled_cache[key]

    for condition, repeat in jobs:
        b_source = float(condition["b_source"])
        b_text = float(condition["b_text"])
        path = record_path(run_dir, sample["sample_id"], layer_id, step_index, b_source, b_text, repeat)
        if resume and path.is_file():
            record = load_json(path)
            completed_records.append(record)
            continue
        result, controlled, velocity_map, attention_map, head_mean = run_condition(
            pipeline.transformer, snapshot, layer_id, config, b_source, b_text
        )
        if repeat == 0:
            controlled_cache[(b_source, b_text)] = controlled
        if b_source == 0.0 and b_text == 0.0 and repeat == 0:
            zero_native_error = float(result["relative_l2"])
            if (
                zero_native_error > float(config["zero_native_relative_l2_max"])
                or float(result["cosine"]) < float(config["zero_native_cosine_min"])
            ):
                raise RuntimeError(
                    f"custom-zero/native mismatch at {sample['sample_id']} {layer_id} step {step_index}: "
                    f"relative_l2={zero_native_error}, cosine={result['cosine']}"
                )
            corrected = velocity_metrics(controlled, controlled)
        else:
            controlled_zero = get_controlled(0.0, 0.0)
            if zero_native_error is None:
                zero_native_error = velocity_metrics(controlled_zero, native_velocity_cpu)["relative_l2"]
            corrected = velocity_metrics(controlled, controlled_zero)
        bias_magnitude = abs(b_source) + abs(b_text)
        if (
            not result["finite"]
            or not result["attention_finite"]
            or float(result["attention_mass_sum_max_error"]) > float(config["mass_sum_error_max"])
        ):
            raise RuntimeError(
                f"non-finite or invalid attention mass at {sample['sample_id']} {layer_id} step {step_index}"
            )
        repeat_metrics: dict[str, float] = {}
        if repeat > 0:
            repeated_reference = get_controlled(b_source, b_text)
            comparison = velocity_metrics(controlled, repeated_reference)
            repeat_metrics = {
                "repeat_relative_l2": float(comparison["relative_l2"]),
                "repeat_cosine": float(comparison["cosine"]),
            }
        record = {
            "schema_version": 1,
            "sample_id": sample["sample_id"],
            "category": sample["category"],
            "source_image": sample["source_image"],
            "instruction": sample["instruction"],
            "seed": int(sample["seed"]),
            "layer_id": layer_id,
            "step_index": step_index,
            "scheduler_timestep": float(snapshot["scheduler_timestep"]),
            "sigma": float(snapshot["sigma"]),
            "next_sigma": float(snapshot["next_sigma"]),
            "b_source": b_source,
            "b_text": b_text,
            "b_target": 0.0,
            "repeat": repeat,
            "baseline_velocity_rms": baseline_velocity_rms,
            **{f"native_{key}": value for key, value in result.items()},
            **{f"corrected_{key}": value for key, value in corrected.items()},
            "absolute_sensitivity": corrected["delta_l2"] / (bias_magnitude + 1e-12),
            "relative_sensitivity": corrected["relative_l2"] / (bias_magnitude + 1e-12),
            "zero_native_relative_l2": zero_native_error,
            **repeat_metrics,
        }
        atomic_write_json(path, record)
        completed_records.append(record)
        key = condition_id(b_source, b_text, repeat)
        outputs[f"{key}__velocity_native"] = velocity_map
        outputs[f"{key}__attention_mass"] = attention_map
        outputs[f"{key}__attention_head_mean"] = head_mean
        print(
            f"[probe] {sample['sample_id']} {layer_id} step={step_index} "
            f"bs={b_source:+.2f} bt={b_text:+.2f} rel={corrected['relative_l2']:.6g}",
            flush=True,
        )

        if not smoke and repeat == 0 and bias_magnitude > 0 and (b_source > 0 or b_text > 0):
            controlled_zero = get_controlled(0.0, 0.0)
            opposite = get_controlled(-b_source, -b_text)
            positive_delta = controlled - controlled_zero
            negative_delta = opposite - controlled_zero
            symmetry_cosine = float(
                torch.nn.functional.cosine_similarity(
                    positive_delta.float().flatten(), -negative_delta.float().flatten(), dim=0
                ).item()
            )
            positive_norm = float(torch.linalg.vector_norm(positive_delta.float()).item())
            negative_norm = float(torch.linalg.vector_norm(negative_delta.float()).item())
            axis = "source" if b_source else "text"
            symmetry_path = (
                run_dir
                / "records"
                / "symmetry"
                / sample["sample_id"]
                / layer_id
                / f"step_{step_index:03d}"
                / f"{axis}_{abs(b_source or b_text):.2f}.json"
            )
            atomic_write_json(
                symmetry_path,
                {
                    "sample_id": sample["sample_id"],
                    "category": sample["category"],
                    "layer_id": layer_id,
                    "step_index": step_index,
                    "axis": axis,
                    "bias_magnitude": abs(b_source or b_text),
                    "direction_symmetry_cosine": symmetry_cosine,
                    "positive_negative_norm_ratio": positive_norm / (negative_norm + 1e-12),
                },
            )

    if smoke:
        native_restored = pipeline.transformer(**{**snapshot["transformer_kwargs"], "return_dict": False})[0][
            :, : RoutingLayout(**snapshot["layout"]).target_tokens
        ]
        restore_metrics = velocity_metrics(native_restored.detach().cpu(), native_velocity_cpu)
        atomic_write_json(
            run_dir / "records" / "smoke_restore_check.json",
            {
                "sample_id": sample["sample_id"],
                "layer_id": layer_id,
                "step_index": step_index,
                **restore_metrics,
            },
        )
        if restore_metrics["relative_l2"] > 1e-7:
            raise RuntimeError(f"processor restoration check failed: {restore_metrics}")
        zero_row = next(row for row in completed_records if row["b_source"] == 0 and row["b_text"] == 0)
        source_row = next(row for row in completed_records if row["b_source"] == 0.1)
        text_row = next(row for row in completed_records if row["b_text"] == 0.1)
        source_large_row = next(row for row in completed_records if row["b_source"] == 0.25)
        text_large_row = next(row for row in completed_records if row["b_text"] == 0.25)
        zero_velocity = get_controlled(0.0, 0.0)
        source_small_delta = get_controlled(0.1, 0.0) - zero_velocity
        source_large_delta = get_controlled(0.25, 0.0) - zero_velocity
        text_small_delta = get_controlled(0.0, 0.1) - zero_velocity
        text_large_delta = get_controlled(0.0, 0.25) - zero_velocity
        source_direction_cosine = float(
            torch.nn.functional.cosine_similarity(
                source_small_delta.float().flatten(), source_large_delta.float().flatten(), dim=0
            ).item()
        )
        text_direction_cosine = float(
            torch.nn.functional.cosine_similarity(
                text_small_delta.float().flatten(), text_large_delta.float().flatten(), dim=0
            ).item()
        )
        smoke_checks = {
            "baseline_completed": True,
            "token_spans_recorded": True,
            "zero_native_consistency": bool(
                zero_row["native_relative_l2"] <= float(config["zero_native_relative_l2_max"])
                and zero_row["native_cosine"] >= float(config["zero_native_cosine_min"])
            ),
            "source_mass_increased": bool(
                source_row["native_attention_source_mean"] > zero_row["native_attention_source_mean"]
            ),
            "text_mass_increased": bool(
                text_row["native_attention_text_mean"] > zero_row["native_attention_text_mean"]
            ),
            "source_mass_monotonic_to_0p25": bool(
                source_large_row["native_attention_source_mean"] > source_row["native_attention_source_mean"]
            ),
            "text_mass_monotonic_to_0p25": bool(
                text_large_row["native_attention_text_mean"] > text_row["native_attention_text_mean"]
            ),
            "source_velocity_changed": bool(source_row["corrected_relative_l2"] > 0),
            "text_velocity_changed": bool(text_row["corrected_relative_l2"] > 0),
            "source_response_direction_positive": bool(source_direction_cosine > 0),
            "text_response_direction_positive": bool(text_direction_cosine > 0),
            "source_response_does_not_collapse": bool(
                source_large_row["corrected_relative_l2"] >= 0.8 * source_row["corrected_relative_l2"]
            ),
            "text_response_does_not_collapse": bool(
                text_large_row["corrected_relative_l2"] >= 0.8 * text_row["corrected_relative_l2"]
            ),
            "source_0p1_to_0p25_direction_cosine": source_direction_cosine,
            "text_0p1_to_0p25_direction_cosine": text_direction_cosine,
            "all_finite": bool(all(row["native_finite"] and row["native_attention_finite"] for row in completed_records)),
            "target_queries_only": "covered_by_test_routing_attention.py",
            "processor_restored": bool(restore_metrics["relative_l2"] <= 1e-7),
            "structured_json_written": True,
        }
        smoke_checks["passed"] = bool(
            all(value for value in smoke_checks.values() if isinstance(value, bool))
        )
        atomic_write_json(run_dir / "success_checks.json", smoke_checks)
        if outputs:
            map_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = map_path.with_suffix(".tmp.npz")
            np.savez_compressed(temp_path, **outputs)
            os.replace(temp_path, map_path)
        return

    # Adaptive axis extension uses custom-zero-corrected response.
    for axis in ("source", "text"):
        axis_records = [
            row
            for row in completed_records
            if row["repeat"] == 0
            and row[f"b_{axis}"] != 0
            and row[f"b_{'text' if axis == 'source' else 'source'}"] == 0
        ]
        if not axis_records:
            continue
        max_response = max(float(row["corrected_relative_l2"]) for row in axis_records)
        zero_error = max(float(row["zero_native_relative_l2"] or 0.0) for row in axis_records)
        threshold = max(
            float(config["adaptive_noise_multiplier"]) * zero_error,
            float(config["adaptive_absolute_floor"]),
        )
        stable = all(
            row["native_finite"]
            and float(row["corrected_relative_l2"]) < float(config["stop_relative_l2"])
            and float(row["native_velocity_rms"])
            < float(config["stop_velocity_rms_ratio"]) * float(row.get("baseline_velocity_rms", float("inf")))
            for row in axis_records
        )
        if max_response > threshold or not stable:
            continue
        for value in config["bias_scan"]["adaptive_extension"]:
            b_source = float(value) if axis == "source" else 0.0
            b_text = float(value) if axis == "text" else 0.0
            path = record_path(run_dir, sample["sample_id"], layer_id, step_index, b_source, b_text, 0)
            if resume and path.is_file():
                continue
            result, controlled, velocity_map, attention_map, head_mean = run_condition(
                pipeline.transformer, snapshot, layer_id, config, b_source, b_text
            )
            controlled_zero = get_controlled(0.0, 0.0)
            corrected = velocity_metrics(controlled, controlled_zero)
            record = {
                "schema_version": 1,
                "sample_id": sample["sample_id"],
                "category": sample["category"],
                "source_image": sample["source_image"],
                "instruction": sample["instruction"],
                "seed": int(sample["seed"]),
                "layer_id": layer_id,
                "step_index": step_index,
                "scheduler_timestep": float(snapshot["scheduler_timestep"]),
                "sigma": float(snapshot["sigma"]),
                "next_sigma": float(snapshot["next_sigma"]),
                "b_source": b_source,
                "b_text": b_text,
                "b_target": 0.0,
                "repeat": 0,
                "adaptive_extension": True,
                "baseline_velocity_rms": baseline_velocity_rms,
                **{f"native_{key}": value for key, value in result.items()},
                **{f"corrected_{key}": value for key, value in corrected.items()},
                "absolute_sensitivity": corrected["delta_l2"] / abs(float(value)),
                "relative_sensitivity": corrected["relative_l2"] / abs(float(value)),
                "zero_native_relative_l2": zero_error,
            }
            atomic_write_json(path, record)
            key = condition_id(b_source, b_text, 0)
            outputs[f"{key}__velocity_native"] = velocity_map
            outputs[f"{key}__attention_mass"] = attention_map
            outputs[f"{key}__attention_head_mean"] = head_mean

    if outputs:
        map_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = map_path.with_suffix(".tmp.npz")
        np.savez_compressed(temp_path, **outputs)
        os.replace(temp_path, map_path)


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    if args.smoke:
        config = dict(config)
        config["num_inference_steps"] = 2
        config["step_indices"] = [0]
        config["run_id"] = "smoke"
    run_dir = Path(args.run_dir or Path(config["output_root"]) / config["run_id"])
    manifest = load_jsonl(config["manifest_path"])
    if args.sample_ids:
        wanted = set(args.sample_ids)
        manifest = [row for row in manifest if row["sample_id"] in wanted]
    if args.smoke:
        manifest = manifest[:1]
    manifest = [row for index, row in enumerate(manifest) if index % args.num_shards == args.shard_index]
    if not manifest:
        raise RuntimeError("no samples selected for this shard")

    layers = args.layers or configured_layers(config)
    if args.smoke:
        layers = ["dual.02"]
    steps = selected_step_indices(int(config["num_inference_steps"]), config.get("step_indices"))
    effective_config = {
        **config,
        "effective_layers": layers,
        "effective_step_indices": steps,
        "effective_sample_ids": [row["sample_id"] for row in load_jsonl(config["manifest_path"])],
        "diffusers_git_revision": git_revision(Path(__file__).resolve().parents[3]),
        "experiment_code_sha256": directory_sha256(Path(__file__).resolve().parent),
        "manifest_sha256": file_sha256(config["manifest_path"]),
        "torch_version": torch.__version__,
        "attention_backend": "diffusers native / PyTorch SDPA",
    }
    ensure_run_config(run_dir / "configs" / "run_config.json", effective_config)
    atomic_write_json(run_dir / "configs" / f"shard_{args.shard_index:02d}.json", {
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "sample_ids": [row["sample_id"] for row in manifest],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    })
    for directory in (
        "attention_stats",
        "velocity_stats",
        "velocity_maps",
        "rollout_preview",
        "tables",
        "plots",
        "logs",
    ):
        (run_dir / directory).mkdir(parents=True, exist_ok=True)
    log_handle = (run_dir / "logs" / f"shard_{args.shard_index:02d}.log").open(
        "a", encoding="utf-8", buffering=1
    )
    sys.stdout = Tee(sys.stdout, log_handle)
    sys.stderr = Tee(sys.stderr, log_handle)

    pipeline = load_pipeline(config, args.device)
    print(
        json.dumps(
            {
                "dual_blocks": len(pipeline.transformer.transformer_blocks),
                "single_blocks": len(pipeline.transformer.single_transformer_blocks),
                "heads": pipeline.transformer.config.num_attention_heads,
                "head_dim": pipeline.transformer.config.attention_head_dim,
                "layers": layers,
                "steps": steps,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for sample in manifest:
        capture_baseline(pipeline, sample, config, run_dir, steps, args.device, args.resume)
        if args.baseline_only:
            continue
        _, snapshot_paths, _ = baseline_paths(run_dir, sample["sample_id"], steps)
        for layer_id in layers:
            for step_index, snapshot_path in zip(steps, snapshot_paths):
                execute_layer_step(
                    pipeline,
                    sample,
                    layer_id,
                    step_index,
                    snapshot_path,
                    config,
                    run_dir,
                    args.resume,
                    args.smoke,
                )


if __name__ == "__main__":
    main()
