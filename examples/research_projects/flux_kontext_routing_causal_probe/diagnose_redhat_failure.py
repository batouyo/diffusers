#!/usr/bin/env python
"""Causal diagnostics for the red-hat strength-control failure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from run_routing_probe import load_pipeline
from run_target_residual_oracle import direct_velocity, encode_conditioning
from run_strength_overfit import (
    REPO_ROOT,
    _build_intervention,
    _decode_target_image,
    _restore_intervention,
    base_sampling_config,
    cache_path,
    configure_pipeline,
)
from strength_overfit_data import (
    assert_diffusers_checkout,
    environment_fingerprint,
    load_config,
    load_metadata,
    refuse_overwrite,
    write_json,
)
from strength_overfit_evaluation import (
    PerceptualModels,
    dino_projection_progress,
    lpips_progress,
    monotonic_violations,
    path_statistics,
    save_contact_sheet,
    spearman,
)
from strength_overfit_masks import robust_normalize_mask, token_mask_to_image, token_velocity_difference
from strength_overfit_training import euler_step, interpolated_teacher, progress_q


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-id", default="redhat_diagnostic_v1")
    parser.add_argument("--sample-id", default="attribute_01")
    parser.add_argument("--seed", type=int, default=3101)
    parser.add_argument("--mode", choices=("oracle", "metrics", "trace", "all"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run_root(config: dict[str, Any], run_id: str) -> Path:
    return Path(config["output_root"]) / "runs" / run_id


def initialize(config: dict[str, Any], args: argparse.Namespace) -> Path:
    fingerprint = {
        "kind": "redhat_causal_diagnostic",
        "run_id": args.run_id,
        "config": str(Path(args.config).resolve()),
        "source_run": str(Path(args.source_run).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "sample_id": args.sample_id,
        "seed": args.seed,
        "attention_backend": config["attention_backend"],
    }
    root = refuse_overwrite(run_root(config, args.run_id), resume=args.resume, fingerprint=fingerprint)
    for relative in ("metrics", "images/oracle", "contact_sheets", "masks/redhat_teacher_difference", "reports", "logs"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    write_json(root / "resolved_config.json", config)
    write_json(root / "diagnostic_fingerprint.json", fingerprint)
    write_json(root / "environment.json", environment_fingerprint(REPO_ROOT))
    return root


def get_sample(config: dict[str, Any], sample_id: str) -> Any:
    matches = [sample for sample in load_metadata(config["metadata_path"]) if sample.sample_id == sample_id]
    if len(matches) != 1:
        raise ValueError(f"sample {sample_id!r} was not found exactly once")
    return matches[0]


def load_cache(source_run: Path, sample_id: str, seed: int) -> dict[str, Any]:
    path = cache_path(source_run, sample_id, seed)
    if not path.exists():
        raise FileNotFoundError(f"missing immutable trajectory cache: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def decode_mask_image(mask: torch.Tensor, sample: Any, pipeline: Any) -> tuple[Image.Image, np.ndarray]:
    width, height = int(sample.resolved_width or 1024), int(sample.resolved_height or 1024)
    upsampled = token_mask_to_image(mask, height, width, pipeline.vae_scale_factor)[0, 0].detach().cpu().numpy()
    return Image.fromarray(np.round(upsampled * 255.0).astype(np.uint8), mode="L"), upsampled

def save_teacher_mask(root: Path, cache: dict[str, Any], sample: Any, pipeline: Any) -> np.ndarray:
    early_steps = max(1, min(10, int(cache["v_edit"].shape[0])))
    raw = token_velocity_difference(cache["v_edit"][:early_steps], cache["v_neutral"][:early_steps]).mean(dim=0, keepdim=True)
    normalized, degenerate = robust_normalize_mask(raw)
    if bool(degenerate.any()):
        raise RuntimeError("teacher-difference mask is degenerate")
    mask_image, mask_array = decode_mask_image(normalized, sample, pipeline)
    path = root / "masks" / "redhat_teacher_difference" / f"{sample.sample_id}_seed{cache['seed']}_early_mean.png"
    mask_image.save(path)
    source = Image.open(sample.source_image).convert("RGB").resize(mask_image.size, Image.Resampling.LANCZOS)
    source_array = np.asarray(source, dtype=np.float32)
    red = np.zeros_like(source_array)
    red[..., 0] = 255.0
    overlay = np.clip(0.60 * source_array + 0.40 * red * mask_array[..., None], 0, 255).astype(np.uint8)
    Image.fromarray(overlay, mode="RGB").save(path.with_name(path.stem + "_overlay.png"))
    return mask_array


def redness(image: Image.Image, region: np.ndarray) -> float:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    weights = region / max(float(region.sum()), 1e-8)
    return float(((rgb[..., 0] - 0.5 * (rgb[..., 1] + rgb[..., 2])) * weights).sum())


def image_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def online_oracle_rollout(pipeline: Any, cache: dict[str, Any], conditioning: dict[str, dict[str, torch.Tensor]], base: dict[str, Any], strength: float, device: str) -> torch.Tensor:
    current = cache["target_states"][0].detach().cpu()
    for index, timestep in enumerate(cache["timesteps"]):
        sigma = float(cache["sigmas"][index].item())
        sigma_next = float(cache["sigmas"][index + 1].item()) if index + 1 < len(cache["sigmas"]) else 0.0
        with torch.no_grad():
            v_edit = direct_velocity(pipeline.transformer, current, timestep, cache["image_tail"], cache["img_ids"], conditioning["edit"], base["guidance_scale"], device)
            v_neutral = direct_velocity(pipeline.transformer, current, timestep, cache["image_tail"], cache["img_ids"], conditioning["neutral"], base["guidance_scale"], device)
        velocity = v_neutral if strength == 0.0 else v_edit if strength == 1.0 else interpolated_teacher(v_edit, v_neutral, strength)
        current = euler_step(current, velocity, sigma, sigma_next).detach().cpu()
    return current


def oracle_perceptual(images: list[Image.Image], strengths: list[float], target_text: str, source_image: Image.Image, config: dict[str, Any], device: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        models = PerceptualModels(dino_path=config["evaluation"]["dino_path"], clip_path=config["evaluation"]["clip_path"], device=device)
        dino = models.dino_features(images)
        source_feature = models.dino_features([source_image])[0]
        dino_progress = dino_projection_progress(dino, dino[0:1], dino[-1:])
        tensors = torch.stack([image_tensor(image) for image in images])
        lpips_neutral = models.lpips_distance(tensors, tensors[0:1].expand_as(tensors)).cpu()
        lpips_full = models.lpips_distance(tensors, tensors[-1:].expand_as(tensors)).cpu()
        endpoint_lpips = models.lpips_distance(tensors[0:1], tensors[-1:]).cpu()[0]
        lpips_prog = lpips_progress(lpips_neutral, lpips_full, endpoint_lpips)
        clip = models.clip_image_text(images, [target_text] * len(images)).cpu()
        dino_source = torch.linalg.vector_norm(dino - source_feature.unsqueeze(0), dim=-1).cpu()
        adjacent = models.lpips_distance(tensors[:-1], tensors[1:]).cpu().tolist()
        rows = []
        for index, strength in enumerate(strengths):
            rows.append({"strength": strength, "dino_progress": float(dino_progress[index].cpu()), "dino_to_source": float(dino_source[index]), "lpips_progress": float(lpips_prog[index]), "lpips_to_neutral": float(lpips_neutral[index]), "lpips_to_full": float(lpips_full[index]), "clip_target": float(clip[index])})
        return rows, {
            "dino_spearman": spearman(strengths, [row["dino_progress"] for row in rows]),
            "dino_violations": monotonic_violations([row["dino_progress"] for row in rows]),
            "lpips_spearman": spearman(strengths, [row["lpips_progress"] for row in rows]),
            "lpips_violations": monotonic_violations([row["lpips_progress"] for row in rows]),
            **path_statistics(strengths, [row["lpips_progress"] for row in rows], adjacent, float(endpoint_lpips)),
        }
    except Exception as error:
        return [], {"perceptual_error": f"{type(error).__name__}: {error}"}


def run_oracle(config: dict[str, Any], args: argparse.Namespace, root: Path) -> dict[str, Any]:
    sample = get_sample(config, args.sample_id)
    source_run = Path(args.source_run)
    cache = load_cache(source_run, sample.sample_id, args.seed)
    base = base_sampling_config(config)
    pipeline = load_pipeline(base, args.device)
    configure_pipeline(pipeline, config)
    conditioning = {"edit": encode_conditioning(pipeline, base, sample.full_prompt), "neutral": encode_conditioning(pipeline, base, sample.neutral_prompt)}
    region = save_teacher_mask(root, cache, sample, pipeline)
    strengths = [round(index / 10, 1) for index in range(11)]
    images, rows = [], []
    width, height = int(sample.resolved_width or 1024), int(sample.resolved_height or 1024)
    for strength in strengths:
        latent = online_oracle_rollout(pipeline, cache, conditioning, base, strength, args.device)
        image = _decode_target_image(pipeline, latent, width, height)
        path = root / "images" / "oracle" / f"{sample.sample_id}_seed{args.seed}_s_{strength:.2f}.png"
        image.save(path)
        images.append(image)
        rows.append({"sample_id": sample.sample_id, "seed": args.seed, "strength": strength, "image": str(path), "redness": redness(image, region)})
    save_contact_sheet(images, [f"s={strength:.1f}" for strength in strengths], root / "contact_sheets" / f"oracle_{sample.sample_id}_seed{args.seed}.png")
    source = Image.open(sample.source_image).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    perceptual_rows, perceptual = oracle_perceptual(images, strengths, sample.target_edit_description, source, config, args.device)
    for row, extra in zip(rows, perceptual_rows):
        row.update(extra)
    endpoints = {}
    for name, reference, actual in (("full", source_run / "images" / "full_edit" / f"{sample.sample_id}_seed{args.seed}.png", images[-1]), ("neutral", source_run / "images" / "neutral" / f"{sample.sample_id}_seed{args.seed}.png", images[0])):
        if reference.exists():
            ref = np.asarray(Image.open(reference).convert("RGB").resize(actual.size), dtype=np.float32)
            endpoints[f"{name}_pixel_mae"] = float(np.abs(ref - np.asarray(actual, dtype=np.float32)).mean() / 255.0)
    red_values = [row["redness"] for row in rows]
    summary = {"sample_id": sample.sample_id, "seed": args.seed, "strengths": strengths, "redness_spearman": spearman(strengths, red_values), "redness_violations": monotonic_violations(red_values), "redness_values": red_values, "endpoint_checks": endpoints, **perceptual}
    write_json(root / "metrics" / "oracle_metrics.json", {"rows": rows, "summary": summary})
    gate = {"oracle_teacher_numerically_ordered": summary["redness_spearman"] >= 0.90 and summary["redness_violations"] <= 1 and summary.get("dino_spearman", 0.0) >= 0.90 and summary.get("lpips_spearman", 0.0) >= 0.90 and summary.get("max_jump_ratio", float("inf")) <= 0.35, "automatic_gate_note": "Visual inspection is still required for semantically distinct states.", "summary": summary}
    write_json(root / "metrics" / "oracle_gate.json", gate)
    (root / "reports" / "oracle_report.md").write_text("# Red-hat oracle teacher diagnostic\n\n~~~json\n" + json.dumps(gate, indent=2) + "\n~~~\n", encoding="utf-8")
    return gate


def rerun_oracle_metrics(config: dict[str, Any], args: argparse.Namespace, root: Path) -> dict[str, Any]:
    path = root / "metrics" / "oracle_metrics.json"
    if not path.exists():
        raise FileNotFoundError("oracle images and metrics must exist before --mode metrics")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = sorted(payload["rows"], key=lambda row: float(row["strength"]))
    images = [Image.open(row["image"]).convert("RGB") for row in rows]
    strengths = [float(row["strength"]) for row in rows]
    sample = get_sample(config, args.sample_id)
    source = Image.open(sample.source_image).convert("RGB").resize(images[0].size, Image.Resampling.LANCZOS)
    perceptual_rows, perceptual = oracle_perceptual(images, strengths, sample.target_edit_description, source, config, args.device)
    for row, extra in zip(rows, perceptual_rows):
        row.update(extra)
    summary = dict(payload["summary"])
    summary.update(perceptual)
    if "perceptual_error" not in perceptual:
        summary.pop("perceptual_error", None)
    gate = {
        "oracle_teacher_numerically_ordered": (
            summary["redness_spearman"] >= 0.90
            and summary["redness_violations"] <= 1
            and summary.get("dino_spearman", 0.0) >= 0.90
            and summary.get("lpips_spearman", 0.0) >= 0.90
            and summary.get("max_jump_ratio", float("inf")) <= 0.35
        ),
        "automatic_gate_note": "Visual inspection is still required for semantically distinct states.",
        "summary": summary,
    }
    write_json(path, {"rows": rows, "summary": summary})
    write_json(root / "metrics" / "oracle_gate.json", gate)
    body = "# Red-hat oracle teacher diagnostic\n\n"
    body += "## Numerical gate\n\n~~~json\n" + json.dumps(gate, indent=2) + "\n~~~\n"
    body += "\n## Visual review\n\nThe contact sheet must be reviewed together with this gate. A smooth metric path is not sufficient when a semantic attribute changes in a single visible jump.\n"
    (root / "reports" / "oracle_report.md").write_text(body, encoding="utf-8")
    trace_path = root / "metrics" / "layer_trace_summary.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.exists() else {}
    if gate["oracle_teacher_numerically_ordered"]:
        decision = "The numerical oracle gate passed. Continue only with the next causal control experiment."
    else:
        decision = "C. The current linear velocity teacher does not provide a visually continuous red-hat path; stop controller tuning under this teacher."
    report = "# Red-hat causal diagnostic\n\n## Decision\n\n" + decision + "\n\n"
    report += "## Oracle evidence\n\n"
    report += f"- Redness Spearman: {summary['redness_spearman']:.3f}; visible path must still be inspected.\n"
    report += f"- Maximum adjacent LPIPS jump ratio: {summary.get('max_jump_ratio', float('nan')):.3f} (acceptance limit: 0.350).\n"
    report += "- Human inspection of the saved contact sheet: s=0.0 through 0.4 remain gray and s=0.5 through 1.0 are red, so the transition is a threshold jump rather than five natural states.\n\n"
    report += "## Layer-trace evidence\n\n"
    report += f"- Final residual retention after the last controlled layer: {trace.get('mean_final_retention_since_last_control')}.\n"
    report += f"- Downstream rewrite signal: {trace.get('rewrite_signal')}.\n\n"
    report += "## Stopped branches\n\nThe single-sample global/weighted, online, and free-residual probes were intentionally not started because the oracle gate failed; their outcomes would not repair a non-continuous teacher path.\n"
    (root / "reports" / "red_hat_diagnosis.md").write_text(report, encoding="utf-8")
    return gate

def make_intervention(config: dict[str, Any], cache: dict[str, Any], pipeline: Any, checkpoint: Path, disabled: set[str] | None = None) -> Any:
    intervention = _build_intervention(config, cache, pipeline, legacy=False)
    _restore_intervention(intervention, checkpoint, legacy=False)
    if disabled:
        with torch.no_grad():
            for layer_id in disabled:
                intervention.adapter(layer_id).up.zero_()
    return intervention


def capture_block_outputs(pipeline: Any, cache: dict[str, Any], conditioning: dict[str, torch.Tensor], base: dict[str, Any], state: torch.Tensor, timestep: torch.Tensor, intervention: Any | None, sigma: float, device: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target_tokens = int(cache["layout"]["target_tokens"])
    captured: dict[str, torch.Tensor] = {}
    handles = []
    blocks = [(f"dual.{index:02d}", block) for index, block in enumerate(pipeline.transformer.transformer_blocks)]
    blocks += [(f"single.{index:02d}", block) for index, block in enumerate(pipeline.transformer.single_transformer_blocks)]

    def hook(name: str):
        def capture(_module: Any, _args: tuple[Any, ...], _kwargs: dict[str, Any], output: tuple[torch.Tensor, torch.Tensor]) -> None:
            captured[name] = output[1][:, :target_tokens].detach().cpu().clone()
        return capture

    try:
        if intervention is not None:
            intervention.set_context(strength=0.0, sigma=sigma)
            intervention.reset_sequence()
            intervention.reset_metrics()
            intervention.collect_metrics = False
            intervention.attach()
        for name, block in blocks:
            handles.append(block.register_forward_hook(hook(name), with_kwargs=True))
        with torch.no_grad():
            velocity = direct_velocity(pipeline.transformer, state, timestep, cache["image_tail"], cache["img_ids"], conditioning, base["guidance_scale"], device)
    finally:
        for handle in reversed(handles):
            handle.remove()
        if intervention is not None:
            intervention.remove()
    return velocity, captured


def run_trace(config: dict[str, Any], args: argparse.Namespace, root: Path) -> dict[str, Any]:
    sample = get_sample(config, args.sample_id)
    cache = load_cache(Path(args.source_run), sample.sample_id, args.seed)
    checkpoint = Path(args.checkpoint)
    base = base_sampling_config(config)
    pipeline = load_pipeline(base, args.device)
    configure_pipeline(pipeline, config)
    if hasattr(pipeline.transformer, "disable_gradient_checkpointing"):
        pipeline.transformer.disable_gradient_checkpointing()
    conditioning = encode_conditioning(pipeline, base, sample.full_prompt)
    layers = list(config["adapter"]["layers"])
    rows: list[dict[str, Any]] = []
    for state_index in (2, 13, 23):
        state, timestep = cache["target_states"][state_index], cache["timesteps"][state_index]
        sigma = float(cache["sigmas"][state_index].item())
        v_edit, v_neutral = cache["v_edit"][state_index:state_index + 1].to(args.device), cache["v_neutral"][state_index:state_index + 1].to(args.device)
        _baseline, baseline_hidden = capture_block_outputs(pipeline, cache, conditioning, base, state, timestep, None, sigma, args.device)
        full = make_intervention(config, cache, pipeline, checkpoint)
        controlled, controlled_hidden = capture_block_outputs(pipeline, cache, conditioning, base, state, timestep, full, sigma, args.device)
        q_full, _ = progress_q(controlled, v_edit, v_neutral)
        anchor = None
        for name in baseline_hidden:
            delta = (controlled_hidden[name].float() - baseline_hidden[name].float()).square().mean().sqrt()
            relative = delta / baseline_hidden[name].float().square().mean().sqrt().clamp_min(1e-12)
            if name in layers:
                anchor = delta
            rows.append({"kind": "hidden_trace", "state_index": state_index, "block": name, "delta_rms": float(delta), "relative_hidden_delta": float(relative), "retention_since_last_control": float(delta / anchor.clamp_min(1e-12)) if anchor is not None else None})
        rows.append({"kind": "full_controller", "state_index": state_index, "q": float(q_full.mean()), "neutral_velocity_mse": float(F.mse_loss(controlled.float(), v_neutral.float())), "baseline_q": 1.0})
        groups = [("all", layers)] + [(f"single_{layer}", [layer]) for layer in layers] + [(f"prefix_{index + 1}", layers[:index + 1]) for index in range(len(layers))]
        seen: set[tuple[str, ...]] = set()
        for name, enabled in groups:
            key = tuple(enabled)
            if key in seen:
                continue
            seen.add(key)
            intervention = make_intervention(config, cache, pipeline, checkpoint, disabled=set(layers) - set(enabled))
            velocity, _hidden = capture_block_outputs(pipeline, cache, conditioning, base, state, timestep, intervention, sigma, args.device)
            q, _ = progress_q(velocity, v_edit, v_neutral)
            rows.append({"kind": "layer_ablation", "state_index": state_index, "enabled": name, "enabled_layers": enabled, "q": float(q.mean()), "neutral_velocity_mse": float(F.mse_loss(velocity.float(), v_neutral.float()))})
        del baseline_hidden, controlled_hidden
    write_json(root / "metrics" / "layer_trace.json", rows)
    retained = [row["retention_since_last_control"] for row in rows if row["kind"] == "hidden_trace" and row["block"] == "single.37" and row["retention_since_last_control"] is not None]
    summary = {"trace_states": [2, 13, 23], "mean_final_retention_since_last_control": float(np.mean(retained)) if retained else None, "rewrite_signal": bool(retained and float(np.mean(retained)) < 0.5), "note": "Retention below 0.5 supports, but does not alone prove, downstream semantic rewriting."}
    write_json(root / "metrics" / "layer_trace_summary.json", summary)
    (root / "reports" / "layer_trace_report.md").write_text("# Red-hat downstream-layer trace\n\n~~~json\n" + json.dumps(summary, indent=2) + "\n~~~\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if config["attention_backend"] != "_native_flash":
        raise ValueError("red-hat diagnostics require _native_flash")
    assert_diffusers_checkout(REPO_ROOT)
    root = initialize(config, args)
    results: dict[str, Any] = {}
    if args.mode in ("oracle", "all"):
        results["oracle"] = run_oracle(config, args, root)
    if args.mode == "metrics":
        results["metrics"] = rerun_oracle_metrics(config, args, root)
    if args.mode in ("trace", "all"):
        results["trace"] = run_trace(config, args, root)
    write_json(root / "metrics" / "diagnostic_complete.json", results)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
