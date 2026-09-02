"""Run VeloEdit at five strengths and analyze FLUX velocities with 2-level DWT."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from VeloEdit import FLUXVelocityAnalyzer, get_config
from VeloEdit.core.wavelet_analysis import analyze_velocity, unpack_flux_velocity


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True)
    p.add_argument("--prompt", default="Add flowers to the dog's mouth")
    p.add_argument("--model-path", default="/data15/hyp/weight/FLUX.1-Kontext-dev")
    p.add_argument("--output-dir", default="./outputs/freqedit_wavelet_dog_flowers")
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance-scale", type=float, default=2.5)
    p.add_argument("--preserve-steps", type=int, default=4)
    p.add_argument("--edit-steps", type=int, default=4)
    p.add_argument("--similarity-threshold", type=float, default=0.8)
    p.add_argument("--similarity-mode", choices=["elementwise", "cosine"], default="elementwise")
    p.add_argument("--intensities", default="0,0.25,0.5,0.75,1.0")
    p.add_argument("--wavelet", default="db4")
    p.add_argument("--levels", type=int, default=2)
    return p.parse_args()


def strength_dir(root: Path, strength: float) -> Path:
    return root / f"strength_{strength:.2f}"


def run_one(analyzer, image, prompt, args, *, enable_interv, blend_weight, out_dir):
    cfg = {
        "num_inference_steps": args.steps,
        "seed": args.seed,
        "preserve_intervention_steps": args.preserve_steps,
        "edit_intervention_steps": args.edit_steps,
        "similarity_threshold": args.similarity_threshold,
        "similarity_mode": args.similarity_mode,
        "enable_interv": enable_interv,
        "blend_weight": blend_weight,
    }
    result = analyzer.analyze(image, prompt, image_path=str(args.image), intervention_config=cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    if result.generated_image:
        result.generated_image.save(out_dir / "final.png", dpi=(300, 300))
    steps_dir = out_dir / "steps"
    steps_dir.mkdir(exist_ok=True)
    for step, img in enumerate(result.step_images or [], 1):
        img.save(steps_dir / f"step_{step:02d}.png", dpi=(300, 300))
    velocities = [d.velocity for d in result.decompositions]
    torch.save(velocities, out_dir / "velocities.pt")
    (out_dir / "sigmas.json").write_text(json.dumps(result.sigmas, indent=2))
    (out_dir / "metadata.json").write_text(json.dumps({
        "strength": None if not enable_interv else round(1.0 - blend_weight, 2),
        "blend_weight": blend_weight,
        "enable_interv": enable_interv,
        "steps": result.num_steps,
        "seed": args.seed,
        "prompt": prompt,
        "model_path": args.model_path,
    }, indent=2))
    return result, velocities


def analyze_run(name, velocities, analyzer, height, width, args, out_dir, native_velocities=None):
    rows = []
    coeff_dir = out_dir / "wavelet_coefficients"
    coeff_dir.mkdir(exist_ok=True)
    for step, packed in enumerate(velocities, 1):
        spatial = unpack_flux_velocity(packed.to(analyzer.device), analyzer.pipeline, height, width).detach().cpu()
        coeffs, stats = analyze_velocity(spatial, levels=args.levels, wavelet=args.wavelet)
        torch.save({key: value.cpu() for key, value in coeffs.items()}, coeff_dir / f"step_{step:02d}.pt")
        row = {"run": name, "step": step}
        row.update(stats)
        if native_velocities is not None:
            native_spatial = unpack_flux_velocity(native_velocities[step - 1].to(analyzer.device), analyzer.pipeline, height, width).detach().cpu()
            row["delta_velocity"] = float(torch.sqrt(torch.mean((spatial - native_spatial).float().square())).item())
            delta_coeffs, delta_stats = analyze_velocity(spatial - native_spatial, levels=args.levels, wavelet=args.wavelet)
            row.update({f"delta_{key}": value for key, value in delta_stats.items() if key.endswith("_rms") or key == "total_rms"})
            torch.save({key: value.cpu() for key, value in delta_coeffs.items()}, coeff_dir / f"step_{step:02d}_delta_native.pt")
        rows.append(row)
    return rows


def save_json_csv(rows, root: Path):
    (root / "wavelet_metrics.json").write_text(json.dumps(rows, indent=2))
    fields = sorted({key for row in rows for key in row})
    with (root / "wavelet_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_outputs(rows, root: Path, strengths):
    names = ["native"] + [f"strength_{s:.2f}" for s in strengths]
    colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(names)))
    bands = ("LL2", "D2", "D1")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    for ax, band in zip(axes, bands):
        for name, color in zip(names, colors):
            points = [r for r in rows if r["run"] == name]
            ax.plot([r["step"] for r in points], [r[f"{band}_rms"] for r in points], label=name, color=color)
        ax.set_title(f"{band} RMS")
        ax.set_xlabel("Denoising step")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Velocity coefficient RMS")
    axes[-1].legend(fontsize=7)
    fig.savefig(root / "wavelet_response_curves.png", dpi=300)
    fig.savefig(root / "wavelet_response_curves.pdf", dpi=300)
    plt.close(fig)

    image_paths = [root / "native" / "final.png"] + [strength_dir(root, s) / "final.png" for s in strengths]
    fig, axes = plt.subplots(1, len(image_paths), figsize=(18, 5), constrained_layout=True)
    for ax, path, name in zip(np.atleast_1d(axes), image_paths, names):
        if path.exists(): ax.imshow(Image.open(path))
        ax.set_title(name); ax.axis("off")
    fig.savefig(root / "strength_grid.png", dpi=300); fig.savefig(root / "strength_grid.pdf", dpi=300); plt.close(fig)

    selected = [1, 2, 3, 8, 12, 15]
    matrix = np.array([[next(r[f"{band}_energy_fraction"] for r in rows if r["run"] == name and r["step"] == step) for name in names] for band in bands for step in selected])
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
    ax.set_yticks(range(len(bands) * len(selected)), [f"{band} / step {step}" for band in bands for step in selected])
    fig.colorbar(im, ax=ax, label="Energy fraction")
    fig.savefig(root / "wavelet_energy_heatmap.png", dpi=300)
    fig.savefig(root / "wavelet_energy_heatmap.pdf", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    final_rows = [r for r in rows if r["step"] == 15]
    x = np.arange(len(names))
    for band in bands:
        ax.plot(x, [r[f"{band}_energy_fraction"] for r in final_rows], marker="o", label=band)
    ax.set_xticks(x, names, rotation=35, ha="right")
    ax.set_ylabel("Energy fraction at step 15")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(root / "wavelet_band_fraction.png", dpi=300)
    fig.savefig(root / "wavelet_band_fraction.pdf", dpi=300)
    plt.close(fig)


def response_summary(rows, strengths):
    native = {r["step"]: r for r in rows if r["run"] == "native"}
    out = []
    for strength in strengths:
        name = f"strength_{strength:.2f}"
        for step in sorted(native):
            current = next(r for r in rows if r["run"] == name and r["step"] == step)
            ratios = [current[f"{band}_rms"] / max(native[step][f"{band}_rms"], 1e-12) for band in ("LL2", "D2", "D1")]
            mean_ratio = sum(ratios) / 3
            ss_res = sum((ratio - mean_ratio) ** 2 for ratio in ratios)
            ss_tot = sum((ratio - 1.0) ** 2 for ratio in ratios)
            out.append({
                "strength": strength,
                "step": step,
                "LL2_response_ratio": ratios[0],
                "D2_response_ratio": ratios[1],
                "D1_response_ratio": ratios[2],
                "shared_scale_fit": mean_ratio,
                "cross_band_ratio_cv": float(np.std(ratios) / max(abs(np.mean(ratios)), 1e-12)),
                "shared_scale_r2": 1.0 - ss_res / max(ss_tot, 1e-12),
            })
    return out


def main():
    args = parse_args()
    if args.steps != 15:
        raise ValueError("This experiment requires --steps 15")
    strengths = [float(value) for value in args.intensities.split(",")]
    if len(strengths) != 5 or any(value < 0 or value > 1 for value in strengths):
        raise ValueError("--intensities must contain five values in [0,1]")
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    image = Image.open(args.image).convert("RGB")
    image.save(root / "source.png", dpi=(300, 300))
    config = get_config("flux")
    config.model.path = args.model_path
    config.sampling.num_inference_steps = args.steps
    config.sampling.guidance_scale = args.guidance_scale
    config.sampling.first_step_align_steps = 0
    analyzer = FLUXVelocityAnalyzer(config, device="cuda", save_tensors=True)
    analyzer.load_model(); analyzer.model_loaded = True

    native_result, native_velocities = run_one(analyzer, image, args.prompt, args, enable_interv=False, blend_weight=0.0, out_dir=root / "native")
    prepared = analyzer._prepare_inputs(image, args.prompt, args.steps, args.seed)
    height = prepared["height"]
    width = prepared["width"]
    rows = analyze_run("native", native_velocities, analyzer, height, width, args, root / "native")
    for strength in strengths:
        result, velocities = run_one(analyzer, image, args.prompt, args, enable_interv=True, blend_weight=1.0 - strength, out_dir=strength_dir(root, strength))
        rows.extend(analyze_run(f"strength_{strength:.2f}", velocities, analyzer, height, width, args, strength_dir(root, strength), native_velocities=native_velocities))
        del result, velocities
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    save_json_csv(rows, root)
    summary = response_summary(rows, strengths)
    (root / "wavelet_response_summary.json").write_text(json.dumps(summary, indent=2))
    plot_outputs(rows, root, strengths)
    (root / "trajectory_metadata.json").write_text(json.dumps({"image": str(args.image), "prompt": args.prompt, "model_path": args.model_path, "steps": args.steps, "seed": args.seed, "guidance_scale": args.guidance_scale, "strengths": strengths, "blend_weight_mapping": "blend_weight=1-strength", "preserve_steps": args.preserve_steps, "edit_steps": args.edit_steps, "similarity_threshold": args.similarity_threshold, "similarity_mode": args.similarity_mode, "wavelet": args.wavelet, "levels": args.levels, "working_resolution": [height, width], "runs": ["native"] + [f"strength_{s:.2f}" for s in strengths]}, indent=2))
    print(f"Saved wavelet experiment to {root}")


if __name__ == "__main__":
    main()
