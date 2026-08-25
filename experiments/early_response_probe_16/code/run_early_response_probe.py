"""Early local-perturbation response probe for frozen FLUX.1-Kontext.

This is intentionally a separate experiment entry point.  It reuses the
numerics of ``run_temporal_probe.py`` but never edits its configuration or its
previous results.  A signed antithetic pair shares the same initial state and
uses the exact opposite SDE noise schedule, so pair-level deltas are finite
differences rather than two unrelated random trials.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.stats import pearsonr, spearmanr
from transformers import AutoImageProcessor, CLIPImageProcessor, CLIPVisionModelWithProjection, Dinov2Model
from diffusers import FluxKontextPipeline

from run_temporal_probe import decode, ode_step, prepare, sigma_pair, velocity


TIE_EPS = 1e-8
MODES = ("first_only", "first_two")


@dataclass(frozen=True)
class Case:
    sample_id: str
    category: str
    subgroup: str
    source_image: str
    instruction: str
    base_seed: int


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cases(path: Path) -> list[Case]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases: list[Case] = []
    for row in payload["cases"]:
        source = Path(row["source_image"])
        if not source.is_absolute():
            source = (path.parent / source).resolve()
        cases.append(Case(
            sample_id=row["sample_id"], category=row["category"], subgroup=row["subgroup"],
            source_image=str(source), instruction=row["instruction"], base_seed=int(row["base_seed"]),
        ))
    return cases


def sde_step_with_noise(
    pipe: FluxKontextPipeline,
    latents: torch.Tensor,
    prediction: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
    scale: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Exact TempFlow-style mean/std from the prior probe, with explicit noise."""
    sigma, next_sigma = sigma_pair(pipe, t, latents.ndim)
    dt = next_sigma - sigma
    sigma_max = pipe.scheduler.sigmas[1]
    safe_sigma = torch.where(sigma == 1, sigma_max, sigma)
    std = torch.sqrt(sigma / (1 - safe_sigma)) * scale
    mean = latents.float() * (1 + std.square() / (2 * sigma) * dt) + prediction.float() * (
        1 + std.square() * (1 - sigma) / (2 * sigma)
    ) * dt
    return (mean + std * torch.sqrt(-dt) * noise.to(mean.dtype)).to(dtype)


def perturb_noise(reference: torch.Tensor, experiment_seed: int, sample_index: int, pair_id: int, step_index: int) -> torch.Tensor:
    """Fixed arithmetic seed; deliberately never depends on Python's randomized hash."""
    seed = int(experiment_seed + sample_index * 100_000 + pair_id * 100 + step_index)
    generator = torch.Generator(device=reference.device).manual_seed(seed)
    return torch.randn(reference.shape, generator=generator, device=reference.device, dtype=torch.float32)


def safe_corr(x: Iterable[float], y: Iterable[float]) -> tuple[float, float]:
    x_arr = np.asarray(list(x), dtype=np.float64)
    y_arr = np.asarray(list(y), dtype=np.float64)
    if len(x_arr) < 2 or not np.isfinite(x_arr).all() or not np.isfinite(y_arr).all():
        return math.nan, math.nan
    if np.ptp(x_arr) <= TIE_EPS or np.ptp(y_arr) <= TIE_EPS:
        return math.nan, math.nan
    return float(spearmanr(x_arr, y_arr).statistic), float(pearsonr(x_arr, y_arr).statistic)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    fd, temp_name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".tmp", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".tmp", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


class ImageEncoders:
    """Frozen image encoders.  DINO failure is recorded and never blocks CLIP."""

    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        self.device = device
        self.status: dict[str, dict[str, str]] = {}
        self.clip_processor = CLIPImageProcessor.from_pretrained(
            args.clip_model, cache_dir=args.encoder_cache, local_files_only=args.encoder_local_files_only
        )
        self.clip = CLIPVisionModelWithProjection.from_pretrained(
            args.clip_model, cache_dir=args.encoder_cache, local_files_only=args.encoder_local_files_only
        ).to(device).eval()
        self.clip.requires_grad_(False)
        self.status["clip"] = {"status": "enabled", "model": args.clip_model}
        self.dino_processor = None
        self.dino = None
        if args.skip_dino:
            self.status["dino"] = {"status": "skipped", "model": args.dino_model, "reason": "--skip-dino"}
            return
        try:
            self.dino_processor = AutoImageProcessor.from_pretrained(
                args.dino_model, cache_dir=args.encoder_cache, local_files_only=args.encoder_local_files_only
            )
            self.dino = Dinov2Model.from_pretrained(
                args.dino_model, cache_dir=args.encoder_cache, local_files_only=args.encoder_local_files_only
            ).to(device).eval()
            self.dino.requires_grad_(False)
            self.status["dino"] = {"status": "enabled", "model": args.dino_model}
        except Exception as exc:  # Optional analysis only.
            self.dino_processor = None
            self.dino = None
            self.status["dino"] = {"status": "unavailable", "model": args.dino_model, "reason": f"{type(exc).__name__}: {str(exc)[:500]}"}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @torch.inference_mode()
    def encode(self, images: list[Image.Image], batch_size: int = 8) -> dict[str, np.ndarray]:
        result: dict[str, list[np.ndarray]] = {"clip": []}
        if self.dino is not None:
            result["dino"] = []
        for start in range(0, len(images), batch_size):
            part = images[start:start + batch_size]
            clip_inputs = self.clip_processor(images=part, return_tensors="pt").to(self.device)
            clip_emb = self.clip(**clip_inputs).image_embeds.float()
            clip_emb = torch.nn.functional.normalize(clip_emb, dim=-1)
            result["clip"].append(clip_emb.cpu().numpy())
            if self.dino is not None and self.dino_processor is not None:
                dino_inputs = self.dino_processor(images=part, return_tensors="pt").to(self.device)
                dino_emb = self.dino(**dino_inputs).last_hidden_state[:, 0, :].float()
                dino_emb = torch.nn.functional.normalize(dino_emb, dim=-1)
                result["dino"].append(dino_emb.cpu().numpy())
        return {name: np.concatenate(chunks, axis=0) for name, chunks in result.items()}


def q_values(source: np.ndarray, normal: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, bool]:
    direction = normal - source
    denom = float(np.dot(direction, direction))
    if denom < TIE_EPS:
        return np.full(values.shape[0], np.nan, dtype=np.float64), True
    return ((values - source) @ direction / denom).astype(np.float64), False


def save_jpeg(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.resize((1024, 1024), Image.Resampling.LANCZOS).save(path, "JPEG", quality=95, subsampling=0)


def label_image_grid(images: list[Image.Image], labels: list[str], path: Path, title: str) -> None:
    if not images:
        return
    cols, thumb, header, gutter = 4, 256, 42, 12
    rows = math.ceil(len(images) / cols)
    canvas = Image.new("RGB", (cols * (thumb + gutter) + gutter, rows * (thumb + header + gutter) + gutter), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(images, labels)):
        row, col = divmod(index, cols)
        x, y = gutter + col * (thumb + gutter), gutter + row * (thumb + header + gutter)
        draw.text((x, y), label, fill="black")
        canvas.paste(image.resize((thumb, thumb), Image.Resampling.LANCZOS), (x, y + header))
    draw.text((gutter, 2), title, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, "PNG")


def mode_dir(output: Path, case: Case, mode: str) -> Path:
    return output / "images" / case.sample_id / mode


@torch.inference_mode()
def normal_edit(pipe: FluxKontextPipeline, state: dict[str, Any]) -> Image.Image:
    current = state["latents"]
    for timestep in state["timesteps"]:
        current = ode_step(pipe, current, velocity(pipe, state, current, timestep), timestep, state["dtype"])
    return decode(pipe, current, state["height"], state["width"])[0]


@torch.inference_mode()
def generate_mode(
    pipe: FluxKontextPipeline, state: dict[str, Any], case: Case, mode: str, sample_index: int, args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[Image.Image], list[Image.Image]]:
    """Generate all signed branches for one schedule, recomputing post-SDE velocity."""
    timesteps = state["timesteps"]
    if mode not in MODES:
        raise ValueError(mode)
    if mode == "first_two" and len(timesteps) < 3:
        raise ValueError("first_two needs at least three inference timesteps")
    sigma0, sigma1 = sigma_pair(pipe, timesteps[0], state["latents"].ndim)
    _, sigma2 = sigma_pair(pipe, timesteps[1], state["latents"].ndim)
    metadata: list[dict[str, Any]] = []
    early_images: list[Image.Image] = []
    final_images: list[Image.Image] = []
    # branch_batch_size=4 naturally yields pair0+, pair0-, pair1+, pair1- ordering.
    for first_pair in range(0, args.pairs, max(1, args.branch_batch_size // 2)):
        pair_ids = list(range(first_pair, min(args.pairs, first_pair + max(1, args.branch_batch_size // 2))))
        noises0, noises1, signs, pairs = [], [], [], []
        for pair_id in pair_ids:
            z0 = perturb_noise(state["latents"], args.experiment_seed, sample_index, pair_id, 0)
            z1 = perturb_noise(state["latents"], args.experiment_seed, sample_index, pair_id, 1)
            for sign in (1, -1):
                noises0.append(z0 * sign)
                noises1.append(z1 * sign)
                signs.append(sign)
                pairs.append(pair_id)
        branch_count = len(signs)
        base = state["latents"].repeat(branch_count, 1, 1)
        initial_v = velocity(pipe, state, state["latents"], timesteps[0]).repeat(branch_count, 1, 1)
        x1 = sde_step_with_noise(pipe, base, initial_v, timesteps[0], torch.cat(noises0), args.sde_scale, state["dtype"])
        v1 = velocity(pipe, state, x1, timesteps[1])  # required branch-specific post-perturbation velocity
        if mode == "first_only":
            readout_latents, readout_v, readout_sigma, readout_timestep = x1, v1, sigma1, timesteps[1]
            final_latents = ode_step(pipe, x1, v1, timesteps[1], state["dtype"])
            suffix_start = 2
            perturb_t1, perturb_sigma1 = "", ""
        else:
            x2 = sde_step_with_noise(pipe, x1, v1, timesteps[1], torch.cat(noises1), args.sde_scale, state["dtype"])
            v2 = velocity(pipe, state, x2, timesteps[2])  # velocity is recomputed at sigma_2
            readout_latents, readout_v, readout_sigma, readout_timestep = x2, v2, sigma2, timesteps[2]
            final_latents = ode_step(pipe, x2, v2, timesteps[2], state["dtype"])
            suffix_start = 3
            perturb_t1, perturb_sigma1 = float(timesteps[1].item()), float(sigma1.item())
        # FlowMatchEulerDiscreteScheduler's official sample reconstruction convention.
        x0_hat = (readout_latents.float() - readout_sigma * readout_v.float()).to(state["dtype"])
        early_images.extend(decode(pipe, x0_hat, state["height"], state["width"]))
        for timestep in timesteps[suffix_start:]:
            final_latents = ode_step(pipe, final_latents, velocity(pipe, state, final_latents, timestep), timestep, state["dtype"])
        final_images.extend(decode(pipe, final_latents, state["height"], state["width"]))
        for pair_id, sign in zip(pairs, signs):
            metadata.append({
                "pair_id": pair_id, "perturbation_id": f"pair_{pair_id:02d}_{'pos' if sign > 0 else 'neg'}",
                "perturbation_sign": sign, "timestep_mode": mode,
                "perturb_timestep_0": float(timesteps[0].item()), "perturb_sigma_0": float(sigma0.item()),
                "perturb_timestep_1": perturb_t1, "perturb_sigma_1": perturb_sigma1,
                "readout_timestep": float(readout_timestep.item()), "readout_sigma": float(readout_sigma.item()),
            })
    return metadata, early_images, final_images


def attach_scores(rows: list[dict[str, Any]], source: Image.Image, normal: Image.Image, early: list[Image.Image], final: list[Image.Image], encoders: ImageEncoders) -> set[str]:
    embeddings = encoders.encode([source, normal] + early + final)
    active: set[str] = set()
    count = len(rows)
    for name, values in embeddings.items():
        early_q, degenerate = q_values(values[0], values[1], values[2:2 + count])
        final_q, ignored = q_values(values[0], values[1], values[2 + count:2 + 2 * count])
        assert degenerate == ignored
        for row, q_early, q_final in zip(rows, early_q, final_q):
            row[f"q_early_{name}"] = float(q_early)
            row[f"q_final_{name}"] = float(q_final)
            row[f"degenerate_{name}"] = int(degenerate)
        active.add(name)
    return active


def compute_analysis(results: list[dict[str, Any]], encoders: set[str]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[(row["sample_id"], row["timestep_mode"])].append(row)
    correlations: list[dict[str, Any]] = []
    pairs_out: list[dict[str, Any]] = []
    pair_stats: list[dict[str, Any]] = []
    ranges: list[dict[str, Any]] = []
    reversals: list[dict[str, Any]] = []
    for (sample_id, mode), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: (int(row["pair_id"]), -int(row["perturbation_sign"])))
        base = {key: rows[0][key] for key in ("sample_id", "category", "subgroup", "timestep_mode")}
        pair_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            pair_groups[int(row["pair_id"])] .append(row)
        for encoder in sorted(encoders):
            qe = np.array([float(row[f"q_early_{encoder}"]) for row in rows])
            qf = np.array([float(row[f"q_final_{encoder}"]) for row in rows])
            degenerate = bool(int(rows[0][f"degenerate_{encoder}"]))
            overall_s, overall_p = (math.nan, math.nan) if degenerate else safe_corr(qe, qf)
            correlations.append({**base, "encoder": encoder, "spearman": overall_s, "pearson": overall_p, "n_signed": len(rows), "degenerate": int(degenerate)})
            deltas_e, deltas_f, valid_deltas = [], [], []
            for pair_id, pair_rows in sorted(pair_groups.items()):
                pos = next(row for row in pair_rows if int(row["perturbation_sign"]) == 1)
                neg = next(row for row in pair_rows if int(row["perturbation_sign"]) == -1)
                de = float(pos[f"q_early_{encoder}"]) - float(neg[f"q_early_{encoder}"])
                df = float(pos[f"q_final_{encoder}"]) - float(neg[f"q_final_{encoder}"])
                valid = (not degenerate and abs(de) >= TIE_EPS and abs(df) >= TIE_EPS)
                sign_agree = int(de * df > 0) if valid else ""
                pairs_out.append({**base, "encoder": encoder, "pair_id": pair_id, "delta_q_early": de, "delta_q_final": df, "valid": int(valid), "sign_agree": sign_agree})
                deltas_e.append(de); deltas_f.append(df)
                if valid:
                    valid_deltas.append((de, df, sign_agree))
            ds, dp = safe_corr([item[0] for item in valid_deltas], [item[1] for item in valid_deltas]) if len(valid_deltas) >= 2 else (math.nan, math.nan)
            pair_stats.append({**base, "encoder": encoder, "total_pairs": len(pair_groups), "valid_pairs": len(valid_deltas), "invalid_pairs": len(pair_groups) - len(valid_deltas), "sign_agreement_ratio": float(np.mean([item[2] for item in valid_deltas])) if valid_deltas else math.nan, "delta_spearman": ds, "delta_pearson": dp, "degenerate": int(degenerate)})
            ranges.append({**base, "encoder": encoder, "std_q_early": float(np.nanstd(qe)), "std_q_final": float(np.nanstd(qf)), "range_q_early": float(np.nanmax(qe) - np.nanmin(qe)), "range_q_final": float(np.nanmax(qf) - np.nanmin(qf)), "median_abs_delta_q_early": float(np.nanmedian(np.abs(deltas_e))), "median_abs_delta_q_final": float(np.nanmedian(np.abs(deltas_f))), "degenerate": int(degenerate)})
            valid_pairs = reversal_pairs = 0
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    de, df = qe[i] - qe[j], qf[i] - qf[j]
                    if abs(de) < TIE_EPS or abs(df) < TIE_EPS or degenerate:
                        continue
                    valid_pairs += 1
                    reversal_pairs += int(de * df < 0)
            reversals.append({**base, "encoder": encoder, "valid_unordered_pairs": valid_pairs, "reversal_pairs": reversal_pairs, "reversal_ratio": reversal_pairs / valid_pairs if valid_pairs else math.nan, "degenerate": int(degenerate)})
    return {"correlations": correlations, "pairs": pairs_out, "pair_stats": pair_stats, "ranges": ranges, "reversals": reversals}


def subgroup_summary(analysis: dict[str, list[dict[str, Any]]], encoders: set[str]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for table_name in ("correlations", "pair_stats", "ranges"):
        for row in analysis[table_name]:
            by_key[(row["encoder"], row["timestep_mode"], row["subgroup"])][table_name].append(row)
    output: list[dict[str, Any]] = []
    for (encoder, mode, subgroup), tables in sorted(by_key.items()):
        corr, pair, response = tables["correlations"], tables["pair_stats"], tables["ranges"]
        median = lambda values: float(np.nanmedian(values)) if np.isfinite(values).any() else math.nan
        output.append({"encoder": encoder, "timestep_mode": mode, "subgroup": subgroup, "samples": len(corr), "median_overall_spearman": median(np.array([x["spearman"] for x in corr], float)), "median_antithetic_sign_agreement": median(np.array([x["sign_agreement_ratio"] for x in pair], float)), "median_delta_spearman": median(np.array([x["delta_spearman"] for x in pair], float)), "median_early_response_range": median(np.array([x["range_q_early"] for x in response], float)), "median_final_response_range": median(np.array([x["range_q_final"] for x in response], float)), "degenerate_or_invalid_samples": int(sum(bool(x["degenerate"]) for x in corr) + sum(int(x["valid_pairs"]) == 0 for x in pair if not bool(x["degenerate"])))})
    return output


def wide_encoder_table(rows: list[dict[str, Any]], shared: tuple[str, ...]) -> list[dict[str, Any]]:
    """Keep the prescribed one-row-per-sample(/pair) CSV shape.

    Branch-level scores already use wide ``*_clip``/``*_dino`` fields.  The
    aggregate CSVs follow the same convention rather than doubling their row
    count when the optional DINO analysis is available.
    """
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in shared)].append(row)
    output: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        merged = dict(zip(shared, key))
        for row in group:
            encoder = row["encoder"]
            for column, value in row.items():
                if column not in shared and column != "encoder":
                    merged[f"{column}_{encoder}"] = value
        output.append(merged)
    return output


def make_scatters(output: Path, case: Case, results: list[dict[str, Any]], analysis: dict[str, list[dict[str, Any]]], encoder: str) -> None:
    rows_by_mode = {mode: sorted([r for r in results if r["sample_id"] == case.sample_id and r["timestep_mode"] == mode], key=lambda r: (int(r["pair_id"]), -int(r["perturbation_sign"]))) for mode in MODES}
    corrs = {(r["sample_id"], r["timestep_mode"]): r for r in analysis["correlations"] if r["encoder"] == encoder}
    revs = {(r["sample_id"], r["timestep_mode"]): r for r in analysis["reversals"] if r["encoder"] == encoder}
    pst = {(r["sample_id"], r["timestep_mode"]): r for r in analysis["pair_stats"] if r["encoder"] == encoder}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for axis, mode in zip(axes, MODES):
        rows = rows_by_mode[mode]
        colors = ["#e76f51" if int(r["perturbation_sign"]) > 0 else "#457b9d" for r in rows]
        axis.scatter([float(r[f"q_early_{encoder}"]) for r in rows], [float(r[f"q_final_{encoder}"]) for r in rows], c=colors, alpha=.8, s=28)
        c, rv, ps = corrs[(case.sample_id, mode)], revs[(case.sample_id, mode)], pst[(case.sample_id, mode)]
        axis.set(title=mode, xlabel="q early", ylabel="q final")
        axis.text(.02, .98, f"Spearman={c['spearman']:.3f}\nPearson={c['pearson']:.3f}\nreversal={rv['reversal_ratio']:.3f}\nsign agree={ps['sign_agreement_ratio']:.3f}", transform=axis.transAxes, va="top", fontsize=8, bbox={"facecolor":"white", "alpha":.8, "edgecolor":"none"})
        axis.axhline(0, color="0.8", lw=.7); axis.axvline(0, color="0.8", lw=.7)
    fig.suptitle(f"{case.sample_id} — {encoder} overall")
    path = output / "plots" / case.sample_id / f"overall_scatter_{encoder}.png"; path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=180); plt.close(fig)
    pair_rows = [r for r in analysis["pairs"] if r["sample_id"] == case.sample_id and r["encoder"] == encoder]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for axis, mode in zip(axes, MODES):
        rows = [r for r in pair_rows if r["timestep_mode"] == mode]
        axis.scatter([float(r["delta_q_early"]) for r in rows], [float(r["delta_q_final"]) for r in rows], c=["#2a9d8f" if int(r["valid"]) else "0.65" for r in rows], alpha=.85, s=30)
        ps = pst[(case.sample_id, mode)]
        axis.set(title=mode, xlabel="Δq early (+ − −)", ylabel="Δq final (+ − −)")
        axis.text(.02, .98, f"delta Spearman={ps['delta_spearman']:.3f}\ndelta Pearson={ps['delta_pearson']:.3f}\nsign agree={ps['sign_agreement_ratio']:.3f}", transform=axis.transAxes, va="top", fontsize=8, bbox={"facecolor":"white", "alpha":.8, "edgecolor":"none"})
        axis.axhline(0, color="0.8", lw=.7); axis.axvline(0, color="0.8", lw=.7)
    fig.suptitle(f"{case.sample_id} — {encoder} antithetic finite differences")
    path = output / "plots" / case.sample_id / f"antithetic_scatter_{encoder}.png"; fig.savefig(path, dpi=180); plt.close(fig)


def make_grids(output: Path, case: Case, rows: list[dict[str, Any]]) -> None:
    for mode in MODES:
        branch = sorted([r for r in rows if r["sample_id"] == case.sample_id and r["timestep_mode"] == mode], key=lambda r: (int(r["pair_id"]), -int(r["perturbation_sign"])))
        labels = [f"pair {int(r['pair_id']):02d} {'+' if int(r['perturbation_sign']) > 0 else '-'}" for r in branch]
        early = [Image.open(output / r["early_image_path"]).convert("RGB") for r in branch]
        final = [Image.open(output / r["final_image_path"]).convert("RGB") for r in branch]
        base = output / "grids" / case.sample_id
        label_image_grid(early, labels, base / f"{mode}_early_grid.png", f"{case.sample_id} | {mode} | early predicted-clean")
        label_image_grid(final, labels, base / f"{mode}_final_grid.png", f"{case.sample_id} | {mode} | final")


def summary_markdown(analysis: dict[str, list[dict[str, Any]]], subgroup: list[dict[str, Any]]) -> str:
    clip_corr = [r for r in analysis["correlations"] if r["encoder"] == "clip" and not int(r["degenerate"])]
    clip_pair = [r for r in analysis["pair_stats"] if r["encoder"] == "clip" and not int(r["degenerate"])]
    clip_ranges = [r for r in analysis["ranges"] if r["encoder"] == "clip" and not int(r["degenerate"])]
    clip_rev = [r for r in analysis["reversals"] if r["encoder"] == "clip" and not int(r["degenerate"])]
    def med(rows: list[dict[str, Any]], key: str, mode: str) -> float:
        vals = np.array([r[key] for r in rows if r["timestep_mode"] == mode], float)
        return float(np.nanmedian(vals)) if len(vals) and np.isfinite(vals).any() else math.nan
    lines = ["# Early-response mechanism probe", "", "The statements below describe frozen-encoder directional-progress proxies only; they are not editing-strength ground truth.", ""]
    lines += ["## Question A — early ranking predicts final ranking?", "", f"CLIP median overall Spearman: first_only={med(clip_corr, 'spearman', 'first_only'):.3f}; first_two={med(clip_corr, 'spearman', 'first_two'):.3f}. Median reversal ratio: first_only={med(clip_rev, 'reversal_ratio', 'first_only'):.3f}; first_two={med(clip_rev, 'reversal_ratio', 'first_two'):.3f}.", ""]
    lines += ["## Question B — which schedule is stronger/stabler?", "", f"Compare the same statistics above together with median antithetic sign agreement: first_only={med(clip_pair, 'sign_agreement_ratio', 'first_only'):.3f}; first_two={med(clip_pair, 'sign_agreement_ratio', 'first_two'):.3f}. This is an engineering comparison of the two probe/readout schedules, not a causal timestep claim.", ""]
    lines += ["## Question C — do antithetic finite-difference directions persist?", "", f"Median delta Spearman: first_only={med(clip_pair, 'delta_spearman', 'first_only'):.3f}; first_two={med(clip_pair, 'delta_spearman', 'first_two'):.3f}. Near-ties are excluded from sign-agreement denominators.", ""]
    lines += ["## Question D — is response dynamic range sufficient?", "", f"Median early/final q ranges, first_only=({med(clip_ranges, 'range_q_early', 'first_only'):.4f}, {med(clip_ranges, 'range_q_final', 'first_only'):.4f}); first_two=({med(clip_ranges, 'range_q_early', 'first_two'):.4f}, {med(clip_ranges, 'range_q_final', 'first_two'):.4f}).", ""]
    lines += ["## Question E — appearance-like vs structural/discrete", "", "See `subgroup_summary.csv` for the predeclared subgroup comparison. It is descriptive: no independent-branch p-values are reported.", ""]
    return "\n".join(lines)


def configuration(args: argparse.Namespace, cases: list[Case]) -> dict[str, Any]:
    payload = {"cases": [asdict(x) for x in cases], "steps": args.steps, "guidance": args.guidance, "sde_scale": args.sde_scale, "pairs": args.pairs, "branch_batch_size": args.branch_batch_size, "experiment_seed": args.experiment_seed, "modes": list(MODES), "clip_model": args.clip_model, "dino_model": args.dino_model}
    payload["fingerprint"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def self_test() -> None:
    ref = torch.zeros((1, 3, 4)); z = perturb_noise(ref, 19, 2, 3, 0); z_repeat = perturb_noise(ref, 19, 2, 3, 0)
    # The noise helper deliberately has no mode argument: first_only and
    # first_two therefore consume identical z0 for an equal sample/pair/step.
    assert torch.equal(z, z_repeat) and torch.equal(z, perturb_noise(ref, 19, 2, 3, 0))
    assert torch.allclose(torch.linalg.vector_norm(z), torch.linalg.vector_norm(-z))
    src = np.array([1., 0.]); edit = np.array([0., 1.]); values = np.array([src, edit, np.array([2., -1.])])
    q, degenerate = q_values(src, edit, values)
    assert not degenerate and abs(q[0]) < 1e-12 and abs(q[1] - 1) < 1e-12 and q[2] < 0
    assert safe_corr([1, 2, 3], [1, 2, 3])[0] > .99
    base = {"sample_id": "unit", "category": "unit", "subgroup": "unit", "timestep_mode": "first_only", "degenerate_clip": 0}
    unit_rows = [
        {**base, "pair_id": 0, "perturbation_sign": 1, "q_early_clip": .2, "q_final_clip": .2},
        {**base, "pair_id": 0, "perturbation_sign": -1, "q_early_clip": -.2, "q_final_clip": -.2},
        {**base, "pair_id": 1, "perturbation_sign": 1, "q_early_clip": .3, "q_final_clip": -.3},
        {**base, "pair_id": 1, "perturbation_sign": -1, "q_early_clip": -.3, "q_final_clip": .3},
    ]
    check = compute_analysis(unit_rows, {"clip"})
    assert [r["sign_agree"] for r in check["pairs"]] == [1, 0]
    assert check["reversals"][0]["reversal_pairs"] > 0
    tie_rows = [dict(row) for row in unit_rows]
    tie_rows[0]["q_early_clip"] = tie_rows[1]["q_early_clip"]
    assert compute_analysis(tie_rows, {"clip"})["pairs"][0]["valid"] == 0
    print("Self-test passed: deterministic antithetic noise, q anchors, reversal/near-tie, and correlation utilities.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("configs/early_response_16_cases.json"))
    parser.add_argument("--model", type=Path, default=Path("/root/autodl-tmp/FLUX.1-Kontext-dev"))
    parser.add_argument("--output", type=Path, default=Path("outputs/early_response_probe_16"))
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--dino-model", default="facebook/dinov2-base")
    parser.add_argument("--encoder-cache", type=Path, default=Path("/root/autodl-tmp/hf_encoder_cache"))
    parser.add_argument("--encoder-local-files-only", action="store_true")
    parser.add_argument("--skip-dino", action="store_true")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--sde-scale", type=float, default=.7)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--branch-batch-size", type=int, default=4)
    parser.add_argument("--experiment-seed", type=int, default=20260825)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for FLUX generation. Restore the H800 before running this probe.")
    if args.steps != 10 or args.guidance != 3.5:
        raise ValueError("This revised experiment is locked to --steps 10 and --guidance 3.5.")
    if args.pairs < 1 or args.branch_batch_size < 2 or args.branch_batch_size % 2:
        raise ValueError("pairs must be positive and branch_batch_size must be an even integer >= 2.")
    cases = load_cases(args.cases)
    if args.limit:
        cases = cases[:args.limit]
    cfg = configuration(args, cases)
    run_config_path = args.output / "run_config.json"
    if args.output.exists() and any(args.output.iterdir()) and not args.resume and not args.overwrite:
        raise FileExistsError(f"{args.output} is nonempty. Use --resume for matching config or --overwrite to replace it.")
    if args.overwrite and args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.resume and run_config_path.exists():
        old = json.loads(run_config_path.read_text(encoding="utf-8"))
        if old.get("fingerprint") != cfg["fingerprint"]:
            raise ValueError("Resume configuration fingerprint differs from the existing run.")
    atomic_text(run_config_path, json.dumps({**cfg, "encoder_status": "initializing"}, indent=2) + "\n")
    shutil.copy2(args.cases, args.output / "early_response_16_cases.json")
    existing = read_csv(args.output / "early_response_results.csv") if args.resume else []
    completed = {(r["sample_id"], r["timestep_mode"]) for r in existing}
    device = torch.device("cuda")
    seed_all(args.experiment_seed)
    pipe = FluxKontextPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    pipe.set_progress_bar_config(disable=True)
    encoders = ImageEncoders(args, device)
    atomic_text(run_config_path, json.dumps({**cfg, "encoder_status": encoders.status}, indent=2) + "\n")
    all_rows: list[dict[str, Any]] = existing
    for sample_index, case in enumerate(cases):
        state = prepare(pipe, case.source_image, case.instruction, case.base_seed, args.steps, args.guidance, device)
        sample_root = args.output / "images" / case.sample_id
        sample_root.mkdir(parents=True, exist_ok=True)
        state["source"].save(sample_root / "source.png")
        normal = normal_edit(pipe, state)
        normal.save(sample_root / "normal_full_edit.png")
        for mode in MODES:
            if (case.sample_id, mode) in completed:
                continue
            metadata, early, final = generate_mode(pipe, state, case, mode, sample_index, args)
            for row, early_image, final_image in zip(metadata, early, final):
                tag = row["perturbation_id"]
                early_rel = Path("images") / case.sample_id / mode / "early" / f"{tag}.jpg"
                final_rel = Path("images") / case.sample_id / mode / "final" / f"{tag}.jpg"
                save_jpeg(early_image, args.output / early_rel); save_jpeg(final_image, args.output / final_rel)
                row.update({"sample_id": case.sample_id, "category": case.category, "subgroup": case.subgroup, "instruction": case.instruction, "base_seed": case.base_seed, "steps": args.steps, "guidance": args.guidance, "sde_scale": args.sde_scale, "early_image_path": str(early_rel), "final_image_path": str(final_rel)})
            attach_scores(metadata, state["source"], normal, early, final, encoders)
            all_rows.extend(metadata)
            atomic_csv(args.output / "early_response_results.csv", all_rows)
            print(f"completed {case.sample_id} {mode}: {len(metadata)} branches", flush=True)
        # Rebuild all per-sample artifacts only after both mode results are present.
        sample_rows = [r for r in all_rows if r["sample_id"] == case.sample_id]
        if len(sample_rows) == args.pairs * 2 * len(MODES):
            active = {name for name, value in encoders.status.items() if value["status"] == "enabled"}
            analysis = compute_analysis(all_rows, active)
            make_grids(args.output, case, sample_rows)
            for encoder in active:
                make_scatters(args.output, case, all_rows, analysis, encoder)
    active = {name for name, value in encoders.status.items() if value["status"] == "enabled"}
    analysis = compute_analysis(all_rows, active)
    tables = {
        "sample_correlations.csv": wide_encoder_table(analysis["correlations"], ("sample_id", "category", "subgroup", "timestep_mode", "n_signed")),
        "antithetic_pair_responses.csv": wide_encoder_table(analysis["pairs"], ("sample_id", "category", "subgroup", "timestep_mode", "pair_id")),
        "antithetic_response_stats.csv": wide_encoder_table(analysis["pair_stats"], ("sample_id", "category", "subgroup", "timestep_mode", "total_pairs")),
        "sample_response_ranges.csv": wide_encoder_table(analysis["ranges"], ("sample_id", "category", "subgroup", "timestep_mode")),
        "sample_reversal_stats.csv": wide_encoder_table(analysis["reversals"], ("sample_id", "category", "subgroup", "timestep_mode")),
        "subgroup_summary.csv": subgroup_summary(analysis, active),
    }
    for filename, rows in tables.items():
        atomic_csv(args.output / filename, rows)
    atomic_text(args.output / "summary.md", summary_markdown(analysis, tables["subgroup_summary.csv"]))
    atomic_text(run_config_path, json.dumps({**cfg, "encoder_status": encoders.status, "completed_rows": len(all_rows)}, indent=2) + "\n")
    print(f"Done. Wrote {len(all_rows)} branch rows to {args.output}")


if __name__ == "__main__":
    main()
