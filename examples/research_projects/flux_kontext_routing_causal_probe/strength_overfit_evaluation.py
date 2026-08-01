"""Evaluation primitives and artifact writers for strength trajectories."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw


def relative_rms(first: torch.Tensor, second: torch.Tensor, eps: float = 1e-12) -> float:
    return float((first.float() - second.float()).square().mean().sqrt().item() / (second.float().square().mean().sqrt().item() + eps))


def rankdata(values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(values), dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def spearman(strengths: Iterable[float], progress: Iterable[float]) -> float:
    x, y = rankdata(strengths), rankdata(progress)
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def monotonic_violations(progress: Iterable[float], tolerance: float = 0.0) -> int:
    values = list(progress)
    return sum(right + tolerance < left for left, right in zip(values, values[1:]))


def dino_projection_progress(features: torch.Tensor, neutral_feature: torch.Tensor, full_feature: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    direction = full_feature.float() - neutral_feature.float()
    displacement = features.float() - neutral_feature.float()
    return (displacement * direction).flatten(1).sum(dim=1) / direction.flatten(1).square().sum(dim=1).clamp_min(eps)


def lpips_progress(distance_to_neutral: torch.Tensor, distance_to_full: torch.Tensor, endpoint_distance: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (distance_to_neutral - distance_to_full + endpoint_distance) / (2 * endpoint_distance.clamp_min(eps))


def path_statistics(strengths: list[float], progress: list[float], adjacent_lpips: list[float], endpoint_lpips: float) -> dict[str, float | int]:
    max_jump = max(adjacent_lpips, default=0.0)
    return {
        "spearman": spearman(strengths, progress),
        "monotonic_violations": monotonic_violations(progress),
        "max_adjacent_lpips": max_jump,
        "max_jump_ratio": max_jump / max(endpoint_lpips, 1e-8),
        "path_length_ratio": sum(adjacent_lpips) / max(endpoint_lpips, 1e-8),
    }


def save_contact_sheet(images: list[Image.Image], labels: list[str], destination: str | Path, panel: int = 256) -> None:
    if len(images) != len(labels):
        raise ValueError("images and labels differ in length")
    label_height = 28
    canvas = Image.new("RGB", (panel * len(images), panel + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(images, labels)):
        fitted = image.convert("RGB").copy()
        fitted.thumbnail((panel, panel), Image.Resampling.LANCZOS)
        x = index * panel + (panel - fitted.width) // 2
        y = (panel - fitted.height) // 2
        canvas.paste(fitted, (x, y))
        draw.text((index * panel + 4, panel + 4), label, fill="black")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def append_metrics(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def write_summary_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class PerceptualModels:
    """Lazy local DINO/CLIP/LPIPS evaluator. Network download is disabled."""

    def __init__(
        self,
        *,
        dino_path: str = "/data15/hyp/weight/dinov2-large",
        clip_path: str = "/data15/hyp/weight/huggingface/hub/models--openai--clip-vit-large-patch14",
        device: str = "cuda",
    ) -> None:
        from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
        import lpips

        self.device = torch.device(device)
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_path, local_files_only=True)
        self.dino = AutoModel.from_pretrained(dino_path, local_files_only=True).to(self.device).eval().requires_grad_(False)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_path, local_files_only=True)
        self.clip = CLIPModel.from_pretrained(clip_path, local_files_only=True).to(self.device).eval().requires_grad_(False)
        self.lpips = lpips.LPIPS(net="alex", spatial=True).to(self.device).eval().requires_grad_(False)

    @torch.no_grad()
    def dino_features(self, images: list[Image.Image]) -> torch.Tensor:
        batch = self.dino_processor(images=images, return_tensors="pt").to(self.device)
        return self.dino(**batch).last_hidden_state[:, 0].float()

    @torch.no_grad()
    def clip_image_text(self, images: list[Image.Image], texts: list[str]) -> torch.Tensor:
        batch = self.clip_processor(images=images, text=texts, return_tensors="pt", padding=True).to(self.device)
        output = self.clip(**batch)
        return torch.nn.functional.cosine_similarity(output.image_embeds.float(), output.text_embeds.float(), dim=-1)

    @torch.no_grad()
    def lpips_distance(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return self.lpips(first.to(self.device) * 2 - 1, second.to(self.device) * 2 - 1).mean(dim=(1, 2, 3))


def report_stage(path: str | Path, title: str, body: str, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(
        f"# {title}\n\nGenerated: {timestamp}\n\n{body}\n\n## Raw summary\n\n~~~json\n{json.dumps(rows, indent=2, default=str)}\n~~~\n",
        encoding="utf-8",
    )

