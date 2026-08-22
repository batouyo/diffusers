from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _image_tensor(images: list[Image.Image], device: torch.device) -> torch.Tensor:
    values = [np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0 for image in images]
    return torch.from_numpy(np.stack(values)).permute(0, 3, 1, 2).to(device)


def _mask_for_image(mask: np.ndarray, image: Image.Image) -> torch.Tensor:
    if mask.shape != (image.height, image.width):
        mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        mask_image = mask_image.resize(image.size, Image.Resampling.NEAREST)
        mask = np.asarray(mask_image, dtype=np.uint8) > 0
    return torch.from_numpy(mask.astype(np.float32))


def masked_errors(
    source: Image.Image,
    outputs: list[Image.Image],
    edit_mask: np.ndarray,
) -> tuple[list[float | None], list[float | None]]:
    source_tensor = _image_tensor([source], torch.device("cpu"))[0]
    output_tensor = _image_tensor(outputs, torch.device("cpu"))
    preserve = 1.0 - _mask_for_image(edit_mask, source)
    area = float(preserve.sum().item())
    if area == 0.0:
        return [None] * len(outputs), [None] * len(outputs)
    diff = output_tensor - source_tensor.unsqueeze(0)
    weights = preserve.unsqueeze(0).unsqueeze(0)
    l1 = (diff.abs() * weights).sum(dim=(1, 2, 3)) / (area * 3.0)
    l2 = (diff.square() * weights).sum(dim=(1, 2, 3)).sqrt() / (area * 3.0) ** 0.5
    return [float(value) for value in l1], [float(value) for value in l2]


@torch.no_grad()
def clip_directional_scores(models: Any, source: Image.Image, outputs: list[Image.Image], text: str) -> list[float]:
    device = models.device
    source_batch = models.clip_processor(images=[source], return_tensors="pt")
    output_batch = models.clip_processor(images=outputs, return_tensors="pt")
    text_batch = models.clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    source_batch = {key: value.to(device) for key, value in source_batch.items()}
    output_batch = {key: value.to(device) for key, value in output_batch.items()}
    text_batch = {key: value.to(device) for key, value in text_batch.items()}
    source_hidden = models.clip.vision_model(pixel_values=source_batch["pixel_values"]).pooler_output
    output_hidden = models.clip.vision_model(pixel_values=output_batch["pixel_values"]).pooler_output
    text_hidden = models.clip.text_model(
        input_ids=text_batch["input_ids"], attention_mask=text_batch.get("attention_mask")
    ).pooler_output
    source_feature = F.normalize(models.clip.visual_projection(source_hidden).float(), dim=-1)
    output_feature = F.normalize(models.clip.visual_projection(output_hidden).float(), dim=-1)
    text_feature = F.normalize(models.clip.text_projection(text_hidden).float(), dim=-1)
    direction = F.normalize(output_feature - source_feature, dim=-1)
    return [float(value) for value in (direction @ text_feature[0]).cpu()]


def _spearman(values: list[float]) -> float:
    if len(values) < 2 or np.std(values) < 1e-12:
        return 0.0
    return float(np.corrcoef(np.arange(len(values)), np.argsort(np.argsort(values)))[0, 1])


def compute_metrics(
    *,
    source: Image.Image,
    outputs: list[Image.Image],
    strengths: list[float],
    edit_mask: np.ndarray,
    models: Any | None,
    target_text: str,
) -> dict[str, Any]:
    l1, l2 = masked_errors(source, outputs, edit_mask)
    result: dict[str, Any] = {
        "preserve_l1_mean": None if l1[0] is None else float(np.mean(l1)),
        "preserve_l2_mean": None if l2[0] is None else float(np.mean(l2)),
        "preserve_l1_s0": l1[0],
        "preserve_l1_s1": l1[-1],
        "preserve_l2_s0": l2[0],
        "preserve_l2_s1": l2[-1],
        "preserve_area_fraction": float(1.0 - edit_mask.mean()),
    }
    if models is None:
        result.update({
            "edit_dynamic_range": None,
            "monotonicity": None,
            "spearman": None,
            "monotonic_violations": None,
        })
        return result
    scores = clip_directional_scores(models, source, outputs, target_text)
    result["clip_scores"] = scores
    result["edit_dynamic_range"] = float(scores[-1] - scores[0])
    result["monotonicity"] = float(sum(right > left for left, right in zip(scores, scores[1:])) / max(1, len(scores) - 1))
    result["spearman"] = _spearman(scores)
    result["monotonic_violations"] = int(sum(right <= left for left, right in zip(scores, scores[1:])))
    image_tensor = _image_tensor(outputs, models.device)
    adjacent = models.lpips_distance(image_tensor[:-1], image_tensor[1:]).detach().cpu().tolist()
    endpoint = float(models.lpips_distance(image_tensor[0:1], image_tensor[-1:]).item())
    result["adjacent_lpips_mean"] = float(np.mean(adjacent)) if adjacent else 0.0
    result["adjacent_lpips_max"] = float(max(adjacent, default=0.0))
    result["path_length_ratio"] = float(sum(adjacent) / max(endpoint, 1e-8))
    result["endpoint_lpips"] = endpoint
    preserve = 1.0 - _mask_for_image(edit_mask, outputs[0]).to(models.device)
    if float(preserve.sum()) == 0.0:
        result["pairwise_preserve_lpips"] = None
    else:
        pairwise: list[float] = []
        for index in range(len(outputs)):
            for other in range(index + 1, len(outputs)):
                delta = (image_tensor[index] - image_tensor[other]).abs() * preserve.unsqueeze(0)
                pairwise.append(float(delta.sum().item() / (preserve.sum().item() * 3.0)))
        result["pairwise_preserve_lpips"] = float(np.mean(pairwise)) if pairwise else 0.0
    for strength, score in zip(strengths, scores):
        result[f"clip_q_s{strength:.2f}"] = score
    return result
