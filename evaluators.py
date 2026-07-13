"""Reproducible VLM edit scoring and source-preservation metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageOps

LOGGER = logging.getLogger("flux_probe.evaluator")

RUBRIC = """You are a strict image-edit completion judge.
Image 1 is the source. Image 2 is the edited output.
Instruction: {instruction}
Target description: {target_description}

Judge only whether the requested target edit is visibly present on the correct object or region.
Do not reward aesthetics, general image quality, or unrelated changes.
Return exactly one JSON object with:
- target_present: integer 0, 1, or 2 (absent, partial/ambiguous, clearly present)
- correct_object: integer 0 or 1
- localized_as_requested: integer 0 or 1
- score_0_to_4: integer sum of the three fields
- evidence: a brief visual observation, maximum 20 words
"""


def resolve_snapshot(path: str | Path) -> str:
    root = Path(path)
    if (root / "config.json").exists():
        return str(root)
    snapshots = sorted((root / "snapshots").glob("*"))
    usable = [item for item in snapshots if (item / "config.json").exists()]
    if not usable:
        raise FileNotFoundError(f"No Hugging Face snapshot below {root}")
    return str(usable[-1])


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluation_hash(config: dict) -> str:
    value = {
        "source_sha256": file_sha256(__file__),
        "rubric": RUBRIC,
        "qwen_model": resolve_snapshot(config["evaluation"]["qwen_model"]),
        "dino_model": resolve_snapshot(config["evaluation"]["dino_model"]),
        "max_new_tokens": config["evaluation"]["vlm_max_new_tokens"],
        "lpips": "alex-v0.1",
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def parse_json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"VLM did not return JSON: {text!r}")
    value = json.loads(match.group(0))
    required = {"target_present", "correct_object", "localized_as_requested", "score_0_to_4"}
    if not required.issubset(value):
        raise ValueError(f"VLM JSON missing {sorted(required - set(value))}: {value}")
    target = int(value["target_present"])
    correct = int(value["correct_object"])
    localized = int(value["localized_as_requested"])
    score = int(value["score_0_to_4"])
    if target not in {0, 1, 2} or correct not in {0, 1} or localized not in {0, 1}:
        raise ValueError(f"VLM rubric values out of range: {value}")
    computed = target + correct + localized
    if score != computed:
        score = computed
        value["score_corrected"] = True
    value["score_0_to_4"] = score
    return value


class QwenEditJudge:
    def __init__(self, model_path: str, device: str, max_new_tokens: int = 160):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        snapshot = resolve_snapshot(model_path)
        self.processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            snapshot,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        ).to(device)
        self.model.eval()
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model_path = snapshot

    def score(self, source_path: str, output_path: str, instruction: str, target_description: str) -> tuple[dict, str]:
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": source_path},
                    {"type": "image", "image": output_path},
                    {
                        "type": "text",
                        "text": RUBRIC.format(
                            instruction=instruction,
                            target_description=target_description,
                        ),
                    },
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        trimmed = [output[len(input_ids) :] for input_ids, output in zip(inputs.input_ids, generated)]
        raw = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
        return parse_json_object(raw), raw


class DinoPreservation:
    def __init__(self, model_path: str, device: str):
        from transformers import AutoImageProcessor, AutoModel

        snapshot = resolve_snapshot(model_path)
        self.processor = AutoImageProcessor.from_pretrained(snapshot, local_files_only=True)
        self.model = AutoModel.from_pretrained(snapshot, local_files_only=True).to(device)
        self.model.eval()
        self.device = device
        self.model_path = snapshot

    def embed(self, path: str) -> torch.Tensor:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        vector = outputs.last_hidden_state[:, 0].float()
        return torch.nn.functional.normalize(vector, dim=-1)

    def similarity(self, source_path: str, output_path: str) -> float:
        source = self.embed(source_path)
        output = self.embed(output_path)
        return float((source * output).sum().cpu())


class LpipsDistance:
    def __init__(self, device: str):
        import lpips

        self.model = lpips.LPIPS(net="alex", version="0.1", verbose=False).to(device)
        self.model.eval()
        self.device = device

    def _tensor(self, path: str) -> torch.Tensor:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
            array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def distance(self, source_path: str, output_path: str) -> float:
        with torch.inference_mode():
            value = self.model(self._tensor(source_path), self._tensor(output_path))
        return float(value.squeeze().cpu())


def quality_flags(path: str) -> dict:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    finite = bool(np.isfinite(array).all())
    luminance = array.mean(axis=2)
    mean = float(luminance.mean())
    std = float(luminance.std())
    near_clip = float(((array <= 1 / 255) | (array >= 254 / 255)).mean())
    return {
        "finite": finite,
        "luminance_mean": mean,
        "luminance_std": std,
        "near_clip_fraction": near_clip,
        "all_black": mean < 0.01 and std < 0.01,
        "all_white": mean > 0.99 and std < 0.01,
        "severe_saturation": near_clip > 0.95,
    }


def load_dataset(path: str) -> dict[str, dict]:
    result = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            result[row["id"]] = row
    return result


def iter_metadata(run_root: Path):
    for artifact_root in [run_root / "images", run_root / "joint"]:
        if not artifact_root.exists():
            continue
        for path in sorted(artifact_root.rglob("*.json")):
            if path.name.endswith(".eval.json"):
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if value.get("status") == "complete" and value.get("output_path"):
                yield path, value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="probe_config.yaml")
    parser.add_argument("--run-id")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-items", type=int)
    parser.add_argument(
        "--metadata-list",
        help="UTF-8 file containing one metadata JSON path per line; order is preserved",
    )
    parser.add_argument("--skip-vlm", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_id = args.run_id or config["project"]["run_id"]
    run_root = Path(config["project"]["output_root"]) / run_id
    dataset = load_dataset(config["project"]["dataset_manifest"])
    metadata = list(iter_metadata(run_root))
    if args.metadata_list:
        run_root_resolved = run_root.resolve()
        requested = [
            Path(line.strip()).resolve()
            for line in Path(args.metadata_list).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(requested) != len(set(requested)):
            raise ValueError("metadata list contains duplicate paths")
        if any(not path.is_relative_to(run_root_resolved) for path in requested):
            raise ValueError("every requested metadata path must be inside the selected run root")
        available = {path.resolve(): (path, value) for path, value in metadata}
        missing = [str(path) for path in requested if path not in available]
        if missing:
            raise FileNotFoundError(f"metadata list contains {len(missing)} unavailable items: {missing[:3]}")
        metadata = [available[path] for path in requested]
    if args.max_items is not None:
        metadata = metadata[: args.max_items]
    LOGGER.info("selected %d metadata items for evaluation", len(metadata))
    judge = None if args.skip_vlm else QwenEditJudge(
        config["evaluation"]["qwen_model"], args.device, config["evaluation"]["vlm_max_new_tokens"]
    )
    dino = DinoPreservation(config["evaluation"]["dino_model"], args.device)
    lpips_metric = LpipsDistance(args.device)
    evaluator_hash = evaluation_hash(config)
    for index, (meta_path, meta) in enumerate(metadata, 1):
        eval_path = meta_path.with_suffix(".eval.json")
        if eval_path.exists():
            try:
                prior = json.loads(eval_path.read_text(encoding="utf-8"))
                if (
                    prior.get("output_sha256") == meta.get("output_sha256")
                    and prior.get("evaluation_hash") == evaluator_hash
                ):
                    continue
            except Exception:
                pass
        row = dataset[meta["sample_id"]]
        result = {
            "sample_id": meta["sample_id"],
            "output_path": meta["output_path"],
            "output_sha256": meta["output_sha256"],
            "quality": quality_flags(meta["output_path"]),
            "dino_similarity": dino.similarity(row["image"], meta["output_path"]),
            "dino_model": dino.model_path,
            "lpips_distance": lpips_metric.distance(row["image"], meta["output_path"]),
            "evaluation_hash": evaluator_hash,
        }
        if judge is not None:
            try:
                rubric, raw = judge.score(
                    row["image"], meta["output_path"], row["instruction"], row["target_description"]
                )
                result.update(
                    {
                        "s_edit": rubric["score_0_to_4"] / 4.0,
                        "vlm_rubric": rubric,
                        "vlm_raw": raw,
                        "vlm_model": judge.model_path,
                        "vlm_parse_ok": True,
                    }
                )
            except Exception as exc:
                LOGGER.exception("VLM failed for %s", meta["output_path"])
                result.update({"s_edit": None, "vlm_parse_ok": False, "vlm_error": repr(exc)})
        atomic_json(eval_path, result)
        LOGGER.info("[%d/%d] evaluated %s", index, len(metadata), meta["output_path"])


if __name__ == "__main__":
    main()
