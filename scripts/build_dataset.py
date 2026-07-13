"""Build a fresh, deterministic 8-category manifest from raw UltraEdit pairs.

This script intentionally does not read any earlier probing manifest or candidate list.
Reference target images are retained only for dataset QC and are never used by the block scorer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps

CATEGORY_PATTERNS = {
    "material_or_texture": [
        r"\b(material|texture|textured|metallic|metal|wooden|wood|glass|gold|silver|marble|fabric|fur|furry|leather|stone|rust|plastic|ceramic|velvet|denim)\b",
        r"made (?:out )?of",
    ],
    "lighting": [
        r"\b(light|lighting|lit|illuminat|bright|dark|shadow|sunset|sunrise|night|glow|glowing|neon|exposure|backlit)\w*\b",
    ],
    "background_or_environment": [
        r"\b(background|environment|surroundings|sky|weather|season|beach|forest|desert|mountain|snow|rain|fog|mist|cloud|ocean|underwater|space|street|room|landscape)\w*\b",
    ],
    "global_style": [
        r"\b(style|stylized|painting|watercolor|oil paint|cartoon|anime|sketch|drawing|illustration|vintage|retro|cinematic|pixel art|black and white|monochrome|impressionist|surreal)\w*\b",
    ],
    "shape_or_geometry": [
        r"\b(shape|geometry|round|circular|square|rectangular|triangle|longer|shorter|wider|narrower|thin|thick|enlarge|shrink|smaller|larger|bend|bent|straight|curve|curved|reshape)\w*\b",
    ],
    "pose_or_spatial_relation": [
        r"\b(move|position|pose|facing|sitting|standing|lying|kneeling|jumping|walking|rotate|left|right|above|below|behind|front|next to|beside|between|near|far)\w*\b",
    ],
    "local_attribute": [
        r"\b(attribute|wearing|wear|hat|glasses|sunglasses|beard|mustache|smile|expression|eyes|hair|sleeves|collar|pattern|logo|stripe|spots|wings|horns|tail)\w*\b",
    ],
}


def normalize_path(path: str) -> Path:
    if path.startswith("/data/hyp/"):
        path = "/data15/hyp/" + path[len("/data/hyp/") :]
    return Path(path)


def classify(row: dict) -> str | None:
    if row.get("task") == "change_color":
        return "color"
    text = row["instruction"].lower()
    order = [
        "material_or_texture",
        "lighting",
        "background_or_environment",
        "global_style",
        "shape_or_geometry",
        "pose_or_spatial_relation",
        "local_attribute",
    ]
    for category in order:
        if any(re.search(pattern, text) for pattern in CATEGORY_PATTERNS[category]):
            return category
    if row.get("task") == "change_global":
        return "global_style"
    if row.get("task") == "change_local":
        return "local_attribute"
    return None


def acceptable(row: dict) -> bool:
    instruction = " ".join(row.get("instruction", "").split())
    words = instruction.split()
    if not 4 <= len(words) <= 28:
        return False
    lowered = instruction.lower()
    banned = ["make it better", "more beautiful", "more appealing", "multiple changes"]
    if any(value in lowered for value in banned):
        return False
    if lowered.count(" and ") > 1 or ";" in instruction:
        return False
    source = normalize_path(row["sourceImage"])
    target = normalize_path(row["targetImage"])
    return source.is_file() and target.is_file()


def stable_order(identifier: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest()


def build(args) -> None:
    pools = defaultdict(list)
    with open(args.source_manifest, "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not acceptable(row):
                continue
            category = classify(row)
            if category:
                pools[category].append(row)

    categories = list(CATEGORY_PATTERNS) + ["color"]
    categories = [
        "color",
        "material_or_texture",
        "local_attribute",
        "shape_or_geometry",
        "pose_or_spatial_relation",
        "background_or_environment",
        "global_style",
        "lighting",
    ]
    output_root = Path(args.output_root)
    image_root = output_root / "source_512"
    image_root.mkdir(parents=True, exist_ok=True)
    output_rows = []
    summary = {"source_manifest": args.source_manifest, "categories": {}, "selection_seed": args.seed}

    for category in categories:
        candidates = sorted(pools[category], key=lambda row: stable_order(row["id"], args.seed))
        selected = []
        seen_hashes = set()
        for row in candidates:
            source = normalize_path(row["sourceImage"])
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if digest in seen_hashes:
                continue
            try:
                with Image.open(source) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    if min(image.size) < 256:
                        continue
                    prepared = ImageOps.fit(image, (512, 512), method=Image.Resampling.LANCZOS)
            except Exception:
                continue
            seen_hashes.add(digest)
            selected.append((row, prepared, digest))
            if len(selected) == args.discovery + args.heldout:
                break
        if len(selected) < args.discovery + args.heldout:
            raise RuntimeError(f"{category}: only {len(selected)} usable candidates")
        for index, (row, prepared, source_hash) in enumerate(selected):
            sample_id = f"{category}_{index:03d}_{row['id']}"
            prepared_path = image_root / f"{sample_id}.png"
            prepared.save(prepared_path, format="PNG")
            split = "discovery" if index < args.discovery else "heldout"
            instruction = " ".join(row["instruction"].split())
            output_rows.append(
                {
                    "id": sample_id,
                    "image": str(prepared_path),
                    "instruction": instruction,
                    "category": category,
                    "target_description": f"The requested edit is visibly completed: {instruction}",
                    "split": split,
                    "source_dataset": row.get("datasetSource", "ultraedit"),
                    "source_id": row["id"],
                    "source_sha256": source_hash,
                    "prepared_sha256": hashlib.sha256(prepared_path.read_bytes()).hexdigest(),
                    "reference_target_image": str(normalize_path(row["targetImage"])),
                    "input_description": row.get("inputDescription"),
                    "license": "UltraEdit research dataset; verify upstream terms before redistribution",
                    "qc_status": "automatic_pass_pending_human_review",
                }
            )
        summary["categories"][category] = {
            "pool": len(candidates),
            "discovery": args.discovery,
            "heldout": args.heldout,
        }

    output_rows.sort(key=lambda row: (categories.index(row["category"]), row["split"], row["id"]))
    manifest = Path(args.output_manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_root / "dataset_build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest",
        default="/data15/hyp/dataset/gptImageEdit15MAdapterSubset/metadata/adapterTrainManifest.jsonl",
    )
    parser.add_argument("--output-root", default="/data15/hyp/dataset/flux-kontext-block-probing")
    parser.add_argument("--output-manifest", default="/home/hyp/Code/flux-kontext-block-probing/dataset.jsonl")
    parser.add_argument("--discovery", type=int, default=20)
    parser.add_argument("--heldout", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260714)
    build(parser.parse_args())


if __name__ == "__main__":
    main()

