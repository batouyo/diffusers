"""Render pre-registered source/baseline/candidate/disable/random/all image grids."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps
import yaml


ROOT = Path("/home/hyp/Code/flux-kontext-block-probing")


def load_metadata(root: Path) -> list[dict]:
    result = []
    folder = root / "joint"
    if not folder.exists():
        return result
    for path in folder.rglob("*.json"):
        if path.name.endswith(".eval.json"):
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if value.get("status") == "complete" and value.get("output_path"):
            result.append(value)
    return result


def panel(path: str | None, label: str, size: int = 384) -> Image.Image:
    canvas = Image.new("RGB", (size, size + 42), "white")
    if not path or not Path(path).exists():
        raise FileNotFoundError(f"required grid panel is missing: {label} -> {path}")
    with Image.open(path) as image:
        image = ImageOps.fit(ImageOps.exif_transpose(image).convert("RGB"), (size, size), Image.Resampling.LANCZOS)
        canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, size + 12), label, fill="black")
    return canvas


def find_arm(records: list[dict], sample_id: str, arm: str, candidate: int) -> str | None:
    candidates = [row for row in records if row.get("sample_id") == sample_id and row.get("seed") == 42]
    if arm == "baseline":
        rows = [row for row in candidates if row.get("arm") == "baseline"]
    elif arm == "candidate":
        rows = [row for row in candidates if row.get("arm") == "candidate_combo"]
    elif arm == "disable":
        rows = [row for row in candidates if row.get("arm") == f"candidate_disable_g{candidate:03d}"]
    elif arm == "random":
        rows = [row for row in candidates if row.get("arm") == "random_00"]
    elif arm == "all":
        rows = [row for row in candidates if row.get("arm") == "all_blocks"]
    else:
        rows = []
    return rows[0]["output_path"] if rows else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "probe_config.yaml"))
    parser.add_argument("--split", default="heldout")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_root = Path(config["project"]["output_root"]) / config["project"]["run_id"]
    selection = json.loads((run_root / "selected_blocks.json").read_text(encoding="utf-8"))
    candidates = [int(value) for value in selection.get("selected_global_blocks") or []]
    if not candidates:
        raise RuntimeError("no independently selected candidate is available for held-out grids")
    candidate = candidates[0]
    joint = json.loads((run_root / "joint_validation.json").read_text(encoding="utf-8"))
    if (
        joint.get("execution_status") != "complete"
        or joint.get("exact_job_matrix_verified") is not True
        or joint.get("image_checksums_verified") is not True
        or joint.get("protocol_fingerprint") != joint.get("expected_protocol_fingerprint")
    ):
        raise RuntimeError("held-out image grids require exact current joint-validation evidence")
    dataset = [json.loads(line) for line in Path(config["project"]["dataset_manifest"]).read_text(encoding="utf-8").splitlines()]
    records = load_metadata(run_root)
    grids = run_root / "plots" / "image_grids"
    grids.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for category in config["dataset"]["categories"]:
        row = sorted(
            (item for item in dataset if item["split"] == args.split and item["category"] == category),
            key=lambda item: item["id"],
        )[0]
        sample_id = row["id"]
        paths = {
            "source": row["image"],
            "baseline": find_arm(records, sample_id, "baseline", candidate),
            "candidate": find_arm(records, sample_id, "candidate", candidate),
            "disable": find_arm(records, sample_id, "disable", candidate),
            "random": find_arm(records, sample_id, "random", candidate),
            "all_blocks": find_arm(records, sample_id, "all", candidate),
        }
        items = [
            panel(paths["source"], "source"),
            panel(paths["baseline"], "baseline"),
            panel(paths["candidate"], f"candidate combo {candidates}"),
            panel(paths["disable"], f"disable g{candidate}"),
            panel(paths["random"], "random matched"),
            panel(paths["all_blocks"], "all blocks"),
        ]
        grid = Image.new("RGB", (3 * items[0].width, 2 * items[0].height), "white")
        for index, item in enumerate(items):
            grid.paste(item, ((index % 3) * item.width, (index // 3) * item.height))
        grid_path = grids / f"{category}_{sample_id}.png"
        grid.save(grid_path)
        manifest_rows.append(
            {
                "category": category,
                "sample_id": sample_id,
                "candidate_global_blocks": candidates,
                "panels": {
                    name: {
                        "path": str(path),
                        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                    }
                    for name, path in paths.items()
                },
                "grid_path": str(grid_path),
                "grid_sha256": hashlib.sha256(grid_path.read_bytes()).hexdigest(),
            }
        )
    if len(manifest_rows) != len(config["dataset"]["categories"]):
        raise RuntimeError("one held-out grid per edit category is required")
    manifest = {
        "status": "complete",
        "candidate_global_blocks": candidates,
        "joint_protocol_fingerprint": joint["protocol_fingerprint"],
        "categories": list(config["dataset"]["categories"]),
        "grids": manifest_rows,
    }
    temporary = (grids / "image_grid_manifest.json").with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(grids / "image_grid_manifest.json")
    print(grids)


if __name__ == "__main__":
    main()
