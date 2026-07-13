"""Lock the exact 80 pilot outputs that will be human/VLM calibrated."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import yaml

from make_calibration_bundle import stable_key


ROOT = Path("/home/hyp/Code/flux-kontext-block-probing")


def main() -> None:
    config = yaml.safe_load((ROOT / "probe_config.yaml").read_text(encoding="utf-8"))
    run_root = Path(config["project"]["output_root"]) / config["project"]["run_id"]
    grouped: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    for meta_path in (run_root / "images").rglob("*.json"):
        if meta_path.name.endswith(".eval.json"):
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") == "complete" and meta.get("split") == "discovery":
            grouped[meta["category"]].append((meta_path.resolve(), meta))

    selected: list[Path] = []
    selection_key: dict[str, dict] = {}
    for category in config["dataset"]["categories"]:
        items = grouped[category]
        baselines = sorted(
            (item for item in items if item[1]["mode"] == "baseline"),
            key=lambda item: stable_key(item[1]["sample_id"]),
        )
        enhanced = sorted(
            (item for item in items if item[1]["mode"] == "enhance_text"),
            key=lambda item: stable_key(f"{item[1]['sample_id']}:{item[1]['global_block_index']}"),
        )
        category_items = baselines[:2]
        used_samples = {item[1]["sample_id"] for item in category_items}
        for item in enhanced:
            if item[1]["sample_id"] in used_samples and len(used_samples) < 5:
                continue
            category_items.append(item)
            used_samples.add(item[1]["sample_id"])
            if len(category_items) == 10:
                break
        if len(category_items) != 10:
            raise RuntimeError(f"{category}: expected 10 calibration items, got {len(category_items)}")
        for meta_path, meta in category_items:
            selected.append(meta_path)
            selection_key[str(meta_path)] = {
                "category": category,
                "sample_id": meta["sample_id"],
                "mode": meta["mode"],
                "global_block_index": meta["global_block_index"],
                "output_sha256": meta["output_sha256"],
            }

    if len(selected) != config["evaluation"]["human_calibration_examples"]:
        raise RuntimeError(f"expected 80 calibration paths, got {len(selected)}")
    output = run_root / "calibration"
    output.mkdir(parents=True, exist_ok=True)
    (output / "metadata_manifest.txt").write_text(
        "\n".join(map(str, selected)) + "\n", encoding="utf-8"
    )
    (output / "metadata_selection_key.json").write_text(
        json.dumps(selection_key, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"selected": len(selected), "manifest": str(output / "metadata_manifest.txt")}, indent=2))


if __name__ == "__main__":
    main()
