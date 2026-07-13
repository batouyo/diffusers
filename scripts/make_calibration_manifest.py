"""Lock the exact 80 pilot outputs that will be human/VLM calibrated."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import yaml

try:
    from make_calibration_bundle import stable_key
except ModuleNotFoundError:  # package import during tests
    from scripts.make_calibration_bundle import stable_key


ROOT = Path("/home/hyp/Code/flux-kontext-block-probing")


def balanced_category_items(items: list[tuple[Path, dict]]) -> list[tuple[Path, dict, str]]:
    """Select ten items with one baseline and four enhanced outputs in each blind half."""
    baselines = sorted(
        (item for item in items if item[1]["mode"] == "baseline"),
        key=lambda item: stable_key(item[1]["sample_id"]),
    )
    enhanced = sorted(
        (item for item in items if item[1]["mode"] == "enhance_text"),
        key=lambda item: stable_key(f"{item[1]['sample_id']}:{item[1]['global_block_index']}"),
    )
    if len(baselines) < 2:
        raise RuntimeError("each calibration category requires at least two baselines")
    chosen_enhanced = []
    used_samples = {baselines[0][1]["sample_id"], baselines[1][1]["sample_id"]}
    for item in enhanced:
        if item[1]["sample_id"] in used_samples and len(used_samples) < 5:
            continue
        chosen_enhanced.append(item)
        used_samples.add(item[1]["sample_id"])
        if len(chosen_enhanced) == 8:
            break
    if len(chosen_enhanced) != 8:
        raise RuntimeError(f"expected eight enhanced calibration items, got {len(chosen_enhanced)}")
    prompt_half = [baselines[0], *chosen_enhanced[:4]]
    locked_half = [baselines[1], *chosen_enhanced[4:]]
    return [
        *(item + ("prompt_calibration",) for item in prompt_half),
        *(item + ("locked_validation",) for item in locked_half),
    ]


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
        for meta_path, meta, subset in balanced_category_items(grouped[category]):
            selected.append(meta_path)
            selection_key[str(meta_path)] = {
                "subset": subset,
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
