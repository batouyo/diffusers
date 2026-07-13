"""Derive pipeline job counts from runtime structure, config, and dataset rows."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml


ROOT = Path("/home/hyp/Code/flux-kontext-block-probing")


def compute_counts(config: dict, structure: dict, dataset: list[dict]) -> dict[str, int]:
    categories = list(config["dataset"]["categories"])
    by_category_split = Counter((row["category"], row["split"]) for row in dataset)
    pilot_samples = sum(
        min(by_category_split[(category, "discovery")], config["dataset"]["pilot_per_category"])
        for category in categories
    )
    discovery_samples = sum(by_category_split[(category, "discovery")] for category in categories)
    heldout_samples = sum(by_category_split[(category, "heldout")] for category in categories)
    total_blocks = int(structure["total_block_count"])
    seed_count = len(config["inference"]["seeds"])
    return {
        "runtime_blocks": total_blocks,
        "pilot_samples": pilot_samples,
        "pilot_stage1_jobs": pilot_samples * (1 + total_blocks),
        "discovery_samples": discovery_samples,
        "heldout_samples": heldout_samples,
        "formal_seed_count": seed_count,
        "formal_baseline_jobs": discovery_samples * seed_count,
        "formal_enhance_jobs": discovery_samples * seed_count * total_blocks,
        "formal_disable_jobs": discovery_samples * seed_count * int(config["probing"]["stage2_blocks"]),
        "formal_remove_jobs": discovery_samples * seed_count * int(config["probing"]["stage3_blocks"]),
    }


def load_counts(root: Path = ROOT) -> dict[str, int]:
    config = yaml.safe_load((root / "probe_config.yaml").read_text(encoding="utf-8"))
    structure_path = Path(config["project"]["output_root"]) / "preflight" / "structure_report.json"
    structure = json.loads(structure_path.read_text(encoding="utf-8"))
    dataset = [
        json.loads(line)
        for line in Path(config["project"]["dataset_manifest"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return compute_counts(config, structure, dataset)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", choices=sorted(load_counts()))
    args = parser.parse_args()
    counts = load_counts()
    print(counts[args.field] if args.field else json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
