#!/usr/bin/env python3
"""Select fixed local HQ-Edit train/test manifests without loading all shards."""
from __future__ import annotations
import argparse
from pathlib import Path
from early_edit_reward_distillation.data import select_local_subset, write_manifest

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, help="directory containing HQ-Edit parquet shards")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--train-count", type=int, default=16)
    parser.add_argument("--test-count", type=int, default=8)
    parser.add_argument("--min-area", type=float, default=0.02)
    parser.add_argument("--max-area", type=float, default=0.40)
    args = parser.parse_args()
    train, test = select_local_subset(args.dataset_root, args.train_count, args.test_count, args.seed, args.min_area, args.max_area)
    root = Path(args.output_root); root.mkdir(parents=True, exist_ok=True)
    write_manifest(root / "hqedit_train16.json", train)
    write_manifest(root / "hqedit_test8.json", test)
    for record in train + test:
        folder = root / "samples" / str(record["sample_id"]); folder.mkdir(parents=True, exist_ok=True)
        record["source"].save(folder / "source.png")
        record["target"].save(folder / "target.png")
        record["mask"].save(folder / "pseudo_edit_mask.png")
    print(f"selected train={len(train)} test={len(test)} output={root}")

if __name__ == "__main__":
    main()
