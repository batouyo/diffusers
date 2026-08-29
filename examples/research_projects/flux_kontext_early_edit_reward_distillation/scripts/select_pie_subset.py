#!/usr/bin/env python3
"""Select a tiny, category-balanced PIE-Bench subset with native masks."""
from __future__ import annotations
import argparse
from pathlib import Path
from early_edit_reward_distillation.pie import select_pie_subset, write_pie_manifest

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--dataset-root", required=True); p.add_argument("--output-root", required=True); p.add_argument("--seed", type=int, default=20260830); p.add_argument("--train-count", type=int, default=16); p.add_argument("--test-count", type=int, default=8); p.add_argument("--per-category", type=int, default=3); p.add_argument("--min-area", type=float, default=0.02); p.add_argument("--max-area", type=float, default=0.40); args = p.parse_args()
    root = Path(args.output_root); root.mkdir(parents=True, exist_ok=True)
    train, test = select_pie_subset(args.dataset_root, args.train_count, args.test_count, args.seed, args.per_category, args.min_area, args.max_area)
    write_pie_manifest(root / "pie_train16.json", train); write_pie_manifest(root / "pie_test8.json", test)
    for record in train + test:
        folder = root / "samples" / str(record["sample_id"]); folder.mkdir(parents=True, exist_ok=True)
        record["source"].save(folder / "source.png"); record["mask"].save(folder / "edit_mask.png")
    print(f"selected train={len(train)} test={len(test)} categories={len(set(r['category'] for r in train + test))} output={root}")

if __name__ == "__main__": main()
