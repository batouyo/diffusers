"""Audit source dimensions and FLUX-Kontext packed token geometry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from early_edit_reward_distillation.pie import decode_image
from early_edit_reward_distillation.resolution import choose_preferred_source_size, resolve_dimensions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--vae-scale-factor", type=int, default=8)
    args = parser.parse_args()

    records = []
    for manifest_path in args.manifest:
        for item in json.loads(Path(manifest_path).read_text()):
            table = pd.read_parquet(item["shard"]).iloc[[int(item["row_index"])]]
            source = decode_image(table.iloc[0]["image"])
            source_height, source_width = choose_preferred_source_size(source.height, source.width, args.vae_scale_factor)
            geometry = resolve_dimensions(args.height, args.width, args.vae_scale_factor)
            source_geometry = resolve_dimensions(source_height, source_width, args.vae_scale_factor)
            records.append({
                "sample_id": item["sample_id"],
                "category": item.get("category", ""),
                "shard": item["shard"],
                "row_index": item["row_index"],
                "source_width": source.width,
                "source_height": source.height,
                "source_conditioning_height": source_geometry["resolved_height"],
                "source_conditioning_width": source_geometry["resolved_width"],
                "source_conditioning_tokens": source_geometry["generated_image_tokens"],
                **geometry,
            })
    payload = {
        "requested_height": args.height,
        "requested_width": args.width,
        "vae_scale_factor": args.vae_scale_factor,
        "records": records,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"count": len(records), "output": args.output, "geometry": geometry}, indent=2))


if __name__ == "__main__":
    main()
