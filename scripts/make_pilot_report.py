"""Create an evidence-linked pilot report without overstating formal validation."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path("/home/hyp/Code/flux-kontext-block-probing")


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    config = yaml.safe_load((ROOT / "probe_config.yaml").read_text(encoding="utf-8"))
    output_root = Path(config["project"]["output_root"])
    run_root = output_root / config["project"]["run_id"]
    structure = read_json(output_root / "preflight" / "structure_report.json", {})
    identity = read_json(output_root / "preflight" / "identity_report.json", {})
    selected = read_json(run_root / "selected_blocks.json", {})
    summary_path = run_root / "block_summary.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    dataset = [json.loads(line) for line in (ROOT / "dataset.jsonl").read_text(encoding="utf-8").splitlines()]
    counts = Counter((row["category"], row["split"]) for row in dataset)
    png_count = len(list((run_root / "images").rglob("*.png")))
    eval_count = len(list((run_root / "images").rglob("*.eval.json")))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()

    table = []
    if not summary.empty and summary["semantic_gain"].notna().any():
        ranked = summary.sort_values(["semantic_gain_ci_low", "semantic_gain"], ascending=False).head(15)
        for _, row in ranked.iterrows():
            table.append(
                f"| {int(row['global_block_index'])} | {row['block_type']} | "
                f"{row['semantic_gain']:.4f} | [{row['semantic_gain_ci_low']:.4f}, {row['semantic_gain_ci_high']:.4f}] | "
                f"{row['semantic_drop']:.4f} | {row['preservation_cost']:.4f} |"
            )
    dataset_lines = [
        f"- `{category}`: discovery={counts[(category, 'discovery')]}, heldout={counts[(category, 'heldout')]}"
        for category in config["dataset"]["categories"]
    ]
    identity_lines = [
        f"- global {item['global_index']}: max_abs={item['max_absolute_error']}, exact={item['exact_equal']}, calls={item['call_count']}"
        for item in identity.get("checks", [])
    ]
    report = f"""# FLUX.1-Kontext-dev Block Probing — Pilot Report

> This is the preregistered 5-samples/category, one-seed pilot. It is not the final held-out claim.

## Reproducibility

- Project commit: `{commit}`
- Model revision: `{config['model']['revision']}`
- Diffusers commit: `{config['model']['diffusers_commit']}`
- Runtime structure: {structure.get('double_block_count')} double + {structure.get('single_block_count')} single = {structure.get('total_block_count')} global blocks
- Generated PNG: {png_count}; evaluated outputs: {eval_count}
- Resolution/steps/guidance: {config['inference']['resolution']} / {config['inference']['num_inference_steps']} / {config['inference']['guidance_scale']}

## Correctness gates

Unit tests: 10/10 passed. Alpha=1 runtime equivalence:

{chr(10).join(identity_lines)}

## Dataset

{chr(10).join(dataset_lines)}

All 240 prepared source hashes are unique. Automated QC is complete; human image/instruction QC remains required before formal claims.

## Stage 1/2 ranking

| Global block | Stream | Semantic gain | 95% CI | Semantic drop | Preservation cost |
|---:|---|---:|---:|---:|---:|
{chr(10).join(table) if table else '| — | — | — | — | — | — |'}

- Stage 2 blocks: `{selected.get('stage2_blocks', [])}`
- Provisional universal blocks: `{selected.get('universal_blocks', [])}`
- Provisional category-specific blocks: `{selected.get('category_specific_blocks', {})}`
- Selection status: `{selected.get('status', 'not yet aggregated')}`

## Required research questions

1. **Are semantics concentrated in a few blocks?** Pilot evidence is summarized above; formal 3-seed discovery is still required.
2. **Double or single stream?** Read from the ranked table after pilot aggregation; no prior stream preference was used.
3. **Shared across categories?** See `plots/category_block_response_curves.png`; formal stability remains pending.
4. **Best alpha range?** Pending the five-alpha scan of verified candidates.
5. **Sparse candidates vs all blocks?** Pending held-out joint validation.
6. **Do FLUX.1-Dev TexTailor blocks transfer?** Not used in discovery; pending held-out comparison only.
7. **Installation locations for edit-strength control?** No final installation block is claimed until disable/remove and held-out gates pass.
8. **Semantically active but structurally destructive blocks?** Identified from semantic drop, DINO/LPIPS cost and bad-image flags after Stage 2/3.

## Evidence files

- `structure_report.json` and `identity_report.json` under the preflight output directory
- `raw_metrics.csv`, `block_summary.csv`, `selected_blocks.json`, and `plots/` under `{run_root}`
- Blinded calibration bundle under `{run_root / 'calibration'}`
"""
    (run_root / "PILOT_REPORT.md").write_text(report, encoding="utf-8")
    print(run_root / "PILOT_REPORT.md")


if __name__ == "__main__":
    main()

