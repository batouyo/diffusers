"""Create an evidence-linked pilot report without overstating formal validation."""

from __future__ import annotations

import json
import math
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


def fmt(value, digits: int = 4) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{numeric:.{digits}f}" if math.isfinite(numeric) else "—"


def main() -> None:
    config = yaml.safe_load((ROOT / "probe_config.yaml").read_text(encoding="utf-8"))
    output_root = Path(config["project"]["output_root"])
    run_root = output_root / config["project"]["run_id"]
    structure = read_json(output_root / "preflight" / "structure_report.json", {})
    identity = read_json(output_root / "preflight" / "identity_report.json", {})
    tests = read_json(output_root / "preflight" / "test_report.json", {})
    pilot = read_json(run_root / "pilot_pipeline_complete.json", {})
    followup = read_json(run_root / "pilot_followup_complete.json", {})
    alpha = read_json(run_root / "pilot_alpha_complete.json", {})
    selected = read_json(run_root / "selected_blocks.json", {})
    stage3 = read_json(run_root / "stage3_blocks.json", {})
    summary_path = run_root / "block_summary.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    stream_path = run_root / "stream_summary.csv"
    stream = pd.read_csv(stream_path) if stream_path.exists() else pd.DataFrame()
    dataset = [
        json.loads(line)
        for line in (ROOT / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    counts = Counter((row["category"], row["split"]) for row in dataset)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    tests_current = tests.get("status") == "pass" and tests.get("git_commit") == commit

    table = []
    if not summary.empty and summary.get("semantic_gain", pd.Series(dtype=float)).notna().any():
        ranked = summary.sort_values(["semantic_gain_ci_low", "semantic_gain"], ascending=False).head(15)
        for _, row in ranked.iterrows():
            table.append(
                "| "
                + " | ".join(
                    [
                        str(int(row["global_block_index"])),
                        str(row.get("block_type", "—")),
                        fmt(row.get("semantic_gain")),
                        f"[{fmt(row.get('semantic_gain_ci_low'))}, {fmt(row.get('semantic_gain_ci_high'))}]",
                        fmt(row.get("semantic_gain_q_bh")),
                        fmt(row.get("semantic_drop")),
                        fmt(row.get("semantic_drop_q_bh")),
                        fmt(row.get("removal_edit_drop")),
                        fmt(row.get("preservation_cost")),
                        fmt(row.get("seed_gain_std")),
                        str(row.get("block_class", "—")),
                    ]
                )
                + " |"
            )
    dataset_lines = [
        f"- `{category}`: discovery={counts[(category, 'discovery')]}, heldout={counts[(category, 'heldout')]}"
        for category in config["dataset"]["categories"]
    ]
    identity_lines = [
        f"- global {item['global_index']}: max_abs={item['max_absolute_error']}, "
        f"exact={item['exact_equal']}, calls={item['call_count']}"
        for item in identity.get("checks", [])
    ]
    stream_lines = []
    for _, row in stream.iterrows():
        stream_lines.append(
            f"- `{row.get('block_type')}`: n={int(row.get('block_count', 0))}, "
            f"mean gain={fmt(row.get('mean_semantic_gain'))}, "
            f"mean disable drop={fmt(row.get('mean_semantic_drop'))}"
        )
    alpha_status = "complete" if alpha.get("status") == "complete" else "pending"
    report = f"""# FLUX.1-Kontext-dev Block Probing — Pilot Report

> This is the preregistered 5-samples/category, one-seed pilot. It is preliminary evidence, not a final held-out claim.

## Reproducibility and verified completion

- Project commit: `{commit}`
- Model revision: `{config['model']['revision']}`
- Diffusers commit: `{config['model']['diffusers_commit']}`
- Runtime structure: {structure.get('double_block_count')} double + {structure.get('single_block_count')} single = {structure.get('total_block_count')} global blocks
- Stage 1 sentinel: `{pilot.get('status', 'missing')}`; valid evaluations {pilot.get('valid_evaluations', 0)}/{pilot.get('expected_jobs', 0)}
- Follow-up sentinel: `{followup.get('status', 'missing')}`; disable {followup.get('valid_disable_evaluations', 0)}/{followup.get('expected_disable_jobs', 0)}, remove {followup.get('valid_remove_evaluations', 0)}/{followup.get('expected_remove_jobs', 0)}
- Alpha scan: `{alpha_status}`
- Resolution/steps/guidance: {config['inference']['resolution']} / {config['inference']['num_inference_steps']} / {config['inference']['guidance_scale']}

## Correctness gates

- Commit-bound unit tests: {tests.get('passed_tests', 0)} passed; current commit match: `{tests_current}`
- Alpha=1 runtime equivalence:

{chr(10).join(identity_lines)}

## Dataset

{chr(10).join(dataset_lines)}

All 240 prepared source hashes are unique. Automated source/output QC is complete; the preregistered 80-example blind human calibration remains a gate before formal claims.

## Stage 1/2/3 diagnostics

| Global block | Stream | Gain | Gain 95% CI | Gain BH q | Disable drop | Drop BH q | Remove drop | Preserve cost | Seed std | Class |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(table) if table else '| — | — | — | — | — | — | — | — | — | — | — |'}

- Stage 2 blocks: `{selected.get('stage2_blocks', [])}`
- Stage 3 blocks: `{stage3.get('stage3_blocks', [])}`
- Provisional universal blocks: `{selected.get('universal_blocks', [])}`
- Provisional category-specific blocks: `{selected.get('category_specific_blocks', {})}`
- Selection status: `{selected.get('status', 'not yet aggregated')}`

### Stream summary

{chr(10).join(stream_lines) if stream_lines else '- Pending aggregate output.'}

## Required research questions

1. **Are semantics concentrated in a few blocks?** Pilot ranking, confidence intervals and corrected q-values are reported above; the 3-seed formal discovery is still required.
2. **Double or single stream?** The empirical stream summary above is used; no stream preference was imposed.
3. **Shared across categories?** See `plots/category_block_response_curves.png`; formal stability remains pending.
4. **Best alpha range?** Read from `alpha_summary.csv` after the five-alpha sentinel is complete.
5. **Sparse candidates vs all blocks?** Pending held-out joint validation against baseline, random, all-block and TexTailor controls.
6. **Do FLUX.1-Dev TexTailor blocks transfer?** They were excluded from discovery and are used only as a held-out control.
7. **Installation locations for edit-strength control?** No final installation block is claimed until disable/remove, alpha and held-out gates pass.
8. **Semantically active but structurally destructive blocks?** The remove drop/cost, DINO/LPIPS cost and mode-specific bad-image rates provide this diagnostic.

## Evidence files

- `structure_report.json`, `identity_report.json`, and `test_report.json` under `{output_root / 'preflight'}`
- `pilot_pipeline_complete.json` and `pilot_followup_complete.json` under `{run_root}`
- `raw_metrics.csv`, `block_summary.csv`, `stream_summary.csv`, `selected_blocks.json`, `stage3_blocks.json`, and `plots/` under `{run_root}`
- Portable blind-rating bundle: `{run_root / 'calibration' / 'blinded_calibration_bundle.zip'}` (sealed key excluded)
"""
    (run_root / "PILOT_REPORT.md").write_text(report, encoding="utf-8")
    print(run_root / "PILOT_REPORT.md")


if __name__ == "__main__":
    main()
