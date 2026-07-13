"""Continue a completed pilot with disable/remove diagnostics on ranked blocks."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pandas as pd

from probe_flux_kontext_blocks import load_config, run_jobs


ROOT = Path("/home/hyp/Code/flux-kontext-block-probing")


def run_command(*args: str) -> None:
    subprocess.run([str(ROOT / ".venv/bin/python"), *args], cwd=ROOT, check=True)


def pilot_config(config: dict) -> dict:
    rows = [json.loads(line) for line in Path(config["project"]["dataset_manifest"]).read_text(encoding="utf-8").splitlines()]
    selected = []
    for category in config["dataset"]["categories"]:
        candidates = sorted(
            (row for row in rows if row["split"] == "discovery" and row["category"] == category),
            key=lambda row: row["id"],
        )
        selected.extend(candidates[: config["dataset"]["pilot_per_category"]])
    manifest = Path(config["project"]["output_root"]) / "preflight" / "pilot_dataset.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    derived = copy.deepcopy(config)
    derived["project"]["dataset_manifest"] = str(manifest)
    derived["inference"]["seeds"] = [derived["inference"]["pilot_seed"]]
    return derived


def main() -> None:
    config = load_config(ROOT / "probe_config.yaml")
    run_root = Path(config["project"]["output_root"]) / config["project"]["run_id"]
    selection_path = run_root / "selected_blocks.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    stage2_blocks = [int(value) for value in selection.get("stage2_blocks", [])]
    if len(stage2_blocks) != config["probing"]["stage2_blocks"]:
        raise RuntimeError(f"expected {config['probing']['stage2_blocks']} stage2 blocks, got {stage2_blocks}")
    derived = pilot_config(config)
    run_jobs(derived, "disable_text", "cuda:0", 0, 1, stage2_blocks, "discovery", None)
    run_command("evaluators.py", "--config", "probe_config.yaml", "--device", "cuda:0")
    run_command("aggregate_results.py", "--config", "probe_config.yaml")

    summary = pd.read_csv(run_root / "block_summary.csv")
    eligible = summary[summary["global_block_index"].isin(stage2_blocks)].copy()
    eligible["diagnostic_score"] = (
        eligible["semantic_gain_ci_low"].fillna(-1)
        + eligible["semantic_drop_ci_low"].fillna(-1)
        - eligible["preservation_cost"].fillna(0).clip(lower=0)
    )
    stage3_blocks = [int(value) for value in eligible.nlargest(config["probing"]["stage3_blocks"], "diagnostic_score")["global_block_index"]]
    (run_root / "stage3_blocks.json").write_text(
        json.dumps({"stage3_blocks": stage3_blocks}, indent=2), encoding="utf-8"
    )
    run_jobs(derived, "remove_block", "cuda:0", 0, 1, stage3_blocks, "discovery", None)
    run_command("evaluators.py", "--config", "probe_config.yaml", "--device", "cuda:0")
    run_command("aggregate_results.py", "--config", "probe_config.yaml")


if __name__ == "__main__":
    main()

