"""Run the preregistered alpha grid for provisional or final candidate blocks."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path

from probe_flux_kontext_blocks import load_config
from run_pilot_stage2 import run_parallel_evaluation, run_parallel_generation


ROOT = Path("/home/hyp/Code/flux-kontext-block-probing")


def pilot_manifest(config: dict) -> Path:
    target = Path(config["project"]["output_root"]) / "preflight" / "pilot_dataset.jsonl"
    if target.exists():
        return target
    rows = [json.loads(line) for line in Path(config["project"]["dataset_manifest"]).read_text(encoding="utf-8").splitlines()]
    selected = []
    for category in config["dataset"]["categories"]:
        candidates = sorted(
            (row for row in rows if row["split"] == "discovery" and row["category"] == category),
            key=lambda row: row["id"],
        )
        selected.extend(candidates[: config["dataset"]["pilot_per_category"]])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "probe_config.yaml"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    base = load_config(args.config)
    candidates = [int(value) for value in args.candidates.split(",") if value]
    if not candidates:
        raise ValueError("alpha scan requires candidate blocks")
    for alpha in base["inference"]["alpha_grid"]:
        if float(alpha) == float(base["inference"]["alpha"]):
            continue
        config = copy.deepcopy(base)
        config["inference"]["alpha"] = float(alpha)
        if args.pilot:
            config["project"]["dataset_manifest"] = str(pilot_manifest(base))
            config["inference"]["seeds"] = [config["inference"]["pilot_seed"]]
        alpha_label = str(alpha).replace(".", "p")
        run_parallel_generation(config, "enhance_text", candidates, f"alpha_{alpha_label}")
    run_parallel_evaluation()
    subprocess.run(
        [str(ROOT / ".venv/bin/python"), "aggregate_results.py", "--config", "probe_config.yaml"],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
