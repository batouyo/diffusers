"""End-to-end formal discovery, diagnostics, alpha selection, and held-out validation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

from probe_flux_kontext_blocks import load_config, run_jobs


ROOT = Path("/home/hyp/Code/flux-kontext-block-probing")


def command(*args: str) -> None:
    subprocess.run([str(ROOT / ".venv/bin/python"), *args], cwd=ROOT, check=True)


def evaluate_and_aggregate(device: str) -> None:
    command("evaluators.py", "--config", "probe_config.yaml", "--device", device)
    command("aggregate_results.py", "--config", "probe_config.yaml")


def stage3_ranking(config: dict, stage2: list[int]) -> list[int]:
    run_root = Path(config["project"]["output_root"]) / config["project"]["run_id"]
    summary = pd.read_csv(run_root / "block_summary.csv")
    eligible = summary[summary["global_block_index"].isin(stage2)].copy()
    eligible["diagnostic_score"] = (
        eligible["semantic_gain_ci_low"].fillna(-1)
        + eligible["semantic_drop_ci_low"].fillna(-1)
        - eligible["preservation_cost"].fillna(0).clip(lower=0)
    )
    return [
        int(value)
        for value in eligible.nlargest(config["probing"]["stage3_blocks"], "diagnostic_score")[
            "global_block_index"
        ]
    ]


def select_alpha(config: dict, candidates: list[int]) -> float:
    run_root = Path(config["project"]["output_root"]) / config["project"]["run_id"]
    table = pd.read_csv(run_root / "alpha_summary.csv")
    table = table[table["global_block_index"].isin(candidates)]
    grouped = table.groupby("alpha", as_index=False).agg(
        semantic_gain=("semantic_gain", "mean"),
        preservation_cost=("preservation_cost", "mean"),
        bad_image_rate=("bad_image_rate", "mean"),
    )
    eligible = grouped[
        (grouped["preservation_cost"] <= -config["statistics"]["dino_noninferiority_margin"])
        & (grouped["bad_image_rate"] <= config["statistics"]["bad_image_rate_max"])
    ]
    if eligible.empty:
        raise RuntimeError("no alpha passes preservation and bad-image gates")
    best_gain = float(eligible["semantic_gain"].max())
    near_best = eligible[eligible["semantic_gain"] >= best_gain - 0.01]
    return float(near_best["alpha"].min())


def main() -> None:
    command("scripts/run_tests_with_report.py")
    config = load_config(ROOT / "probe_config.yaml")
    device = "cuda:0"
    run_root = Path(config["project"]["output_root"]) / config["project"]["run_id"]

    run_jobs(config, "baseline", device, 0, 1, None, "discovery", None)
    run_jobs(config, "enhance_text", device, 0, 1, None, "discovery", None)
    evaluate_and_aggregate(device)

    selection = json.loads((run_root / "selected_blocks.json").read_text(encoding="utf-8"))
    stage2 = [int(value) for value in selection["stage2_blocks"]]
    if len(stage2) != config["probing"]["stage2_blocks"]:
        raise RuntimeError(f"formal stage2 ranking incomplete: {stage2}")
    run_jobs(config, "disable_text", device, 0, 1, stage2, "discovery", None)
    evaluate_and_aggregate(device)

    stage3 = stage3_ranking(config, stage2)
    (run_root / "stage3_blocks.json").write_text(json.dumps({"stage3_blocks": stage3}, indent=2), encoding="utf-8")
    run_jobs(config, "remove_block", device, 0, 1, stage3, "discovery", None)
    evaluate_and_aggregate(device)
    command("scripts/verify_formal_complete.py")

    selection = json.loads((run_root / "selected_blocks.json").read_text(encoding="utf-8"))
    candidates = [int(value) for value in selection.get("selected_global_blocks", [])]
    if not candidates:
        (run_root / "FORMAL_NO_GO.json").write_text(
            json.dumps(
                {
                    "status": "validated_no_go",
                    "reason": "no block passed preregistered gain/drop/preservation gates",
                    "selection_status": selection.get("status"),
                    "selected_global_blocks": [],
                    "universal_blocks": selection.get("universal_blocks", []),
                    "category_specific_blocks": selection.get("category_specific_blocks", {}),
                    "evaluated_stage2_blocks": stage2,
                    "evaluated_stage3_blocks": stage3,
                    "preregistered_alpha": config["inference"]["alpha"],
                    "policy": "do not force a top-k candidate when no block clears all gates",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        command("scripts/make_final_report.py")
        command("scripts/audit_completion.py")
        return

    command(
        "scripts/run_alpha_scan.py",
        "--config",
        "probe_config.yaml",
        "--device",
        device,
        "--candidates",
        ",".join(map(str, candidates)),
    )
    command("scripts/verify_alpha_scan.py", "--scope", "formal")
    alpha = select_alpha(config, candidates)
    (run_root / "selected_alpha.json").write_text(json.dumps({"alpha": alpha}, indent=2), encoding="utf-8")
    command(
        "scripts/run_joint_validation.py",
        "--config",
        "probe_config.yaml",
        "--device",
        device,
        "--candidates",
        ",".join(map(str, candidates)),
        "--alpha",
        str(alpha),
        "--split",
        "heldout",
        "--random-sets",
        str(config["probing"]["random_control_sets"]),
    )
    command("evaluators.py", "--config", "probe_config.yaml", "--device", device)
    command("scripts/aggregate_joint.py", "--config", "probe_config.yaml")
    command("scripts/make_image_grids.py", "--config", "probe_config.yaml", "--split", "heldout")
    command("scripts/make_final_report.py")
    command("scripts/audit_completion.py")


if __name__ == "__main__":
    main()
