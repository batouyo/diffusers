"""Requirement-by-requirement audit for the original probing specification."""

from __future__ import annotations

import json
import csv
import subprocess
from collections import Counter
from pathlib import Path

import yaml

from evaluators import evaluation_hash
from verify_pilot_complete import sentinel_current


ROOT = Path("/home/hyp/Code/flux-kontext-block-probing")


def check(name: str, passed: bool, evidence, required: bool = True) -> dict:
    return {"requirement": name, "passed": bool(passed), "required": required, "evidence": evidence}


def main() -> None:
    config = yaml.safe_load((ROOT / "probe_config.yaml").read_text(encoding="utf-8"))
    output_root = Path(config["project"]["output_root"])
    run_root = output_root / config["project"]["run_id"]
    structure_path = output_root / "preflight" / "structure_report.json"
    identity_path = output_root / "preflight" / "identity_report.json"
    test_report_path = output_root / "preflight" / "test_report.json"
    structure = json.loads(structure_path.read_text(encoding="utf-8")) if structure_path.exists() else {}
    identity = json.loads(identity_path.read_text(encoding="utf-8")) if identity_path.exists() else {}
    test_report = json.loads(test_report_path.read_text(encoding="utf-8")) if test_report_path.exists() else {}
    current_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dataset = [json.loads(line) for line in (ROOT / "dataset.jsonl").read_text(encoding="utf-8").splitlines()]
    data_counts = Counter((row["category"], row["split"]) for row in dataset)
    metadata = []
    for path in (run_root / "images").rglob("*.json") if (run_root / "images").exists() else []:
        if path.name.endswith(".eval.json"):
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if value.get("status") == "complete":
            metadata.append(value)
    mode_counts = Counter(value.get("mode") for value in metadata)
    eval_paths = list((run_root / "images").rglob("*.eval.json")) if (run_root / "images").exists() else []
    eval_count = len(eval_paths)
    expected_evaluation_hash = evaluation_hash(config)
    valid_eval_count = 0
    for path in eval_paths:
        try:
            evaluation = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (
            evaluation.get("evaluation_hash") == expected_evaluation_hash
            and evaluation.get("vlm_parse_ok") is True
            and evaluation.get("s_edit") is not None
            and evaluation.get("dino_similarity") is not None
            and evaluation.get("lpips_distance") is not None
            and evaluation.get("quality")
        ):
            valid_eval_count += 1
    required_code = [
        "probe_config.yaml",
        "dataset.jsonl",
        "probe_flux_kontext_blocks.py",
        "interventions.py",
        "evaluators.py",
        "aggregate_results.py",
        "tests/test_interventions.py",
        "tests/test_aggregation.py",
        "tests/test_joint_validation.py",
        "tests/test_expected_counts.py",
        "scripts/verify_pilot_complete.py",
    ]
    required_outputs = ["raw_metrics.csv", "block_summary.csv", "stream_summary.csv", "selected_blocks.json"]
    required_block_columns = {
        "semantic_gain",
        "semantic_gain_q_bh",
        "semantic_drop",
        "semantic_drop_q_bh",
        "removal_edit_drop",
        "preservation_cost",
        "removal_preservation_cost",
        "seed_gain_std",
        "category_gain_json",
        "block_type",
        "block_class",
    }
    block_summary_path = run_root / "block_summary.csv"
    block_columns = set()
    if block_summary_path.exists():
        with block_summary_path.open("r", encoding="utf-8", newline="") as handle:
            block_columns = set(next(csv.reader(handle), []))
    metric_tables_complete = all((run_root / path).exists() for path in required_outputs) and required_block_columns <= block_columns
    core_plots = [
        "semantic_gain_vs_global_block.png",
        "semantic_drop_vs_global_block.png",
        "preservation_cost_vs_global_block.png",
        "category_block_response_curves.png",
        "alpha_sensitivity_curves.png",
    ]
    total_blocks = int(structure.get("total_block_count", 0))
    expected_discovery = 8 * config["dataset"]["discovery_per_category"] * len(config["inference"]["seeds"])
    selection_path = run_root / "selected_blocks.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else {}
    candidates = selection.get("selected_global_blocks", [])
    no_go_path = run_root / "FORMAL_NO_GO.json"
    no_go = json.loads(no_go_path.read_text(encoding="utf-8")) if no_go_path.exists() else {}
    formal_counts_complete = (
        mode_counts["baseline"] >= expected_discovery
        and mode_counts["enhance_text"] >= expected_discovery * total_blocks
        and mode_counts["disable_text"] >= expected_discovery * config["probing"]["stage2_blocks"]
        and mode_counts["remove_block"] >= expected_discovery * config["probing"]["stage3_blocks"]
    )
    validated_no_go = (
        formal_counts_complete
        and selection.get("status") == "no_go"
        and candidates == []
        and no_go.get("status") == "validated_no_go"
        and no_go.get("selected_global_blocks") == []
    )
    required_plots = core_plots + ([] if validated_no_go else ["candidate_vs_random_and_all.png"])
    grid_manifest_path = run_root / "plots" / "image_grids" / "image_grid_manifest.json"
    grid_manifest = json.loads(grid_manifest_path.read_text(encoding="utf-8")) if grid_manifest_path.exists() else {}
    valid_grids = validated_no_go or (
        grid_manifest.get("status") == "complete"
        and set(grid_manifest.get("categories", [])) == set(config["dataset"]["categories"])
        and len(grid_manifest.get("grids", [])) == len(config["dataset"]["categories"])
        and all(
            Path(item.get("grid_path", "")).exists()
            and set(item.get("panels", {})) == {"source", "baseline", "candidate", "disable", "random", "all_blocks"}
            and all(Path(panel.get("path", "")).exists() and panel.get("sha256") for panel in item.get("panels", {}).values())
            for item in grid_manifest.get("grids", [])
        )
    )
    joint_path = run_root / "joint_validation.json"
    joint = json.loads(joint_path.read_text(encoding="utf-8")) if joint_path.exists() else {}
    valid_joint = (
        joint.get("execution_status") == "complete"
        and joint.get("expected_total", 0) > 0
        and joint.get("evaluated_total") == joint.get("expected_total")
        and joint.get("arm_counts")
        and all(count == joint.get("expected_per_arm") for count in joint["arm_counts"].values())
    )
    joint_or_no_go = valid_joint or validated_no_go
    checks = [
        check(
            "independent required source files and commit-bound tests",
            all((ROOT / path).exists() for path in required_code)
            and test_report.get("status") == "pass"
            and int(test_report.get("passed_tests") or 0) >= 20
            and test_report.get("git_commit") == current_commit,
            {
                "required_code": required_code,
                "test_report": test_report_path,
                "tested_commit": test_report.get("git_commit"),
                "current_commit": current_commit,
                "passed_tests": test_report.get("passed_tests"),
            },
        ),
        check("runtime structure report", total_blocks > 0 and len(structure.get("blocks", [])) == total_blocks, structure_path),
        check(
            "alpha=1 numerical equivalence",
            identity.get("status") == "pass"
            and identity.get("checks")
            and max(item["max_absolute_error"] for item in identity["checks"]) <= 1e-6,
            identity,
        ),
        check(
            "research dataset 20 discovery + 10 heldout per category",
            len(dataset) == 240
            and all(
                data_counts[(category, "discovery")] == 20 and data_counts[(category, "heldout")] == 10
                for category in config["dataset"]["categories"]
            ),
            {f"{category}:{split}": count for (category, split), count in data_counts.items()},
        ),
        check("formal baselines complete", mode_counts["baseline"] >= expected_discovery, dict(mode_counts)),
        check(
            "all-block enhance discovery complete",
            mode_counts["enhance_text"] >= expected_discovery * total_blocks,
            dict(mode_counts),
        ),
        check(
            "top-15 disable complete",
            mode_counts["disable_text"] >= expected_discovery * config["probing"]["stage2_blocks"],
            dict(mode_counts),
        ),
        check(
            "top-10 remove complete",
            mode_counts["remove_block"] >= expected_discovery * config["probing"]["stage3_blocks"],
            dict(mode_counts),
        ),
        check(
            "all generated outputs evaluated with current evaluator and valid metrics",
            bool(metadata)
            and eval_count >= len(metadata)
            and valid_eval_count >= len(metadata)
            and sentinel_current(ROOT),
            {
                "generated": len(metadata),
                "evaluation_files": eval_count,
                "valid_current_evaluations": valid_eval_count,
                "expected_evaluation_hash": expected_evaluation_hash,
                "pilot_pipeline_sentinel_current": sentinel_current(ROOT),
            },
        ),
        check(
            "required metric tables and protocol columns",
            metric_tables_complete,
            {
                "files": required_outputs,
                "required_block_columns": sorted(required_block_columns),
                "observed_block_columns": sorted(block_columns),
            },
        ),
        check(
            "required plots (candidate comparison conditional on non-empty selection)",
            all((run_root / "plots" / path).exists() for path in required_plots) and valid_grids,
            {
                "required": required_plots,
                "validated_no_go": validated_no_go,
                "image_grid_manifest": grid_manifest_path,
                "valid_image_grids": valid_grids,
            },
        ),
        check("80-example calibration bundle", (run_root / "calibration" / "blinded_labels.csv").exists(), run_root / "calibration"),
        check(
            "human calibration gate",
            (run_root / "calibration" / "calibration_report.json").exists()
            and json.loads((run_root / "calibration" / "calibration_report.json").read_text(encoding="utf-8")).get("gate_pass", False),
            run_root / "calibration" / "calibration_report.json",
        ),
        check(
            "joint heldout validation or preregistered no-go",
            joint_or_no_go,
            {
                "joint": joint_path,
                "valid_joint_execution": valid_joint,
                "joint_status": joint.get("status"),
                "evaluated_total": joint.get("evaluated_total"),
                "expected_total": joint.get("expected_total"),
                "no_go": no_go_path,
                "validated_no_go": validated_no_go,
            },
        ),
        check("final report", (run_root / "FINAL_REPORT.md").exists(), run_root / "FINAL_REPORT.md"),
    ]
    required = [item for item in checks if item["required"]]
    result = {
        "status": "complete" if all(item["passed"] for item in required) else "incomplete",
        "passed": sum(item["passed"] for item in required),
        "required": len(required),
        "checks": checks,
    }
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "completion_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = ["# Completion Audit", "", f"Status: **{result['status']}** ({result['passed']}/{result['required']})", ""]
    for item in checks:
        lines.append(f"- {'PASS' if item['passed'] else 'MISSING'} — {item['requirement']}: `{item['evidence']}`")
    (run_root / "completion_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
