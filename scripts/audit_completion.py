"""Requirement-by-requirement audit for the original probing specification."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import yaml


ROOT = Path("/home/hyp/Code/flux-kontext-block-probing")


def check(name: str, passed: bool, evidence, required: bool = True) -> dict:
    return {"requirement": name, "passed": bool(passed), "required": required, "evidence": evidence}


def main() -> None:
    config = yaml.safe_load((ROOT / "probe_config.yaml").read_text(encoding="utf-8"))
    output_root = Path(config["project"]["output_root"])
    run_root = output_root / config["project"]["run_id"]
    structure_path = output_root / "preflight" / "structure_report.json"
    identity_path = output_root / "preflight" / "identity_report.json"
    structure = json.loads(structure_path.read_text(encoding="utf-8")) if structure_path.exists() else {}
    identity = json.loads(identity_path.read_text(encoding="utf-8")) if identity_path.exists() else {}
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
    eval_count = len(list((run_root / "images").rglob("*.eval.json"))) if (run_root / "images").exists() else 0
    required_code = [
        "probe_config.yaml",
        "dataset.jsonl",
        "probe_flux_kontext_blocks.py",
        "interventions.py",
        "evaluators.py",
        "aggregate_results.py",
        "tests/test_interventions.py",
    ]
    required_outputs = ["raw_metrics.csv", "block_summary.csv", "selected_blocks.json"]
    required_plots = [
        "semantic_gain_vs_global_block.png",
        "semantic_drop_vs_global_block.png",
        "preservation_cost_vs_global_block.png",
        "category_block_response_curves.png",
        "alpha_sensitivity_curves.png",
        "candidate_vs_random_and_all.png",
    ]
    total_blocks = int(structure.get("total_block_count", 0))
    expected_discovery = 8 * config["dataset"]["discovery_per_category"] * len(config["inference"]["seeds"])
    checks = [
        check("independent required source files", all((ROOT / path).exists() for path in required_code), required_code),
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
        check("all generated outputs evaluated", bool(metadata) and eval_count >= len(metadata), {"generated": len(metadata), "evaluated": eval_count}),
        check("required metric tables", all((run_root / path).exists() for path in required_outputs), required_outputs),
        check("required plots", all((run_root / "plots" / path).exists() for path in required_plots), required_plots),
        check("80-example calibration bundle", (run_root / "calibration" / "blinded_labels.csv").exists(), run_root / "calibration"),
        check(
            "human calibration gate",
            (run_root / "calibration" / "calibration_report.json").exists()
            and json.loads((run_root / "calibration" / "calibration_report.json").read_text(encoding="utf-8")).get("gate_pass", False),
            run_root / "calibration" / "calibration_report.json",
        ),
        check("joint heldout validation", (run_root / "joint_validation.json").exists(), run_root / "joint_validation.json"),
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
