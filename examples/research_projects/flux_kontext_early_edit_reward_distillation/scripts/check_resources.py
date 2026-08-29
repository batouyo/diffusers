#!/usr/bin/env python3
"""Fail-fast resource gate for P1/P2; no fallback model is silently used."""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--editscore-root", required=True)
    parser.add_argument("--syncsde-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model = Path(args.model_path); editscore = Path(args.editscore_root); syncsde = Path(args.syncsde_root)
    report = {
        "flux_kontext_model": {"path": str(model), "exists": model.exists()},
        "editscore_root": {"path": str(editscore), "exists": editscore.exists()},
        "syncsde_root": {"path": str(syncsde), "exists": syncsde.exists()},
        "peft_importable": importlib.util.find_spec("peft") is not None,
        "pyarrow_importable": importlib.util.find_spec("pyarrow") is not None,
    }
    report["p1_ready"] = all((report["flux_kontext_model"]["exists"], report["editscore_root"]["exists"], report["syncsde_root"]["exists"]))
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["p1_ready"]: raise SystemExit("P1 blocked: official resources are not all available")

if __name__ == "__main__": main()
