"""Run the full test suite and bind the result to the exact Git commit."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path("/home/hyp/Code/flux-kontext-block-probing")


def main() -> None:
    config = yaml.safe_load((ROOT / "probe_config.yaml").read_text(encoding="utf-8"))
    collection = subprocess.run(
        [str(ROOT / ".venv/bin/python"), "-m", "pytest", "--collect-only", "-q"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    node_count = sum("::" in line for line in collection.stdout.splitlines())
    file_counts = [
        int(match.group(1))
        for line in collection.stdout.splitlines()
        if (match := re.search(r":\s*(\d+)\s*$", line))
    ]
    collected_tests = node_count or sum(file_counts)
    if collection.returncode != 0 or collected_tests == 0:
        print(collection.stdout + collection.stderr, end="")
        raise RuntimeError("pytest collection failed or returned zero test nodes")
    process = subprocess.run(
        [str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    combined = process.stdout + process.stderr
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    report = {
        "status": "pass" if process.returncode == 0 else "fail",
        "returncode": process.returncode,
        "passed_tests": collected_tests if process.returncode == 0 else None,
        "git_commit": commit,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": ".venv/bin/python -m pytest -q",
        "output": combined,
    }
    path = Path(config["project"]["output_root"]) / "preflight" / "test_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    print(combined, end="")
    print(json.dumps({key: report[key] for key in ["status", "passed_tests", "git_commit"]}, indent=2))
    if process.returncode:
        raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
