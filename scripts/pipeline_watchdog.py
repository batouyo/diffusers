#!/usr/bin/env python3
"""Conservative tmux watchdog for the overnight FLUX probing pipeline.

It only restarts idempotent, hash-checked stages after their tmux session has
disappeared.  It never selects a different GPU and never kills a process.
"""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from expected_counts import load_counts
from verify_pilot_complete import sentinel_current
from verify_pilot_followup import sentinel_current as followup_sentinel_current


ROOT = Path("/home/hyp/Code/flux-kontext-block-probing")
RUN_ROOT = Path("/data15/hyp/project_storage/flux-kontext-block-probing/main_512")
LOG = ROOT / "logs" / "watchdog.log"
MAX_RESTARTS = 3
POLL_SECONDS = 120


def log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    line = f"{timestamp} {message}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def session_exists(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def count(pattern: str) -> int:
    root = RUN_ROOT / "images"
    return sum(1 for _ in root.rglob(pattern)) if root.exists() else 0


def start_session(name: str, command: str) -> None:
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", name, command],
        cwd=ROOT,
        check=True,
    )
    log(f"started {name}: {command}")


def audit_complete() -> bool:
    path = RUN_ROOT / "completion_audit.json"
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def json_flag(path: Path, key: str) -> bool:
    if not path.exists():
        return False
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get(key))
    except (OSError, json.JSONDecodeError):
        return False


def write_status(payload: dict) -> None:
    path = RUN_ROOT / "pipeline_status.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    pilot_restarts = 0
    followup_restarts = 0
    alpha_restarts = 0
    calibration_restarts = 0
    report_restarts = 0
    formal_restarts = 0
    pilot_expected = load_counts(ROOT)["pilot_stage1_jobs"]
    log("watchdog active; fixed GPU selection remains inside stage scripts")
    while True:
        png_count = count("*.png")
        eval_count = count("*.eval.json")
        pilot = session_exists("flux_probe_pilot")
        followup = session_exists("flux_probe_followup")
        alpha = session_exists("flux_probe_alpha")
        calibration = session_exists("flux_probe_calibration")
        report = session_exists("flux_probe_report")
        formal = session_exists("flux_probe_formal")
        stage3_ready = (RUN_ROOT / "stage3_blocks.json").exists()
        alpha_ready = json_flag(RUN_ROOT / "pilot_alpha_complete.json", "status")
        calibration_ready = (RUN_ROOT / "calibration" / "blinded_labels.csv").exists()
        calibration_gate = json_flag(
            RUN_ROOT / "calibration" / "calibration_report.json", "gate_pass"
        )
        pilot_report_ready = (RUN_ROOT / "PILOT_REPORT.md").exists()
        pilot_verified = sentinel_current(ROOT)
        followup_verified = followup_sentinel_current(ROOT) if stage3_ready else False
        status_payload = {
            "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "pilot_png": png_count,
            "pilot_expected": pilot_expected,
            "pilot_eval": eval_count,
            "pilot_verified": pilot_verified,
            "followup_verified": followup_verified,
            "sessions": {
                "pilot": pilot,
                "followup": followup,
                "alpha": alpha,
                "calibration": calibration,
                "report": report,
                "formal": formal,
            },
            "stage3_ready": stage3_ready,
            "alpha_ready": alpha_ready,
            "calibration_bundle_ready": calibration_ready,
            "calibration_gate_pass": calibration_gate,
            "pilot_report_ready": pilot_report_ready,
            "completion_audit_complete": audit_complete(),
        }
        write_status(status_payload)
        log(
            f"pilot_png={png_count}/{pilot_expected} pilot_eval={eval_count}/{pilot_expected} "
            f"pilot_session={pilot} followup_session={followup} stage3_ready={stage3_ready} "
            f"pilot_verified={pilot_verified} alpha_ready={alpha_ready} "
            f"followup_verified={followup_verified} calibration_ready={calibration_ready} gate={calibration_gate}"
        )

        if audit_complete():
            log("completion audit is complete; watchdog exiting")
            return

        pilot_incomplete = not pilot_verified
        if not pilot and pilot_incomplete and pilot_restarts < MAX_RESTARTS:
            pilot_restarts += 1
            start_session("flux_probe_pilot", str(ROOT / "scripts" / "run_pilot_pipeline.sh"))
        elif (
            not pilot
            and pilot_verified
            and not followup
            and not followup_verified
            and followup_restarts < MAX_RESTARTS
        ):
            followup_restarts += 1
            start_session("flux_probe_followup", str(ROOT / "scripts" / "run_after_pilot.sh"))

        if (
            followup_verified
            and not followup
            and not alpha_ready
            and not alpha
            and alpha_restarts < MAX_RESTARTS
        ):
            alpha_restarts += 1
            start_session("flux_probe_alpha", str(ROOT / "scripts" / "run_alpha_after_followup.sh"))

        if (
            pilot_verified
            and not calibration_ready
            and not calibration
            and calibration_restarts < MAX_RESTARTS
        ):
            calibration_restarts += 1
            start_session(
                "flux_probe_calibration", str(ROOT / "scripts" / "run_calibration_after_pilot.sh")
            )

        if (
            followup_verified
            and not pilot_report_ready
            and not report
            and report_restarts < MAX_RESTARTS
        ):
            report_restarts += 1
            start_session("flux_probe_report", str(ROOT / "scripts" / "run_report_after_followup.sh"))

        if (
            alpha_ready
            and calibration_gate
            and not formal
            and not audit_complete()
            and formal_restarts < MAX_RESTARTS
        ):
            formal_restarts += 1
            start_session("flux_probe_formal", str(ROOT / "scripts" / "run_formal_after_alpha.sh"))

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
