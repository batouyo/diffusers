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


ROOT = Path("/home/hyp/Code/flux-kontext-block-probing")
RUN_ROOT = Path("/data15/hyp/project_storage/flux-kontext-block-probing/main_512")
LOG = ROOT / "logs" / "watchdog.log"
PILOT_EXPECTED = 2320
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


def main() -> None:
    pilot_restarts = 0
    followup_restarts = 0
    log("watchdog active; fixed GPU selection remains inside stage scripts")
    while True:
        png_count = count("*.png")
        eval_count = count("*.eval.json")
        pilot = session_exists("flux_probe_pilot")
        followup = session_exists("flux_probe_followup")
        stage3_ready = (RUN_ROOT / "stage3_blocks.json").exists()
        log(
            f"pilot_png={png_count}/{PILOT_EXPECTED} pilot_eval={eval_count}/{PILOT_EXPECTED} "
            f"pilot_session={pilot} followup_session={followup} stage3_ready={stage3_ready}"
        )

        if audit_complete():
            log("completion audit is complete; watchdog exiting")
            return

        pilot_incomplete = png_count < PILOT_EXPECTED or eval_count < PILOT_EXPECTED
        if not pilot and pilot_incomplete and pilot_restarts < MAX_RESTARTS:
            pilot_restarts += 1
            start_session("flux_probe_pilot", str(ROOT / "scripts" / "run_pilot_pipeline.sh"))
        elif (
            not pilot
            and png_count >= PILOT_EXPECTED
            and eval_count >= PILOT_EXPECTED
            and not followup
            and not stage3_ready
            and followup_restarts < MAX_RESTARTS
        ):
            followup_restarts += 1
            start_session("flux_probe_followup", str(ROOT / "scripts" / "run_after_pilot.sh"))

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
