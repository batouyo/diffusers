"""Scheduler introspection with no hard-coded timestep indices."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from .core import critical_nonzero_steps

def inspect_scheduler(scheduler: Any, steps: int = 28) -> dict[str, Any]:
    scheduler.set_timesteps(steps)
    sigmas = [float(x) for x in scheduler.sigmas.detach().cpu().flatten().tolist()]
    timesteps = [float(x) for x in scheduler.timesteps.detach().cpu().flatten().tolist()]
    transitions = critical_nonzero_steps(sigmas)
    for row in transitions:
        index = int(row["index"])
        row["timestep"] = timesteps[index]
        row["post_timestep"] = timesteps[index + 1] if index + 1 < len(timesteps) else None
    return {"num_inference_steps": steps, "timesteps": timesteps, "sigmas": sigmas, "nonzero_transitions": transitions, "selected": transitions[:2]}

def write_scheduler_audit(path: str | Path, scheduler: Any, steps: int = 28) -> dict[str, Any]:
    payload = inspect_scheduler(scheduler, steps)
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
