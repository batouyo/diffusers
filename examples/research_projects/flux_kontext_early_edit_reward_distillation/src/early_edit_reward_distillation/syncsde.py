"""Explicit adapter for the pinned official syncSDE scheduler."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

OFFICIAL_SYNCSDE_COMMIT = "36cf2a38b1c08425257d7bdbe359c6afd2fbd4c5"


def load_official_scheduler(syncsde_root: str | Path, base_scheduler: Any) -> Any:
    """Load only the official scheduler, leaving Kontext pipeline code native."""
    path = Path(syncsde_root) / "scheduling_flow_match_euler_discrete_sde.py"
    if not path.exists():
        raise FileNotFoundError(f"official syncSDE scheduler not found: {path}")
    spec = importlib.util.spec_from_file_location("official_syncsde_scheduler", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import syncSDE scheduler from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FlowMatchEulerDiscreteSDEScheduler.from_config(base_scheduler.config)
