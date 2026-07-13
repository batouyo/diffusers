"""Select five provisional pilot blocks for the full alpha grid."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path("/home/hyp/Code/flux-kontext-block-probing")


def main() -> None:
    path = Path("/data15/hyp/project_storage/flux-kontext-block-probing/main_512/stage3_blocks.json")
    blocks = json.loads(path.read_text(encoding="utf-8"))["stage3_blocks"][:5]
    if not blocks:
        raise RuntimeError("no provisional blocks available for alpha scan")
    subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            "scripts/run_alpha_scan.py",
            "--config",
            "probe_config.yaml",
            "--device",
            "cuda:0",
            "--candidates",
            ",".join(map(str, blocks)),
            "--pilot",
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()

