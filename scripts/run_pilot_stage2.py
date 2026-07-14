"""Continue a completed pilot with disable/remove diagnostics on ranked blocks."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import time
from pathlib import Path

import pandas as pd
import yaml

from probe_flux_kontext_blocks import load_config


ROOT = Path("/home/hyp/Code/flux-kontext-block-probing")
GPU_UUIDS = [
    "GPU-2cd22c91-025f-16c6-f54a-0947f721d15e",
    "GPU-40d0b0a1-543f-cdd7-09a0-3b8d348198f4",
    "GPU-2f3e1340-c0fc-668b-e619-5e32ec72b99c",
    "GPU-1a8948f7-f590-3773-34d5-3abaeacfa367",
    "GPU-d44367f8-885c-b179-43a8-8c1e8e8eaa6a",
]


def run_command(*args: str) -> None:
    subprocess.run([str(ROOT / ".venv/bin/python"), *args], cwd=ROOT, check=True)


def run_parallel_generation(config: dict, stage: str, blocks: list[int], label: str) -> None:
    run_root = Path(config["project"]["output_root"]) / config["project"]["run_id"]
    config_path = run_root / "parallel_configs" / "pilot_followup.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    processes: list[tuple[int, subprocess.Popen, object]] = []
    try:
        for shard_id, gpu_uuid in enumerate(GPU_UUIDS):
            log_handle = (ROOT / "logs" / f"{label}_shard_{shard_id:02d}.log").open("w", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
            command = [
                str(ROOT / ".venv/bin/python"),
                str(ROOT / "probe_flux_kontext_blocks.py"),
                "--config",
                str(config_path),
                "--device",
                "cuda:0",
                "run",
                "--stage",
                stage,
                "--blocks",
                ",".join(map(str, blocks)),
                "--split",
                "discovery",
                "--shard-id",
                str(shard_id),
                "--num-shards",
                str(len(GPU_UUIDS)),
            ]
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
            processes.append((shard_id, process, log_handle))
        active = {shard_id for shard_id, _, _ in processes}
        while active:
            for shard_id, process, _ in processes:
                if shard_id not in active:
                    continue
                returncode = process.poll()
                if returncode is None:
                    continue
                active.remove(shard_id)
                if returncode != 0:
                    for other_id, other, _ in processes:
                        if other_id in active:
                            other.terminate()
                    raise RuntimeError(f"generation shard {shard_id} exited with status {returncode}")
            time.sleep(1)
    finally:
        for _, process, log_handle in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            log_handle.close()


def run_parallel_evaluation() -> None:
    run_command(
        "scripts/run_parallel_evaluators.py",
        "--config",
        "probe_config.yaml",
        "--gpu-uuids",
        ",".join(GPU_UUIDS),
    )


def pilot_config(config: dict) -> dict:
    rows = [json.loads(line) for line in Path(config["project"]["dataset_manifest"]).read_text(encoding="utf-8").splitlines()]
    selected = []
    for category in config["dataset"]["categories"]:
        candidates = sorted(
            (row for row in rows if row["split"] == "discovery" and row["category"] == category),
            key=lambda row: row["id"],
        )
        selected.extend(candidates[: config["dataset"]["pilot_per_category"]])
    manifest = Path(config["project"]["output_root"]) / "preflight" / "pilot_dataset.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    derived = copy.deepcopy(config)
    derived["project"]["dataset_manifest"] = str(manifest)
    derived["inference"]["seeds"] = [derived["inference"]["pilot_seed"]]
    return derived


def main() -> None:
    config = load_config(ROOT / "probe_config.yaml")
    run_root = Path(config["project"]["output_root"]) / config["project"]["run_id"]
    selection_path = run_root / "selected_blocks.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    stage2_blocks = [int(value) for value in selection.get("stage2_blocks", [])]
    if len(stage2_blocks) != config["probing"]["stage2_blocks"]:
        raise RuntimeError(f"expected {config['probing']['stage2_blocks']} stage2 blocks, got {stage2_blocks}")
    derived = pilot_config(config)
    run_parallel_generation(derived, "disable_text", stage2_blocks, "pilot_disable")
    run_parallel_evaluation()
    run_command("aggregate_results.py", "--config", "probe_config.yaml")

    summary = pd.read_csv(run_root / "block_summary.csv")
    eligible = summary[summary["global_block_index"].isin(stage2_blocks)].copy()
    eligible["diagnostic_score"] = (
        eligible["semantic_gain_ci_low"].fillna(-1)
        + eligible["semantic_drop_ci_low"].fillna(-1)
        - eligible["preservation_cost"].fillna(0).clip(lower=0)
    )
    stage3_blocks = [int(value) for value in eligible.nlargest(config["probing"]["stage3_blocks"], "diagnostic_score")["global_block_index"]]
    (run_root / "stage3_blocks.json").write_text(
        json.dumps({"stage3_blocks": stage3_blocks}, indent=2), encoding="utf-8"
    )
    run_parallel_generation(derived, "remove_block", stage3_blocks, "pilot_remove")
    run_parallel_evaluation()
    run_command("aggregate_results.py", "--config", "probe_config.yaml")


if __name__ == "__main__":
    main()
