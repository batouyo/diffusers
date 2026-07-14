#!/usr/bin/env python3
"""Evaluate one run on disjoint metadata shards across explicitly selected GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluators import iter_metadata  # noqa: E402


def partition_paths(paths: list[Path], shard_count: int) -> list[list[Path]]:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    return [paths[index::shard_count] for index in range(shard_count)]


def write_manifest(path: Path, paths: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(f"{item}\n" for item in paths), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="probe_config.yaml")
    parser.add_argument("--run-id")
    parser.add_argument("--gpu-uuids", required=True, help="Comma-separated physical GPU UUIDs")
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_id = args.run_id or config["project"]["run_id"]
    run_root = Path(config["project"]["output_root"]) / run_id
    gpu_uuids = [value.strip() for value in args.gpu_uuids.split(",") if value.strip()]
    if not gpu_uuids or len(gpu_uuids) != len(set(gpu_uuids)):
        raise ValueError("gpu UUID list must be non-empty and unique")

    metadata_paths = [path.resolve() for path, _ in iter_metadata(run_root)]
    if not metadata_paths or len(metadata_paths) != len(set(metadata_paths)):
        raise RuntimeError("metadata inventory must be non-empty and unique")
    shards = partition_paths(metadata_paths, len(gpu_uuids))
    manifest_root = run_root / "parallel_eval_manifests"
    log_root = ROOT / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    processes: list[tuple[int, subprocess.Popen, object]] = []
    try:
        for shard_id, (gpu_uuid, paths) in enumerate(zip(gpu_uuids, shards)):
            manifest = manifest_root / f"shard_{shard_id:02d}_of_{len(shards):02d}.txt"
            write_manifest(manifest, paths)
            log_handle = (log_root / f"pilot_eval_shard_{shard_id:02d}.log").open("w", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
            command = [
                str(ROOT / ".venv/bin/python"),
                str(ROOT / "evaluators.py"),
                "--config",
                str(config_path),
                "--run-id",
                run_id,
                "--device",
                "cuda:0",
                "--metadata-list",
                str(manifest),
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
                    for other_id, other, _ in processes:
                        if other_id in active:
                            other.wait(timeout=30)
                    raise RuntimeError(f"evaluation shard {shard_id} exited with status {returncode}")
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

    print(json.dumps({"status": "complete", "metadata": len(metadata_paths), "shards": len(shards)}))


if __name__ == "__main__":
    main()
