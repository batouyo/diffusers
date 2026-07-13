#!/usr/bin/env bash
set -euo pipefail

while tmux has-session -t flux_probe_alpha 2>/dev/null; do
  sleep 60
done

cd /home/hyp/Code/flux-kontext-block-probing
calibration=/data15/hyp/project_storage/flux-kontext-block-probing/main_512/calibration/calibration_report.json
while [[ ! -f "$calibration" ]] || ! grep -q '"gate_pass": true' "$calibration"; do
  sleep 300
done
export CUDA_VISIBLE_DEVICES=GPU-2cd22c91-025f-16c6-f54a-0947f721d15e
.venv/bin/python scripts/run_formal_pipeline.py > logs/formal_pipeline.log 2>&1
