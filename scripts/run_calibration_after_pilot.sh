#!/usr/bin/env bash
set -euo pipefail

while tmux has-session -t flux_probe_pilot 2>/dev/null; do
  sleep 60
done

cd /home/hyp/Code/flux-kontext-block-probing
.venv/bin/python scripts/verify_pilot_complete.py --check-only
.venv/bin/python scripts/make_calibration_bundle.py > logs/calibration_bundle.log 2>&1
