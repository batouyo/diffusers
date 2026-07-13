#!/usr/bin/env bash
set -euo pipefail

while tmux has-session -t flux_probe_followup 2>/dev/null; do
  sleep 60
done

cd /home/hyp/Code/flux-kontext-block-probing
.venv/bin/python scripts/verify_pilot_followup.py --check-only
.venv/bin/python aggregate_results.py --config probe_config.yaml > logs/final_pilot_aggregate.log 2>&1
.venv/bin/python scripts/make_pilot_report.py > logs/pilot_report.log 2>&1
