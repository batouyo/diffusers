#!/usr/bin/env bash
set -euo pipefail

while tmux has-session -t flux_probe_followup 2>/dev/null; do
  sleep 60
done

cd /home/hyp/Code/flux-kontext-block-probing
if [[ ! -f /data15/hyp/project_storage/flux-kontext-block-probing/main_512/stage3_blocks.json ]]; then
  echo "stage3 block selection missing" >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES=GPU-2cd22c91-025f-16c6-f54a-0947f721d15e
.venv/bin/python scripts/run_alpha_after_followup.py > logs/pilot_alpha.log 2>&1
