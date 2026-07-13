#!/usr/bin/env bash
set -euo pipefail

while tmux has-session -t flux_probe_pilot 2>/dev/null; do
  sleep 60
done

cd /home/hyp/Code/flux-kontext-block-probing
expected=$(.venv/bin/python scripts/expected_counts.py --field pilot_stage1_jobs)
png_count=$(find /data15/hyp/project_storage/flux-kontext-block-probing/main_512/images -name '*.png' | wc -l)
eval_count=$(find /data15/hyp/project_storage/flux-kontext-block-probing/main_512/images -name '*.eval.json' | wc -l)
if [[ "$png_count" -lt "$expected" || "$eval_count" -lt "$expected" ]]; then
  echo "pilot incomplete: png=$png_count eval=$eval_count expected=$expected" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=GPU-2cd22c91-025f-16c6-f54a-0947f721d15e
.venv/bin/python scripts/run_pilot_stage2.py > logs/pilot_stage2.log 2>&1
