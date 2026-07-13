#!/usr/bin/env bash
set -euo pipefail

while tmux has-session -t flux_probe_pilot 2>/dev/null; do
  sleep 60
done

cd /home/hyp/Code/flux-kontext-block-probing
expected=$(.venv/bin/python scripts/expected_counts.py --field pilot_stage1_jobs)
eval_count=$(find /data15/hyp/project_storage/flux-kontext-block-probing/main_512/images -name '*.eval.json' | wc -l)
if [[ "$eval_count" -lt "$expected" ]]; then
  echo "pilot evaluation incomplete: eval=$eval_count expected=$expected" >&2
  exit 1
fi
.venv/bin/python scripts/make_calibration_bundle.py > logs/calibration_bundle.log 2>&1
