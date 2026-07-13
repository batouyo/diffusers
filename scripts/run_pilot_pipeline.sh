#!/usr/bin/env bash
set -euo pipefail

cd /home/hyp/Code/flux-kontext-block-probing
export CUDA_VISIBLE_DEVICES=GPU-2cd22c91-025f-16c6-f54a-0947f721d15e

.venv/bin/python probe_flux_kontext_blocks.py \
  --config probe_config.yaml --device cuda:0 run \
  --stage pilot --split discovery >> logs/pilot_gpu2.log 2>&1

.venv/bin/python evaluators.py \
  --config probe_config.yaml --device cuda:0 > logs/pilot_eval.log 2>&1

.venv/bin/python aggregate_results.py \
  --config probe_config.yaml > logs/pilot_aggregate.log 2>&1
