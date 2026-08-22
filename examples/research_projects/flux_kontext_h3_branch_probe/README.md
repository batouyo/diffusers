# FLUX-Kontext H3 Branching Probe

This research project tests whether FLUX-Kontext retains semantic editing
freedom after a shared sampling prefix. It uses the default 28-step
FlowMatch-Euler schedule, branches at `k={0,1,2,3,4,5,8,14}`, and probes
strengths `s={0,0.25,0.5,0.75,1}` with a VKeep-style velocity interpolation.

The runner reads PIE-Bench++ parquet data from `/data15/hyp/dataset/PIE-Bench`
and writes trajectories, strength images, metrics, plots, and checkpoint
summaries under the requested output directory.

Run on H20 with the SEAdapter environment:

```bash
/home/gem/anaconda3/envs/SEAdapter/bin/python run_h3_probe.py \
  --limit 50 --per-category 10 \
  --output-root /data15/hyp/experiments/flux_kontext_h3_branch_probe/pilot50
```

Use `--resume` to reuse cached trajectories and `--skip-perceptual` for a
sampling-only smoke test. `monitor_h3_checkpoints.py` writes summaries after
every ten complete samples when the two-shard pilot is running.

Unit tests:

```bash
/home/gem/anaconda3/envs/SEAdapter/bin/python -m pytest -q tests/test_h3_probe.py
```
