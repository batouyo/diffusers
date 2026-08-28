# Coupled-SDE mechanism diagnostic

`src/run_mechanism_diagnostic.py` is independent of the existing
coupling-as-strength runner. It produces a native deterministic baseline,
two TempFlow-style positive controls, RF-SDE local-window and ODE-suffix
counterfactuals, and a global/all-stochastic RF-SDE sensitivity diagnostic.

The run must use the previously validated FLUX-Kontext environment
(`diffusers==0.35.2`) and must provide the historical PIE-Bench source outside
the repository; source data and model weights are never committed.

```bash
python src/run_mechanism_diagnostic.py \
  --manifest configs/coupling_strength_4_cases.json \
  --historical-source /path/to/historical_positive_control.png \
  --output results_mechanism_diagnostic \
  all
```

Before generation, run `validate`; after generation inspect
`trajectory_diagnostics.csv`, `final_metrics.csv`, `conclusions.json`, the four
plots, `legacy_replay.json`, and `report.md`. The historical replay uses the
archived four-branch CUDA random-draw shape before selecting branch 0, so that
its pre-registered numeric comparison remains meaningful.

