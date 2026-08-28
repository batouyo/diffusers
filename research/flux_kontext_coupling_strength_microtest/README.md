# FLUX-Kontext Coupling-as-Strength Micro-Test

This directory contains the independent implementation and machine-readable artifacts for the four-sample FLUX-Kontext coupling-as-strength micro-test.

The upload is limited to:

- the fixed four-sample manifest;
- the standalone runner and its unit tests;
- generation configuration and provenance records;
- validation JSON files;
- raw numeric results and per-seed/per-sample metric tables;
- numeric generation diagnostics.

The upload intentionally excludes subjective conclusions, manual visual-review notes, summary prose, model weights, source/target image files, and generated image grids.

The experiment configuration is fixed at 28 inference steps, guidance 3.5, rho values `[1.00, 0.75, 0.50, 0.25, 0.00]`, three stochastic seed slots, and the RF-equivalent SDE implementation recorded in `results/implementation_notes.md`.

The expected completed numeric tables contain 60 raw branch rows, 12 per-seed rows, and 4 per-sample rows. Validation artifacts record the schedule, explicit-noise coupling, packed-mask mapping, ROI construction, and smoke-test checks.

The code is an independent experiment entry point and does not modify the existing temporal-probe runner.

