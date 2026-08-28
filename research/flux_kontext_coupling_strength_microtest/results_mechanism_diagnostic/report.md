# Coupled-SDE mechanism diagnostic

## Outcome

**Primary classification: `RF_SDE_BASELINE_EDITING_DEGRADATION`; no evidence that the tested early local scalar-rho coupling is an editing-strength control.**

The native ODE output is a visually valid blue-ball edit. Target-image similarity was recorded only as a diagnostic and was not used as a gate, because the dataset target is not assumed to be a required attainable rendering.

## Q1–Q9 evidence

1. Native ODE gate: `PASS`. Preserve-region L1 vs source was 0.0117568; diagnostic global/ROI progress was 0.2595.
2. B-historical replay: `ENVIRONMENT_OR_VERSION_REPRODUCTION_FAILED`. The local latent error met its tolerance (0.3556%), but final latent error (6.6211%) and pixel-RMS error (0.0216855) exceeded the registered limits. This is an environment/version reproduction failure, not a Coupled-SDE conclusion.
3. B-matched changed the native output modestly: global progress 0.2595 → 0.340867, pixel RMS 0.0353169, LPIPS 0.0356018.
4. C (masked early three non-zero-diffusion steps): R_C=0.195548; the early rho difference contracts strongly by the endpoint. The rho0–rho1 global progress delta is -0.0245913, which is small and not a usable semantic-strength effect.
5. D (same early prefix, deterministic suffix): R_D=0.266069; neither the registered preservation criterion nor the amplification criterion is met. The rho0–rho1 global progress delta is -0.00721519.
6. E (global, all stochastic steps): endpoint global progress changes by 0.278742 between rho1 and rho0, but this is accompanied by RF-SDE baseline degradation and the qualitative difference is chiefly texture/sample variation. It is not evidence of a robust editing-strength control.
7. Baseline comparison: native global progress is 0.2595, whereas C rho1 is -0.449336; `RF_SDE_BASELINE_EDITING_DEGRADATION=True`.
8. No formula/noise/mask/scheduler/prefix-identity sanity check failed: the deterministic scheduler check has max absolute error 0, the first RF step is deterministic, and C/D share exactly cloned boundary states.
9. There are no p-values: this is one sample and one seed. The conclusion is a mechanism diagnosis, not a claim of cross-sample statistical generality.

## Interpretation boundary

This run does not establish a code-level Coupled-SDE implementation failure in the checked formula, noise coupling, mask packing, scheduler ordering, or C/D prefix identity. It does show a marginal historical replay mismatch and a degraded RF-SDE rho1 baseline. Therefore the tested local early scalar-rho parameterization is not currently supported as a strength-control signal. The result does not rule out other coupling fields, other stochastic editing formulations, or other trajectories.
