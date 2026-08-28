# Coupled-SDE mechanism diagnostic — stopped at native baseline gate

## Status

`NATIVE_BASELINE_INCONCLUSIVE`

The diagnostic stopped before B-historical, B-matched, C, D, and E. This is the
pre-registered stop condition: both allowed native deterministic ODE candidates
failed the fixed native gate. No coupled-SDE endpoint claim is made from this
run.

## Checks that did pass

- The five CPU numerical tests passed: RF first-step determinism, non-first RF
  formula, deterministic SeedSequence noise, empty-region handling, and invalid
  sigma rejection.
- The explicit-noise validation passed; the first RF-SDE step has diffusion
  coefficient exactly zero.
- The hand-instrumented deterministic update matched an independently
  initialized `FlowMatchEulerDiscreteScheduler.step()` for all 28 schedule
  indices with maximum absolute difference `0.0`.

## Native gate evidence

| Candidate | global DINO progress | ROI-DINO progress | target-edit L1 improvement | preserve L1 vs source | Result |
|---|---:|---:|---:|---:|---|
| `native_candidate_0` | 0.259500 | 0.259500 | -0.158341 | 0.011757 | FAIL |
| `native_candidate_1` | 0.233900 | 0.233900 | -0.224450 | 0.018000 | FAIL |

Both outputs have positive directional progress and low preserve-region change,
but neither improves edit-region L1 to the fixed MagicBrush target by the
required 10%; each is farther from the target in that region. Visual inspection
of the ball example is consistent with a target-alignment issue: the model
produced a simple blue ball for the text instruction, while the paired target
contains a more complex teal/multicolour ball transformation. This observation
does not relax or replace the locked numeric gate.

## Q1–Q9 disposition

1. Native baseline construction did not meet the pre-registered gate for either
   permitted candidate.
2. The historical TempFlow positive control was not run, because the stop gate
   occurred first.
3. The matched TempFlow control was not run.
4. The local-window RF-SDE comparison C was not run.
5. The ODE-suffix counterfactual D was not run.
6. The global/all-stochastic comparison E was not run.
7. No RF-SDE baseline-degradation claim can be made; no RF-SDE endpoint was
   generated after the gate.
8. The numerical and scheduler sanity checks did not identify an implementation
   bug. They do not prove the absence of all possible implementation issues.
9. The current evidence neither supports nor refutes rho as an editing-strength
   variable: the experiment did not reach the rho diagnostic stage.

## Scope boundary

This stopped run must not be interpreted as evidence against all trajectory
coupling methods, coupling fields, or stochastic editing formulations. It only
establishes that the present diagnostic cannot proceed under its fixed native
baseline gate and its two allowed candidate samples.

