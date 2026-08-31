# Dynamic VeloEdit SDE-Off Control

This directory contains the extra control requested by the mask/Reward
diagnostic review. It uses the shared sampler with all stochastic search and
Reward paths disabled.

## Configuration

- sample: pie_8_change_background_80_814000000007
- instruction: a painting of [moon] rising above the sea level
- seed: 20260830
- resolution: 512x512
- sampling: 30 steps, guidance 2.5
- VeloEdit intervention: first 4 steps
- SDE search: disabled
- Reward: disabled
- Coupled SDE: disabled

## Results

| Strength | Outside-mask L1 | Edit-region L1 |
| ---: | ---: | ---: |
| 0.00 | 0.01527 | 0.01158 |
| 0.50 | 0.11783 | 0.11218 |
| 1.00 | 0.25685 | 0.33537 |

The outside-mask metric is `preserve_l1` and the edit-region metric is
`edit_l1` in `strength_metrics.csv`.

## Interpretation

`s=0` is close to the source (whole-image mean absolute difference 0.01507),
while `s=0.5` is intermediate and `s=1` has strong edit and background drift.
The `s=1` outside-mask L1 is 0.25685, nearly identical to the previous Dynamic
+ Reward winner value of 0.25697. This indicates that most background drift in
this sample is already present in the deterministic Dynamic VeloEdit controller
before SDE search or Reward selection.

The dynamic mask editing ratios in the first four steps are 80.86%, 83.63%,
83.84%, and 84.41%. This is a single-sample diagnostic, not a significance
claim.

## Artifacts

The sample directory contains source, edit mask, preservation, the three
strength images, the contact sheet, trajectory metadata, and `branch_scores.json`.
