# Mask x Reward Diagnostic

This directory stores the single-sample 2x2 diagnostic for mask selection and
Reward selection. It is a mechanism diagnostic, not a statistical ablation.

## Experiment

- sample: pie_8_change_background_80_814000000007
- instruction: a painting of [moon] rising above the sea level
- seed: 20260830
- sampler: FLUX-Kontext, 30 steps, guidance 2.5
- resolution: 512x512
- candidates: 4 early SDE candidates
- coupling: enabled; shared noise in the preservation region
- Reward: local official EditScore, one inference per candidate
- preservation-aware score: EditScore - 5.0 * outside-mask pixel L1

The four cases are the Cartesian product of dynamic versus GT edit masks and
original versus preservation-aware Reward scoring.

| Mask | Reward | Winner | Winner outside-mask L1 |
| --- | --- | ---: | ---: |
| Dynamic | Original EditScore | 3 | 0.25697 |
| Dynamic | Preservation-aware | 3 | 0.25697 |
| GT | Original EditScore | 0 | 0.01555 |
| GT | Preservation-aware | 1 | 0.01550 |

Reward values by candidate (candidate order 0..3):

- Dynamic + original: [6.3875, 6.1188, 6.5727, 7.2000]
- Dynamic + preservation-aware: [5.1041, 4.8342, 5.2903, 5.9151]
- GT + original: [5.5426, 5.5426, 5.5426, 5.5426]
- GT + preservation-aware: [5.4648, 5.4651, 5.4647, 5.3482]

## Observations

1. The dynamic velocity-similarity mask marks most generated elements as edited:
   80.86%, 83.63%, 83.84%, and 84.41% in the first four steps.
2. The GT mask marks 4.79% of generated elements as edited. Its outside-mask
   L1 is about 0.0155, versus 0.2570 for the dynamic mask.
3. Under the dynamic mask, preservation-aware scoring does not change the
   winner. Under the GT mask it changes the winner from 0 to 1, but the four
   original EditScore values are identical, so Reward discrimination is weak.
4. All four cases have four unique corrected candidate state hashes at each
   search stage. Dynamic delta residual norms are about 6.0 and 8.2; GT norms
   are about 1.47 and 2.0 for stages one and two.

## Files

Each case directory contains `source.png`, `winner.png`, `trajectory.json`,
and `branch_scores.json`. The root `summary.json` contains the compact result
record and timings. The diagnostic runner is:

`examples/research_projects/flux_kontext_early_edit_reward_distillation/scripts/run_mask_reward_diagnostic.py`

## Scope

This result is a quick single-sample diagnostic and does not support claims of
statistical significance. The GT mask is used only by a runtime similarity-map
override for diagnosis; it does not change the sampler implementation. A
1024x1024 official VeloEdit parity run was not completed because of H20 memory
limits, so these observations should not be interpreted as a refutation of the
VeloEdit mechanism.
