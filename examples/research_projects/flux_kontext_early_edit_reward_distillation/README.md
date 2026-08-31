# FLUX-Kontext Early Edit Reward Distillation

Research-only validation primitives for early SDE branching, selective coupling,
EditScore ranking, and early-only LoRA distillation.

The P0 numerics are model-independent and tested on CPU. P1+ remains gated on
official syncSDE and Qwen3-VL 4B EditScore resources.

## Training-free continuous strength prototype

`src/early_edit_reward_distillation/continuous_strength.py` provides paired
preservation/edit trajectories, early regional SDE branching, a pluggable
`RewardScorer.score(source, candidate, instruction)` contract, and deterministic
velocity interpolation for strengths in `[0, 1]`.

The preservation trajectory uses the fixed neutral prompt
`preserve the source image without any edit` and reuses the edited trajectory's
initial latent. This is a runnable approximation, not an oracle preservation
velocity. Editing masks are estimated from normalized early generated-token
differences; source conditioning tokens are never branched. At configured
critical transitions, preservation receives shared noise and edited states use
the same noise outside the edit mask plus independent noise inside it. The
Pilot token masks are diagnostic metadata only. At the single search transition
(step 1 by default), preservation receives shared noise while the edit region
explores independently; Reward selects one early branch and only its masked edit-direction velocity correction is deployed.

Strength rollout re-evaluates model velocities on the current state and applies
the short VeloEdit-style controller: preserve intervention lasts 4 steps and
edit-strength interpolation lasts 2 steps. The selected correction is added to
the edit velocity once at the search step and is then absent from later steps.
There is no state residual replay. Strength scans do not resample or call Reward.

Run a single sample with a reward factory:

```bash
PYTHONPATH=src python scripts/run_continuous_strength.py \
  --model /path/to/FLUX.1-Kontext-dev \
  --source /path/to/source.png \
  --instruction "replace the red object with a blue object" \
  --reward-factory my_reward:make_scorer \
  --output /tmp/continuous_strength
```

For mechanism-only debugging, replace `--reward-factory` with
`--candidate-index 0`. Such runs are explicitly marked as not Reward-selected.
SDE noise is injected only during early branch search; strength rollout is
deterministic and does not resample or call Reward again. Intermediate values
are recomputed from the initial latent like every other strength. Endpoint
parity is checked by regression tests rather than enforced through caching.
