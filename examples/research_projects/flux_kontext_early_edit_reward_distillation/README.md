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
selected branch residual is cached as `winner_state - deterministic_edit_state`.

Strength rollout re-evaluates both model velocities on the current interpolated
state, then applies `v_preserve + s * (v_edit - v_preserve)` and the residual
`(1-s) * preservation_residual + s * reward_residual`. Thus strength scans do
not resample and do not replay velocities cached at other states. Set
`--critical-step-indices 1,2` to override the fallback critical transitions.

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
use paired cached velocities as an engineering approximation; `s=0` and `s=1`
return the cached preservation and selected full-edit endpoints exactly.
