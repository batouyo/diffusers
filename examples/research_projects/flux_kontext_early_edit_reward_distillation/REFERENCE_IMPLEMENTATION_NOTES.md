# Reference implementation notes

- Diffusers base: `48fb9912b`, branch `codex/flux-kontext-h3-branch-probe`.
- SliderEdit snapshot: `ee6b1fc`; reused training, selective LoRA and pipeline
  adapter conventions. Rank-4 and token/timestep gating are research changes.
- TempFlow-GRPO snapshot: `63e4def`; reused only single SDE branch followed by
  deterministic rollout. PPO/GRPO/log-prob optimization is excluded.
- Official syncSDE: `Z-Jianxin/syncSDE-release`, commit
  `36cf2a38b1c08425257d7bdbe359c6afd2fbd4c5`.
- Official reward: `VectorSpaceLab/EditScore`, commit
  `4609c5d2ebb62fdebf665d3c924686d896ef1f74`; Qwen3-VL-4B base and
  `EditScore-Qwen3-VL-4B-Instruct` LoRA are installed under `/data15/hyp/weight`.

The implementation is research-local and does not modify `src/diffusers`.
