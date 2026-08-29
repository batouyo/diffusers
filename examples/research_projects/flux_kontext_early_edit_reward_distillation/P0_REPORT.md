# P0 validation report

## Dataset and resolution

- Dataset: PIE-Bench native masks, category-balanced 16 train + 8 held-out.
- Target/generated resolution: 512x512, rounded to the VAE packing multiple.
- Generated image latent grid: 64x64 before 2x2 packing, 1024 image tokens after packing.
- Kontext source conditioning uses the same explicit 512x512 preprocessing in this P0 runner. The official `_auto_resize` preferred-resolution path is separately audited and recorded; it must not be conflated with the target resolution.

## Mechanism checks

- 28 scheduler steps completed.
- First two non-zero diffusion transitions were scheduler indices 1 and 2.
- Native source-image conditioning remained concatenated with generated latents.
- K=4 branching, shared preserve noise, independent edit noise, top-2 repeated scoring, and post-branch metadata all completed.
- CPU regression suite: 14 passed.

## Reward gate

For sample `pie_5_change_attribute_pose_40_513000000003`, seed `20260830`:

| candidate | EditScore overall |
|---|---:|
| deterministic native baseline | 8.1388 |
| selected SDE winner | 0.0000 |

The official Qwen3-VL 4B EditScore explanation reports severe distortion and loss of recognizable source content for the selected SDE winner. This is a negative P0 result. P1 and LoRA distillation are gated until the SDE candidate path is corrected and a repeat P0 demonstrates no baseline degradation.

Artifacts are under `/data15/hyp/experiments/flux_kontext_early_edit_reward_distill/p0_smoke/sample0/`.
