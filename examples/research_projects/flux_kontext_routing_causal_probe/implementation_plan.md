# FLUX-Kontext continuous-strength overfit experiment

## Scope and invariants

- Branch: `exp/strength-conditioned-overfit-v1`.
- Code root: `/home/hyp/Code/diffusers/examples/research_projects/flux_kontext_routing_causal_probe`.
- Large artifacts: `/data15/hyp/experiments/flux_kontext_strength_overfit_v1`.
- Base model, VAE, and text encoders stay frozen. Only target-image-token residual adapters and their gates train.
- All runs force `PYTHONPATH=/home/hyp/Code/diffusers/src` and `_native_flash`; the process aborts if the imported diffusers checkout is not this checkout.
- Existing Oracle/VKeep files and historical output directories are read-only.

## Verified model contract

- Pipeline: `FluxKontextPipeline`; transformer: `FluxTransformer2DModel`.
- Blocks: `dual.00`--`dual.18` at `transformer.transformer_blocks`, and `single.00`--`single.37` at `transformer.single_transformer_blocks`.
- The image stream is `[target noisy tokens, source condition tokens]`. The target slice is validated at run time from the actual layout.
- Scheduler: `FlowMatchEulerDiscreteScheduler`, dynamic shift, 28 steps, guidance 2.5, true CFG 1.0, max sequence length 512.
- Base checkpoint: `/data15/hyp/weight/FLUX.1-Kontext-dev`.
- Legacy `TargetLowRankResidual` is RMSNorm/down/up with zero-initialized up factor. It remains unchanged for Stage0 and the residual-scaling baseline.

## New components

- `strength_residual.py`: target-only RMSNorm/down/SiLU/up adapters, time-strength gate, hooks, context contract, and layer statistics.
- `strength_overfit_data.py`: metadata validation, pipeline-native preprocessing, trajectory cache, and fingerprints.
- `strength_overfit_training.py`: frozen paired teachers, static-state and detached online-two-step objectives, checkpointing, and safety logging.
- `strength_overfit_masks.py`: velocity-difference masks, robust normalization, hard/soft weighting, EMA, and spatial conversion.
- `strength_overfit_evaluation.py`: endpoints, velocity metrics, monotonicity/smoothness metrics, contact sheets, reports, and output bookkeeping.
- `run_strength_overfit.py`: preflight/prepare/train/evaluate/report CLI.

The adapter is:

```
residual = (1 - strength) * spatial_weight * gate(sigma, strength) * adapter(target_hidden)
```

The gate consumes `[sigma, strength, sigma * strength]`, uses `Linear(3, 32) -> SiLU -> Linear(32, num_layers)`, and returns `1 + tanh(raw)`. Its final linear layer and every adapter up factor are zero initialized. Therefore strength 1.0 is an exact no-op.

## Dataset and state protocol

The core set contains five generated source images: `attribute_01`, `pose_01`, `addition_01`, `material_01`, and `season_01`. The expansion adds one sample per category, with the second addition/removal example being a removal. The metadata schema includes source/full/neutral prompts, target phrase and description, category, generation provenance and all seed lists.

The neutral prompt is uniformly the empty string. Images use only the pipeline native preprocessing; resolved dimensions and preprocessed images are saved. The train, validation, and rollout seeds are respectively `[1101,1102,1103,1104]`, `[2101,2102]`, and `[3101,3102,3103]`. State indices 4, 13 and 23 are held out for unseen-timestep validation.

At a fixed current state the frozen teachers are:

```
v_edit = base(z_t, full_prompt)
v_neutral = base(z_t, neutral_prompt)
v_target(s) = s * v_edit + (1 - s) * v_neutral
```

Every paired call validates the same state, timestep/sigma, source condition, noise definition and input fingerprint.

## Configured experiment sequence

1. `00_stage0_repro.json`: historical VKeep Experiment B reproduction.
2. `10_previous_scaling_core5.json`: s=0-only shared residual baseline, with inference-time residual scaling.
3. `11_stage1a_pilot2.json` then `12_stage1a_early_core5.json`: explicit strength at dual.00/01/02.
4. `13_stage1b_distributed_core5.json`: dual.00/01/10/18 plus single.19/34.
5. `20_stage2_online2_core5.json`: 50% static and 50% detached online two-step training.
6. `30_stage3_hard_core5.json` and `31_stage3_soft_core5.json`.
7. `40_stage4_expand10.json` only after a core configuration passes.

Strength sampling is 20% zero, 20% one and 60% Uniform(0,1), with a paired `sa < sb` sample on 50% of updates. The default loss is velocity MSE plus monotonicity (0.05), progress (0.10), and residual regularization (1e-4). The first 300 updates use velocity only; mono/progress ramp for 200 updates.

## Acceptance and safety

Stage0 requires a numerically exact strength-1 endpoint, scheduler parity, target-only hooks, and no CUDA error. Stage1 requires at least four of five samples to satisfy endpoint fidelity, rho >= 0.90, <=1 monotonic violation, <=35% maximum adjacent LPIPS jump, five visually distinct ordered states, and non-collapsed unseen-state velocity fit. Later stages use the documented 30% corrected-state, 20% spatial-preservation, and 8/10 expansion gates.

All runs save resolved config, environment/git fingerprints, commands, checkpoints, raw JSONL/CSV metrics, contact sheets, masks, reports, peak memory, timing, per-layer RMS/gate/parameter/gradient statistics. Existing results are never overwritten.

## First command after implementation

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/hyp/Code/diffusers/src /home/gem/anaconda3/envs/SEAdapter/bin/python examples/research_projects/flux_kontext_routing_causal_probe/run_strength_overfit.py --config examples/research_projects/flux_kontext_routing_causal_probe/configs/strength_overfit_v1/00_stage0_repro.json --mode preflight
```

