# FLUX.1-Kontext Block Probing

Independent, training-free probing of which FLUX.1-Kontext-dev transformer blocks are most responsive to text hidden-state enhancement.

The project does not import code, manifests, candidate blocks, or results from earlier `semantic-strength-editing` experiments. Diffusers is installed from a fresh official checkout and its commit is pinned in `probe_config.yaml`. TexTailor FLUX.1-Dev indices are introduced only after candidate locking as a held-out control.

## Fixed locations

- Code: `/home/hyp/Code/flux-kontext-block-probing`
- Model: `/data15/hyp/weight/FLUX.1-Kontext-dev`
- Dataset: `/data15/hyp/dataset/flux-kontext-block-probing`
- Artifacts: `/data15/hyp/project_storage/flux-kontext-block-probing/main_512`
- Machine-readable live status: `main_512/pipeline_status.json`

## Safety and research gates

1. Runtime inspection records every actual block, signature, input/output shape, dtype, device, and stream boundary.
2. Unit tests plus the real-model identity test must pass before scanning; `alpha=1` must be numerically identical to baseline.
3. Baseline and intervention jobs reuse the same source, instruction, seed, packed initial latents, scheduler settings, resolution, and inference parameters.
4. Stage 1 probes one block at a time. Multi-block hooks are enabled only in the explicitly separate joint-validation phase.
5. Candidate selection requires gain, disable loss, bootstrap confidence, preservation, bad-image, category, seed, and correlated-adjacency redundancy gates. No minimum top-k is forced.
6. A complete negative joint result and a preregistered no-go are valid scientific outcomes, but both require complete execution evidence.
7. The final completion audit must pass all 15 requirements; file presence alone is insufficient.

## Reproduce the preflight

```bash
cd /home/hyp/Code/flux-kontext-block-probing
source .venv/bin/activate

python probe_flux_kontext_blocks.py --config probe_config.yaml --device cuda:0 inspect
python probe_flux_kontext_blocks.py --config probe_config.yaml --device cuda:0 identity-test
python -m pytest -q
```

The active overnight pipeline is started by:

```bash
scripts/run_pilot_pipeline.sh
```

It performs resumable 57-block pilot generation, evaluates the locked 80-example calibration subset first, exports the blind-rating bundle, finishes all automatic evaluation, and aggregates Stage 1. Waiting tmux stages then run top-15 `disable_text`, top-10 `remove_block`, pilot alpha sensitivity, the report, and—after the human calibration gate—the formal three-seed pipeline.

## Safe manual sharding

The runner supports deterministic disjoint shards. Only use GPU UUIDs confirmed to be idle and assigned to this project; do not use numeric CUDA ordering as a physical-GPU identity.

```bash
CUDA_VISIBLE_DEVICES=<approved-gpu-uuid> python probe_flux_kontext_blocks.py \
  --config probe_config.yaml --device cuda:0 run \
  --stage enhance_text --split discovery --shard-id 0 --num-shards 2
```

Run the complementary shard with `--shard-id 1`. Atomic metadata, configuration hashes, output hashes, and latent hashes prevent valid work from being regenerated or incompatible work from being silently reused.

## Human calibration gate

When ready, open:

`/data15/hyp/project_storage/flux-kontext-block-probing/main_512/calibration/index.html`

For every blinded example, select an integer score from 0 to 4 and enter brief visible evidence. The page saves progress in browser local storage and exports `blinded_labels.csv` only after all 80 rows are complete. Replace the blank CSV in the same calibration directory, then run:

```bash
cd /home/hyp/Code/flux-kontext-block-probing
.venv/bin/python scripts/score_calibration.py
```

The locked 40-example validation subset must reach Spearman correlation at least 0.7. The formal pipeline waits for a hash-recorded `calibration_report.json` with `gate_pass: true`; it does not bypass a failed or incomplete human gate.

## Acceptance artifacts

Primary outputs under `main_512` are:

- `raw_metrics.csv`, `block_summary.csv`, `stream_summary.csv`, `alpha_summary.csv`
- `selected_blocks.json`, `stage3_blocks.json`, and `selected_alpha.json` when applicable
- `joint_metrics.csv`, `joint_summary.csv`, `joint_category_summary.csv`, `joint_seed_summary.csv`, `joint_validation.json` when candidates exist
- `FORMAL_NO_GO.json` when no block clears all preregistered gates
- `plots/`, including response curves, alpha sensitivity, joint controls, and image grids when applicable
- `FINAL_REPORT.md`
- `completion_audit.json` and `completion_audit.md`

Inspect status and audit without loading a model:

```bash
python -m json.tool /data15/hyp/project_storage/flux-kontext-block-probing/main_512/pipeline_status.json
.venv/bin/python scripts/audit_completion.py
```
