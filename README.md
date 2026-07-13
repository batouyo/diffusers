# FLUX.1-Kontext Block Probing

Independent, training-free probing of which FLUX.1-Kontext-dev transformer blocks are most responsive to text hidden-state enhancement.

The project never imports code, manifests, candidate blocks, or results from earlier `semantic-strength-editing` experiments. Diffusers is installed from a fresh official checkout pinned in `probe_config.yaml`.

## Safety gates

1. `inspect` records the runtime model structure and tensor contracts.
2. `pytest` and `identity-test` must pass before any scan.
3. `pilot` uses the formal inference configuration and can be resumed.
4. Discovery, alpha selection, and held-out validation are separate stages.

All large artifacts are written under `/data15/hyp/project_storage/flux-kontext-block-probing`.

## Commands

```bash
python probe_flux_kontext_blocks.py inspect --config probe_config.yaml
pytest tests/test_interventions.py
python probe_flux_kontext_blocks.py run --config probe_config.yaml --stage pilot --shard-id 0 --num-shards 2
python evaluators.py --config probe_config.yaml --run-id main
python aggregate_results.py --config probe_config.yaml --run-id main
```

