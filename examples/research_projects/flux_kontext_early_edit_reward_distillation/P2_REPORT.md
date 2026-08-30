# P2 status

The 16-sample teacher cache is complete at
`/data15/hyp/experiments/flux_kontext_early_edit_reward_distill/teacher_cache/`.
Each record contains baseline/winner states, velocities and delta velocities at
scheduler indices 1 and 2, the generated-token mask, and all Kontext
conditioning tensors. The cache builder performs no reward evaluation.

The current cache winner policy is explicitly recorded as
`fixed_coupled_candidate_zero_no_reward`; P1 did not persist train-set reward
winners, so these records must not be described as EditScore-selected teacher
trajectories.

Selective LoRA training is not complete. The 1-step smoke reached transformer
forward but failed with CUDA OOM because unrelated Ray jobs occupied the H20
GPUs. It must be retried after GPU resources are available; no 250-step result
or LoRA validation claim is made.
