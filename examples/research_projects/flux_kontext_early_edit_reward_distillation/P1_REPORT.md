# P1 A/B/C/D report

P1 used the fixed PIE-Bench held-out manifest (`pie_test8.json`), explicit
512x512 source and generated resolution, 28 steps, guidance 3.5, alpha 0.2,
and one generation seed per sample because official EditScore inference was
resource-intensive. The downgrade is recorded here; it is not a 2-3 seed
estimate.

| method | n | EditScore mean | std | preserve L1 | edit L1 |
|---|---:|---:|---:|---:|---:|
| A native deterministic | 8 | 4.6388 | 3.7737 | 0.2396 | 0.2975 |
| B random independent SDE | 8 | 5.5860 | 3.0490 | 0.2273 | 0.2902 |
| C random coupled SDE | 8 | 5.1838 | 3.2723 | 0.2407 | 0.2939 |
| D EditScore-selected coupled SDE | 8 | 5.2484 | 3.4949 | 0.2406 | 0.2933 |

D is 0.6096 above A and 0.0646 above C, which satisfies the weak-positive
trend gate for entering the cache/LoRA stage. These values are exploratory
because each sample has one generation seed.

Artifacts: `/data15/hyp/experiments/flux_kontext_early_edit_reward_distill/p1_abcd/`.
