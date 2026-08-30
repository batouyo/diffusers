import math
import torch
from early_edit_reward_distillation.core import align_image_token_mask, coupled_noise, critical_nonzero_steps, greedy_two_stage_branch, native_euler_sde_step, noise_correlations, rf_sde_step
from early_edit_reward_distillation.lora import masked_residual, timestep_gate

def test_first_step_is_deterministic():
    x, v = torch.randn(1, 4, 3), torch.randn(1, 4, 3)
    a, meta = rf_sde_step(x, v, 1.0, 0.9, torch.randn_like(x), first_step=True); b, _ = rf_sde_step(x, v, 1.0, 0.9, torch.randn_like(x), first_step=True)
    assert torch.equal(a, b) and meta["diffusion_coeff"] == 0.0

def test_rf_formula_and_critical_steps():
    x, v, noise = torch.tensor([[[.5, -.2]]]), torch.tensor([[[.1, .3]]]), torch.tensor([[[.7, -1.1]]])
    out, meta = rf_sde_step(x, v, .8, .7, noise); c = math.sqrt(2*.8/.2*.1); expected = x + (-.1)*(2*v + x/.2) + c*noise
    assert torch.allclose(out, expected) and math.isclose(meta["diffusion_coeff"], c, rel_tol=1e-12)
    assert [r["index"] for r in critical_nonzero_steps([1., .9, .8, .7])] == [1, 2]

def test_native_euler_alpha_zero_matches_ode_mean():
    sample = torch.randn(1, 4, 2)
    velocity = torch.randn_like(sample)
    noise = torch.randn_like(sample)
    updated, meta = native_euler_sde_step(sample, velocity, .9, .8, noise, alpha=0.0)
    expected = sample - .1 * velocity
    assert torch.equal(updated, expected.to(updated.dtype))
    assert meta["perturbation_norm"] == 0.0

def test_noise_correlation_separates_preserve_and_edit_regions():
    shared = torch.randn(1, 32, 2)
    independent = torch.randn(1, 32, 2)
    mask = torch.zeros(1, 32, dtype=torch.bool); mask[:, 16:] = True
    mixed = coupled_noise(shared, independent, mask)
    stats = noise_correlations(shared, mixed, independent, mask)
    assert stats["preserve_shared_correlation"] > .999
    assert abs(stats["edit_independent_correlation"] - 1.) < 1e-6

def test_selective_noise_correlation_endpoints():
    torch.manual_seed(3); shared = torch.randn(1, 10000, 1); independent = torch.randn_like(shared); edit = torch.zeros_like(shared); edit[:, :5000] = 1
    assert torch.equal(coupled_noise(shared, independent, edit, 1.), shared)
    target = coupled_noise(shared, independent, edit, 0.); assert torch.equal(target[:, 5000:], shared[:, 5000:])
    corr = torch.corrcoef(torch.stack([shared[:, :5000, 0].flatten(), target[:, :5000, 0].flatten()]))[0, 1]; assert abs(float(corr)) < .05

def test_two_stage_greedy_starts_stage_two_at_winner():
    calls = []
    make = lambda state, step: [state + i for i in range(4)]
    score = lambda candidates, stage: [float(x.item()) for x in candidates]
    def rollout(state, post_step, stage): calls.append((stage, float(state.item()), post_step)); return state
    final, records = greedy_two_stage_branch(torch.tensor(0.), [1, 4], make, score, rollout)
    assert records[0].winner_index == 3 and records[1].branch_step_index == 4 and calls[4][1] == 3. and final.item() == 6.

def test_mask_alignment_and_lora_gates():
    mask = torch.zeros(4, 4); mask[:2, :2] = 1; assert align_image_token_mask(mask, (2, 2)).tolist() == [True, False, False, False]
    applied = masked_residual(torch.ones(1, 4, 2), torch.tensor([[True, False, True, False]]), .5); assert applied[0, :, 0].tolist() == [.5, 0., .5, 0.]
    assert timestep_gate(2, {2, 3}) == 1. and timestep_gate(1, {2, 3}) == 0.
