import torch

from early_edit_reward_distillation.trajectory import deterministic_rollout
from early_edit_reward_distillation.core import coupled_noise


def test_trajectory_module_imports_without_pipeline():
    assert callable(deterministic_rollout)

def test_regional_noise_mask_uses_batch_and_token_axes():
    shared = torch.zeros(4, 8, 2)
    independent = torch.ones(4, 8, 2)
    mask = torch.zeros(1, 8, dtype=torch.bool)
    mask[:, 3:] = True
    result = coupled_noise(shared, independent, mask)
    assert torch.equal(result[:, :3], torch.zeros(4, 3, 2))
    assert torch.equal(result[:, 3:], torch.ones(4, 5, 2))
