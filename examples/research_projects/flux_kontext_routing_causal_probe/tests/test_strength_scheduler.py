import torch

from strength_overfit_training import euler_step
from vkeep_control import euler_update


def test_euler_step_matches_verified_update():
    sample = torch.randn(1, 5, 4, dtype=torch.bfloat16)
    velocity = torch.randn_like(sample)
    expected = euler_update(sample, velocity, 0.8, 0.7).to(sample.dtype)
    actual = euler_step(sample, velocity, 0.8, 0.7)
    assert torch.equal(actual, expected)

