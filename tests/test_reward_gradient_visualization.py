import numpy as np
import torch

from scripts.reward_gradient_visualization import (
    clean_prediction_from_flow,
    compute_image_gradient,
    directional_reward,
    global_percentile_normalize,
    make_overlay,
    per_step_percentile_normalize,
    rgb_gradient_magnitude,
    select_visualization_steps,
)


def test_clean_prediction_matches_flow_interpolation():
    sample = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    velocity = torch.tensor([[[0.5, -1.0], [2.0, -3.0]]])
    assert torch.equal(clean_prediction_from_flow(sample, velocity, 0.25), sample - 0.25 * velocity)


def test_image_gradient_and_rgb_l2_are_image_space():
    image = torch.zeros(1, 3, 4, 5)
    reward, gradient = compute_image_gradient(image, lambda value: value.sum())
    assert not reward.requires_grad
    assert gradient.shape == image.shape
    assert torch.equal(gradient, torch.ones_like(image))
    assert rgb_gradient_magnitude(gradient).shape == (1, 4, 5)


def test_directional_reward_gradient_is_difference():
    image = torch.zeros(1, 3, 2, 2, requires_grad=True)
    target = (image * 3.0).sum()
    source = (image * 0.5).sum()
    reward = directional_reward(target, source)
    assert torch.equal(torch.autograd.grad(reward, image)[0], torch.full_like(image, 2.5))


def test_per_step_and_shared_normalization():
    values = {1: np.array([[0.0, 1.0]], dtype=np.float32), 2: np.array([[2.0, 3.0]], dtype=np.float32)}
    normalized, bounds = per_step_percentile_normalize(values, 0, 100)
    assert bounds[1] == (0.0, 1.0) and bounds[2] == (2.0, 3.0)
    assert np.isclose(normalized[1][0, 1], 1.0)
    shared, low, high = global_percentile_normalize(values, 0, 100)
    assert low == 0.0 and high == 3.0
    assert np.isclose(shared[1][0, 1], 1 / 3)


def test_overlay_and_step_selection():
    assert select_visualization_steps(15) == [1, 2, 3, 4, 8, 12, 15]
    image = np.zeros((4, 5, 3), dtype=np.float32)
    overlay = make_overlay(image, np.ones((4, 5), dtype=np.float32))
    assert overlay.shape == image.shape and np.isfinite(overlay).all()
