import numpy as np
import torch

from scripts.reward_gradient_visualization import (
    clean_prediction_from_flow,
    compute_image_gradient,
    global_percentile_normalize,
    make_overlay,
    rgb_gradient_magnitude,
    select_visualization_steps,
)


def test_clean_prediction_matches_flow_interpolation():
    sample = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    velocity = torch.tensor([[[0.5, -1.0], [2.0, -3.0]]])
    assert torch.equal(clean_prediction_from_flow(sample, velocity, 0.25), sample - 0.25 * velocity)


def test_image_gradient_and_rgb_l2_are_image_space():
    image = torch.zeros(1, 3, 4, 5)
    reward, gradient = compute_image_gradient(image, lambda value: value.square().sum())
    assert not reward.requires_grad
    assert gradient.shape == image.shape
    assert torch.isfinite(gradient).all()
    assert rgb_gradient_magnitude(gradient).shape == (1, 4, 5)


def test_global_percentile_uses_one_range_for_all_steps():
    values = {1: np.array([[0.0, 1.0]], dtype=np.float32), 2: np.array([[2.0, 3.0]], dtype=np.float32)}
    normalized, low, high = global_percentile_normalize(values, 0, 100)
    assert low == 0.0 and high == 3.0
    assert np.isclose(normalized[1][0, 1], 1 / 3)
    assert np.isclose(normalized[2][0, 1], 1.0)


def test_overlay_and_step_selection():
    assert select_visualization_steps(15) == [1, 2, 3, 4, 8, 12, 15]
    image = np.zeros((4, 5, 3), dtype=np.float32)
    overlay = make_overlay(image, np.ones((4, 5), dtype=np.float32))
    assert overlay.shape == image.shape and np.isfinite(overlay).all()
    zero_overlay = make_overlay(image, np.zeros((4, 5), dtype=np.float32))
    assert np.allclose(zero_overlay, image)
