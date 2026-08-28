from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_coupling_strength_microtest import (  # noqa: E402
    RHOS,
    coupled_noise,
    derived_torch_seed,
    directional_progress,
    expanded_roi,
    explicit_noise,
    l1_regions,
    orientation,
    rf_diffusion_coefficient,
    rf_sde_step,
    safe_spearman,
)
from PIL import Image  # noqa: E402


def test_first_step_matches_deterministic_euler_exactly() -> None:
    sample = torch.randn(2, 5, 7, dtype=torch.float32)
    model_output = torch.randn_like(sample)
    noise = torch.randn_like(sample) * 1000
    result, diagnostics = rf_sde_step(sample, model_output, 1.0, 0.98, noise, True)
    expected = sample + (0.98 - 1.0) * model_output
    assert torch.equal(result, expected)
    assert diagnostics["diffusion_coeff"] == 0.0
    result_other_noise, _ = rf_sde_step(sample, model_output, 1.0, 0.98, -noise, True)
    assert torch.equal(result, result_other_noise)


def test_nonfirst_step_matches_official_formula() -> None:
    sample = torch.tensor([[[0.5, -0.2]]], dtype=torch.float64)
    prediction = torch.tensor([[[0.1, 0.3]]], dtype=torch.float64)
    noise = torch.tensor([[[0.7, -1.1]]], dtype=torch.float64)
    sigma, sigma_next = 0.8, 0.7
    result, diagnostics = rf_sde_step(sample, prediction, sigma, sigma_next, noise, False)
    drift = 2 * prediction.float() + sample.float() / (1 - sigma)
    coefficient = math.sqrt(2 * sigma / (1 - sigma) * (sigma - sigma_next))
    expected = sample.float() + (sigma_next - sigma) * drift + coefficient * noise.float()
    assert torch.equal(result, expected.to(torch.float64))
    assert diagnostics["diffusion_coeff"] == coefficient


def test_seedsequence_noise_is_reproducible_and_stream_specific() -> None:
    shape = (1, 64, 16)
    first = explicit_noise(shape, torch.device("cpu"), 2, 1, 3, 0)
    repeat = explicit_noise(shape, torch.device("cpu"), 2, 1, 3, 0)
    other_stream = explicit_noise(shape, torch.device("cpu"), 2, 1, 3, 1)
    assert torch.equal(first, repeat)
    assert not torch.equal(first, other_stream)
    assert derived_torch_seed(2, 1, 3, 0) == derived_torch_seed(2, 1, 3, 0)


def test_coupling_preserves_shared_noise_outside_edit_mask() -> None:
    shape = (1, 200_000, 1)
    shared = explicit_noise(shape, torch.device("cpu"), 0, 0, 1, 0)
    independent = explicit_noise(shape, torch.device("cpu"), 0, 0, 1, 1)
    mask = torch.zeros(shape)
    mask[:, :100_000] = 1
    edit = mask.bool().reshape(-1).numpy()
    for rho in RHOS:
        target = coupled_noise(shared, independent, mask, rho)
        assert torch.equal(target[mask == 0], shared[mask == 0])
        empirical = np.corrcoef(shared.reshape(-1).numpy()[edit], target.reshape(-1).numpy()[edit])[0, 1]
        assert abs(empirical - rho) < 0.01
        assert abs(float(target.mean())) < 0.01
        assert abs(float(target.std()) - 1) < 0.01
        if rho == 1:
            assert torch.equal(target, shared)


def test_roi_expansion_and_full_image_fallback() -> None:
    mask = np.zeros((100, 200), dtype=bool)
    mask[20:60, 50:150] = True
    original, roi, full = expanded_roi(mask)
    assert original == (50, 20, 150, 60)
    assert roi == (43, 17, 157, 63)
    assert not full
    large = np.ones((100, 200), dtype=bool)
    original, roi, full = expanded_roi(large)
    assert roi == (0, 0, 200, 100)
    assert full


def test_directional_progress_endpoints_and_reverse_direction() -> None:
    source = np.array([1.0, 0.0, 0.0])
    target = np.array([0.0, 1.0, 0.0])
    values = np.stack([source, target, 2 * source - target])
    progress, denominator, degenerate = directional_progress(source, target, values)
    assert denominator > 1e-8
    assert not degenerate
    assert abs(progress[0]) < 1e-7
    assert abs(progress[1] - 1) < 1e-7
    assert progress[2] < 0


def test_statistics_ties_and_orientation() -> None:
    assert abs(safe_spearman([1, 0.75, 0.5, 0.25, 0], [5, 4, 3, 2, 1]) - 1.0) < 1e-12
    assert abs(safe_spearman([1, 0.75, 0.5, 0.25, 0], [1, 2, 3, 4, 5]) + 1.0) < 1e-12
    assert math.isnan(safe_spearman([1, 0.5, 0], [2, 2, 2]))
    assert orientation(0.7) == "positive"
    assert orientation(-0.7) == "negative"
    assert orientation(float("nan")) == "none"


def test_region_l1_is_area_normalized() -> None:
    first = np.zeros((4, 4, 3), dtype=np.uint8)
    second = first.copy()
    second[:2, :, :] = 255
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2, :] = True
    edit, preserve = l1_regions(Image.fromarray(first), Image.fromarray(second), mask)
    assert edit == 1.0
    assert preserve == 0.0


def test_rf_diffusion_coefficient_rejects_invalid_schedule() -> None:
    assert rf_diffusion_coefficient(1.0, 0.9, True) == 0.0
    assert rf_diffusion_coefficient(0.9, 0.8, False) > 0
    try:
        rf_diffusion_coefficient(1.0, 0.9, False)
    except ValueError:
        pass
    else:
        raise AssertionError("Non-first sigma=1 must be rejected")

