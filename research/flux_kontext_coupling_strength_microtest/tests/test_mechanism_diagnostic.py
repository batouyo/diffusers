"""CPU unit tests for the Coupled-SDE mechanism diagnostic helpers."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_mechanism_diagnostic import EPS, explicit_noise, regional_difference, rf_coefficient, rf_step, seed_for  # noqa: E402


def test_rf_first_step_is_exact_deterministic_euler() -> None:
    x = torch.randn(1, 4, 6)
    velocity = torch.randn_like(x)
    result, coefficient = rf_step(x, velocity, torch.tensor(1.0), torch.tensor(0.9), torch.randn_like(x), True)
    assert coefficient == 0.0
    assert torch.equal(result, (x.float() + (torch.tensor(0.9) - torch.tensor(1.0)) * velocity).to(x.dtype))


def test_rf_nonfirst_step_matches_registered_formula() -> None:
    x = torch.tensor([[[.2, -.3]]])
    v = torch.tensor([[[.4, .1]]])
    noise = torch.tensor([[[.5, -.7]]])
    result, coefficient = rf_step(x, v, torch.tensor(.8), torch.tensor(.7), noise, False)
    sigma, sigma_next = float(torch.tensor(.8)), float(torch.tensor(.7))
    expected_coefficient = math.sqrt(2 * sigma / (1 - sigma) * (sigma - sigma_next))
    expected = x + (sigma_next - sigma) * (2 * v + x / (1 - sigma)) + expected_coefficient * noise
    assert abs(coefficient - expected_coefficient) < 1e-7
    assert torch.allclose(result, expected, atol=1e-6)


def test_noise_seeds_are_repeatable_and_stream_distinct() -> None:
    a = explicit_noise((1, 64, 8), torch.device("cpu"), 0, 1, 0)
    b = explicit_noise((1, 64, 8), torch.device("cpu"), 0, 1, 0)
    c = explicit_noise((1, 64, 8), torch.device("cpu"), 0, 1, 1)
    assert seed_for(0, 1, 0) == seed_for(0, 1, 0)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_regional_difference_reports_nan_for_empty_preserve() -> None:
    a, b = torch.zeros(1, 4, 2), torch.ones(1, 4, 2)
    all_edit = torch.ones_like(a)
    value = regional_difference(a, b, all_edit)
    assert math.isfinite(value["edit_latent_l2"])
    assert math.isnan(value["preserve_latent_l2"])


def test_rf_coefficient_rejects_nonfirst_sigma_one() -> None:
    try:
        rf_coefficient(1.0, .9, False)
    except ValueError:
        pass
    else:
        raise AssertionError("sigma=1 is valid only for the deterministic first step")

