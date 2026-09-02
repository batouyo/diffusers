import pytest
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.wavelet_analysis import analyze_velocity, band_statistics, decompose_velocity


def test_two_level_db4_statistics_are_finite():
    pytest.importorskip("pytorch_wavelets")
    velocity = torch.randn(1, 8, 12, 16)
    coeffs, stats = analyze_velocity(velocity)
    assert set(coeffs) == {"LL2", "D1", "D2"}
    assert all(torch.isfinite(value).all() for value in coeffs.values())
    assert all(torch.isfinite(torch.tensor(value)) for value in stats.values())
    assert abs(sum(stats[f"{band}_energy_fraction"] for band in ("LL2", "D2", "D1")) - 1.0) < 1e-6


def test_band_statistics_support_zero_velocity():
    pytest.importorskip("pytorch_wavelets")
    coeffs = {"LL2": torch.zeros(1, 1, 2, 2), "D1": torch.zeros(1, 1, 3, 2, 2), "D2": torch.zeros(1, 1, 3, 1, 1)}
    stats = band_statistics(coeffs)
    assert stats["LL2_rms"] == 0.0
    assert stats["D1_energy_fraction"] == 0.0


def test_invalid_layout_and_levels():
    pytest.importorskip("pytorch_wavelets")
    with pytest.raises(ValueError):
        decompose_velocity(torch.zeros(1, 10, 4), levels=2)
    with pytest.raises(ValueError):
        decompose_velocity(torch.zeros(1, 4, 8, 8), levels=1)
