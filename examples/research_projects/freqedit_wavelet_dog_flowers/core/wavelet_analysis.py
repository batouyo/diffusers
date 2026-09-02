"""Wavelet diagnostics for FLUX velocity trajectories."""

from __future__ import annotations

from typing import Dict, Mapping

import torch


def unpack_flux_velocity(velocity: torch.Tensor, pipeline, height: int, width: int) -> torch.Tensor:
    """Unpack a packed FLUX sequence velocity into ``[B, C, H, W]``."""
    if velocity.ndim != 3:
        raise ValueError(f"Expected packed velocity [B, tokens, channels], got {tuple(velocity.shape)}")
    unpacked = pipeline._unpack_latents(velocity, height, width, pipeline.vae_scale_factor)
    if unpacked.ndim != 4:
        raise ValueError(f"Kontext _unpack_latents must return NCHW, got {tuple(unpacked.shape)}")
    return unpacked


def _wavelet_modules(levels: int, wavelet: str, device: torch.device):
    try:
        from pytorch_wavelets import DWTForward
    except ImportError as exc:
        raise ImportError("Install pytorch_wavelets==1.3.0 for wavelet analysis") from exc
    return DWTForward(J=levels, wave=wavelet, mode="symmetric").to(device)


def decompose_velocity(velocity_nchw: torch.Tensor, *, levels: int = 2, wavelet: str = "db4") -> Dict[str, torch.Tensor]:
    """Return LL2 and grouped detail coefficients for a spatial velocity."""
    if velocity_nchw.ndim != 4:
        raise ValueError(f"Expected NCHW velocity, got {tuple(velocity_nchw.shape)}")
    if levels != 2:
        raise ValueError("This experiment is defined for exactly two DWT levels")
    ll2, details = _wavelet_modules(levels, wavelet, velocity_nchw.device)(velocity_nchw.float())
    return {"LL2": ll2, "D1": details[0], "D2": details[1]}


def _rms(value: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(value.float().square())).item())


def _energy(value: torch.Tensor) -> float:
    return float(value.float().square().sum().item())


def band_statistics(coefficients: Mapping[str, torch.Tensor]) -> Dict[str, float]:
    """Compute RMS, energy, fractions and orientation RMS for LL2/D2/D1."""
    bands = {name: coefficients[name] for name in ("LL2", "D2", "D1")}
    energies = {name: _energy(value) for name, value in bands.items()}
    total_energy = sum(energies.values())
    result: Dict[str, float] = {}
    for name, value in bands.items():
        result[f"{name}_rms"] = _rms(value)
        result[f"{name}_energy"] = energies[name]
        result[f"{name}_energy_fraction"] = energies[name] / max(total_energy, 1e-12)
    for level in (1, 2):
        details = coefficients[f"D{level}"]
        for index, orientation in enumerate(("LH", "HL", "HH")):
            result[f"{orientation}{level}_rms"] = _rms(details[:, :, index])
    result["total_energy"] = total_energy
    result["total_rms"] = _rms(torch.cat([value.flatten() for value in bands.values()]))
    return result


def analyze_velocity(velocity_nchw: torch.Tensor, *, levels: int = 2, wavelet: str = "db4"):
    coefficients = decompose_velocity(velocity_nchw, levels=levels, wavelet=wavelet)
    if not all(torch.isfinite(value).all().item() for value in coefficients.values()):
        raise ValueError("Non-finite wavelet coefficient encountered")
    return coefficients, band_statistics(coefficients)
