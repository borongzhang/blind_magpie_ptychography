from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PhaseRampAlignment:
    """Affine phase ramp and scalar gauge used to align a blind solution."""

    slope_y_rad_per_px: float
    slope_x_rad_per_px: float
    object_scale: complex


def _as_complex_2d(array: np.ndarray, name: str) -> np.ndarray:
    array = np.squeeze(np.asarray(array, dtype=np.complex64))
    if array.ndim != 2:
        raise ValueError(f"Expected {name} to squeeze to 2D, got {array.shape}.")
    return array


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cutoff = 0.5 * float(np.sum(weights))
    return float(values[np.searchsorted(np.cumsum(weights), cutoff, side="left")])


def _robust_circular_location(
    angles: np.ndarray,
    weights: np.ndarray,
    *,
    iterations: int = 6,
) -> float:
    valid = np.isfinite(angles) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        raise ValueError("Phase-ramp estimation has no valid neighboring pixels.")

    angles = angles[valid].astype(np.float32, copy=False)
    weights = weights[valid].astype(np.float32, copy=False)
    vector_sum = np.sum(weights * np.exp(1j * angles))
    center = float(np.angle(vector_sum)) if vector_sum != 0 else 0.0

    # Iterated circular medians reject object-detail differences while retaining
    # a spatially constant ramp increment.
    for _ in range(iterations):
        residual = np.angle(np.exp(1j * (angles - center)))
        shift = _weighted_median(residual, weights)
        center = float(np.angle(np.exp(1j * (center + shift))))
        if abs(shift) < 1e-10:
            break
    return center


def _phase_ramp(
    shape: tuple[int, int],
    slope_y_rad_per_px: float,
    slope_x_rad_per_px: float,
) -> np.ndarray:
    height, width = shape
    y = np.arange(height, dtype=np.float32) - 0.5 * (height - 1)
    x = np.arange(width, dtype=np.float32) - 0.5 * (width - 1)
    return slope_y_rad_per_px * y[:, None] + slope_x_rad_per_px * x[None, :]


def estimate_phase_ramp(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> tuple[float, float]:
    """Estimate the affine phase in ``candidate / reference``.

    The two slopes are robust circular medians of neighboring phase
    increments. Amplitude weighting suppresses unreliable low-signal pixels,
    while ``mask`` can restrict estimation to the reconstructed specimen.
    """
    reference = _as_complex_2d(reference, "reference")
    candidate = _as_complex_2d(candidate, "candidate")
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate must have the same shape.")

    valid = np.isfinite(reference) & np.isfinite(candidate)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != reference.shape:
            raise ValueError("mask must have the same shape as reference.")
        valid &= mask

    amplitude_weight = np.abs(reference) * np.abs(candidate)
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float32)
        if weights.shape != reference.shape:
            raise ValueError("weights must have the same shape as reference.")
        amplitude_weight *= np.clip(weights, 0.0, None)
    amplitude_weight = np.where(valid, amplitude_weight, 0.0)

    cross = candidate * np.conj(reference)
    dx = np.angle(cross[:, 1:] * np.conj(cross[:, :-1]))
    dy = np.angle(cross[1:, :] * np.conj(cross[:-1, :]))
    weight_x = np.sqrt(amplitude_weight[:, 1:] * amplitude_weight[:, :-1])
    weight_y = np.sqrt(amplitude_weight[1:, :] * amplitude_weight[:-1, :])

    slope_x = _robust_circular_location(dx, weight_x)
    slope_y = _robust_circular_location(dy, weight_y)
    return slope_y, slope_x


def align_complex_scalar(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, complex]:
    """Least-squares complex-scalar alignment, optionally on a weighted mask."""
    reference = np.asarray(reference, dtype=np.complex64)
    candidate = np.asarray(candidate, dtype=np.complex64)
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate must have the same shape.")

    fit_weight = np.ones(reference.shape, dtype=np.float32)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != reference.shape:
            raise ValueError("mask must have the same shape as reference.")
        fit_weight *= mask
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float32)
        if weights.shape != reference.shape:
            raise ValueError("weights must have the same shape as reference.")
        fit_weight *= np.clip(weights, 0.0, None)

    valid = np.isfinite(reference) & np.isfinite(candidate) & (fit_weight > 0)
    if not np.any(valid):
        raise ValueError("Complex-scalar alignment has no valid pixels.")
    numerator = np.sum(fit_weight[valid] * np.conj(candidate[valid]) * reference[valid])
    denominator = float(np.sum(fit_weight[valid] * np.abs(candidate[valid]) ** 2))
    scale = complex(numerator / denominator) if denominator > 0 else 0.0 + 0.0j
    return scale * candidate, scale


def align_object_phase_ramp(
    reference_object: np.ndarray,
    candidate_object: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    scale_amplitude: bool = True,
) -> tuple[np.ndarray, PhaseRampAlignment]:
    """Remove an affine phase ramp and align the remaining global phase.

    Set ``scale_amplitude=False`` when an algorithm fixes the blind scale
    gauge and the native object/probe amplitudes should remain observable.
    """
    reference_object = _as_complex_2d(reference_object, "reference_object")
    candidate_object = _as_complex_2d(candidate_object, "candidate_object")
    slope_y, slope_x = estimate_phase_ramp(
        reference_object,
        candidate_object,
        mask=mask,
        weights=weights,
    )
    ramp = _phase_ramp(candidate_object.shape, slope_y, slope_x)
    object_deramped = candidate_object * np.exp(-1j * ramp)
    object_aligned, scale = align_complex_scalar(
        reference_object,
        object_deramped,
        mask=mask,
        weights=weights,
    )
    if not scale_amplitude:
        scale = scale / abs(scale) if scale != 0 else 1.0 + 0.0j
        object_aligned = scale * object_deramped
    alignment = PhaseRampAlignment(slope_y, slope_x, scale)
    return object_aligned, alignment


def align_blind_object_probe(
    reference_object: np.ndarray,
    candidate_object: np.ndarray,
    candidate_probe: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    scale_amplitude: bool = True,
) -> tuple[np.ndarray, np.ndarray, PhaseRampAlignment]:
    """Align a blind object/probe pair without changing their product gauge.

    The object ramp is removed and the opposite ramp is applied to the probe.
    The remaining object scalar is applied inversely to the probe.
    """
    candidate_probe = _as_complex_2d(candidate_probe, "candidate_probe")
    object_aligned, alignment = align_object_phase_ramp(
        reference_object,
        candidate_object,
        mask=mask,
        weights=weights,
        scale_amplitude=scale_amplitude,
    )
    probe_ramp = _phase_ramp(
        candidate_probe.shape,
        alignment.slope_y_rad_per_px,
        alignment.slope_x_rad_per_px,
    )
    probe_deramped = candidate_probe * np.exp(1j * probe_ramp)
    if alignment.object_scale == 0:
        probe_aligned = probe_deramped
    else:
        probe_aligned = probe_deramped / alignment.object_scale
    return object_aligned, probe_aligned, alignment
