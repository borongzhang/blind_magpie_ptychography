from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from skimage.transform import resize

from algorithms.probe_shift import shift_images_with_replicate_padding
from utils.common import PROJECT_ROOT
from utils.optics import generate_probe
from utils.phase_alignment import align_blind_object_probe, align_complex_scalar

__all__ = [
    "ExperimentConfig",
    "SyntheticDataset",
    "build_synthetic_dataset",
    "registered_object_truth_and_mask",
    "score_blind_reconstruction",
]


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration shared by the synthetic rPIE, LSQML, and MAGPIE runs."""

    asset_dir: Path = PROJECT_ROOT / "assets"

    object_size: int = 1024
    probe_size: int = 128
    overlap_ratio: float = 0.5
    position_jitter_px: float = 0.49
    position_seed: int = 1
    poisson_eta: float = 0.05
    detector_centered_data: bool = True
    data_seed: int = 0

    probe_source_size: int = 512
    wavelength_m: float = 1e-10
    sample_pixel_size_m: float = 1e-8
    total_probe_power: float = 300.0
    truth_probe_defocus_m: float = -7e-4
    initial_probe_defocus_m: float = -6e-4

    reconstruction_seed: int = 11
    num_epochs: int = 300
    batch_size: int = 25
    object_extra_pixels: int = 10

    object_alpha: float = 0.01
    object_step_size: float = 1.0
    probe_alpha: float = 0.3
    probe_step_size: float = 1.0
    remove_object_probe_ambiguity: bool = False

    def __post_init__(self) -> None:
        if self.object_size < self.probe_size:
            raise ValueError("object_size must be at least probe_size.")
        if self.probe_size < 2:
            raise ValueError("probe_size must be at least two.")
        if self.object_size % 2 or self.probe_size % 2:
            raise ValueError(
                "object_size and probe_size must both be even so synthetic "
                "generation and reconstruction use the same pixel origin."
            )
        if self.probe_source_size % self.probe_size != 0:
            raise ValueError("probe_source_size must be divisible by probe_size.")
        if not 0 <= self.overlap_ratio < 1:
            raise ValueError("overlap_ratio must lie in [0, 1).")
        if not np.isfinite(self.position_jitter_px) or self.position_jitter_px < 0:
            raise ValueError("position_jitter_px must be finite and nonnegative.")
        available_scan_span = self.object_size - self.probe_size
        if self.position_jitter_px > 0 and (
            available_scan_span <= 0
            or 2 * self.position_jitter_px >= available_scan_span
        ):
            raise ValueError(
                "position_jitter_px must be less than half the available "
                "object-minus-probe scan span."
            )
        if not np.isfinite(self.poisson_eta) or self.poisson_eta < 0:
            raise ValueError("poisson_eta must be finite and nonnegative.")
        if not np.isfinite(self.total_probe_power) or self.total_probe_power <= 0:
            raise ValueError("total_probe_power must be finite and positive.")
        if self.num_epochs < 1 or self.batch_size < 1:
            raise ValueError("num_epochs and batch_size must be positive.")
        if self.object_extra_pixels < 0:
            raise ValueError("object_extra_pixels must be nonnegative.")
        for name, value in (
            ("object_alpha", self.object_alpha),
            ("probe_alpha", self.probe_alpha),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name, value in (
            ("object_step_size", self.object_step_size),
            ("probe_step_size", self.probe_step_size),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")


@dataclass(frozen=True)
class SyntheticDataset:
    truth: np.ndarray
    probe_truth: np.ndarray
    probe_init: np.ndarray
    data: np.ndarray
    positions_px: np.ndarray
    position_shifts_px: np.ndarray


def _read_gray(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing image file: {path}")
    image = imageio.imread(path).astype(np.float32)
    if image.ndim == 3:
        image = image[..., :3].mean(axis=2)
    return image


def _resize(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if image.shape == shape:
        return image.astype(np.float32, copy=False)
    return resize(
        image,
        shape,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)


def _load_object(cfg: ExperimentConfig) -> np.ndarray:
    shape = (cfg.object_size, cfg.object_size)
    magnitude_image = _resize(_read_gray(cfg.asset_dir / "baboon.tiff"), shape)
    phase_image = _resize(_read_gray(cfg.asset_dir / "cameraman.tif"), shape)
    magnitude = magnitude_image / np.max(magnitude_image)
    phase = phase_image / np.max(phase_image) * (np.pi / 2)
    return np.asarray(magnitude * np.exp(1j * phase), dtype=np.complex64)


def _make_probe(
    cfg: ExperimentConfig,
    defocus_m: float,
) -> np.ndarray:
    source = generate_probe(
        size=cfg.probe_source_size,
        wavelength_m=cfg.wavelength_m,
        sample_pixel_size_m=cfg.sample_pixel_size_m,
        defocus_m=defocus_m,
    )
    stride = cfg.probe_source_size // cfg.probe_size
    return np.asarray(source[::stride, ::stride], dtype=np.complex64)


def _normalize_probe_power(probe: np.ndarray, target_power: float) -> np.ndarray:
    current_power = np.sum(np.abs(probe) ** 2, dtype=np.float32)
    if current_power <= 0:
        raise ValueError("Cannot normalize a zero-power probe.")
    scale = np.sqrt(np.float32(target_power) / current_power)
    return np.asarray(probe * scale, dtype=np.complex64)


def _make_raster_scan(
    object_shape: tuple[int, int],
    probe_shape: tuple[int, int],
    overlap_ratio: float,
    position_jitter_px: float,
    position_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return center-relative scan positions and their uniform random shifts.

    Each retained nominal raster coordinate receives an independent
    perturbation in ``[-position_jitter_px, position_jitter_px]`` along both
    axes. Nominal boundary rows and columns that cannot support the full shift
    interval are omitted, so the raster spacing itself is unchanged and no
    shifted probe leaves the object.
    """
    object_height, object_width = object_shape
    probe_height, probe_width = probe_shape
    step_y = int(round(probe_height * (1 - overlap_ratio)))
    step_x = int(round(probe_width * (1 - overlap_ratio)))
    if step_y < 1 or step_x < 1:
        raise ValueError("The scan step is zero; decrease overlap_ratio.")

    maximum_top_left = np.asarray(
        [object_height - probe_height, object_width - probe_width],
        dtype=np.float32,
    )
    if position_jitter_px > 0 and np.any(2 * position_jitter_px >= maximum_top_left):
        raise ValueError(
            "position_jitter_px must be less than half the available scan "
            "span along each axis."
        )

    nominal_y = np.arange(0, maximum_top_left[0] + 1, step_y, dtype=np.float32)
    nominal_x = np.arange(0, maximum_top_left[1] + 1, step_x, dtype=np.float32)
    if position_jitter_px > 0:
        nominal_y = nominal_y[
            (nominal_y >= position_jitter_px)
            & (nominal_y <= maximum_top_left[0] - position_jitter_px)
        ]
        nominal_x = nominal_x[
            (nominal_x >= position_jitter_px)
            & (nominal_x <= maximum_top_left[1] - position_jitter_px)
        ]

    nominal_top_left = np.asarray(
        [(y, x) for y in nominal_y for x in nominal_x],
        dtype=np.float32,
    )
    if len(nominal_top_left) == 0:
        raise ValueError("The object and probe shapes produce no scan positions.")

    position_shifts_px = np.zeros_like(nominal_top_left)
    if position_jitter_px > 0:
        rng = np.random.default_rng(position_seed)
        position_shifts_px = rng.uniform(
            -position_jitter_px,
            position_jitter_px,
            size=nominal_top_left.shape,
        ).astype(np.float32)
    top_left = nominal_top_left + position_shifts_px

    centers_y = top_left[:, 0] + probe_height / 2
    centers_x = top_left[:, 1] + probe_width / 2
    positions_px = np.stack(
        [
            centers_y - object_height / 2,
            centers_x - object_width / 2,
        ],
        axis=1,
    )
    return positions_px.astype(np.float32), position_shifts_px


def _generate_measurements(
    obj: np.ndarray,
    probe: np.ndarray,
    positions_px: np.ndarray,
    *,
    poisson_eta: float,
    detector_centered: bool,
    seed: int,
) -> np.ndarray:
    """Generate data using the fixed one-pixel replicate-padded Fourier shift."""
    rng = np.random.default_rng(seed)
    top_left_float = positions_px.copy()
    top_left_float[:, 0] += obj.shape[0] / 2 - probe.shape[0] / 2
    top_left_float[:, 1] += obj.shape[1] / 2 - probe.shape[1] / 2
    top_left = np.rint(top_left_float).astype(np.int64)
    fractional_shifts = (top_left_float - top_left).astype(np.float32)
    maximum_top_left = np.asarray(obj.shape) - np.asarray(probe.shape)
    if np.any(top_left < 0) or np.any(top_left > maximum_top_left):
        raise ValueError("Scan positions place at least one probe outside the object.")

    probe_batch = torch.from_numpy(probe)[None].expand(len(positions_px), -1, -1)
    shifted_probes = shift_images_with_replicate_padding(
        probe_batch,
        torch.from_numpy(fractional_shifts),
        method="fourier",
    ).numpy(force=True)

    patterns = []
    for index, (y0, x0) in enumerate(top_left):
        local_probe = shifted_probes[index]
        object_patch = obj[
            y0 : y0 + probe.shape[0],
            x0 : x0 + probe.shape[1],
        ]
        exit_wave = local_probe * object_patch
        intensity = np.abs(np.fft.fft2(exit_wave)) ** 2
        if detector_centered:
            intensity = np.fft.fftshift(intensity)
        if poisson_eta > 0:
            intensity = poisson_eta * rng.poisson(
                np.maximum(intensity / poisson_eta, 0)
            )
        patterns.append(intensity.astype(np.float32))
    return np.asarray(patterns, dtype=np.float32)


def build_synthetic_dataset(cfg: ExperimentConfig) -> SyntheticDataset:
    """Build the fixed FZP experiment shared by all comparison algorithms."""
    truth = _load_object(cfg)
    probe_truth = _normalize_probe_power(
        _make_probe(
            cfg,
            cfg.truth_probe_defocus_m,
        ),
        cfg.total_probe_power,
    )
    positions_px, position_shifts_px = _make_raster_scan(
        truth.shape,
        probe_truth.shape,
        cfg.overlap_ratio,
        cfg.position_jitter_px,
        cfg.position_seed,
    )
    data = _generate_measurements(
        truth,
        probe_truth,
        positions_px,
        poisson_eta=cfg.poisson_eta,
        detector_centered=cfg.detector_centered_data,
        seed=cfg.data_seed,
    )

    initial_probe_power = float(
        np.mean(data.sum(axis=(-2, -1))) / np.prod(data.shape[-2:])
    )
    probe_init = _normalize_probe_power(
        _make_probe(
            cfg,
            cfg.initial_probe_defocus_m,
        ),
        initial_probe_power,
    )
    return SyntheticDataset(
        truth=truth,
        probe_truth=probe_truth,
        probe_init=probe_init,
        data=data,
        positions_px=positions_px,
        position_shifts_px=position_shifts_px,
    )


def _as_complex_2d(array: np.ndarray, label: str) -> np.ndarray:
    array = np.squeeze(np.asarray(array, dtype=np.complex64))
    if array.ndim != 2:
        raise ValueError(f"Expected {label} to squeeze to 2D, got {array.shape}.")
    return array


def _overlapping_object_views(
    reference: np.ndarray,
    estimate: np.ndarray,
    positions_px: np.ndarray,
    probe_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return registered truth/reconstruction views and their measured mask.

    Synthetic measurements anchor scan positions to the truth-array center,
    whereas Pty-Chi anchors them to the midpoint of their extrema in its
    reconstruction buffer.  Those origins can differ by a pixel even when a
    centered crop has the right shape.  Recover the integer offset by matching
    the truth and reconstruction top-left patch coordinates used by the two
    forward models.
    """
    positions_px = np.asarray(positions_px, dtype=np.float32)
    if positions_px.ndim != 2 or positions_px.shape[1] != 2 or len(positions_px) == 0:
        raise ValueError("positions_px must be a nonempty array of shape (N, 2).")
    if not np.isfinite(positions_px).all():
        raise ValueError("positions_px must be finite.")

    reference_shape = np.asarray(reference.shape, dtype=np.int64)
    estimate_shape = np.asarray(estimate.shape, dtype=np.int64)
    probe_shape_array = np.asarray(probe_shape, dtype=np.int64)
    if probe_shape_array.shape != (2,) or np.any(probe_shape_array < 1):
        raise ValueError("probe_shape must contain two positive dimensions.")

    # Match SyntheticDataset generation, which rounds the top-left coordinate
    # after adding the truth-centered position.
    truth_top_left_float = positions_px.copy()
    truth_top_left_float += (
        reference_shape.astype(np.float32) / 2
        - probe_shape_array.astype(np.float32) / 2
    )
    truth_top_left = np.rint(truth_top_left_float).astype(np.int64)

    # Match PlanarObject.update_pos_origin_coordinates(POSITIONS), followed by
    # integer object-patch extraction when fractional shifts act on the probe.
    position_center = (positions_px.max(axis=0) + positions_px.min(axis=0)) / 2
    buffer_center = estimate_shape.astype(np.float32) / 2
    reconstruction_origin = np.rint(buffer_center - position_center) + np.float32(0.5)
    reconstruction_centers = np.rint(positions_px) + reconstruction_origin
    reconstruction_top_left = np.rint(
        reconstruction_centers - (probe_shape_array.astype(np.float32) - 1) / 2
    ).astype(np.int64)

    illuminated = np.zeros(tuple(estimate_shape), dtype=bool)
    for top_left in reconstruction_top_left:
        start = np.maximum(top_left, 0)
        stop = np.minimum(top_left + probe_shape_array, estimate_shape)
        if np.all(stop > start):
            illuminated[
                int(start[0]) : int(stop[0]),
                int(start[1]) : int(stop[1]),
            ] = True

    offsets = truth_top_left - reconstruction_top_left
    if not np.all(offsets == offsets[0]):
        raise ValueError(
            "Synthetic truth and reconstruction coordinates do not have a "
            "single integer offset."
        )
    offset = offsets[0]

    estimate_start = np.maximum(-offset, 0)
    reference_start = np.maximum(offset, 0)
    overlap_shape = np.minimum(
        estimate_shape - estimate_start,
        reference_shape - reference_start,
    )
    if np.any(overlap_shape <= 0):
        raise ValueError("Synthetic truth and reconstruction supports do not overlap.")

    reference_slices = tuple(
        slice(int(start), int(start + length))
        for start, length in zip(reference_start, overlap_shape)
    )
    estimate_slices = tuple(
        slice(int(start), int(start + length))
        for start, length in zip(estimate_start, overlap_shape)
    )
    illuminated = illuminated[estimate_slices]
    if not np.any(illuminated):
        raise ValueError("Synthetic scan positions do not illuminate the overlap.")
    return reference[reference_slices], estimate[estimate_slices], illuminated


def _normalized_squared_error(
    reference: np.ndarray,
    estimate: np.ndarray,
) -> float:
    denominator = float(np.sum(np.abs(reference) ** 2, dtype=np.float32))
    if denominator <= 0:
        raise ValueError("Cannot compute NMSE against a zero reference.")
    numerator = float(np.sum(np.abs(reference - estimate) ** 2, dtype=np.float32))
    return numerator / denominator


def registered_object_truth_and_mask(
    recon_object: np.ndarray,
    dataset: SyntheticDataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Return truth registered to a reconstruction and its illuminated mask.

    The returned truth view has the same overlap geometry used by
    :func:`score_blind_reconstruction`. The boolean mask is the union of object
    pixels covered by at least one measured probe footprint.
    """
    reference_object, _, illuminated = _overlapping_object_views(
        _as_complex_2d(dataset.truth, "truth"),
        _as_complex_2d(recon_object, "object"),
        dataset.positions_px,
        _as_complex_2d(dataset.probe_truth, "probe_truth").shape,
    )
    return reference_object, illuminated


def score_blind_reconstruction(
    recon_object: np.ndarray,
    recon_probe: np.ndarray,
    dataset: SyntheticDataset,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Return truth errors after removing unobservable blind ambiguities.

    Object fitting and errors use only the registered union of measured scan
    footprints. Probe error retains the object-derived amplitude gauge while
    removing the independent, intensity-invisible exit-wave phase.
    """
    object_estimate = _as_complex_2d(recon_object, "object")
    probe_truth = _as_complex_2d(dataset.probe_truth, "probe_truth")
    reference_object, object_estimate, illuminated = _overlapping_object_views(
        _as_complex_2d(dataset.truth, "truth"),
        object_estimate,
        dataset.positions_px,
        probe_truth.shape,
    )
    probe_estimate = _as_complex_2d(recon_probe, "probe")
    object_aligned, probe_aligned, _ = align_blind_object_probe(
        reference_object,
        object_estimate,
        probe_estimate,
        mask=illuminated,
    )
    # Intensities leave one global exit-wave phase unobservable in addition to
    # the reciprocal object/probe scalar gauge. The object-derived gauge above
    # fixes the observable probe amplitude; remove only the remaining
    # unit-modulus phase from the probe comparison.
    _, probe_scale = align_complex_scalar(probe_truth, probe_aligned)
    if probe_scale != 0:
        probe_aligned = probe_aligned * (probe_scale / abs(probe_scale))

    phase_error = np.angle(
        object_aligned[illuminated] * np.conj(reference_object[illuminated])
    )
    phase_weights = np.abs(reference_object[illuminated]) ** 2
    weight_sum = float(np.sum(phase_weights, dtype=np.float32))
    if weight_sum <= 0:
        raise ValueError("Cannot compute phase RMSE for a zero object.")
    phase_rmse = math.sqrt(
        float(
            np.sum(
                phase_weights * phase_error**2,
                dtype=np.float32,
            )
            / weight_sum
        )
    )

    metrics = {
        "phase_rmse_rad": phase_rmse,
        "object_nmse": _normalized_squared_error(
            reference_object[illuminated],
            object_aligned[illuminated],
        ),
        "object_magnitude_nrmse": math.sqrt(
            _normalized_squared_error(
                np.abs(reference_object[illuminated]),
                np.abs(object_aligned[illuminated]),
            )
        ),
        "probe_nmse": _normalized_squared_error(
            probe_truth,
            probe_aligned,
        ),
    }
    return metrics, object_aligned, probe_aligned
