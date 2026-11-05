"""Synthetic arrays, scaled-Poisson perturbations, and fixed-region errors.

Dataset identity depends only on acquisition settings, this module, and its
two image assets. Reconstruction methods, batches, and epochs never affect it.
All arrays returned here are NumPy arrays; importing this module needs no Torch.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

import imageio.v2 as imageio
import numpy as np
from skimage.transform import resize

from .protocol import case_token, float_token


# Fixed generator constants are internal. The notebooks expose only the
# mathematical fields, their dimensions, normalization, and initial error.
_PROBE_TRUTH_PARAMETER = -7e-4
_PROBE_INITIAL_PARAMETER = -4e-4
_PROBE_SQUARED_NORM = 300.0
_PROBE_SOURCE_SIZE = 512
_PROBE_TARGET_SIZE = 128


@dataclass(frozen=True)
class DataConfig:
    overlap_ratio: float = 0.75
    eta: float = 0.1
    seed: int = 42
    smoke: bool = False

    def __post_init__(self):
        if self.overlap_ratio not in (0.75, 0.875):
            raise ValueError("Choose overlap_ratio from 0.75 or 0.875.")
        if not np.isfinite(self.eta) or self.eta < 0:
            raise ValueError("eta must be finite and nonnegative; zero means exact data.")
        if not isinstance(self.seed, (int, np.integer)) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        object.__setattr__(self, "overlap_ratio", float(self.overlap_ratio))
        object.__setattr__(self, "eta", float(self.eta))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "smoke", bool(self.smoke))

    @property
    def object_size(self):
        return 64 if self.smoke else 1024

    @property
    def probe_size(self):
        return 16 if self.smoke else 128

    @property
    def roi_bounds(self):
        # Retain the reference 75% study's predeclared evaluation region at
        # every overlap. Weak internal sensitivity is part of the new problem.
        return (4, 60, 4, 60) if self.smoke else (46, 978, 46, 978)


@dataclass(frozen=True)
class SyntheticData:
    config: DataConfig
    intensities: np.ndarray
    noiseless_intensities: np.ndarray
    truth: np.ndarray
    probe_truth: np.ndarray
    probe_init: np.ndarray
    object_init: np.ndarray
    positions_px: np.ndarray
    top_left_px: np.ndarray
    roi_mask: np.ndarray
    metadata: dict
    path: Path

    @property
    def fingerprint(self):
        return self.metadata["fingerprint"]

    @property
    def mask(self):
        """Compatibility alias; always the predefined evaluation ROI."""
        return self.roi_mask


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array_sha256(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.tobytes()).hexdigest()


def _identity(config):
    assets = Path(__file__).resolve().parent / "assets"
    identity = {
        "schema_version": 2,
        "study": "blind_magpie_synthetic",
        "config": asdict(config),
        "data_protocol": {
            "object_size": config.object_size,
            "probe_size": config.probe_size,
            "roi_bounds": list(config.roi_bounds),
            "probe_source_size": _PROBE_SOURCE_SIZE,
            "probe_target_size": _PROBE_TARGET_SIZE,
            "probe_truth_parameter": _PROBE_TRUTH_PARAMETER,
            "probe_initial_parameter": _PROBE_INITIAL_PARAMETER,
            "probe_squared_norm": _PROBE_SQUARED_NORM,
            "probe_resampling": "centered Fourier truncation; no spatial crop",
            "object_formula": "(0.2+0.8*a)*exp(1j*pi*(b-0.5)); min-max image normalization",
            "full_integer_perturbation_bound": 4,
            "scan_rule": "stride 2/1 of the shared step-16 master grid, including final endpoint",
            "noise_order": "75%-subset row-major first, then remaining master indices; independent 64-row phases",
        },
        "data_source_sha256": _sha256(__file__),
        "assets_sha256": {name: _sha256(assets / name)
                          for name in ("baboon.tiff", "cameraman.tif")},
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True,
                                           separators=(",", ":")).encode()).hexdigest()
    return identity, fingerprint


def bandlimited_resize(field, size):
    """Centered Fourier truncation of an even square complex array."""
    field = np.asarray(field)
    old = field.shape[-1]
    if field.ndim != 2 or field.shape[0] != old or old % 2 or size % 2 or size > old:
        raise ValueError("Require an even square source and an even smaller target.")
    start = (old - size) // 2
    spectrum = np.fft.fftshift(np.fft.fft2(field))
    result = np.fft.ifft2(np.fft.ifftshift(spectrum[start:start+size, start:start+size]))
    return np.asarray(result * (size / old)**2, dtype=np.complex64)


def make_probes(smoke=False):
    """Return the fixed truth and hard initialization, each with squared norm 300.

    The full setup truncates the 512-square Fourier array to 128-square, with
    no spatial crop. Only the diagnostic smoke geometry further resamples
    the entire 128-square array to 16-square.
    """
    probes = []
    for parameter in (_PROBE_TRUTH_PARAMETER, _PROBE_INITIAL_PARAMETER):
        source = generate_probe(_PROBE_SOURCE_SIZE, 1e-10, 1e-8, parameter)
        probe = bandlimited_resize(source, _PROBE_TARGET_SIZE)
        if smoke:
            probe = bandlimited_resize(probe, 16)
        norm = np.sum(np.abs(probe)**2, dtype=np.float64)
        probe = np.asarray(probe * np.sqrt(_PROBE_SQUARED_NORM / norm), dtype=np.complex64)
        probes.append(probe)
    return tuple(probes)


def _make_object(config):
    assets = Path(__file__).resolve().parent / "assets"
    images = []
    for name in ("baboon.tiff", "cameraman.tif"):
        image = imageio.imread(assets / name).astype(np.float32)
        if image.ndim == 3:
            image = image[..., :3].mean(axis=2)
        image = resize(image, (config.object_size, config.object_size),
                       preserve_range=True, anti_aliasing=True).astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min())
        images.append(image)
    return np.asarray((0.2 + 0.8 * images[0]) *
                      np.exp(1j * np.pi * (images[1] - 0.5)), dtype=np.complex64)


def scan_geometry(config):
    """Return integer top-left positions and their nested master-grid indices.

    Both scans are subsets of one perturbed master grid. Boundary coordinate
    components remain fixed.
    """
    span = config.object_size - config.probe_size
    step = config.probe_size // 8
    nominal = np.arange(0, span + 1, step, dtype=np.int32)
    yy, xx = np.meshgrid(nominal, nominal, indexing="ij")
    master = np.stack((yy, xx), axis=-1)
    rng = np.random.default_rng(config.seed)
    bound = 1 if config.smoke else 4
    jitter = rng.integers(-bound, bound + 1, size=master.shape, dtype=np.int32)
    jitter[(master == 0) | (master == span)] = 0
    master = master + jitter
    stride = {0.75: 2, 0.875: 1}[config.overlap_ratio]
    indices = np.arange(len(nominal)**2).reshape(len(nominal), len(nominal))
    axis = np.unique(np.r_[np.arange(0, len(nominal), stride), len(nominal)-1])
    selected = indices[np.ix_(axis, axis)].ravel()
    return (master.reshape(-1, 2)[selected], selected, master.reshape(-1, 2))


def _shared_measurements(config, truth, probe_truth, selected, master):
    """Create matching noise on shared sites, preserving original 75% data.

    The reference subset is processed first with exactly the original
    64-pattern grouping, then the remaining sites. Every overlap consumes
    the same random sequence. Only its own measurements are retained.
    """
    side = int(round(np.sqrt(len(master))))
    grid = np.arange(len(master)).reshape(side, side)
    reference = grid[::2, ::2].ravel()
    remaining = np.setdiff1d(np.arange(len(master)), reference, assume_unique=True)
    lookup = np.full(len(master), -1, dtype=np.int32)
    lookup[selected] = np.arange(len(selected))
    shape = (len(selected), config.probe_size, config.probe_size)
    exact = np.empty(shape, dtype=np.float32)
    measured = np.empty(shape, dtype=np.float32) if config.eta else exact
    rng = np.random.default_rng(config.seed + 100_000)
    for phase in (reference, remaining):
        for start in range(0, len(phase), 64):
            indices = phase[start:start+64]
            values = forward_intensities(truth, probe_truth, master[indices])
            noisy = (config.eta * rng.poisson(
                np.maximum(values.astype(np.float64) / config.eta, 0.0))
                if config.eta else values)
            destinations = lookup[indices]
            keep = destinations >= 0
            exact[destinations[keep]] = values[keep]
            measured[destinations[keep]] = noisy[keep]
    return exact, measured


def forward_intensities(truth, probe, top_left_px, batch_size=64):
    """Integer patch products followed by |FFT2|² (unscaled, centered output)."""
    n = probe.shape[-1]
    output = np.empty((len(top_left_px), n, n), dtype=np.float32)
    for start in range(0, len(top_left_px), batch_size):
        positions = top_left_px[start:start + batch_size]
        patches = np.stack([truth[y:y+n, x:x+n] for y, x in positions])
        values = np.abs(np.fft.fft2(patches * probe[None], axes=(-2, -1)))**2
        output[start:start + len(positions)] = np.fft.fftshift(values, axes=(-2, -1))
    return output


def illumination_sum(probe, top_left_px, object_size):
    """Sum of shifted squared magnitudes; a geometry diagnostic only."""
    weights = np.zeros((object_size, object_size), dtype=np.float64)
    n = probe.shape[-1]
    squared = np.abs(probe)**2
    for y, x in top_left_px:
        weights[y:y+n, x:x+n] += squared
    return weights


def shape_scalar_nmse(reference, estimate):
    aligned, _ = align_complex_scalar(reference, estimate)
    return float(np.sum(np.abs(reference - aligned)**2, dtype=np.float64) /
                 np.sum(np.abs(reference)**2, dtype=np.float64))


def _generate(config, identity, fingerprint, path):
    truth = _make_object(config)
    probe_truth, probe_init = make_probes(config.smoke)
    top_left, selected, master = scan_geometry(config)
    exact, measured = _shared_measurements(config, truth, probe_truth, selected, master)
    roi = np.zeros(truth.shape, dtype=bool)
    y0, y1, x0, x1 = config.roi_bounds
    roi[y0:y1, x0:x1] = True
    weights = illumination_sum(probe_truth, top_left, config.object_size)
    middle = config.object_size // 4
    central_mean = float(weights[middle:-middle, middle:-middle].mean())
    minimum_relative_sensitivity = float(weights[roi].min() / central_mean)
    if not config.smoke and np.any(weights[roi] <= 0):
        raise ValueError("The predefined common ROI has unobserved pixels.")
    metadata = {
        **identity,
        "fingerprint": fingerprint,
        "object_shape": list(truth.shape),
        "probe_shape": list(probe_truth.shape),
        "num_patterns": len(top_left),
        "nominal_overlap": config.overlap_ratio,
        "nominal_step": int(round(config.probe_size * (1-config.overlap_ratio))),
        "positions": "integer; stride 2/1 subset of perturbed master grid; final endpoint included; boundary coordinate components fixed",
        "boundary_nominal_gap": int((config.object_size-config.probe_size) %
                                    round(config.probe_size*(1-config.overlap_ratio)) or
                                    round(config.probe_size*(1-config.overlap_ratio))),
        "noise_coupling": "identical noisy values at shared master sites, preserving original 75% random draw order",
        "measurements": "d = eta * Poisson(d_tilde / eta); eta=0 returns d_tilde",
        "forward_transform": "numpy.fft.fft2(norm='backward'), squared modulus, fftshift",
        "object_initialization": "exact complex ones",
        "probe_initialization": "fixed hard-error array; squared norm 300, independent of measurements",
        "probe_truth_squared_norm": float(np.sum(abs(probe_truth)**2, dtype=np.float64)),
        "probe_initial_squared_norm": float(np.sum(abs(probe_init)**2, dtype=np.float64)),
        "probe_initial_shape_nmse": shape_scalar_nmse(probe_truth, probe_init),
        "evaluation_roi_bounds": list(config.roi_bounds),
        "evaluation_roi_pixels": int(roi.sum()),
        "evaluation_roi_criterion": (
            "fixed 932-square ROI inherited from the 75% reference study; "
            "all pixels must have positive shifted squared-probe sum; "
            "the reference 1% sensitivity criterion is not imposed at other overlaps"
            if not config.smoke else "fixed diagnostic ROI; no sensitivity threshold required"
        ),
        "sensitivity_reference_bounds": [middle, config.object_size-middle,
                                          middle, config.object_size-middle],
        "roi_min_illumination_to_central_mean": minimum_relative_sensitivity,
        "roi_fraction_below_reference_1pct_sensitivity": float(
            (weights[roi] < 0.01 * central_mean).mean()),
        "roi_coverage_fraction": float((weights[roi] > 0).mean()),
        "next_centered_even_square_min_relative_sensitivity": (
            float(weights[y0-1:y1+1, x0-1:x1+1].min() / central_mean)
            if not config.smoke else None
        ),
        "full_coverage_fraction": float((weights > 0).mean()),
        "full_min_illumination_to_central_mean": float(weights.min() / central_mean),
        "smoke_note": "diagnostic geometry only; not a scientific experiment" if config.smoke else None,
    }
    arrays = dict(intensities=measured,
                  noiseless_intensities=exact, truth=truth,
                  probe_truth=probe_truth, probe_init=probe_init,
                  object_init=np.ones(truth.shape, dtype=np.complex64),
                  positions_px=np.asarray(top_left + (config.probe_size-config.object_size)/2,
                                          dtype=np.float32),
                  top_left_px=top_left.astype(np.int32), roi_mask=roi)
    metadata["array_sha256"] = {name: _array_sha256(array) for name, array in arrays.items()}
    return SyntheticData(config=config, metadata=metadata, path=path, **arrays)


_ARRAY_FIELDS = ("intensities", "noiseless_intensities", "truth", "probe_truth",
                 "probe_init", "object_init", "positions_px", "top_left_px", "roi_mask")


def _read_cache(path, config, expected_fingerprint):
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata_json"].item()))
        if metadata.get("fingerprint") != expected_fingerprint:
            raise ValueError(f"{path} belongs to a different data source or configuration. "
                             "Use another output folder; the cache was not changed.")
        arrays = {name: saved[name] for name in _ARRAY_FIELDS}
    for name, array in arrays.items():
        if _array_sha256(array) != metadata["array_sha256"][name]:
            raise ValueError(f"Cached {name} failed its SHA256 integrity check: {path}")
    return SyntheticData(config=config, metadata=metadata, path=path, **arrays)


def _atomic_json(path, payload):
    fd, temporary = tempfile.mkstemp(prefix=".metadata-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_or_make_data_unlocked(root, config=None):
    """Load an integrity-checked dataset or atomically create an independent cache."""
    config = config or DataConfig()
    identity, fingerprint = _identity(config)
    root = Path(root).expanduser().resolve()
    if config.smoke:
        root = root / "smoke"
    directory = root / "datasets" / (f"{case_token(config.overlap_ratio)}_eta_"
                                     f"{float_token(config.eta)}_seed{config.seed}")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "dataset.npz"
    if path.exists():
        dataset = _read_cache(path, config, fingerprint)
    else:
        dataset = _generate(config, identity, fingerprint, path)
        fd, temporary = tempfile.mkstemp(prefix=".dataset-", suffix=".npz", dir=directory)
        os.close(fd)
        try:
            np.savez_compressed(temporary, metadata_json=json.dumps(dataset.metadata, sort_keys=True),
                                **{name: getattr(dataset, name) for name in _ARRAY_FIELDS})
            try:
                # Linking installs a completed file without overwriting another
                # process's independently completed first-time cache.
                os.link(temporary, path)
            except FileExistsError:
                dataset = _read_cache(path, config, fingerprint)
        finally:
            os.unlink(temporary)
    if dataset.truth.shape != (config.object_size, config.object_size):
        raise ValueError("Cached object shape is inconsistent.")
    if dataset.intensities.shape != (len(dataset.positions_px), config.probe_size, config.probe_size):
        raise ValueError("Cached measurements and scan positions are inconsistent.")
    if not np.all(dataset.object_init == 1) or dataset.roi_mask.sum() != (config.roi_bounds[1]-config.roi_bounds[0])**2:
        raise ValueError("Cached initialization or evaluation ROI is inconsistent.")
    _atomic_json(directory / "metadata.json", dataset.metadata)
    return dataset


def load_or_make_data(root, config=None):
    """Serialize first-time generation and reuse an integrity-checked cache.

    Different methods sharing one overlap wait for the same data cache,
    avoiding duplicated first-time FFT and random-array work in array jobs.
    """
    config = config or DataConfig()
    base = Path(root).expanduser().resolve()
    if config.smoke:
        base = base / "smoke"
    directory = base / "datasets" / (f"{case_token(config.overlap_ratio)}_eta_"
                                     f"{float_token(config.eta)}_seed{config.seed}")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".dataset.lock").open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            return _load_or_make_data_unlocked(root, config)
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def score(object_estimate, probe_estimate, data):
    """Return (metrics, aligned object, aligned probe), using the fixed ROI.

    The object scalar and affine phase ramp are fitted only on this same ROI.
    Their inverse transformation is applied to the probe; only the remaining
    global probe phase is removed. No independent probe magnitude fit is made.
    Phase RMSE weights the squared wrapped phase difference by |O_truth|²,
    matching the manuscript, on the same fixed ROI.
    """
    def array(value, name):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return _as_complex_2d(value, name)
    obj = array(object_estimate, "object")
    probe = array(probe_estimate, "probe")
    if obj.shape != data.truth.shape or probe.shape != data.probe_truth.shape:
        raise ValueError("Scoring requires the full saved object and probe shapes.")
    if not np.isfinite(obj).all() or not np.isfinite(probe).all():
        raise ValueError("Cannot score nonfinite reconstruction arrays.")
    aligned_obj, aligned_probe, _ = align_blind_object_probe(
        data.truth, obj, probe, mask=data.roi_mask)
    _, probe_scale = align_complex_scalar(data.probe_truth, aligned_probe)
    if probe_scale:
        aligned_probe = aligned_probe * (probe_scale / abs(probe_scale))
    def nmse(reference, candidate):
        return float(np.sum(abs(reference-candidate)**2, dtype=np.float64) /
                     np.sum(abs(reference)**2, dtype=np.float64))
    phase_difference = np.angle(aligned_obj[data.roi_mask] * np.conj(data.truth[data.roi_mask]))
    phase_weight = np.abs(data.truth[data.roi_mask]).astype(np.float64)**2
    metrics = {
        "object_nmse": nmse(data.truth[data.roi_mask], aligned_obj[data.roi_mask]),
        "probe_nmse": nmse(data.probe_truth, aligned_probe),
        "phase_rmse": float(np.sqrt(np.sum(phase_weight * phase_difference.astype(np.float64)**2)
                                    / np.sum(phase_weight))),
    }
    return metrics, aligned_obj, aligned_probe


# Adapted from src/utils/optics.py (data generation / evaluation only).


def fresnel_propagation(
    wavefield: np.ndarray,
    pixel_size_m: float,
    propagation_distance_m: float,
    wavelength_m: float,
) -> np.ndarray:
    """Propagate a 2D field with the single-FFT Fresnel transform.

    The returned field is unnormalized because probe power is calibrated after
    propagation. Its natural output sampling is
    ``wavelength_m * propagation_distance_m / (N * pixel_size_m)``.
    """
    wavefield = np.asarray(wavefield, dtype=np.complex64)
    if wavefield.ndim != 2:
        raise ValueError("wavefield must be a two-dimensional array.")
    if not np.isfinite(pixel_size_m) or pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be finite and positive.")
    if not np.isfinite(wavelength_m) or wavelength_m <= 0:
        raise ValueError("wavelength_m must be finite and positive.")
    if not np.isfinite(propagation_distance_m) or propagation_distance_m <= 0:
        raise ValueError("propagation_distance_m must be finite and positive.")

    height, width = wavefield.shape
    pixel_size = np.float32(pixel_size_m)
    propagation_distance = np.float32(propagation_distance_m)
    wavelength = np.float32(wavelength_m)
    wave_number = np.float32(2 * np.pi / wavelength_m)
    imaginary_unit = np.complex64(1j)

    x_axis = (np.arange(width, dtype=np.float32) - width // 2) * pixel_size
    y_axis = (np.arange(height, dtype=np.float32) - height // 2) * pixel_size
    x, y = np.meshgrid(x_axis, y_axis)

    u_axis = (
        wavelength
        * propagation_distance
        * np.fft.fftfreq(width, d=pixel_size).astype(np.float32)
    ).astype(np.float32, copy=False)
    v_axis = (
        wavelength
        * propagation_distance
        * np.fft.fftfreq(height, d=pixel_size).astype(np.float32)
    ).astype(np.float32, copy=False)
    u, v = np.meshgrid(u_axis, v_axis)

    output_phase = np.exp(imaginary_unit * wave_number * propagation_distance)
    output_phase = output_phase * np.exp(
        imaginary_unit
        * wave_number
        * (u**2 + v**2)
        / (np.float32(2) * propagation_distance)
    )
    input_chirp = wavefield * np.exp(
        imaginary_unit
        * wave_number
        * (x**2 + y**2)
        / (np.float32(2) * propagation_distance)
    )
    propagated = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(input_chirp)) * output_phase
    )
    return propagated.astype(np.complex64)


def generate_probe(
    size: int,
    wavelength_m: float,
    sample_pixel_size_m: float,
    defocus_m: float,
) -> np.ndarray:
    """Generate an idealized Velociprobe FZP field at the sample plane."""
    if not isinstance(size, (int, np.integer)) or size < 2:
        raise ValueError("size must be an integer greater than one.")
    if not np.isfinite(wavelength_m) or wavelength_m <= 0:
        raise ValueError("wavelength_m must be finite and positive.")
    if not np.isfinite(sample_pixel_size_m) or sample_pixel_size_m <= 0:
        raise ValueError("sample_pixel_size_m must be finite and positive.")
    if not np.isfinite(defocus_m):
        raise ValueError("defocus_m must be finite.")
    wavelength = np.float32(wavelength_m)
    sample_pixel_size = np.float32(sample_pixel_size_m)
    outer_radius_m = np.float32(90e-6)
    outermost_zone_width_m = np.float32(50e-9)
    beamstop_diameter_m = np.float32(60e-6)
    focal_length_m = (
        np.float32(2) * outer_radius_m * outermost_zone_width_m / wavelength
    )
    propagation_distance = focal_length_m + np.float32(defocus_m)
    if propagation_distance <= 0:
        raise ValueError("defocus_m must place the sample downstream of the FZP.")

    fzp_pixel_size_m = (
        wavelength * propagation_distance / (np.float32(size) * sample_pixel_size)
    )
    axis = (np.arange(size, dtype=np.float32) - size // 2) * fzp_pixel_size_m
    x_fzp, y_fzp = np.meshgrid(axis, axis)
    radius = np.sqrt(x_fzp**2 + y_fzp**2)

    lens_phase = np.exp(
        np.complex64(-1j)
        * np.float32(2 * np.pi)
        / wavelength
        * radius**2
        / (np.float32(2) * focal_length_m)
    )
    transmitting_annulus = (radius <= outer_radius_m) & (
        radius >= beamstop_diameter_m / 2
    )
    wave_at_fzp = transmitting_annulus * lens_phase

    return fresnel_propagation(
        wave_at_fzp,
        fzp_pixel_size_m,
        propagation_distance,
        wavelength,
    )


# Adapted from src/utils/phase_alignment.py (data generation / evaluation only).


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
