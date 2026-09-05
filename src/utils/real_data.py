from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from algorithms.lsqml import (
    LSQMLHyperparameters,
    MPSCompatibleLSQMLTask,
    build_lsqml_options,
)
from algorithms.magpie import BlindMAGPIETask
from algorithms.rpie import (
    ReplicatePaddedRPIETask,
    build_rpie_options as build_base_rpie_options,
)
from utils.common import PROJECT_ROOT
from utils.optics import generate_probe
from utils.reconstruction import (
    FinalMetricFunction,
    ProgressCallback,
    ReconstructionResult,
    StateSnapshotCallback,
    TaskMetricFunction,
    make_negative_complex_gaussian_object_init,
    run_reconstruction_task,
)

import h5py
import numpy as np

import ptychi.api as api
from ptychi.api.task import PtychographyTask

REAL_DATA_FILENAME = "Velo_18c3_comm_chip65nm_scan054_data_roi0_Ndp512_us2.hdf5"


def _default_real_data_path() -> Path:
    return PROJECT_ROOT / "assets" / "ptycho_real_data" / REAL_DATA_FILENAME


@dataclass
class RealDataConfig:
    data_path: Path = field(default_factory=_default_real_data_path)
    fzp_defocus_m: float = -1e-3
    pattern_stride: int = 1
    max_patterns: int | None = None
    fft_shift_data: bool = True

    num_epochs: int = 100
    batch_size: int = 16
    object_extra_pixels: int = 20
    seed: int = 42

    object_alpha: float = 0.5
    object_step_size: float = 1.0
    probe_alpha: float = 1.0
    probe_step_size: float = 1.0
    probe_update_start_epoch: int = 0
    probe_update_stride: int = 1
    save_data_on_device: bool = False
    remove_object_probe_ambiguity: bool = True
    random_pattern_fraction: float | None = None

    def __post_init__(self) -> None:
        if self.pattern_stride < 1:
            raise ValueError("pattern_stride must be at least 1.")
        if self.random_pattern_fraction is not None:
            if not 0 < self.random_pattern_fraction <= 1:
                raise ValueError("random_pattern_fraction must be in (0, 1].")
            if self.pattern_stride != 1:
                raise ValueError(
                    "random_pattern_fraction and pattern_stride cannot both select "
                    "a pattern subset."
                )
        if self.max_patterns is not None and self.max_patterns < 1:
            raise ValueError("max_patterns must be positive when provided.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")


@dataclass
class RealPtychographyDataset:
    data: np.ndarray
    positions_px: np.ndarray
    probe_init: np.ndarray
    dx_m: float
    wavelength_m: float
    selected_pattern_indices: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    source_pattern_count: int = 0


def _selected_pattern_indices(handle: h5py.File, cfg: RealDataConfig) -> np.ndarray:
    num_patterns = int(handle["dp"].shape[0])
    if cfg.random_pattern_fraction is not None:
        requested_count = int(np.floor(num_patterns * cfg.random_pattern_fraction))
        if cfg.max_patterns is not None:
            requested_count = min(requested_count, int(cfg.max_patterns))

        # Random-selection experiments use complete minibatches. Dropping the
        # remainder also gives every method the same number of updates.
        selected_count = requested_count - requested_count % cfg.batch_size
        if selected_count == 0:
            raise ValueError(
                "Random pattern selection retained fewer than one complete batch: "
                f"requested {requested_count} patterns with batch_size={cfg.batch_size}."
            )

        generator = np.random.default_rng(cfg.seed)
        indices = generator.choice(
            num_patterns,
            size=selected_count,
            replace=False,
        ).astype(np.int64, copy=False)
        # h5py requires increasing indices for advanced indexing. Sorting does
        # not change which scan regions were selected.
        return np.sort(indices)

    indices = np.arange(0, num_patterns, cfg.pattern_stride, dtype=np.int64)
    if cfg.max_patterns is not None:
        indices = indices[: min(int(cfg.max_patterns), len(indices))]
    return indices


def _read_pattern_subset(
    patterns: h5py.Dataset,
    indices: np.ndarray,
    chunk_size: int = 32,
) -> np.ndarray:
    """Read selected HDF5 frames directly into bounded float32 chunks."""
    output = np.empty((len(indices), *patterns.shape[-2:]), dtype=np.float32)
    if np.array_equal(indices, np.arange(len(indices))):
        patterns.read_direct(output, source_sel=np.s_[: len(indices)])
        return output

    for start in range(0, len(indices), chunk_size):
        stop = min(start + chunk_size, len(indices))
        output[start:stop] = patterns[indices[start:stop]]
    return output


def preprocess_real_intensities(data: np.ndarray) -> np.ndarray:
    """Prepare measured intensities without inventing detector corrections.

    The supplied HDF5 files contain already cropped, centered intensities but no
    measured dark frame, gain map, bad-pixel mask, or incident-flux monitor.
    Consequently, the defensible shared preprocessing is to use the full
    detector, preserve zero-count pixels, and validate the physical data model.
    Background subtraction, frame normalization, and detector masking require
    calibration data and are intentionally not estimated from the diffraction
    signal itself.
    """
    intensities = np.ascontiguousarray(data, dtype=np.float32)
    if intensities.ndim != 3:
        raise ValueError(
            "Real diffraction data must have shape (patterns, height, width); "
            f"received {intensities.shape}."
        )
    if any(size == 0 for size in intensities.shape):
        raise ValueError("Real diffraction data must not have an empty dimension.")

    # Validate in bounded views so the full test-pattern dataset does not need a
    # second detector-sized temporary array.
    for start in range(0, len(intensities), 32):
        chunk = intensities[start : start + 32]
        if not np.isfinite(chunk).all():
            raise ValueError("Real diffraction data contains NaN or infinite values.")
        if np.any(chunk < 0):
            raise ValueError("Real diffraction intensities must be nonnegative.")

    return intensities


def load_real_dataset(cfg: RealDataConfig) -> RealPtychographyDataset:
    with h5py.File(cfg.data_path, "r") as handle:
        source_pattern_count = int(handle["dp"].shape[0])
        pattern_indices = _selected_pattern_indices(handle, cfg)
        data = preprocess_real_intensities(
            _read_pattern_subset(handle["dp"], pattern_indices)
        )
        position_x_m = np.asarray(handle["ppX"][pattern_indices], dtype=np.float32)
        position_y_m = np.asarray(handle["ppY"][pattern_indices], dtype=np.float32)
        dx_m = float(np.asarray(handle["dx"][()]).squeeze())
        wavelength_m = float(np.asarray(handle["lambda"][()]).squeeze())

    mean_data = np.mean(data, axis=0, dtype=np.float32)

    probe = generate_probe(
        data.shape[-1],
        wavelength_m,
        dx_m,
        cfg.fzp_defocus_m,
    )
    probe_norm = float(np.sqrt(np.sum(np.abs(probe) ** 2, dtype=np.float32)))
    if probe_norm == 0:
        raise ValueError("Generated a zero-norm probe initialization.")
    data_power_scale = np.sqrt(np.sum(mean_data)) / float(probe.shape[-1])
    probe = (probe * (data_power_scale / probe_norm)).astype(np.complex64)

    positions_px = np.stack([position_y_m / dx_m, position_x_m / dx_m], axis=1).astype(
        np.float32
    )
    positions_px = positions_px - positions_px.mean(axis=0, keepdims=True)

    return RealPtychographyDataset(
        data=data,
        positions_px=positions_px,
        probe_init=probe,
        dx_m=dx_m,
        wavelength_m=wavelength_m,
        selected_pattern_indices=pattern_indices,
        source_pattern_count=source_pattern_count,
    )


def build_real_rpie_options(
    dataset: RealPtychographyDataset,
    cfg: RealDataConfig,
    device: api.Devices,
    seed: int | None = None,
) -> api.RPIEOptions:
    reconstruction_seed = cfg.seed if seed is None else seed
    options = build_base_rpie_options(
        data=dataset.data,
        valid_pixel_mask=None,
        positions_px=dataset.positions_px,
        probe_init=dataset.probe_init,
        make_object_initial_guess=make_negative_complex_gaussian_object_init,
        device=device,
        seed=reconstruction_seed,
        fft_shift_data=cfg.fft_shift_data,
        save_data_on_device=cfg.save_data_on_device,
        object_extra_pixels=cfg.object_extra_pixels,
        batch_size=cfg.batch_size,
        num_epochs=cfg.num_epochs,
        object_step_size=cfg.object_step_size,
        probe_step_size=cfg.probe_step_size,
        object_alpha=cfg.object_alpha,
        probe_alpha=cfg.probe_alpha,
        object_pixel_size_m=dataset.dx_m,
        wavelength_m=dataset.wavelength_m,
        remove_object_probe_ambiguity=cfg.remove_object_probe_ambiguity,
        probe_update_start_epoch=cfg.probe_update_start_epoch,
        probe_update_stride=cfg.probe_update_stride,
    )
    options.reconstructor_options.batching_mode = api.BatchingModes.RANDOM
    # Keep online residuals comparable across real-data algorithms even if a
    # future Pty-Chi release changes an algorithm-specific display default.
    options.reconstructor_options.displayed_loss_function = api.LossFunctions.MSE_SQRT
    options.object_options.remove_object_probe_ambiguity.optimization_plan.stride = 1
    return options


def _run_real_task(
    task: PtychographyTask,
    *,
    report_stride: int | None,
    progress_callback: ProgressCallback | None,
    task_metric_function: TaskMetricFunction | None,
    final_metric_function: FinalMetricFunction | None,
    snapshot_callback: StateSnapshotCallback | None,
    snapshot_stride: int | None,
    include_initial_metrics: bool,
    chunk_epochs: bool,
) -> ReconstructionResult:
    return run_reconstruction_task(
        task,
        metric_stride=1 if report_stride is None else report_stride,
        progress_callback=progress_callback,
        task_metric_function=task_metric_function,
        final_metric_function=final_metric_function,
        snapshot_callback=snapshot_callback,
        snapshot_stride=snapshot_stride,
        include_initial_metrics=include_initial_metrics,
        chunk_epochs=chunk_epochs,
    )


def run_real_rpie(
    dataset: RealPtychographyDataset,
    cfg: RealDataConfig,
    device: api.Devices,
    seed: int | None = None,
    report_stride: int | None = None,
    progress_callback: ProgressCallback | None = None,
    task_metric_function: TaskMetricFunction | None = None,
    final_metric_function: FinalMetricFunction | None = None,
    snapshot_callback: StateSnapshotCallback | None = None,
    snapshot_stride: int | None = None,
    include_initial_metrics: bool = False,
    chunk_epochs: bool = False,
) -> ReconstructionResult:
    reconstruction_seed = cfg.seed if seed is None else seed
    task = ReplicatePaddedRPIETask(
        build_real_rpie_options(dataset, cfg, device, seed=reconstruction_seed)
    )
    shuffle_generator = task.reconstructor.dataloader.generator
    if shuffle_generator is None:
        raise RuntimeError("rPIE random batching requires a DataLoader generator.")
    shuffle_generator.manual_seed(reconstruction_seed)
    return _run_real_task(
        task,
        report_stride=report_stride,
        progress_callback=progress_callback,
        task_metric_function=task_metric_function,
        final_metric_function=final_metric_function,
        snapshot_callback=snapshot_callback,
        snapshot_stride=snapshot_stride,
        include_initial_metrics=include_initial_metrics,
        chunk_epochs=chunk_epochs,
    )


def build_real_lsqml_options(
    dataset: RealPtychographyDataset,
    cfg: RealDataConfig,
    device: api.Devices,
    seed: int | None = None,
    hyperparameters: LSQMLHyperparameters | None = None,
) -> api.LSQMLOptions:
    reconstruction_seed = cfg.seed if seed is None else seed
    if hyperparameters is None:
        raise ValueError(
            "Real-data LSQML requires explicit hyperparameters; pass the "
            "noise model and both optimal-step-size scalers."
        )

    options = build_lsqml_options(
        data=dataset.data,
        valid_pixel_mask=None,
        positions_px=dataset.positions_px,
        probe_init=dataset.probe_init,
        make_object_initial_guess=make_negative_complex_gaussian_object_init,
        device=device,
        seed=reconstruction_seed,
        fft_shift_data=cfg.fft_shift_data,
        save_data_on_device=cfg.save_data_on_device,
        object_extra_pixels=cfg.object_extra_pixels,
        batch_size=cfg.batch_size,
        num_epochs=cfg.num_epochs,
        object_step_size=cfg.object_step_size,
        probe_step_size=cfg.probe_step_size,
        noise_model=hyperparameters.noise_model,
        object_optimal_step_size_scaler=hyperparameters.object_step_size_scaler,
        probe_optimal_step_size_scaler=hyperparameters.probe_step_size_scaler,
        gaussian_noise_std=hyperparameters.gaussian_noise_std,
        object_pixel_size_m=dataset.dx_m,
        wavelength_m=dataset.wavelength_m,
        remove_object_probe_ambiguity=cfg.remove_object_probe_ambiguity,
        probe_update_start_epoch=cfg.probe_update_start_epoch,
        probe_update_stride=cfg.probe_update_stride,
    )
    options.object_options.remove_object_probe_ambiguity.optimization_plan.stride = 1
    options.reconstructor_options.displayed_loss_function = api.LossFunctions.MSE_SQRT
    return options


def run_real_lsqml(
    dataset: RealPtychographyDataset,
    cfg: RealDataConfig,
    device: api.Devices,
    seed: int | None = None,
    hyperparameters: LSQMLHyperparameters | None = None,
    report_stride: int | None = None,
    progress_callback: ProgressCallback | None = None,
    task_metric_function: TaskMetricFunction | None = None,
    final_metric_function: FinalMetricFunction | None = None,
    snapshot_callback: StateSnapshotCallback | None = None,
    snapshot_stride: int | None = None,
    include_initial_metrics: bool = False,
    chunk_epochs: bool = False,
) -> ReconstructionResult:
    reconstruction_seed = cfg.seed if seed is None else seed
    task = MPSCompatibleLSQMLTask(
        build_real_lsqml_options(
            dataset,
            cfg,
            device,
            seed=reconstruction_seed,
            hyperparameters=hyperparameters,
        )
    )
    shuffle_generator = task.reconstructor.dataloader.generator
    if shuffle_generator is None:
        raise RuntimeError("LSQML random batching requires a DataLoader generator.")
    shuffle_generator.manual_seed(reconstruction_seed)
    return _run_real_task(
        task,
        report_stride=report_stride,
        progress_callback=progress_callback,
        task_metric_function=task_metric_function,
        final_metric_function=final_metric_function,
        snapshot_callback=snapshot_callback,
        snapshot_stride=snapshot_stride,
        include_initial_metrics=include_initial_metrics,
        chunk_epochs=chunk_epochs,
    )


def run_real_blind_magpie(
    dataset: RealPtychographyDataset,
    cfg: RealDataConfig,
    device: api.Devices,
    multigrid_levels: int | None = None,
    seed: int | None = None,
    report_stride: int | None = None,
    progress_callback: ProgressCallback | None = None,
    task_metric_function: TaskMetricFunction | None = None,
    final_metric_function: FinalMetricFunction | None = None,
    snapshot_callback: StateSnapshotCallback | None = None,
    snapshot_stride: int | None = None,
    include_initial_metrics: bool = False,
    chunk_epochs: bool = False,
) -> ReconstructionResult:
    if cfg.object_step_size != 1.0 or cfg.probe_step_size != 1.0:
        raise ValueError(
            "Blind MAGPIE uses the surrogate minimizer directly, so "
            "object_step_size and probe_step_size must both be 1.0."
        )
    if cfg.probe_update_start_epoch != 0 or cfg.probe_update_stride != 1:
        raise ValueError(
            "Blind MAGPIE updates the object and probe in every minibatch; "
            "probe_update_start_epoch and probe_update_stride must be 0 and 1."
        )

    reconstruction_seed = cfg.seed if seed is None else seed
    task = BlindMAGPIETask(
        build_real_rpie_options(dataset, cfg, device, seed=reconstruction_seed),
        multigrid_levels=multigrid_levels,
    )
    shuffle_generator = task.reconstructor.dataloader.generator
    if shuffle_generator is None:
        raise RuntimeError("MAGPIE random batching requires a shuffle generator.")
    shuffle_generator.manual_seed(reconstruction_seed)
    return _run_real_task(
        task,
        report_stride=report_stride,
        progress_callback=progress_callback,
        task_metric_function=task_metric_function,
        final_metric_function=final_metric_function,
        snapshot_callback=snapshot_callback,
        snapshot_stride=snapshot_stride,
        include_initial_metrics=include_initial_metrics,
        chunk_epochs=chunk_epochs,
    )


def run_real_gm_rpie(
    dataset: RealPtychographyDataset,
    cfg: RealDataConfig,
    device: api.Devices,
    seed: int | None = None,
    report_stride: int | None = None,
    progress_callback: ProgressCallback | None = None,
    task_metric_function: TaskMetricFunction | None = None,
    final_metric_function: FinalMetricFunction | None = None,
    snapshot_callback: StateSnapshotCallback | None = None,
    snapshot_stride: int | None = None,
    include_initial_metrics: bool = False,
    chunk_epochs: bool = False,
) -> ReconstructionResult:
    """Run GM-rPIE using one-level object and probe endpoints."""
    return run_real_blind_magpie(
        dataset,
        cfg,
        device,
        multigrid_levels=1,
        seed=seed,
        report_stride=report_stride,
        progress_callback=progress_callback,
        task_metric_function=task_metric_function,
        final_metric_function=final_metric_function,
        snapshot_callback=snapshot_callback,
        snapshot_stride=snapshot_stride,
        include_initial_metrics=include_initial_metrics,
        chunk_epochs=chunk_epochs,
    )


def run_real_gm_magpie(
    dataset: RealPtychographyDataset,
    cfg: RealDataConfig,
    device: api.Devices,
    seed: int | None = None,
    report_stride: int | None = None,
    progress_callback: ProgressCallback | None = None,
    task_metric_function: TaskMetricFunction | None = None,
    final_metric_function: FinalMetricFunction | None = None,
    snapshot_callback: StateSnapshotCallback | None = None,
    snapshot_stride: int | None = None,
    include_initial_metrics: bool = False,
    chunk_epochs: bool = False,
) -> ReconstructionResult:
    """Run GM-MAGPIE using every valid object multigrid level."""
    return run_real_blind_magpie(
        dataset,
        cfg,
        device,
        multigrid_levels=None,
        seed=seed,
        report_stride=report_stride,
        progress_callback=progress_callback,
        task_metric_function=task_metric_function,
        final_metric_function=final_metric_function,
        snapshot_callback=snapshot_callback,
        snapshot_stride=snapshot_stride,
        include_initial_metrics=include_initial_metrics,
        chunk_epochs=chunk_epochs,
    )
