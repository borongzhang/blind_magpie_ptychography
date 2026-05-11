from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Mapping, Optional

import h5py
import numpy as np
import torch

from common import PROJECT_ROOT, patch_ptychi_compatibility
from common import (
    attach_residual_printer,
    configure_ptychi_device,
    set_random_seed,
)

import matplotlib.pyplot as plt
import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.utils import get_default_complex_dtype, get_suggested_object_size

patch_ptychi_compatibility()


REAL_DATA_FILENAME = "Velo_18c3_comm_chip65nm_scan054_data_roi0_Ndp512_us2.hdf5"


def _default_real_data_path() -> Path:
    return PROJECT_ROOT / "assets" / "ptycho_real_data" / REAL_DATA_FILENAME


@dataclass
class RealDataConfig:
    data_path: Path = field(default_factory=_default_real_data_path)
    probe_grid_size: Optional[int] = None
    fzp_defocus_m: float = -9e-4
    fzp_setup: str = "velo"
    intensity_floor: Optional[float] = None
    cutoff_radius_px: Optional[int] = 128
    max_patterns: Optional[int] = None
    center_positions: bool = True
    fft_shift_data: bool = True
    shift_subpixel_on_probe: bool = True
    pad_for_shift: int = 1

    num_epochs: int = 100
    batch_size: int = 16
    object_extra_pixels: int = 50
    use_mps: bool = True
    seed: int = 42

    object_alpha: float = 0.5
    object_step_size: float = 1.0
    probe_alpha: float = 1.0
    probe_step_size: float = 1.0
    probe_update_start_epoch: int = 0
    probe_update_stride: int = 1
    object_init_noise: float = 1e-2
    save_data_on_device: bool = False
    probe_shift_tol: float = 1e-6
    print_residual_every: int = 1


@dataclass
class RealPtychographyDataset:
    data: np.ndarray
    positions_px: np.ndarray
    probe_init: np.ndarray
    dx_m: float
    wavelength_m: float
    raw_positions_m: np.ndarray


@dataclass
class RealReconstructionResult:
    object: np.ndarray
    probe: np.ndarray
    task: PtychographyTask


def fresnel_propagation(
    wavefield: np.ndarray,
    pixel_size_m: float,
    propagation_distance_m: float,
    wavelength_m: float,
) -> np.ndarray:
    wavefield = np.asarray(wavefield)
    height, width = wavefield.shape
    k = 2 * np.pi / wavelength_m

    lx = np.linspace(-pixel_size_m * width / 2, pixel_size_m * width / 2, width)
    ly = np.linspace(-pixel_size_m * height / 2, pixel_size_m * height / 2, height)
    x, y = np.meshgrid(lx, ly)

    output_width_m = wavelength_m * propagation_distance_m / pixel_size_m
    lu = np.fft.ifftshift(np.linspace(-output_width_m / 2, output_width_m / 2, width))
    lv = np.fft.ifftshift(np.linspace(-output_width_m / 2, output_width_m / 2, height))
    u, v = np.meshgrid(lu, lv)

    if propagation_distance_m > 0:
        phase_out = np.exp(1j * k * propagation_distance_m)
        phase_out *= np.exp(1j * k * (u**2 + v**2) / (2 * propagation_distance_m))
        kernel = wavefield * np.exp(
            1j * k * (x**2 + y**2) / (2 * propagation_distance_m)
        )
        propagated = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(kernel)) * phase_out)
    else:
        z_abs = abs(propagation_distance_m)
        phase_in = np.exp(1j * k * z_abs) * np.exp(1j * k * (x**2 + y**2) / (2 * z_abs))
        phase_out = np.exp(1j * k * z_abs) * np.exp(1j * k * (u**2 + v**2) / (2 * z_abs))
        propagated = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(wavefield) / phase_out))
        propagated = propagated / phase_in

    return propagated.astype(np.complex64)


def generate_probe(
    size: int,
    wavelength_m: float,
    sample_pixel_size_m: float,
    defocus_m: float,
    setup: str,
) -> np.ndarray:
    if setup == "velo":
        radius_outer_m = 90e-6
        outermost_zone_width_m = 50e-9
    elif setup == "barry":
        radius_outer_m = 80e-6
        outermost_zone_width_m = 70e-9
    elif setup == "barry2":
        radius_outer_m = 70e-6
        outermost_zone_width_m = 160e-9
    else:
        radius_outer_m = 90e-6
        outermost_zone_width_m = 50e-9

    focal_length_m = 2 * radius_outer_m * outermost_zone_width_m / wavelength_m
    zone_plate_diameter_m = 180e-6
    beamstop_diameter_m = 60e-6
    zone_plate_pixel_size_m = wavelength_m * focal_length_m / (
        size * sample_pixel_size_m
    )

    axis = np.linspace(
        -zone_plate_pixel_size_m * size / 2,
        zone_plate_pixel_size_m * size / 2,
        size,
    )
    x_fzp, y_fzp = np.meshgrid(axis, axis)
    radius = np.sqrt(x_fzp**2 + y_fzp**2)

    lens_phase = np.exp(
        -1j * 2 * np.pi / wavelength_m * radius**2 / (2 * focal_length_m)
    )
    aperture = radius <= zone_plate_diameter_m / 2
    beamstop = radius >= beamstop_diameter_m / 2
    wave_at_fzp = aperture * lens_phase * beamstop

    return fresnel_propagation(
        wave_at_fzp,
        zone_plate_pixel_size_m,
        focal_length_m + defocus_m,
        wavelength_m,
    )


def apply_circular_cutoff(
    diffraction_data: np.ndarray,
    cutoff_radius_px: Optional[int],
    copy: bool = True,
) -> np.ndarray:
    if cutoff_radius_px is None:
        return np.array(diffraction_data, copy=copy)

    data = np.array(diffraction_data, copy=copy)
    size = data.shape[-1]
    axis = np.arange(-size // 2, size - size // 2)
    yy, xx = np.meshgrid(axis, axis)
    mask = np.sqrt(xx**2 + yy**2) > cutoff_radius_px
    data[..., mask] = 0
    return data


def load_real_dataset(cfg: RealDataConfig) -> RealPtychographyDataset:
    with h5py.File(cfg.data_path, "r") as handle:
        num_patterns = handle["dp"].shape[0]
        if cfg.max_patterns is None:
            pattern_slice = slice(None)
        else:
            pattern_slice = slice(0, min(int(cfg.max_patterns), num_patterns))

        data = np.asarray(handle["dp"][pattern_slice], dtype=np.float32)
        position_x_m = np.asarray(handle["ppX"][pattern_slice], dtype=np.float32)
        position_y_m = np.asarray(handle["ppY"][pattern_slice], dtype=np.float32)
        dx_m = float(np.asarray(handle["dx"][()]).squeeze())
        wavelength_m = float(np.asarray(handle["lambda"][()]).squeeze())

    probe_size = cfg.probe_grid_size or data.shape[-1]
    probe = generate_probe(
        probe_size,
        wavelength_m,
        dx_m,
        cfg.fzp_defocus_m,
        cfg.fzp_setup,
    )
    probe_norm = np.linalg.norm(probe)
    if probe_norm == 0:
        raise ValueError("Generated a zero-norm probe initialization.")
    data_power_scale = np.sqrt(np.sum(np.mean(data, axis=0))) / float(probe.shape[-1])
    probe = (probe * (data_power_scale / probe_norm)).astype(np.complex64)

    if cfg.intensity_floor is not None:
        data[data < cfg.intensity_floor] = 0
    data = apply_circular_cutoff(data, cfg.cutoff_radius_px, copy=False).astype(
        np.float32,
        copy=False,
    )

    positions_px = np.stack([position_y_m / dx_m, position_x_m / dx_m], axis=1).astype(
        np.float32
    )
    if cfg.center_positions:
        positions_px = positions_px - positions_px.mean(axis=0, keepdims=True)

    raw_positions_m = np.stack([position_y_m, position_x_m], axis=1).astype(np.float32)
    return RealPtychographyDataset(
        data=data,
        positions_px=positions_px,
        probe_init=probe,
        dx_m=dx_m,
        wavelength_m=wavelength_m,
        raw_positions_m=raw_positions_m,
    )


def _make_real_object_init(shape: tuple[int, int], noise_level: float) -> torch.Tensor:
    complex_dtype = get_default_complex_dtype()
    real_dtype = torch.float32 if complex_dtype == torch.complex64 else torch.float64
    obj_init = torch.ones((1, *shape), dtype=complex_dtype, device="cpu")
    if noise_level <= 0:
        return obj_init

    noise_real = torch.randn((1, *shape), dtype=real_dtype, device="cpu")
    noise_imag = torch.randn((1, *shape), dtype=real_dtype, device="cpu")
    noise = (noise_real + 1j * noise_imag).to(complex_dtype)
    return obj_init + noise_level * noise


def build_real_rpie_options(
    dataset: RealPtychographyDataset,
    cfg: RealDataConfig,
    device: api.Devices,
    seed: Optional[int] = None,
) -> api.RPIEOptions:
    patch_ptychi_compatibility()
    reconstruction_seed = cfg.seed if seed is None else seed
    set_random_seed(reconstruction_seed)
    obj_shape = get_suggested_object_size(
        dataset.positions_px,
        dataset.probe_init.shape[-2:],
        extra=cfg.object_extra_pixels,
    )

    options = api.RPIEOptions()
    options.data_options.data = dataset.data
    options.data_options.fft_shift = cfg.fft_shift_data
    options.data_options.save_data_on_device = cfg.save_data_on_device
    options.data_options.wavelength_m = dataset.wavelength_m

    options.object_options.initial_guess = _make_real_object_init(
        tuple(obj_shape),
        cfg.object_init_noise,
    )
    options.object_options.optimizable = True
    options.object_options.optimizer = api.Optimizers.SGD
    options.object_options.step_size = cfg.object_step_size
    options.object_options.alpha = cfg.object_alpha
    options.object_options.pixel_size_m = dataset.dx_m
    options.object_options.remove_object_probe_ambiguity.enabled = False
    options.object_options.determine_position_origin_coords_by = (
        api.ObjectPosOriginCoordsMethods.POSITIONS
    )

    options.probe_options.initial_guess = dataset.probe_init[None, None, :, :]
    options.probe_options.optimizable = True
    options.probe_options.optimizer = api.Optimizers.SGD
    options.probe_options.step_size = cfg.probe_step_size
    options.probe_options.alpha = cfg.probe_alpha
    options.probe_options.optimization_plan.start = cfg.probe_update_start_epoch
    options.probe_options.optimization_plan.stride = cfg.probe_update_stride
    options.probe_options.power_constraint.enabled = False
    options.probe_options.support_constraint.enabled = False
    options.probe_options.center_constraint.enabled = False

    options.probe_position_options.position_x_px = dataset.positions_px[:, 1]
    options.probe_position_options.position_y_px = dataset.positions_px[:, 0]
    options.probe_position_options.optimizable = False
    options.probe_position_options.optimizer = api.Optimizers.SGD
    options.probe_position_options.constrain_position_mean = True

    options.reconstructor_options.default_device = device
    options.reconstructor_options.batch_size = cfg.batch_size
    options.reconstructor_options.num_epochs = cfg.num_epochs
    options.reconstructor_options.random_seed = reconstruction_seed
    options.reconstructor_options.allow_nondeterministic_algorithms = False
    forward_model_field_names = {
        item.name
        for item in fields(options.reconstructor_options.forward_model_options)
    }
    if "apply_subpixel_shifts_on_probe" in forward_model_field_names:
        options.reconstructor_options.forward_model_options.apply_subpixel_shifts_on_probe = (
            cfg.shift_subpixel_on_probe
        )
    elif not cfg.shift_subpixel_on_probe:
        raise ValueError(
            "This Pty-Chi installation does not expose "
            "`apply_subpixel_shifts_on_probe` in ForwardModelOptions, so the "
            "chip experiments require shift_subpixel_on_probe=True."
        )
    options.reconstructor_options.forward_model_options.pad_for_shift = cfg.pad_for_shift

    return options


def run_real_rpie(
    dataset: RealPtychographyDataset,
    cfg: RealDataConfig,
    device: api.Devices,
    seed: Optional[int] = None,
) -> RealReconstructionResult:
    task = PtychographyTask(build_real_rpie_options(dataset, cfg, device, seed=seed))
    print_final_residual = attach_residual_printer(
        task,
        "rPIE",
        cfg.print_residual_every,
        cfg.num_epochs,
    )
    task.run()
    print_final_residual()
    return RealReconstructionResult(
        object=task.get_data_to_cpu("object", as_numpy=True),
        probe=task.get_data_to_cpu("probe", as_numpy=True),
        task=task,
    )


def run_real_geometric_magpie(
    dataset: RealPtychographyDataset,
    cfg: RealDataConfig,
    device: api.Devices,
    multigrid_levels: Optional[int] = None,
    seed: Optional[int] = None,
) -> RealReconstructionResult:
    from algorithms.geometric_magpie import GeometricMAGPIETask

    if cfg.object_step_size != 1.0 or cfg.probe_step_size != 1.0:
        raise ValueError(
            "Geometric MAGPIE uses the surrogate minimizer directly, so "
            "object_step_size and probe_step_size must both be 1.0."
        )

    task = GeometricMAGPIETask(
        build_real_rpie_options(dataset, cfg, device, seed=seed),
        multigrid_levels=multigrid_levels,
        probe_shift_tol=cfg.probe_shift_tol,
        assume_no_subpixel_shifts=False,
    )
    print_final_residual = attach_residual_printer(
        task,
        "MAGPIE",
        cfg.print_residual_every,
        cfg.num_epochs,
    )
    task.run()
    print_final_residual()
    return RealReconstructionResult(
        object=task.get_data_to_cpu("object", as_numpy=True),
        probe=task.get_data_to_cpu("probe", as_numpy=True),
        task=task,
    )


def summarize_real_dataset(dataset: RealPtychographyDataset) -> dict[str, float | tuple[int, ...]]:
    fractional = dataset.positions_px - np.round(dataset.positions_px)
    return {
        "data_shape": dataset.data.shape,
        "probe_shape": dataset.probe_init.shape,
        "position_shape": dataset.positions_px.shape,
        "max_fractional_position_px": float(np.max(np.abs(fractional))),
        "mean_fractional_position_px": float(np.mean(np.abs(fractional))),
        "dx_m": dataset.dx_m,
        "wavelength_m": dataset.wavelength_m,
    }


def print_real_dataset_summary(dataset: RealPtychographyDataset) -> None:
    for key, value in summarize_real_dataset(dataset).items():
        print(f"{key}: {value}")


def _squeeze_2d(arr: np.ndarray, label: str) -> np.ndarray:
    squeezed = np.squeeze(np.asarray(arr))
    if squeezed.ndim != 2:
        raise ValueError(f"Expected {label} to squeeze to 2D, got {squeezed.shape}.")
    return squeezed


def _demeaned_phase(
    arr: np.ndarray,
    amplitude_mask_fraction: float | None = None,
) -> np.ma.MaskedArray:
    arr = np.asarray(arr)
    phase = np.angle(arr)
    valid = np.isfinite(phase)

    if amplitude_mask_fraction is not None:
        amplitude = np.abs(arr)
        threshold = float(amplitude_mask_fraction) * float(np.nanmax(amplitude))
        valid &= amplitude >= threshold

    if np.any(valid):
        phase_center = np.angle(np.mean(np.exp(1j * phase[valid])))
    else:
        phase_center = 0.0

    phase = np.angle(np.exp(1j * (phase - phase_center)))
    return np.ma.masked_where(~valid, phase)


def _phase_cmap(name: str):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(color="#202020")
    return cmap


def _phase_values(phase_images: list[np.ndarray | np.ma.MaskedArray]) -> np.ndarray:
    values = []
    for phase in phase_images:
        if np.ma.isMaskedArray(phase):
            arr = np.asarray(phase.compressed(), dtype=np.float64)
        else:
            arr = np.asarray(phase, dtype=np.float64).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size:
            values.append(arr)
    if not values:
        return np.array([], dtype=np.float64)
    return np.concatenate(values)


def _phase_limits(
    phase_images: list[np.ndarray | np.ma.MaskedArray],
    phase_vlim_rad: float | None,
    phase_percentile: float,
) -> dict[str, float]:
    if phase_vlim_rad is not None:
        vlim = float(phase_vlim_rad)
    else:
        values = _phase_values(phase_images)
        if values.size == 0:
            vlim = np.pi
        else:
            percentile = float(np.clip(phase_percentile, 50.0, 100.0))
            vlim = float(np.nanpercentile(np.abs(values), percentile))
            if not np.isfinite(vlim) or vlim <= 0:
                vlim = np.pi
    vlim = float(np.clip(vlim, 0.05, np.pi))
    return {"vmin": -vlim, "vmax": vlim}


def plot_real_data_overview(dataset: RealPtychographyDataset) -> None:
    mean_pattern = np.mean(dataset.data, axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)

    im = axes[0].imshow(np.log1p(mean_pattern), cmap="magma")
    axes[0].set_title("mean log diffraction")
    axes[0].axis("off")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    im = axes[1].imshow(np.abs(dataset.probe_init), cmap="magma")
    axes[1].set_title("init |probe|")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].scatter(dataset.positions_px[:, 1], dataset.positions_px[:, 0], s=5)
    axes[2].set_aspect("equal")
    axes[2].invert_yaxis()
    axes[2].set_title("scan positions")
    axes[2].set_xlabel("x px")
    axes[2].set_ylabel("y px")
    plt.show()


def plot_real_loss_comparison(tasks: Mapping[str, PtychographyTask]) -> None:
    plt.figure(figsize=(7, 4))
    for name, task in tasks.items():
        loss_table = task.reconstructor.loss_tracker.table.copy()
        epochs = np.asarray(loss_table["epoch"], dtype=np.float64) + 1
        losses = np.asarray(loss_table["loss"], dtype=np.float64)
        valid = np.isfinite(losses) & (losses > 0)
        plt.loglog(
            epochs[valid],
            losses[valid],
            marker="o",
            lw=1.2,
            ms=3,
            label=name,
        )
    plt.xlabel("epoch")
    plt.ylabel("displayed loss")
    plt.grid(True, which="both", ls=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_real_object_comparison(
    results: Mapping[str, RealReconstructionResult],
    phase_vlim_rad: float | None = None,
    phase_percentile: float = 99.5,
) -> None:
    fig, axes = plt.subplots(2, len(results), figsize=(4 * len(results), 7), constrained_layout=True)
    if len(results) == 1:
        axes = axes[:, None]
    panels = [
        (name, _squeeze_2d(result.object, name))
        for name, result in results.items()
    ]
    phase_images = [_demeaned_phase(obj) for _, obj in panels]
    phase_limits = _phase_limits(phase_images, phase_vlim_rad, phase_percentile)
    phase_cmap = _phase_cmap("viridis")

    for col, ((name, obj), phase) in enumerate(zip(panels, phase_images)):
        im = axes[0, col].imshow(np.abs(obj), cmap="viridis")
        axes[0, col].set_title(f"{name} |object|")
        axes[0, col].axis("off")
        fig.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)

        im = axes[1, col].imshow(phase, cmap=phase_cmap, **phase_limits)
        axes[1, col].set_title(f"{name} demeaned phase")
        axes[1, col].axis("off")
        fig.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)
    plt.show()


def plot_real_probe_comparison(
    probe_init: np.ndarray,
    results: Mapping[str, RealReconstructionResult],
    phase_vlim_rad: float | None = None,
    phase_percentile: float = 99.5,
    phase_mask_fraction: float | None = 0.03,
) -> None:
    panels = [("init", probe_init)]
    panels.extend((name, _squeeze_2d(result.probe, name)) for name, result in results.items())

    fig, axes = plt.subplots(2, len(panels), figsize=(4 * len(panels), 7), constrained_layout=True)
    if len(panels) == 1:
        axes = axes[:, None]
    phase_images = [
        _demeaned_phase(probe, amplitude_mask_fraction=phase_mask_fraction)
        for _, probe in panels
    ]
    phase_limits = _phase_limits(phase_images, phase_vlim_rad, phase_percentile)
    phase_cmap = _phase_cmap("twilight")

    for col, ((name, probe), phase) in enumerate(zip(panels, phase_images)):
        im = axes[0, col].imshow(np.abs(probe), cmap="magma")
        axes[0, col].set_title(f"{name} |probe|")
        axes[0, col].axis("off")
        fig.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)

        im = axes[1, col].imshow(
            phase,
            cmap=phase_cmap,
            **phase_limits,
        )
        axes[1, col].set_title(f"{name} demeaned phase")
        axes[1, col].axis("off")
        fig.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)
    plt.show()
