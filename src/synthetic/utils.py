from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from common import PROJECT_ROOT

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
from numpy.fft import fft2, fftshift
from scipy.ndimage import gaussian_filter
from skimage.transform import resize

import ptychi.api as api
from ptychi.utils import get_default_complex_dtype

from common import configure_ptychi_device, set_random_seed


@dataclass
class ExperimentConfig:
    asset_dir: Path = PROJECT_ROOT / "assets"
    object_size: int = 512
    probe_size: int = 128
    overlap_ratio: float = 0.75
    poisson_eta: float = 0.0
    detector_centered_data: bool = True
    seed: int = 0

    num_epochs: int = 1000
    batch_size: int = 32
    object_extra_pixels: int = 10
    use_mps: bool = True
    print_residual_every: int = 1

    object_alpha: float = 0.1
    object_step_size: float = 1.0
    probe_alpha: float = 1.0
    probe_step_size: float = 1.0

    probe_init_phase_noise_rad: float = 0.05
    probe_init_blur_sigma: float = 2.0
    probe_init_amp_noise: float = 0.05
    probe_init_phase_ramp_rad: float = 0.5

    @property
    def probe_mat_path(self) -> Path:
        return self.asset_dir / "recon_zoneplate_Yi.mat"

    @property
    def baboon_path(self) -> Path:
        return self.asset_dir / "baboon.tiff"

    @property
    def cameraman_path(self) -> Path:
        return self.asset_dir / "cameraman.tif"


@dataclass
class SyntheticDataset:
    truth: np.ndarray
    probe_truth: np.ndarray
    probe_init: np.ndarray
    data: np.ndarray
    top_left_positions: list[tuple[int, int]]
    positions_px: np.ndarray


def _read_gray(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing image file: {path}")

    img = imageio.imread(path).astype(np.float32)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=2)
    return img


def _resize_if_needed(img: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if img.shape == shape:
        return img.astype(np.float32, copy=False)

    return resize(
        img,
        shape,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)


def load_synthetic_object(cfg: ExperimentConfig) -> np.ndarray:
    target_shape = (cfg.object_size, cfg.object_size)
    baboon = _resize_if_needed(_read_gray(cfg.baboon_path), target_shape)
    cameraman = _resize_if_needed(_read_gray(cfg.cameraman_path), target_shape)

    mag = baboon / np.max(baboon)
    phase = (cameraman / np.max(cameraman)) * (np.pi / 2.0)
    return (mag * np.exp(1j * phase)).astype(np.complex64)


def load_probe(cfg: ExperimentConfig) -> np.ndarray:
    if not cfg.probe_mat_path.exists():
        raise FileNotFoundError(f"Missing probe file: {cfg.probe_mat_path}")

    mat_data = scipy.io.loadmat(cfg.probe_mat_path)
    if "probe" not in mat_data:
        raise KeyError(f"{cfg.probe_mat_path} does not contain a variable named 'probe'.")

    original_probe = mat_data["probe"]
    ratio = original_probe.shape[0] // cfg.probe_size
    if ratio < 1:
        raise ValueError(
            f"Probe in {cfg.probe_mat_path} has shape {original_probe.shape}, "
            f"smaller than requested probe_size={cfg.probe_size}."
        )

    probe = original_probe[::ratio, ::ratio][: cfg.probe_size, : cfg.probe_size]
    expected_shape = (cfg.probe_size, cfg.probe_size)
    if probe.shape != expected_shape:
        raise ValueError(f"Downsampled probe has shape {probe.shape}; expected {expected_shape}.")
    return probe.astype(np.complex64)


def make_raster_scan(
    object_shape: tuple[int, int],
    probe_shape: tuple[int, int],
    overlap_ratio: float,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    height, width = object_shape
    probe_height, probe_width = probe_shape

    if not (0.0 <= overlap_ratio < 1.0):
        raise ValueError("overlap_ratio must satisfy 0 <= overlap_ratio < 1.")

    shift_y = int(round(probe_height * (1.0 - overlap_ratio)))
    shift_x = int(round(probe_width * (1.0 - overlap_ratio)))
    if shift_y <= 0 or shift_x <= 0:
        raise ValueError("The scan shift is non-positive; decrease overlap_ratio.")

    top_left_positions = [
        (y, x)
        for y in range(0, height - probe_height + 1, shift_y)
        for x in range(0, width - probe_width + 1, shift_x)
    ]

    y_center = np.array([y + probe_height / 2 for y, _ in top_left_positions])
    x_center = np.array([x + probe_width / 2 for _, x in top_left_positions])
    y_px = y_center - height / 2.0
    x_px = x_center - width / 2.0
    positions_px = np.stack([y_px, x_px], axis=1).astype(np.float32)
    return top_left_positions, positions_px


def generate_intensity_measurements(
    obj: np.ndarray,
    probe: np.ndarray,
    top_left_positions: Iterable[tuple[int, int]],
    poisson_eta: float = 0.0,
    detector_centered: bool = True,
    seed: int | None = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    probe_height, probe_width = probe.shape
    patterns = []

    for y0, x0 in top_left_positions:
        patch = obj[y0 : y0 + probe_height, x0 : x0 + probe_width]
        intensity = np.abs(fft2(probe * patch)) ** 2
        if detector_centered:
            intensity = fftshift(intensity)

        if poisson_eta is not None and poisson_eta > 0:
            lam = np.maximum(intensity / poisson_eta, 0.0)
            intensity = poisson_eta * rng.poisson(lam)

        patterns.append(intensity.astype(np.float32))

    return np.asarray(patterns, dtype=np.float32)


def _probe_power_from_data(data: np.ndarray, probe_shape: tuple[int, int]) -> float:
    return float(np.mean(data.sum(axis=(-2, -1))) / np.prod(probe_shape))


def _power_normalize_probe(
    probe: np.ndarray,
    data: np.ndarray,
    probe_shape: tuple[int, int],
) -> np.ndarray:
    target_power = _probe_power_from_data(data, probe_shape)
    probe = probe.astype(np.complex64, copy=False)
    probe *= np.sqrt(target_power / np.sum(np.abs(probe) ** 2))
    return probe


def make_perturbed_probe_init(
    probe_truth: np.ndarray,
    data: np.ndarray,
    cfg: ExperimentConfig,
) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed)
    probe_init = probe_truth.astype(np.complex64, copy=True)

    if cfg.probe_init_blur_sigma > 0:
        probe_init = (
            gaussian_filter(probe_init.real, sigma=cfg.probe_init_blur_sigma)
            + 1j * gaussian_filter(probe_init.imag, sigma=cfg.probe_init_blur_sigma)
        ).astype(np.complex64)

    if cfg.probe_init_amp_noise > 0:
        amp_noise = gaussian_filter(rng.standard_normal(probe_truth.shape), sigma=3.0)
        amp_noise = amp_noise / np.std(amp_noise)
        probe_init = probe_init * np.clip(1.0 + cfg.probe_init_amp_noise * amp_noise, 0.05, None)

    if cfg.probe_init_phase_noise_rad > 0:
        phase_noise = gaussian_filter(rng.standard_normal(probe_truth.shape), sigma=5.0)
        phase_noise = phase_noise / np.std(phase_noise)
        probe_init = probe_init * np.exp(1j * cfg.probe_init_phase_noise_rad * phase_noise)

    if cfg.probe_init_phase_ramp_rad != 0:
        height, width = probe_truth.shape
        y = np.linspace(-1.0, 1.0, height)
        x = np.linspace(-1.0, 1.0, width)
        yy, xx = np.meshgrid(y, x, indexing="ij")
        ramp = cfg.probe_init_phase_ramp_rad * (0.6 * xx - 0.4 * yy)
        probe_init = probe_init * np.exp(1j * ramp)

    return _power_normalize_probe(probe_init, data, data.shape[-2:]).astype(np.complex64)


def build_synthetic_dataset(cfg: ExperimentConfig) -> SyntheticDataset:
    set_random_seed(cfg.seed)
    probe_truth = load_probe(cfg)
    truth = load_synthetic_object(cfg)
    top_left_positions, positions_px = make_raster_scan(
        object_shape=truth.shape,
        probe_shape=probe_truth.shape,
        overlap_ratio=cfg.overlap_ratio,
    )
    data = generate_intensity_measurements(
        obj=truth,
        probe=probe_truth,
        top_left_positions=top_left_positions,
        poisson_eta=cfg.poisson_eta,
        detector_centered=cfg.detector_centered_data,
        seed=cfg.seed,
    )
    probe_init = make_perturbed_probe_init(probe_truth, data, cfg)
    return SyntheticDataset(truth, probe_truth, probe_init, data, top_left_positions, positions_px)


def _make_complex_object_init(shape: tuple[int, int], amplitude_noise: float = 1e-2) -> torch.Tensor:
    complex_dtype = get_default_complex_dtype()
    real_dtype = torch.float32 if complex_dtype == torch.complex64 else torch.float64
    obj_init = torch.ones((1, *shape), dtype=complex_dtype, device="cpu")
    noise = amplitude_noise * torch.rand((1, *shape), dtype=real_dtype, device="cpu")
    return obj_init + noise.to(complex_dtype)


def center_crop_2d(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    height, width = arr.shape[-2:]
    target_height, target_width = target_shape
    if target_height > height or target_width > width:
        raise ValueError(f"Cannot crop shape {(height, width)} to larger target {target_shape}.")

    y0 = (height - target_height) // 2
    x0 = (width - target_width) // 2
    return arr[..., y0 : y0 + target_height, x0 : x0 + target_width]


def squeeze_object(recon_obj: np.ndarray) -> np.ndarray:
    obj = np.squeeze(np.asarray(recon_obj))
    if obj.ndim != 2:
        raise ValueError(f"Expected a single 2D object after squeeze, got shape {obj.shape}.")
    return obj


def align_complex_scale(recon: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, complex]:
    denom = np.vdot(recon, recon)
    if np.abs(denom) == 0:
        raise ValueError("Cannot align a zero reconstruction.")
    scale = np.vdot(recon, truth) / denom
    return scale * recon, scale


def align_reconstructed_object(
    recon_obj: np.ndarray,
    truth: np.ndarray,
) -> tuple[np.ndarray, complex]:
    recon_2d = squeeze_object(recon_obj)
    recon_crop = center_crop_2d(recon_2d, truth.shape)
    return align_complex_scale(recon_crop, truth)


def summarize_errors(recon_aligned: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    mag_err = np.linalg.norm(np.abs(recon_aligned) - np.abs(truth)) / np.linalg.norm(np.abs(truth))
    complex_err = np.linalg.norm(recon_aligned - truth) / np.linalg.norm(truth)
    phase_err = np.angle(recon_aligned * np.conj(truth))
    phase_rmse = math.sqrt(np.mean(phase_err**2))
    return {
        "relative_magnitude_l2": float(mag_err),
        "relative_complex_l2_after_scale": float(complex_err),
        "phase_rmse_rad": float(phase_rmse),
    }


def summarize_reconstruction_results(
    recon_objects: Mapping[str, np.ndarray],
    truth: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float | complex]]]:
    aligned_objects = {}
    summary = {}
    for name, recon_obj in recon_objects.items():
        aligned, scale = align_reconstructed_object(recon_obj, truth)
        aligned_objects[name] = aligned
        summary[name] = {"alignment_scale": scale, **summarize_errors(aligned, truth)}
    return aligned_objects, summary


def summarize_probe_results(
    recon_probes: Mapping[str, np.ndarray],
    truth_probe: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float | complex]]]:
    aligned_probes = {}
    summary = {}
    for name, recon_probe in recon_probes.items():
        probe_2d = np.squeeze(recon_probe)
        if probe_2d.ndim != 2:
            raise ValueError(f"Expected a 2D probe for {name}, got shape {probe_2d.shape}.")
        aligned, scale = align_complex_scale(probe_2d, truth_probe)
        aligned_probes[name] = aligned
        probe_err = np.linalg.norm(aligned - truth_probe) / np.linalg.norm(truth_probe)
        summary[name] = {
            "alignment_scale": scale,
            "relative_probe_l2_after_scale": float(probe_err),
        }
    return aligned_probes, summary


def _format_metric_value(value: float | complex) -> str:
    if np.iscomplexobj(value):
        value = complex(value)
        return f"{value.real:.6e}{value.imag:+.6e}j"
    return f"{float(value):.6e}"


def print_metric_summary(summary: Mapping[str, Mapping[str, float | complex]]) -> None:
    for name, metrics in summary.items():
        print(name)
        for key, val in metrics.items():
            print(f"  {key}: {_format_metric_value(val)}")


def compute_task_reconstruction_errors(task, dataset: SyntheticDataset) -> dict[str, float]:
    recon_obj = task.reconstructor.parameter_group.object.data.detach().cpu().numpy()
    aligned_obj, _ = align_reconstructed_object(recon_obj, dataset.truth)
    object_errors = summarize_errors(aligned_obj, dataset.truth)

    recon_probe = task.reconstructor.parameter_group.probe.data.detach().cpu().numpy()
    probe_2d = np.squeeze(recon_probe)
    if probe_2d.ndim != 2:
        raise ValueError(f"Expected a 2D probe after squeeze, got shape {probe_2d.shape}.")
    aligned_probe, _ = align_complex_scale(probe_2d, dataset.probe_truth)
    probe_err = np.linalg.norm(aligned_probe - dataset.probe_truth) / np.linalg.norm(
        dataset.probe_truth
    )

    return {
        "object_relative_magnitude_l2": object_errors["relative_magnitude_l2"],
        "object_relative_complex_l2_after_scale": object_errors[
            "relative_complex_l2_after_scale"
        ],
        "object_phase_rmse_rad": object_errors["phase_rmse_rad"],
        "probe_relative_l2_after_scale": float(probe_err),
    }


def attach_reconstruction_error_tracker(task, dataset: SyntheticDataset) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    reconstructor = task.reconstructor
    original_run_post_epoch_hooks = reconstructor.run_post_epoch_hooks

    def run_post_epoch_hooks_with_error_tracking() -> None:
        original_run_post_epoch_hooks()
        history.append(
            {
                "epoch": int(reconstructor.current_epoch),
                **compute_task_reconstruction_errors(task, dataset),
            }
        )

    reconstructor.run_post_epoch_hooks = run_post_epoch_hooks_with_error_tracking
    reconstructor.ground_truth_error_history = history
    return history


def plot_reconstruction(recon_aligned: np.ndarray, truth: np.ndarray) -> None:
    mag_err = np.abs(recon_aligned) - np.abs(truth)
    phase_err = np.angle(recon_aligned * np.conj(truth))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    panels = [
        (np.abs(truth), "truth |object|", "viridis"),
        (np.abs(recon_aligned), "blind rPIE |object|", "viridis"),
        (mag_err, "magnitude error", "seismic"),
        (np.angle(truth), "truth phase", "twilight"),
        (np.angle(recon_aligned), "blind rPIE phase", "twilight"),
        (phase_err, "relative phase error", "twilight"),
    ]

    for ax, (image, title, cmap) in zip(axes.ravel(), panels):
        im = ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.show()


def plot_reconstruction_comparison(
    truth: np.ndarray,
    aligned_objects: Mapping[str, np.ndarray],
    phase_vlim_rad: float | None = 0.75,
) -> None:
    panels = [("truth", truth), *aligned_objects.items()]
    fig, axes = plt.subplots(2, len(panels), figsize=(4 * len(panels), 7), constrained_layout=True)
    if len(panels) == 1:
        axes = axes[:, None]
    phase_limits = (
        {"vmin": -np.pi, "vmax": np.pi}
        if phase_vlim_rad is None
        else {"vmin": -float(phase_vlim_rad), "vmax": float(phase_vlim_rad)}
    )

    for col, (name, obj) in enumerate(panels):
        im = axes[0, col].imshow(np.abs(obj), cmap="viridis")
        axes[0, col].set_title(f"{name} |object|")
        axes[0, col].axis("off")
        fig.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)

        phase = np.angle(obj)
        phase = np.angle(np.exp(1j * (phase - np.mean(phase))))
        im = axes[1, col].imshow(phase, cmap="viridis", **phase_limits)
        axes[1, col].set_title(f"{name} demeaned phase")
        axes[1, col].axis("off")
        fig.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)

    plt.show()


def plot_probe_comparison(
    probe_init: np.ndarray,
    recon_probe: np.ndarray,
    truth_probe: np.ndarray,
) -> None:
    recon_probe_2d = np.squeeze(recon_probe)
    recon_probe_aligned, _ = align_complex_scale(recon_probe_2d, truth_probe)
    init_probe_aligned, _ = align_complex_scale(probe_init, truth_probe)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    panels = [
        (np.abs(truth_probe), "truth |probe|", "magma"),
        (np.abs(init_probe_aligned), "init |probe|", "magma"),
        (np.abs(recon_probe_aligned), "recon |probe|", "magma"),
        (np.angle(truth_probe), "truth phase", "twilight"),
        (np.angle(init_probe_aligned), "init phase", "twilight"),
        (np.angle(recon_probe_aligned), "recon phase", "twilight"),
    ]

    for ax, (image, title, cmap) in zip(axes.ravel(), panels):
        im = ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.show()

    probe_err = np.linalg.norm(recon_probe_aligned - truth_probe) / np.linalg.norm(truth_probe)
    print(f"relative_probe_l2_after_scale: {probe_err:.6e}")


def plot_probe_result_comparison(
    probe_init: np.ndarray,
    aligned_probes: Mapping[str, np.ndarray],
    truth_probe: np.ndarray,
) -> None:
    init_probe_aligned, _ = align_complex_scale(probe_init, truth_probe)
    panels = [("truth", truth_probe), ("init", init_probe_aligned), *aligned_probes.items()]
    fig, axes = plt.subplots(2, len(panels), figsize=(4 * len(panels), 7), constrained_layout=True)
    if len(panels) == 1:
        axes = axes[:, None]

    for col, (name, probe) in enumerate(panels):
        im = axes[0, col].imshow(np.abs(probe), cmap="magma")
        axes[0, col].set_title(f"{name} |probe|")
        axes[0, col].axis("off")
        fig.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)

        im = axes[1, col].imshow(np.angle(probe), cmap="twilight")
        axes[1, col].set_title(f"{name} phase")
        axes[1, col].axis("off")
        fig.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)

    plt.show()


def plot_probe_initialization(probe_init: np.ndarray, truth_probe: np.ndarray) -> None:
    init_probe_aligned, _ = align_complex_scale(probe_init, truth_probe)
    fig, axes = plt.subplots(2, 2, figsize=(8, 7), constrained_layout=True)
    panels = [
        (np.abs(truth_probe), "truth |probe|", "magma"),
        (np.abs(init_probe_aligned), "init |probe|", "magma"),
        (np.angle(truth_probe), "truth phase", "twilight"),
        (np.angle(init_probe_aligned), "init phase", "twilight"),
    ]

    for ax, (image, title, cmap) in zip(axes.ravel(), panels):
        im = ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.show()

    init_err = np.linalg.norm(init_probe_aligned - truth_probe) / np.linalg.norm(truth_probe)
    print(f"relative_initial_probe_l2_after_scale: {init_err:.6e}")


def plot_loss(task) -> None:
    loss_table = task.reconstructor.loss_tracker.table.copy()
    plt.figure(figsize=(6, 4))
    plt.semilogy(loss_table["epoch"] + 1, loss_table["loss"], marker="o", lw=1.2, ms=3)
    plt.xlabel("epoch")
    plt.ylabel("displayed loss")
    plt.grid(True, which="both", ls=":", alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_loss_comparison(tasks: Mapping[str, object]) -> None:
    plt.figure(figsize=(7, 4))
    for name, task in tasks.items():
        loss_table = task.reconstructor.loss_tracker.table.copy()
        plt.loglog(
            loss_table["epoch"] + 1,
            loss_table["loss"],
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


def _get_error_history(result_or_task) -> list[dict[str, float]]:
    task = getattr(result_or_task, "task", result_or_task)
    history = getattr(task.reconstructor, "ground_truth_error_history", None)
    if history is None:
        history = getattr(result_or_task, "error_history", None)
    if history is None:
        raise ValueError(
            "No ground-truth error history found. Re-run the reconstruction cells "
            "so the epoch-end tracker can record errors."
        )
    return history


def plot_error_comparison(
    results_or_tasks: Mapping[str, object],
    metric: str = "object_relative_complex_l2_after_scale",
) -> None:
    metric_labels = {
        "object_relative_magnitude_l2": "object relative magnitude L2",
        "object_relative_complex_l2_after_scale": "object relative complex L2 after scale",
        "object_phase_rmse_rad": "object phase RMSE (rad)",
        "probe_relative_l2_after_scale": "probe relative L2 after scale",
    }

    plt.figure(figsize=(7, 4))
    for name, result_or_task in results_or_tasks.items():
        history = _get_error_history(result_or_task)
        epochs = np.asarray([row["epoch"] + 1 for row in history], dtype=np.float64)
        values = np.asarray([row[metric] for row in history], dtype=np.float64)
        valid = np.isfinite(values) & (values > 0)
        plt.loglog(
            epochs[valid],
            values[valid],
            marker="o",
            lw=1.2,
            ms=3,
            label=name,
        )

    plt.xlabel("epoch")
    plt.ylabel(metric_labels.get(metric, metric))
    plt.grid(True, which="both", ls=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
