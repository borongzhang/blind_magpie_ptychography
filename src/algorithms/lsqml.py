from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import ptychi.api as api
from ptychi.api.task import PtychographyTask
import torch

from utils.reconstruction import (
    ProgressCallback,
    ReconstructionResult,
    build_ptychi_options,
    make_complex_gaussian_object_init,
    run_reconstruction_task,
    set_if_field,
)

if TYPE_CHECKING:
    from utils.synthetic import ExperimentConfig, SyntheticDataset


def _normalize_noise_model(noise_model: api.NoiseModels | str) -> api.NoiseModels:
    if isinstance(noise_model, api.NoiseModels):
        return noise_model
    return api.NoiseModels(str(noise_model))


@dataclass(frozen=True)
class LSQMLHyperparameters:
    object_step_size_scaler: float = 0.9
    probe_step_size_scaler: float = 0.9
    gaussian_noise_std: float = 0.5
    noise_model: api.NoiseModels | str = api.NoiseModels.POISSON

    def __post_init__(self) -> None:
        for name, value in (
            ("object_step_size_scaler", self.object_step_size_scaler),
            ("probe_step_size_scaler", self.probe_step_size_scaler),
            ("gaussian_noise_std", self.gaussian_noise_std),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        object.__setattr__(
            self, "noise_model", _normalize_noise_model(self.noise_model)
        )


def safe_lsqml_device(device: api.Devices) -> api.Devices:
    if device == api.Devices.GPU and torch.backends.mps.is_available():
        print("LSQML: using CPU fallback for Apple MPS compatibility.")
        return api.Devices.CPU
    return device


def build_lsqml_options(
    *,
    noise_model: api.NoiseModels | str,
    object_optimal_step_size_scaler: float,
    probe_optimal_step_size_scaler: float,
    gaussian_noise_std: float,
    **kwargs,
) -> api.LSQMLOptions:
    options = build_ptychi_options(api.LSQMLOptions, **kwargs)
    set_if_field(
        options.reconstructor_options,
        "noise_model",
        _normalize_noise_model(noise_model),
    )
    set_if_field(
        options.object_options,
        "optimal_step_size_scaler",
        object_optimal_step_size_scaler,
    )
    set_if_field(
        options.probe_options,
        "optimal_step_size_scaler",
        probe_optimal_step_size_scaler,
    )
    set_if_field(
        options.reconstructor_options, "gaussian_noise_std", gaussian_noise_std
    )
    options.reconstructor_options.batching_mode = api.BatchingModes.RANDOM
    options.reconstructor_options.rescale_probe_intensity_in_first_epoch = False
    options.probe_options.orthogonalize_incoherent_modes.enabled = False
    options.probe_options.orthogonalize_opr_modes.enabled = False
    options.opr_mode_weight_options.optimizable = False
    return options


def build_synthetic_lsqml_options(
    dataset: "SyntheticDataset",
    cfg: "ExperimentConfig",
    device: api.Devices,
    seed: int | None = None,
    hyperparameters: LSQMLHyperparameters | None = None,
) -> api.LSQMLOptions:
    if hyperparameters is None:
        hyperparameters = LSQMLHyperparameters()

    options = build_lsqml_options(
        data=dataset.data,
        positions_px=dataset.positions_px,
        probe_init=dataset.probe_init,
        make_object_initial_guess=make_complex_gaussian_object_init,
        device=safe_lsqml_device(device),
        seed=cfg.reconstruction_seed if seed is None else seed,
        fft_shift_data=cfg.detector_centered_data,
        save_data_on_device=False,
        object_extra_pixels=cfg.object_extra_pixels,
        batch_size=cfg.batch_size,
        num_epochs=cfg.num_epochs,
        object_step_size=cfg.object_step_size,
        probe_step_size=cfg.probe_step_size,
        noise_model=hyperparameters.noise_model,
        object_optimal_step_size_scaler=hyperparameters.object_step_size_scaler,
        probe_optimal_step_size_scaler=hyperparameters.probe_step_size_scaler,
        gaussian_noise_std=hyperparameters.gaussian_noise_std,
        remove_object_probe_ambiguity=cfg.remove_object_probe_ambiguity,
        pad_for_shift=0,
    )
    options.object_options.remove_object_probe_ambiguity.optimization_plan.stride = 1
    return options


def run_lsqml(
    dataset: "SyntheticDataset",
    cfg: "ExperimentConfig",
    device: api.Devices,
    seed: int | None = None,
    hyperparameters: LSQMLHyperparameters | None = None,
    error_stride: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ReconstructionResult:
    reconstruction_seed = cfg.reconstruction_seed if seed is None else seed
    task = PtychographyTask(
        build_synthetic_lsqml_options(
            dataset,
            cfg,
            device,
            seed=reconstruction_seed,
            hyperparameters=hyperparameters,
        )
    )
    shuffle_generator = task.reconstructor.dataloader.generator
    if shuffle_generator is not None:
        shuffle_generator.manual_seed(reconstruction_seed)

    if error_stride is None:
        if progress_callback is not None:
            raise ValueError("progress_callback requires error_stride.")
        return run_reconstruction_task(task)

    from utils.synthetic import score_blind_reconstruction

    def score_errors(
        recon_object,
        recon_probe,
    ) -> dict[str, float]:
        metrics, _, _ = score_blind_reconstruction(
            recon_object,
            recon_probe,
            dataset,
        )
        return metrics

    return run_reconstruction_task(
        task,
        metric_function=score_errors,
        metric_stride=error_stride,
        progress_callback=progress_callback,
    )
