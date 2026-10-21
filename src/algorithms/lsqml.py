from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import ptychi.api as api
import ptychi.data_structures.parameter_group as paramgrp
from ptychi.api.task import PtychographyTask
from ptychi.reconstructors.lsqml import (
    LSQMLReconstructor,
    MultiprocessLSQMLReconstructor,
)

from algorithms.probe_shift import ReplicatePaddedProbeShiftMixin
from utils.reconstruction import (
    FinalMetricFunction,
    ProgressCallback,
    ReconstructionResult,
    TaskMetricFunction,
    build_ptychi_options,
    make_complex_gaussian_object_init,
    run_reconstruction_task,
    set_if_field,
)

if TYPE_CHECKING:
    from utils.synthetic import ExperimentConfig, SyntheticDataset


class ReplicatePaddedLSQMLReconstructor(
    ReplicatePaddedProbeShiftMixin,
    LSQMLReconstructor,
):
    """LSQML with the fixed replicate-padded probe shift and exact adjoint."""


class ReplicatePaddedMultiprocessLSQMLReconstructor(
    ReplicatePaddedProbeShiftMixin,
    MultiprocessLSQMLReconstructor,
):
    """Multiprocess LSQML with the fixed replicate-padded probe shift."""


class ReplicatePaddedLSQMLTask(PtychographyTask):
    """Use fixed replicate-padded LSQML for one or multiple CUDA processes."""

    def build_reconstructor(self) -> None:
        parameter_group = paramgrp.PlanarPtychographyParameterGroup(
            object=self.object,
            probe=self.probe,
            probe_positions=self.probe_positions,
            opr_mode_weights=self.opr_mode_weights,
        )
        if self.n_ranks == 1:
            reconstructor_class = ReplicatePaddedLSQMLReconstructor
        else:
            reconstructor_class = ReplicatePaddedMultiprocessLSQMLReconstructor
        self.reconstructor = reconstructor_class(
            parameter_group=parameter_group,
            dataset=self.dataset,
            options=self.reconstructor_options,
        )
        self.reconstructor.build()


def _normalize_noise_model(noise_model: api.NoiseModels | str) -> api.NoiseModels:
    if isinstance(noise_model, api.NoiseModels):
        return noise_model
    return api.NoiseModels(str(noise_model))


@dataclass(frozen=True)
class LSQMLHyperparameters:
    object_step_size_scaler: float
    probe_step_size_scaler: float
    noise_model: api.NoiseModels | str
    gaussian_noise_std: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("object_step_size_scaler", self.object_step_size_scaler),
            ("probe_step_size_scaler", self.probe_step_size_scaler),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        noise_model = _normalize_noise_model(self.noise_model)
        object.__setattr__(self, "noise_model", noise_model)
        if noise_model == api.NoiseModels.GAUSSIAN:
            if (
                self.gaussian_noise_std is None
                or not math.isfinite(self.gaussian_noise_std)
                or self.gaussian_noise_std <= 0
            ):
                raise ValueError(
                    "gaussian_noise_std must be finite and positive for the "
                    "Gaussian noise model."
                )
        elif self.gaussian_noise_std is not None:
            raise ValueError(
                "gaussian_noise_std must be None for the Poisson noise model."
            )


def build_lsqml_options(
    *,
    noise_model: api.NoiseModels | str,
    object_optimal_step_size_scaler: float,
    probe_optimal_step_size_scaler: float,
    gaussian_noise_std: float | None = None,
    **kwargs,
) -> api.LSQMLOptions:
    noise_model = _normalize_noise_model(noise_model)
    if noise_model == api.NoiseModels.GAUSSIAN:
        if (
            gaussian_noise_std is None
            or not math.isfinite(gaussian_noise_std)
            or gaussian_noise_std <= 0
        ):
            raise ValueError(
                "gaussian_noise_std must be finite and positive for the "
                "Gaussian noise model."
            )
    elif gaussian_noise_std is not None:
        raise ValueError("gaussian_noise_std must be None for the Poisson noise model.")

    options = build_ptychi_options(api.LSQMLOptions, **kwargs)
    set_if_field(
        options.reconstructor_options,
        "noise_model",
        noise_model,
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
    if noise_model == api.NoiseModels.GAUSSIAN:
        set_if_field(
            options.reconstructor_options,
            "gaussian_noise_std",
            gaussian_noise_std,
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
        # Preserve the original synthetic benchmark defaults while requiring
        # direct LSQMLHyperparameters construction to state its model/scalers.
        hyperparameters = LSQMLHyperparameters(
            object_step_size_scaler=0.9,
            probe_step_size_scaler=0.9,
            noise_model=api.NoiseModels.POISSON,
        )

    options = build_lsqml_options(
        data=dataset.data,
        positions_px=dataset.positions_px,
        probe_init=dataset.probe_init,
        make_object_initial_guess=make_complex_gaussian_object_init,
        device=device,
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
    task_metric_function: TaskMetricFunction | None = None,
    final_metric_function: FinalMetricFunction | None = None,
    include_initial_metrics: bool = False,
) -> ReconstructionResult:
    reconstruction_seed = cfg.reconstruction_seed if seed is None else seed
    task = ReplicatePaddedLSQMLTask(
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
        if progress_callback is not None or task_metric_function is not None:
            raise ValueError(
                "progress_callback and task_metric_function require error_stride."
            )
        return run_reconstruction_task(
            task,
            final_metric_function=final_metric_function,
            include_initial_metrics=include_initial_metrics,
        )

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
        task_metric_function=task_metric_function,
        final_metric_function=final_metric_function,
        include_initial_metrics=include_initial_metrics,
    )
