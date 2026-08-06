from __future__ import annotations

from typing import TYPE_CHECKING

import ptychi.api as api
from ptychi.api.task import PtychographyTask

from utils.reconstruction import (
    ProgressCallback,
    ReconstructionResult,
    build_ptychi_options,
    make_complex_gaussian_object_init,
    run_reconstruction_task,
)

if TYPE_CHECKING:
    from utils.synthetic import ExperimentConfig, SyntheticDataset


def build_rpie_options(**kwargs) -> api.RPIEOptions:
    return build_ptychi_options(api.RPIEOptions, **kwargs)


def build_synthetic_rpie_options(
    dataset: "SyntheticDataset",
    cfg: "ExperimentConfig",
    device: api.Devices,
    seed: int | None = None,
) -> api.RPIEOptions:
    reconstruction_seed = cfg.reconstruction_seed if seed is None else seed
    options = build_rpie_options(
        data=dataset.data,
        positions_px=dataset.positions_px,
        probe_init=dataset.probe_init,
        make_object_initial_guess=make_complex_gaussian_object_init,
        device=device,
        seed=reconstruction_seed,
        fft_shift_data=cfg.detector_centered_data,
        save_data_on_device=False,
        object_extra_pixels=cfg.object_extra_pixels,
        batch_size=cfg.batch_size,
        num_epochs=cfg.num_epochs,
        object_step_size=cfg.object_step_size,
        probe_step_size=cfg.probe_step_size,
        object_alpha=cfg.object_alpha,
        probe_alpha=cfg.probe_alpha,
        remove_object_probe_ambiguity=cfg.remove_object_probe_ambiguity,
        pad_for_shift=0,
    )
    options.reconstructor_options.batching_mode = api.BatchingModes.RANDOM
    options.object_options.remove_object_probe_ambiguity.optimization_plan.stride = 1
    for parameter_options in (
        options.object_options,
        options.probe_options,
    ):
        parameter_options.optimizable = True
        parameter_options.optimizer = api.Optimizers.SGD
        parameter_options.optimizer_params = {}
        parameter_options.optimization_plan.start = 0
        parameter_options.optimization_plan.stop = None
        parameter_options.optimization_plan.stride = 1
        parameter_options.optimization_plan.step_size_scheduler_class = None
        parameter_options.optimization_plan.step_size_scheduler_options = {}

    options.probe_options.orthogonalize_incoherent_modes.enabled = False
    options.probe_options.orthogonalize_opr_modes.enabled = False
    options.opr_mode_weight_options.optimizable = False
    return options


def run_rpie(
    dataset: "SyntheticDataset",
    cfg: "ExperimentConfig",
    device: api.Devices,
    seed: int | None = None,
    error_stride: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ReconstructionResult:
    reconstruction_seed = cfg.reconstruction_seed if seed is None else seed
    task = PtychographyTask(
        build_synthetic_rpie_options(dataset, cfg, device, seed=seed)
    )
    shuffle_generator = task.reconstructor.dataloader.generator
    if shuffle_generator is None:
        raise RuntimeError("rPIE random batching requires a DataLoader generator.")
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
