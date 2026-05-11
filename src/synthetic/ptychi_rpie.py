from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.utils import get_suggested_object_size

from common import patch_ptychi_compatibility
from common import attach_residual_printer
from synthetic.utils import (
    ExperimentConfig,
    SyntheticDataset,
    _make_complex_object_init,
    attach_reconstruction_error_tracker,
    set_random_seed,
)


@dataclass
class ReconstructionResult:
    object: np.ndarray
    probe: np.ndarray
    task: PtychographyTask
    error_history: list[dict[str, float]]


def build_rpie_options(
    dataset: SyntheticDataset,
    cfg: ExperimentConfig,
    device: api.Devices,
    seed: int | None = None,
) -> api.RPIEOptions:
    patch_ptychi_compatibility()
    reconstruction_seed = cfg.seed if seed is None else seed
    set_random_seed(reconstruction_seed)
    probe_0 = dataset.probe_init.astype(np.complex64, copy=True)
    obj_shape = get_suggested_object_size(
        dataset.positions_px,
        probe_0.shape[-2:],
        extra=cfg.object_extra_pixels,
    )

    options = api.RPIEOptions()
    options.data_options.data = dataset.data
    options.data_options.fft_shift = cfg.detector_centered_data
    options.data_options.save_data_on_device = False

    options.object_options.initial_guess = _make_complex_object_init(tuple(obj_shape))
    options.object_options.optimizable = True
    options.object_options.optimizer = api.Optimizers.SGD
    options.object_options.step_size = cfg.object_step_size
    options.object_options.alpha = cfg.object_alpha
    options.object_options.remove_object_probe_ambiguity.enabled = True
    options.object_options.determine_position_origin_coords_by = (
        api.ObjectPosOriginCoordsMethods.POSITIONS
    )

    options.probe_options.initial_guess = probe_0[None, None, :, :]
    options.probe_options.optimizable = True
    options.probe_options.optimizer = api.Optimizers.SGD
    options.probe_options.step_size = cfg.probe_step_size
    options.probe_options.alpha = cfg.probe_alpha
    options.probe_options.power_constraint.enabled = False
    options.probe_options.support_constraint.enabled = False
    options.probe_options.center_constraint.enabled = False

    options.probe_position_options.position_x_px = dataset.positions_px[:, 1]
    options.probe_position_options.position_y_px = dataset.positions_px[:, 0]
    options.probe_position_options.optimizable = False

    options.reconstructor_options.default_device = device
    options.reconstructor_options.batch_size = cfg.batch_size
    options.reconstructor_options.num_epochs = cfg.num_epochs
    options.reconstructor_options.random_seed = reconstruction_seed
    options.reconstructor_options.allow_nondeterministic_algorithms = False

    return options


def run_rpie(
    dataset: SyntheticDataset,
    cfg: ExperimentConfig,
    device: api.Devices,
    seed: int | None = None,
) -> ReconstructionResult:
    task = PtychographyTask(build_rpie_options(dataset, cfg, device, seed=seed))
    error_history = attach_reconstruction_error_tracker(task, dataset)
    print_final_residual = attach_residual_printer(
        task,
        "rPIE",
        cfg.print_residual_every,
        cfg.num_epochs,
    )
    task.run()
    print_final_residual()

    return ReconstructionResult(
        object=task.get_data_to_cpu("object", as_numpy=True),
        probe=task.get_data_to_cpu("probe", as_numpy=True),
        task=task,
        error_history=error_history,
    )
