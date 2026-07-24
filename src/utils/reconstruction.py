from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from dataclasses import fields
from typing import Any

import numpy as np
import torch

from utils.common import patch_ptychi_compatibility, set_random_seed

import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.utils import get_suggested_object_size

ProgressCallback = Callable[[int, float, Mapping[str, float]], None]


def _make_noisy_constant_object_init(
    shape: tuple[int, int],
    center: float,
) -> torch.Tensor:
    real = torch.randn((1, *shape), dtype=torch.float32, device="cpu")
    imaginary = torch.randn((1, *shape), dtype=torch.float32, device="cpu")
    noise = torch.complex(real, imaginary) * (1e-2 * 2.0**-0.5)
    return torch.full_like(noise, center) + noise


def make_complex_gaussian_object_init(shape: tuple[int, int]) -> torch.Tensor:
    """Initialize a synthetic object around +1 with complex Gaussian noise."""
    return _make_noisy_constant_object_init(shape, center=1.0)


def make_negative_complex_gaussian_object_init(
    shape: tuple[int, int],
) -> torch.Tensor:
    """Initialize a real-data object around -1 with complex Gaussian noise."""
    return _make_noisy_constant_object_init(shape, center=-1.0)


@dataclass
class ReconstructionResult:
    object: np.ndarray
    probe: np.ndarray
    residual_history: np.ndarray
    metric_epochs: np.ndarray
    metric_history: dict[str, np.ndarray]


def _has_field(options_group: object, name: str) -> bool:
    try:
        return name in {item.name for item in fields(options_group)}
    except TypeError:
        return hasattr(options_group, name)


def set_if_field(options_group: object, name: str, value: object) -> None:
    if value is not None and _has_field(options_group, name):
        setattr(options_group, name, value)


def build_ptychi_options(
    options_factory: Callable[[], Any],
    *,
    data: np.ndarray,
    valid_pixel_mask: np.ndarray | None = None,
    positions_px: np.ndarray,
    probe_init: np.ndarray,
    make_object_initial_guess: Callable[[tuple[int, int]], object],
    device: api.Devices,
    seed: int,
    fft_shift_data: bool,
    save_data_on_device: bool,
    object_extra_pixels: int,
    batch_size: int,
    num_epochs: int,
    object_step_size: float,
    probe_step_size: float,
    object_alpha: float | None = None,
    probe_alpha: float | None = None,
    object_pixel_size_m: float | None = None,
    wavelength_m: float | None = None,
    remove_object_probe_ambiguity: bool = False,
    probe_update_start_epoch: int | None = None,
    probe_update_stride: int | None = None,
    pad_for_shift: int | None = None,
) -> Any:
    patch_ptychi_compatibility()
    set_random_seed(seed)

    data = np.asarray(data, dtype=np.float32)
    probe_init = probe_init.astype(np.complex64, copy=True)
    positions_px = np.asarray(positions_px, dtype=np.float32)
    obj_shape = get_suggested_object_size(
        positions_px,
        probe_init.shape[-2:],
        extra=object_extra_pixels,
    )

    options = options_factory()
    options.data_options.data = data
    options.data_options.valid_pixel_mask = valid_pixel_mask
    options.data_options.fft_shift = fft_shift_data
    options.data_options.save_data_on_device = save_data_on_device
    set_if_field(options.data_options, "wavelength_m", wavelength_m)

    options.object_options.initial_guess = make_object_initial_guess(tuple(obj_shape))
    options.object_options.optimizable = True
    options.object_options.optimizer = api.Optimizers.SGD
    options.object_options.step_size = object_step_size
    set_if_field(options.object_options, "alpha", object_alpha)
    set_if_field(options.object_options, "pixel_size_m", object_pixel_size_m)
    if _has_field(options.object_options, "remove_object_probe_ambiguity"):
        options.object_options.remove_object_probe_ambiguity.enabled = (
            remove_object_probe_ambiguity
        )
    options.object_options.determine_position_origin_coords_by = (
        api.ObjectPosOriginCoordsMethods.POSITIONS
    )

    options.probe_options.initial_guess = probe_init[None, None, :, :]
    options.probe_options.optimizable = True
    options.probe_options.optimizer = api.Optimizers.SGD
    options.probe_options.step_size = probe_step_size
    set_if_field(options.probe_options, "alpha", probe_alpha)
    if probe_update_start_epoch is not None:
        options.probe_options.optimization_plan.start = probe_update_start_epoch
    if probe_update_stride is not None:
        options.probe_options.optimization_plan.stride = probe_update_stride
    options.probe_options.power_constraint.enabled = False
    options.probe_options.support_constraint.enabled = False
    options.probe_options.center_constraint.enabled = False

    options.probe_position_options.position_x_px = positions_px[:, 1]
    options.probe_position_options.position_y_px = positions_px[:, 0]
    options.probe_position_options.optimizable = False

    options.reconstructor_options.default_device = device
    options.reconstructor_options.default_dtype = api.Dtypes.FLOAT32
    options.reconstructor_options.use_double_precision_for_fft = False
    options.reconstructor_options.batch_size = batch_size
    options.reconstructor_options.num_epochs = num_epochs
    options.reconstructor_options.random_seed = seed
    options.reconstructor_options.allow_nondeterministic_algorithms = False
    set_if_field(
        options.reconstructor_options.forward_model_options,
        "pad_for_shift",
        pad_for_shift,
    )

    return options


def run_reconstruction_task(
    task: PtychographyTask,
    *,
    metric_function: (
        Callable[[np.ndarray | None, np.ndarray | None], Mapping[str, float]] | None
    ) = None,
    metric_stride: int = 1,
    progress_callback: ProgressCallback | None = None,
    metric_requires_reconstruction_arrays: bool = True,
) -> ReconstructionResult:
    """Run a task and optionally report progress or sample metrics."""
    metric_epochs: list[int] = []
    metric_values: dict[str, list[float]] = {}

    if metric_function is None and progress_callback is None:
        task.run()
    else:
        if metric_stride < 1:
            raise ValueError("metric_stride must be positive.")

        num_epochs = int(task.reconstructor.n_epochs)
        for epoch in range(1, num_epochs + 1):
            task.run(n_epochs=1)
            if epoch != 1 and epoch % metric_stride != 0 and epoch != num_epochs:
                continue

            metrics: dict[str, float] = {}
            if metric_function is not None:
                if metric_requires_reconstruction_arrays:
                    current_object = task.get_data_to_cpu("object", as_numpy=True)
                    current_probe = task.get_data_to_cpu("probe", as_numpy=True)
                else:
                    current_object = None
                    current_probe = None
                metrics = {
                    name: float(value)
                    for name, value in metric_function(
                        current_object,
                        current_probe,
                    ).items()
                }
                if not metric_values:
                    metric_values = {name: [] for name in metrics}
                elif set(metrics) != set(metric_values):
                    raise ValueError(
                        "metric_function returned inconsistent metric names."
                    )

                metric_epochs.append(epoch)
                for name, value in metrics.items():
                    metric_values[name].append(value)

            if progress_callback is not None:
                sampled_losses = np.asarray(
                    task.reconstructor.loss_tracker.table["loss"]
                )
                residual = float(sampled_losses[-1])
                progress_callback(epoch, residual, metrics)

    residual_history = np.asarray(
        task.reconstructor.loss_tracker.table["loss"],
        dtype=np.float32,
    )
    return ReconstructionResult(
        object=task.get_data_to_cpu("object", as_numpy=True),
        probe=task.get_data_to_cpu("probe", as_numpy=True),
        residual_history=residual_history,
        metric_epochs=np.asarray(metric_epochs, dtype=np.int64),
        metric_history={
            name: np.asarray(values, dtype=np.float32)
            for name, values in metric_values.items()
        },
    )
