from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
import math
from numbers import Integral
from typing import Any

import numpy as np
import torch

from algorithms.probe_shift import REPLICATE_PADDING_PIXELS
from utils.common import patch_ptychi_compatibility, set_random_seed

import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.utils import get_suggested_object_size

ProgressCallback = Callable[[int, float, Mapping[str, float]], None]
TaskMetricFunction = Callable[[PtychographyTask], Mapping[str, float]]
FinalMetricFunction = TaskMetricFunction
StateSnapshotCallback = Callable[[int, np.ndarray, np.ndarray], None]


def _coerce_finite_metric_mapping(
    evaluated: object,
    *,
    source: str,
) -> dict[str, float]:
    if not isinstance(evaluated, Mapping):
        raise TypeError(f"{source} must return a mapping.")
    metrics: dict[str, float] = {}
    for name, value in evaluated.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{source} metric names must be nonempty strings.")
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{source} metric values must be real numbers.")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"{source} metric values must be real numbers."
            ) from error
        if not math.isfinite(numeric_value):
            raise ValueError(f"{source} metric values must be finite.")
        metrics[name] = numeric_value
    return metrics


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
    final_metrics: dict[str, float] = field(default_factory=dict)


def make_frozen_amplitude_mse_evaluator(
    intensity_targets: Mapping[str, np.ndarray | torch.Tensor],
    *,
    batch_size: int,
    eps: float = 1e-7,
    targets_are_internal: bool = False,
) -> FinalMetricFunction:
    """Build a static all-frame amplitude-MSE evaluator for a task.

    Target arrays normally use the same detector ordering as the arrays passed
    into the task options. The evaluator applies the task's configured FFT
    shift before comparison with its forward model. Set
    ``targets_are_internal=True`` only for arrays already taken from
    ``task.dataset.patterns`` or transformed equivalently.

    The returned function evaluates the task's current object and probe without
    updating them. It visits every scan index in order and transfers at most
    ``batch_size`` target frames to the prediction device at once.
    """
    targets = dict(intensity_targets)
    if not targets:
        raise ValueError("intensity_targets must not be empty.")
    if any(not isinstance(name, str) or not name for name in targets):
        raise ValueError("final metric names must be nonempty strings.")
    if any(
        not isinstance(value, (np.ndarray, torch.Tensor)) for value in targets.values()
    ):
        raise TypeError("intensity targets must be NumPy arrays or Torch tensors.")
    if any(value.ndim < 1 for value in targets.values()):
        raise ValueError("intensity targets must include a scan dimension.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
        raise TypeError("batch_size must be an integer.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    batch_size = int(batch_size)
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and nonnegative.")
    if not isinstance(targets_are_internal, bool):
        raise TypeError("targets_are_internal must be boolean.")

    @torch.no_grad()
    def evaluate(task: PtychographyTask) -> dict[str, float]:
        num_patterns = len(task.dataset)
        if num_patterns < 1:
            raise ValueError("cannot evaluate an empty diffraction dataset.")
        if any(len(target) != num_patterns for target in targets.values()):
            raise ValueError("every target must contain one array per scan position.")

        apply_fft_shift = bool(
            not targets_are_internal
            and getattr(getattr(task, "data_options", None), "fft_shift", False)
        )
        squared_error_sums = {name: 0.0 for name in targets}
        element_counts = {name: 0 for name in targets}

        for start in range(0, num_patterns, batch_size):
            stop = min(start + batch_size, num_patterns)
            indices = torch.arange(start, stop, dtype=torch.long, device="cpu")
            prediction = task.reconstructor.forward_model.forward(indices)
            if torch.is_complex(prediction):
                raise TypeError("frozen forward predictions must be real intensities.")
            if not bool(torch.isfinite(prediction).all().detach().cpu()):
                raise ValueError("frozen forward predictions must be finite.")
            if bool((prediction < 0).any().detach().cpu()):
                raise ValueError("frozen forward predictions must be nonnegative.")

            for name, target in targets.items():
                target_slice = target[start:stop]
                target_is_complex = (
                    torch.is_complex(target_slice)
                    if isinstance(target_slice, torch.Tensor)
                    else np.iscomplexobj(target_slice)
                )
                if target_is_complex:
                    raise TypeError(f"target {name!r} must contain real intensities.")
                # Pty-Chi changes Torch's global default device. Convert to the
                # prediction dtype and device in one step so a NumPy float64
                # target is never first materialized as an unsupported MPS
                # float64 tensor.
                target_batch = torch.as_tensor(
                    target_slice,
                    device=prediction.device,
                    dtype=prediction.dtype,
                )
                if apply_fft_shift:
                    target_batch = torch.fft.fftshift(target_batch, dim=(-2, -1))
                if target_batch.shape != prediction.shape:
                    raise ValueError(
                        f"target {name!r} has batch shape {tuple(target_batch.shape)}; "
                        f"expected {tuple(prediction.shape)}."
                    )
                if not bool(torch.isfinite(target_batch).all().detach().cpu()):
                    raise ValueError(f"target {name!r} must be finite.")
                if bool((target_batch < 0).any().detach().cpu()):
                    raise ValueError(f"target {name!r} must be nonnegative.")

                difference = torch.sqrt(prediction + eps) - torch.sqrt(
                    target_batch + eps
                )
                squared_error_sums[name] += (
                    difference.square().detach().cpu().double().sum().item()
                )
                element_counts[name] += difference.numel()

        return {
            name: squared_error_sums[name] / element_counts[name] for name in targets
        }

    return evaluate


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
    options.reconstructor_options.forward_model_options.pad_for_shift = (
        REPLICATE_PADDING_PIXELS
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
    task_metric_function: TaskMetricFunction | None = None,
    final_metric_function: FinalMetricFunction | None = None,
    snapshot_callback: StateSnapshotCallback | None = None,
    snapshot_stride: int | None = None,
    include_initial_metrics: bool = False,
    chunk_epochs: bool = False,
) -> ReconstructionResult:
    """Run a task and optionally report metrics or stream state snapshots.

    ``metric_function`` evaluates arrays copied from the current completed
    iterate. ``task_metric_function`` evaluates that same completed iterate
    directly on the live task, which supports fixed-dataset forward metrics
    without changing the reconstruction or relying on its online loss table.
    Set ``include_initial_metrics`` to sample those functions once at epoch
    zero, before any reconstruction update; online residuals still begin at
    epoch one.
    ``chunk_epochs=True`` groups task calls between reporting/snapshot events,
    preserving the server real-data execution schedule. The default retains
    the per-epoch calls used by the synthetic experiments.
    ``snapshot_callback`` receives independent CPU copies of the completed
    object and probe at each positive ``snapshot_stride`` and the final epoch.
    """
    if not isinstance(chunk_epochs, bool):
        raise TypeError("chunk_epochs must be boolean.")
    if not isinstance(include_initial_metrics, bool):
        raise TypeError("include_initial_metrics must be boolean.")

    metric_epochs: list[int] = []
    metric_values: dict[str, list[float]] = {}
    reporting_enabled = any(
        function is not None
        for function in (
            metric_function,
            task_metric_function,
            progress_callback,
        )
    )

    def evaluate_current_metrics() -> dict[str, float]:
        metrics: dict[str, float] = {}
        if metric_function is not None:
            if metric_requires_reconstruction_arrays:
                current_object = task.get_data_to_cpu("object", as_numpy=True)
                current_probe = task.get_data_to_cpu("probe", as_numpy=True)
            else:
                current_object = None
                current_probe = None
            metrics = _coerce_finite_metric_mapping(
                metric_function(
                    current_object,
                    current_probe,
                ),
                source="metric_function",
            )

        if task_metric_function is not None:
            task_metrics = _coerce_finite_metric_mapping(
                task_metric_function(task),
                source="task_metric_function",
            )
            duplicate_names = set(metrics).intersection(task_metrics)
            if duplicate_names:
                duplicates = ", ".join(sorted(duplicate_names))
                raise ValueError(f"duplicate sampled metric names: {duplicates}.")
            metrics.update(task_metrics)
        return metrics

    def record_metrics(epoch: int, metrics: Mapping[str, float]) -> None:
        if not metrics:
            return
        if not metric_values:
            metric_values.update({name: [] for name in metrics})
        elif set(metrics) != set(metric_values):
            raise ValueError(
                "sampled metric functions returned inconsistent metric names."
            )

        metric_epochs.append(epoch)
        for name, value in metrics.items():
            metric_values[name].append(value)

    if (snapshot_callback is None) != (snapshot_stride is None):
        raise ValueError(
            "snapshot_callback and snapshot_stride must be provided together."
        )
    if snapshot_stride is not None:
        if isinstance(snapshot_stride, bool) or not isinstance(
            snapshot_stride,
            Integral,
        ):
            raise TypeError("snapshot_stride must be an integer.")
        if snapshot_stride < 1:
            raise ValueError("snapshot_stride must be positive.")
        snapshot_stride = int(snapshot_stride)

    if not reporting_enabled and snapshot_callback is None:
        task.run()
    else:
        if reporting_enabled and metric_stride < 1:
            raise ValueError("metric_stride must be positive.")

        if include_initial_metrics:
            record_metrics(0, evaluate_current_metrics())

        num_epochs = int(task.reconstructor.n_epochs)
        if chunk_epochs:
            event_epochs: set[int] = {num_epochs}
            if reporting_enabled:
                event_epochs.add(1)
                event_epochs.update(range(metric_stride, num_epochs + 1, metric_stride))
            if snapshot_callback is not None:
                event_epochs.update(range(snapshot_stride, num_epochs + 1, snapshot_stride))
            epochs = sorted(event_epochs)
        else:
            epochs = range(1, num_epochs + 1)
        completed_epochs = 0
        for epoch in epochs:
            task.run(n_epochs=epoch - completed_epochs)
            completed_epochs = epoch
            report_due = reporting_enabled and (
                epoch == 1
                or epoch % metric_stride == 0
                or epoch == num_epochs
            )
            snapshot_due = snapshot_callback is not None and (
                epoch % snapshot_stride == 0 or epoch == num_epochs
            )
            if not report_due and not snapshot_due:
                continue

            metrics = evaluate_current_metrics() if report_due else {}
            record_metrics(epoch, metrics)

            if report_due and progress_callback is not None:
                sampled_losses = np.asarray(
                    task.reconstructor.loss_tracker.table["loss"]
                )
                residual = float(sampled_losses[-1])
                progress_callback(epoch, residual, metrics)

            if snapshot_due:
                object_snapshot = np.array(
                    task.get_data_to_cpu("object", as_numpy=True),
                    copy=True,
                )
                probe_snapshot = np.array(
                    task.get_data_to_cpu("probe", as_numpy=True),
                    copy=True,
                )
                snapshot_callback(epoch, object_snapshot, probe_snapshot)

    final_metrics: dict[str, float] = {}
    if final_metric_function is not None:
        final_metrics = _coerce_finite_metric_mapping(
            final_metric_function(task),
            source="final_metric_function",
        )

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
        final_metrics=final_metrics,
    )
