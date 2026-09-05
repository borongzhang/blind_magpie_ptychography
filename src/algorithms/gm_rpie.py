from __future__ import annotations

from typing import TYPE_CHECKING

import ptychi.api as api

from algorithms.magpie import run_blind_magpie
from utils.reconstruction import (
    FinalMetricFunction,
    ProgressCallback,
    ReconstructionResult,
    TaskMetricFunction,
)

if TYPE_CHECKING:
    from utils.synthetic import ExperimentConfig, SyntheticDataset


def run_gm_rpie(
    dataset: "SyntheticDataset",
    cfg: "ExperimentConfig",
    device: api.Devices,
    seed: int | None = None,
    error_stride: int | None = None,
    progress_callback: ProgressCallback | None = None,
    task_metric_function: TaskMetricFunction | None = None,
    final_metric_function: FinalMetricFunction | None = None,
    include_initial_metrics: bool = False,
) -> ReconstructionResult:
    """Run finest-grid GM-rPIE with no multigrid corrections.

    GM-rPIE starts from the usual local rPIE object and probe proposals and
    takes their phase-aligned geometric means with the current estimates. One
    weighted-adjoint synthesis rule combines the local estimates for every
    minibatch size. The same object/probe ambiguity-removal gauge used by rPIE
    is applied after every full epoch.
    """
    return run_blind_magpie(
        dataset,
        cfg,
        device,
        multigrid_levels=1,
        seed=seed,
        error_stride=error_stride,
        progress_callback=progress_callback,
        task_metric_function=task_metric_function,
        final_metric_function=final_metric_function,
        include_initial_metrics=include_initial_metrics,
    )
