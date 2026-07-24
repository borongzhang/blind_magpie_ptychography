from __future__ import annotations

from typing import TYPE_CHECKING

import ptychi.api as api

from algorithms.magpie import run_blind_magpie
from utils.reconstruction import ProgressCallback, ReconstructionResult

if TYPE_CHECKING:
    from utils.synthetic import ExperimentConfig, SyntheticDataset


def run_gm_rpie(
    dataset: "SyntheticDataset",
    cfg: "ExperimentConfig",
    device: api.Devices,
    seed: int | None = None,
    error_stride: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ReconstructionResult:
    """Run finest-grid GM-rPIE with no multigrid corrections.

    GM-rPIE starts from the usual local rPIE object and probe proposals and
    takes their phase-aligned geometric means with the current estimates. One
    position is applied directly; larger minibatches use updated
    counterpart-intensity weights. The same object/probe ambiguity-removal
    gauge used by rPIE is applied after every full epoch.
    """
    return run_blind_magpie(
        dataset,
        cfg,
        device,
        multigrid_levels=1,
        seed=seed,
        error_stride=error_stride,
        progress_callback=progress_callback,
    )
