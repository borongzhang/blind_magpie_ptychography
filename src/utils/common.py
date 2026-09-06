from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

import ptychi.api as api
import ptychi.device as ptychi_device

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def patch_ptychi_compatibility() -> None:
    """
    Keep the experiments runnable with the pip-installed Pty-Chi package.

    The installed package uses Tensor.repeat for complex probe batches and
    rebuilds a bounding-box tensor from four gradient-tracking scalar tensors.
    The local replacements preserve the experiment behavior across supported
    accelerators while avoiding a device incompatibility and a scalar warning.
    """
    import ptychi.forward_models as forward_models
    import ptychi.data_structures.base as data_base
    import ptychi.data_structures.object as object_module
    import ptychi.data_structures.probe_positions as probe_positions_module

    def get_unique_probes(self, indices, always_return_probe_batch=True):
        if self.probe.has_multiple_opr_modes:
            return self.probe.get_unique_probes(
                self.opr_mode_weights.get_weights(indices),
                mode_to_apply=0,
            )
        if always_return_probe_batch:
            return self.probe.data.expand(indices.shape[0], -1, -1, -1).clone()
        return self.probe.get_opr_mode(0)

    forward_models.PlanarPtychographyForwardModel.get_unique_probes = get_unique_probes

    def build_roi_bounding_box(self, positions):
        position_data = positions.data
        bounds = (
            torch.stack(
                (
                    position_data[:, 0].min(),
                    position_data[:, 0].max(),
                    position_data[:, 1].min(),
                    position_data[:, 1].max(),
                )
            )
            .detach()
            .cpu()
            .tolist()
        )
        self.roi_bbox = data_base.BoundingBox(
            *bounds,
            origin=tuple(self.pos_origin_coords.detach().cpu().tolist()),
        )

    object_module.Object.build_roi_bounding_box = build_roi_bounding_box

    # Pty-Chi 1.4.0 calls torch.clip on a Python ``inf`` when the MAD-based
    # position clip is disabled but a finite user cap is retained. Route that
    # configuration through the base parameter implementation, whose scalar
    # cap has the intended per-axis behavior for real-valued (y, x) positions.
    current_position_step = probe_positions_module.ProbePositions.step_optimizer
    if not getattr(current_position_step, "_magpie_no_mad_cap_fix", False):
        original_position_step = current_position_step

        def step_position_optimizer(self, clip_update=True, *args, **kwargs):
            correction = self.options.correction_options
            if clip_update and not correction.clip_update_magnitude_by_mad:
                return data_base.ReconstructParameter.step_optimizer(
                    self,
                    limit=correction.update_magnitude_limit,
                    *args,
                    **kwargs,
                )
            return original_position_step(
                self,
                clip_update=clip_update,
                *args,
                **kwargs,
            )

        step_position_optimizer._magpie_no_mad_cap_fix = True
        probe_positions_module.ProbePositions.step_optimizer = step_position_optimizer


def configure_ptychi_device() -> api.Devices:
    """Select an NVIDIA CUDA GPU and fail before a long run if unavailable."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot see an NVIDIA GPU. "
            "Allocate an NVIDIA GPU and install a CUDA-enabled "
            "PyTorch build before running this notebook."
        )

    # Pty-Chi 1.4.0 normally defaults to CUDA. Set both hooks explicitly so a
    # reused notebook kernel cannot retain accelerator state from another run.
    ptychi_device.set_torch_accelerator_module(torch.cuda)
    ptychi_device.AcceleratorModuleWrapper.get_to_device_string = classmethod(
        lambda cls: "cuda"
    )
    return api.Devices.GPU


def cuda_runtime_preflight() -> dict[str, Any]:
    """Exercise the CUDA runtime before loading a large measured dataset.

    A simple availability check does not load every CUDA runtime component.
    In particular, complex transcendental kernels use NVRTC, which caught a
    broken ``libnvrtc-builtins`` installation in the original server setup.
    Run both operations here so that failure is immediate and inexpensive.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA preflight requires a visible NVIDIA GPU.")

    device_index = int(torch.cuda.current_device())
    device = torch.device("cuda", device_index)
    sample = torch.tensor(
        [1.0 + 0.5j, -0.25 + 1.5j],
        dtype=torch.complex64,
        device=device,
    )
    exponential = torch.exp(sample)
    square_root = torch.sqrt(sample)
    torch.cuda.synchronize(device)

    checks = (("complex exp", exponential), ("complex sqrt", square_root))
    for name, value in checks:
        if not bool(torch.isfinite(torch.view_as_real(value)).all().cpu()):
            raise RuntimeError(f"CUDA preflight produced a nonfinite {name} result.")

    properties = torch.cuda.get_device_properties(device_index)
    capability = torch.cuda.get_device_capability(device_index)
    return {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": True,
        "gpu_index": device_index,
        "gpu_name": properties.name,
        "compute_capability": [int(capability[0]), int(capability[1])],
        "total_memory_bytes": int(properties.total_memory),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "complex_exp_and_sqrt": "passed",
    }


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
