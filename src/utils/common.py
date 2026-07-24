from __future__ import annotations

# Imports below the environment setup are intentional.
# ruff: noqa: E402

import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/magpie-matplotlib")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import torch

import ptychi.api as api
import ptychi.device as ptychi_device


def patch_ptychi_compatibility() -> None:
    """
    Keep the experiments runnable with the pip-installed Pty-Chi package.

    The installed package uses Tensor.repeat for complex probe batches, which
    fails on Apple MPS. It also rebuilds a bounding-box tensor from four
    gradient-tracking scalar tensors. The local replacements preserve both
    results while avoiding the MPS failure and the PyTorch scalar warning.
    """
    import ptychi.forward_models as forward_models
    import ptychi.data_structures.base as data_base
    import ptychi.data_structures.object as object_module

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


class PtychiMPSModule:
    @staticmethod
    def is_available() -> bool:
        return torch.backends.mps.is_available()

    @staticmethod
    def device_count() -> int:
        return 1 if torch.backends.mps.is_available() else 0

    @staticmethod
    def get_device_name(index: int = 0) -> str:
        return "Apple MPS"

    @staticmethod
    def synchronize() -> None:
        torch.mps.synchronize()

    @staticmethod
    def empty_cache() -> None:
        torch.mps.empty_cache()

    @staticmethod
    def ipc_collect() -> None:
        return None

    @staticmethod
    def mem_get_info() -> tuple[int, int]:
        total = int(torch.mps.recommended_max_memory())
        used = int(torch.mps.current_allocated_memory())
        return max(total - used, 0), total


def configure_ptychi_device(use_mps: bool = True) -> api.Devices:
    if use_mps:
        ptychi_device.set_torch_accelerator_module(PtychiMPSModule)
        ptychi_device.AcceleratorModuleWrapper.get_to_device_string = classmethod(
            lambda cls: "mps"
        )
        return api.Devices.GPU
    return api.Devices.CPU


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
