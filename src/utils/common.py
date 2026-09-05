from __future__ import annotations

# Imports below the environment setup are intentional.
# ruff: noqa: E402

import os
from pathlib import Path
import tempfile
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "magpie-matplotlib")
)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
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


def configure_ptychi_device(
    use_mps: bool = True,
    *,
    backend: str | None = None,
) -> api.Devices:
    """Configure Pty-Chi for an explicitly selected accelerator or CPU.

    Existing ``use_mps=True``/``False`` calls retain their MPS/CPU meaning.
    ``backend`` overrides that legacy flag and accepts ``"mps"``, ``"cuda"``,
    or ``"cpu"``. CUDA studies select their backend explicitly so their
    computation cannot silently fall back to a different device.
    """
    selected = ("mps" if use_mps else "cpu") if backend is None else backend
    if selected not in {"mps", "cuda", "cpu"}:
        raise ValueError("backend must be 'mps', 'cuda', or 'cpu'.")
    if selected == "mps":
        if not torch.backends.mps.is_built():
            raise RuntimeError(
                "MPS was requested, but this PyTorch build has no MPS support."
            )
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS was requested, but no Apple MPS device is available."
            )
        ptychi_device.set_torch_accelerator_module(PtychiMPSModule)
        ptychi_device.AcceleratorModuleWrapper.get_to_device_string = classmethod(
            lambda cls: "mps"
        )
        return api.Devices.GPU
    if selected == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot see an NVIDIA GPU. "
            "Allocate a GPU and install a CUDA-enabled PyTorch build before "
            "running this notebook."
        )

    # Reset both accelerator hooks when switching away from MPS in a reused
    # kernel. CPU task options still select the CPU independently of this hook.
    ptychi_device.set_torch_accelerator_module(torch.cuda)
    ptychi_device.AcceleratorModuleWrapper.get_to_device_string = classmethod(
        lambda cls: "cuda"
    )
    return api.Devices.GPU if selected == "cuda" else api.Devices.CPU


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
