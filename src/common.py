from __future__ import annotations

from collections.abc import Callable, Mapping
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/magpie-matplotlib")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import torch

import ptychi.api as api
import ptychi.device as ptychi_device


def patch_ptychi_compatibility() -> None:
    """
    Keep the experiments runnable with the pip-installed Pty-Chi package.

    The installed package uses Tensor.repeat for complex probe batches, which
    fails on Apple MPS. This local patch uses expand(...).clone() instead.
    """
    import ptychi.forward_models as forward_models

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


def _as_detached_float(value: object) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _latest_residual_entry(task: object) -> tuple[int, float] | None:
    table = task.reconstructor.loss_tracker.table
    if table is None or len(table) == 0:
        return None

    if hasattr(table, "iloc"):
        row = table.iloc[-1]
        columns = set(table.columns)
        epoch_value = row["epoch"] if "epoch" in columns else len(table) - 1
        residual_value = row["loss"] if "loss" in columns else row.iloc[-1]
    else:
        row = table[-1]
        if isinstance(row, Mapping):
            epoch_value = row.get("epoch", len(table) - 1)
            residual_value = row.get("loss")
        else:
            epoch_value = len(table) - 1
            residual_value = row
        if residual_value is None:
            return None

    epoch = int(round(_as_detached_float(epoch_value)))
    residual = _as_detached_float(residual_value)
    return epoch, residual


def attach_residual_printer(
    task: object,
    label: str,
    print_every: int,
    total_iterations: int,
) -> Callable[[], None]:
    if print_every <= 0:
        return lambda: None

    original_hook = task.reconstructor.run_post_epoch_hooks
    printed_iterations: set[int] = set()

    def print_latest_residual(force: bool = False) -> None:
        entry = _latest_residual_entry(task)
        if entry is None:
            return

        epoch, residual = entry
        iteration = epoch + 1
        should_print = (
            force
            or iteration == 1
            or iteration == total_iterations
            or iteration % print_every == 0
        )
        if should_print and iteration not in printed_iterations:
            printed_iterations.add(iteration)
            print(
                f"{label}: iteration {iteration}/{total_iterations}, "
                f"residual {residual:.6e}",
                flush=True,
            )

    def run_post_epoch_hooks_with_print(*args, **kwargs):
        result = original_hook(*args, **kwargs)
        print_latest_residual()
        return result

    task.reconstructor.run_post_epoch_hooks = run_post_epoch_hooks_with_print
    return lambda: print_latest_residual(force=True)
