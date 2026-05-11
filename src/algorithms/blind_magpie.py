from __future__ import annotations

import math
from typing import Optional, TYPE_CHECKING

from common import attach_residual_printer

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import ptychi.api as api
import ptychi.data_structures.parameter_group as paramgrp
import ptychi.image_proc as ip
from ptychi.api.task import PtychographyTask
from ptychi.metrics import MSELossOfSqrt
from ptychi.reconstructors.base import AnalyticalIterativePtychographyReconstructor
from ptychi.timing.timer_utils import timer

from algorithms.geometric_mean import aligned_geom_mean_torch
from synthetic.ptychi_rpie import ReconstructionResult, build_rpie_options
from synthetic.utils import (
    ExperimentConfig,
    SyntheticDataset,
    attach_reconstruction_error_tracker,
)

if TYPE_CHECKING:
    import ptychi.data_structures.parameter_group as pg


class BlindMAGPIEReconstructor(AnalyticalIterativePtychographyReconstructor):
    parameter_group: "pg.PlanarPtychographyParameterGroup"

    def __init__(
        self,
        parameter_group: "pg.PlanarPtychographyParameterGroup",
        dataset: Dataset,
        options: Optional["api.options.pie.PIEReconstructorOptions"] = None,
        multigrid_levels: int | None = None,
        probe_shift_tol: float = 1e-6,
        assume_no_subpixel_shifts: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            parameter_group=parameter_group,
            dataset=dataset,
            options=options,
            *args,
            **kwargs,
        )
        self.multigrid_levels = multigrid_levels
        self.probe_shift_tol = probe_shift_tol
        self.assume_no_subpixel_shifts = assume_no_subpixel_shifts

    def build_loss_tracker(self):
        if self.displayed_loss_function is None:
            self.displayed_loss_function = MSELossOfSqrt()
        return super().build_loss_tracker()

    def check_inputs(self, *args, **kwargs):
        for var in self.parameter_group.get_optimizable_parameters():
            if "lr" not in var.optimizer_params:
                raise ValueError(
                    f"Optimizable parameter {var.name} must have 'lr' in optimizer_params."
                )

    @timer()
    def run_minibatch(self, input_data, y_true, *args, **kwargs):
        self.parameter_group.probe.initialize_grad()
        (delta_o, delta_p, delta_pos), y_pred = self.compute_updates(*input_data, y_true)
        self.apply_updates(delta_o, delta_p, delta_pos)
        self.loss_tracker.update_batch_loss_with_metric_function(y_pred, y_true)

    @timer()
    def compute_updates(
        self,
        indices: torch.Tensor,
        y_true: torch.Tensor,
    ) -> tuple[
        tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]],
        torch.Tensor,
    ]:
        object_ = self.parameter_group.object
        probe = self.parameter_group.probe
        probe_positions = self.parameter_group.probe_positions

        if object_.n_slices != 1:
            raise NotImplementedError(
                "The blind MAGPIE implementation assumes a single-slice object."
            )

        indices = indices.cpu()
        positions = probe_positions.tensor[indices]
        y = self.forward_model.forward(indices)

        obj_patches = self.forward_model.intermediate_variables["obj_patches"]
        psi_far = self.forward_model.intermediate_variables["psi_far"]
        shifted_unique_probes = self.forward_model.intermediate_variables.shifted_unique_probes
        p_batch = self._ensure_probe_batch(shifted_unique_probes[0])
        # In the no-subpixel synthetic case, Pty-Chi may still carry one probe
        # per scan in the batch. Keep only the unique probe so MAGPIE probe
        # overheads are built once and then broadcast over object patches.
        # With real subpixel positions, each shifted probe is physically
        # different, so the object surrogate must use the per-position batch.
        q_for_object = self._probes_for_object_update(p_batch, positions)

        psi_prime = self.replace_propagated_exit_wave_magnitude(
            psi_far,
            y_true,
            constrained_pixel_mask=self.get_constrained_pixel_mask(y_true),
        )
        psi_prime = self.forward_model.free_space_propagator.propagate_backward(psi_prime)

        delta_o = self._compute_object_delta(
            object_=object_,
            positions=positions,
            obj_patches=obj_patches,
            q_for_object=q_for_object,
            psi_prime=psi_prime,
        )
        delta_pos = None
        delta_p = self._compute_probe_delta(
            probe=probe,
            indices=indices,
            obj_patches=obj_patches,
            p_batch=p_batch,
            psi_prime=psi_prime,
        )

        return (delta_o, delta_p, delta_pos), y

    @staticmethod
    def _ensure_probe_batch(probe_tensor: torch.Tensor) -> torch.Tensor:
        if probe_tensor.ndim == 3:
            return probe_tensor[None, ...]
        if probe_tensor.ndim == 4:
            return probe_tensor
        raise ValueError(f"Expected a 3D or 4D probe tensor, got shape={probe_tensor.shape}.")

    def _probes_for_object_update(
        self,
        p_batch: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if not self.forward_model.apply_subpixel_shifts_on_probe:
            return p_batch[:1]

        positions_detached = positions.detach()
        max_fractional_shift = torch.max(
            torch.abs(positions_detached - positions_detached.round())
        ).item()
        if max_fractional_shift <= self.probe_shift_tol:
            return p_batch[:1]

        if self.assume_no_subpixel_shifts:
            raise ValueError(
                "Subpixel probe positions were detected, but "
                "assume_no_subpixel_shifts=True. Pass assume_no_subpixel_shifts=False "
                "so MAGPIE uses the shifted probe batch for the object update and the "
                "adjoint shifted frame for the probe update."
            )
        if p_batch.shape[0] != positions.shape[0]:
            raise ValueError(
                "Subpixel probe positions require one shifted probe per scan position; "
                f"got p_batch.shape={p_batch.shape} for {positions.shape[0]} positions."
            )
        return p_batch

    @staticmethod
    def _downsample_complex(x: torch.Tensor) -> torch.Tensor:
        real = F.avg_pool2d(x.real, kernel_size=2, stride=2)
        imag = F.avg_pool2d(x.imag, kernel_size=2, stride=2)
        return torch.complex(real, imag)

    @staticmethod
    def _upsample_complex(x: torch.Tensor) -> torch.Tensor:
        real = F.interpolate(x.real, mode="nearest", scale_factor=2)
        imag = F.interpolate(x.imag, mode="nearest", scale_factor=2)
        return torch.complex(real, imag)

    @staticmethod
    def _downsample_real(x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, kernel_size=2, stride=2)

    @staticmethod
    def _upsample_real(x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, mode="nearest", scale_factor=2)

    @staticmethod
    def _safe_divide_real(
        numerator: torch.Tensor,
        denominator: torch.Tensor,
        fill_value: float = 0.0,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        mask = denominator.abs() > eps
        safe_denominator = torch.where(mask, denominator, torch.ones_like(denominator))
        divided = numerator / safe_denominator
        return torch.where(mask, divided, torch.full_like(numerator, fill_value))

    @staticmethod
    def _safe_divide_complex(
        numerator: torch.Tensor,
        denominator: torch.Tensor,
        fill_value: complex = 0.0 + 0.0j,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        mask = denominator.abs() > eps
        safe_denominator = torch.where(mask, denominator, torch.ones_like(denominator))
        divided = numerator / safe_denominator
        fill = torch.full_like(numerator, fill_value)
        return torch.where(mask, divided, fill)

    @staticmethod
    def _valid_num_multigrid_levels(height: int, width: int) -> int:
        min_dim = min(height, width)
        if min_dim < 2:
            return 1

        levels = int(math.floor(math.log2(min_dim)))
        while levels > 0 and (height % (2**levels) != 0 or width % (2**levels) != 0):
            levels -= 1
        return max(levels, 1)

    def _w_z_from_power(
        self,
        q_power: torch.Tensor,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        denom = self._upsample_real(self._downsample_real(q_power))
        return self._safe_divide_real(q_power, denom, fill_value=0.0, eps=eps)

    def _w_u_from_power(
        self,
        q_power_fine: torch.Tensor,
        q_power_coarse: torch.Tensor,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        denom = self._downsample_real(q_power_fine)
        return self._safe_divide_real(q_power_coarse, denom, fill_value=1.0, eps=eps)

    def _w_rew_from_wz(
        self,
        q: torch.Tensor,
        w_z: torch.Tensor,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        q_coarse = self._downsample_complex(q)
        q_coarse_up = self._upsample_complex(q_coarse)
        return self._safe_divide_complex(q_coarse_up * w_z, q, eps=eps)

    @staticmethod
    def _object_proximal_step(
        z: torch.Tensor,
        q: torch.Tensor,
        r: torch.Tensor,
        u: torch.Tensor,
        q_power: torch.Tensor | None = None,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        if q_power is None:
            q_power = (torch.abs(q) ** 2).sum(dim=1, keepdim=True).real
        residual = r - q * z
        numerator = (q.conj() * residual).sum(dim=1, keepdim=True)
        denominator = q_power + u
        return z + numerator / (denominator.to(numerator.dtype) + eps)

    @staticmethod
    def _object_weight_from_probe(
        q: torch.Tensor,
        alpha: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_power = (torch.abs(q) ** 2).sum(dim=1, keepdim=True).real
        q_max = q_power.amax(dim=(-2, -1), keepdim=True)
        u_q = alpha * (q_max - q_power).clamp_min(0)
        return q_power, u_q, q_power + u_q

    def _multigrid_update_z(
        self,
        z: torch.Tensor,
        q: torch.Tensor,
        r: torch.Tensor,
        alpha: float,
        q_power_0: torch.Tensor | None = None,
        u_0: torch.Tensor | None = None,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        if z.ndim == 3:
            z = z[:, None, ...]
        if q.ndim == 3:
            q = q[None, ...]
        if r.ndim == 3:
            r = r[:, None, ...]

        if q.shape[0] not in (1, z.shape[0]):
            raise ValueError(
                "q must contain either one shared probe or one shifted probe per object patch; "
                f"got z.shape={z.shape}, q.shape={q.shape}."
            )
        if z.shape[1] != 1:
            raise ValueError(f"Expected object patch channel dimension 1, got z.shape={z.shape}.")
        if q.shape[-2:] != z.shape[-2:] or r.shape[-2:] != z.shape[-2:]:
            raise ValueError(
                f"Spatial-size mismatch: z.shape={z.shape}, q.shape={q.shape}, r.shape={r.shape}."
            )

        _, _, height, width = z.shape
        max_levels = self._valid_num_multigrid_levels(height, width)
        if self.multigrid_levels is None:
            num_levels = max_levels
        else:
            num_levels = min(max(int(self.multigrid_levels), 1), max_levels)

        if q_power_0 is None or u_0 is None:
            q_power_0, u_0, _ = self._object_weight_from_probe(q, alpha)

        if num_levels <= 1:
            return self._object_proximal_step(
                z,
                q,
                r,
                u_0,
                q_power=q_power_0,
                eps=eps,
            )

        z_levels = [z]
        q_levels = [q]
        r_levels = [r]

        for _ in range(1, num_levels):
            q_levels.append(self._downsample_complex(q_levels[-1]))

        q_power_levels = [q_power_0]
        q_power_levels.extend(
            (torch.abs(q_level) ** 2).sum(dim=1, keepdim=True).real
            for q_level in q_levels[1:]
        )

        u_levels = [u_0]
        for level in range(1, num_levels):
            w_u = self._w_u_from_power(
                q_power_levels[level - 1],
                q_power_levels[level],
                eps=eps,
            )
            u_levels.append(w_u * self._downsample_real(u_levels[level - 1]))

        for level in range(1, num_levels):
            w_z = self._w_z_from_power(q_power_levels[level - 1], eps=eps)
            w_r = self._w_rew_from_wz(q_levels[level - 1], w_z, eps=eps)

            z_levels.append(self._downsample_complex(w_z * z_levels[level - 1]))
            r_levels.append(self._downsample_complex(w_r * r_levels[level - 1]))

        level = num_levels - 1
        z_new = self._object_proximal_step(
            z_levels[level],
            q_levels[level],
            r_levels[level],
            u_levels[level],
            q_power=q_power_levels[level],
            eps=eps,
        )

        for level in range(num_levels - 2, -1, -1):
            correction = self._upsample_complex(z_new - z_levels[level + 1])
            z_new = z_levels[level] + correction
            z_new = self._object_proximal_step(
                z_new,
                q_levels[level],
                r_levels[level],
                u_levels[level],
                q_power=q_power_levels[level],
                eps=eps,
            )

        return z_new

    def _compute_object_delta(
        self,
        object_,
        positions: torch.Tensor,
        obj_patches: torch.Tensor,
        q_for_object: torch.Tensor,
        psi_prime: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if not object_.optimization_enabled(self.current_epoch):
            return None

        eps = 1e-12
        obj_old = object_.get_slice(0)
        obj_patches_old = obj_patches[:, 0:1, ...]
        alpha_o = object_.options.alpha
        q_power, u_q, obj_weight_patches = self._object_weight_from_probe(q_for_object, alpha_o)

        z_mg_patches = self._multigrid_update_z(
            z=obj_patches_old,
            q=q_for_object,
            r=psi_prime,
            alpha=alpha_o,
            q_power_0=q_power,
            u_0=u_q,
            eps=eps,
        )

        z_fuse_patches = aligned_geom_mean_torch(
            obj_patches_old,
            z_mg_patches,
            obj_patches_old,
        )
        obj_num_patches = obj_weight_patches * z_fuse_patches

        obj_num = self._place_object_patches_on_buffer(
            object_,
            torch.zeros_like(obj_old),
            positions,
            obj_num_patches[:, 0],
        )
        obj_den = self._place_object_patches_on_buffer(
            object_,
            torch.zeros_like(obj_old.real),
            positions,
            obj_weight_patches.expand_as(z_mg_patches)[:, 0],
        ).real

        obj_candidate = obj_num / (obj_den.to(obj_num.dtype) + eps)
        obj_new = torch.where(obj_den > eps, obj_candidate, obj_old)

        delta_o = torch.zeros_like(object_.data)
        delta_o[0, ...] = obj_new - obj_old
        return delta_o

    def _place_object_patches_on_buffer(
        self,
        object_,
        buffer: torch.Tensor,
        positions: torch.Tensor,
        patches: torch.Tensor,
    ) -> torch.Tensor:
        if self.forward_model.apply_subpixel_shifts_on_probe:
            return ip.place_patches_integer(
                buffer,
                positions.round().int() + object_.pos_origin_coords,
                patches,
                op="add",
            )

        return object_.place_patches_function(
            buffer,
            positions + object_.pos_origin_coords,
            patches,
            op="add",
            pad=self.options.forward_model_options.pad_for_shift,
        )

    def _compute_probe_delta(
        self,
        probe,
        indices: torch.Tensor,
        obj_patches: torch.Tensor,
        p_batch: torch.Tensor,
        psi_prime: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if not probe.optimization_enabled(self.current_epoch):
            return None

        eps = 1e-12
        alpha_p = float(probe.options.alpha)
        uses_shifted_probe_frame = (
            not self.assume_no_subpixel_shifts
            and self.forward_model.apply_subpixel_shifts_on_probe
        )
        if uses_shifted_probe_frame and abs(alpha_p - 1.0) > 1e-12:
            raise ValueError(
                "Shifted-probe MAGPIE currently requires probe_alpha == 1.0. "
                "Synthetic integer-shift tests can use probe_alpha < 1.0."
            )

        obj_patch_batch = obj_patches[:, 0, ...]
        mode_slicer = probe._get_probe_mode_slicer(None)
        probe_old = probe.data[0, mode_slicer]

        z_power = (torch.abs(obj_patch_batch) ** 2)[:, None, ...].real
        z_max = z_power.amax(dim=(-2, -1), keepdim=True)
        u_z = alpha_p * (z_max - z_power).clamp_min(0)
        probe_weight = z_power + u_z

        probe_plus_shift = (
            obj_patch_batch[:, None, ...].conj() * psi_prime
            + u_z * p_batch
        ) / (probe_weight.to(psi_prime.dtype) + eps)

        probe_fuse_shift = aligned_geom_mean_torch(
            p_batch,
            probe_plus_shift,
            p_batch,
        )

        probe_num_i = probe_weight.to(probe_fuse_shift.dtype) * probe_fuse_shift
        probe_num_i = self._adjoint_shift_probe_update_direction_if_needed(
            indices,
            probe_num_i,
        )

        probe_num = probe_num_i.sum(0)
        probe_den = probe_weight.sum(0).real.clamp_min(0).expand_as(probe_old.real)
        probe_candidate = probe_num / (probe_den.to(probe_num.dtype) + eps)
        probe_new = torch.where(probe_den > eps, probe_candidate, probe_old)

        return probe_new - probe_old

    def _adjoint_shift_probe_update_direction_if_needed(
        self,
        indices: torch.Tensor,
        delta_p: torch.Tensor,
    ) -> torch.Tensor:
        if self.assume_no_subpixel_shifts or not self.forward_model.apply_subpixel_shifts_on_probe:
            return delta_p
        return self.adjoint_shift_probe_update_direction(
            indices,
            delta_p,
            first_mode_only=True,
        )

    @timer()
    def apply_updates(self, delta_o, delta_p, delta_pos, *args, **kwargs):
        object_ = self.parameter_group.object
        probe = self.parameter_group.probe
        probe_positions = self.parameter_group.probe_positions

        if delta_o is not None:
            object_.set_grad(-delta_o)
            object_.optimizer.step()

        if delta_p is not None:
            mode_slicer = probe._get_probe_mode_slicer(None)
            probe.set_grad(-delta_p, slicer=(0, mode_slicer))
            probe.optimizer.step()

        if delta_pos is not None:
            probe_positions.set_grad(-delta_pos)
            probe_positions.step_optimizer()


class BlindMAGPIETask(PtychographyTask):
    def __init__(
        self,
        options,
        multigrid_levels: int | None = None,
        probe_shift_tol: float = 1e-6,
        assume_no_subpixel_shifts: bool = True,
        *args,
        **kwargs,
    ):
        self.multigrid_levels = multigrid_levels
        self.probe_shift_tol = probe_shift_tol
        self.assume_no_subpixel_shifts = assume_no_subpixel_shifts
        super().__init__(options, *args, **kwargs)

    def build_reconstructor(self):
        par_group = paramgrp.PlanarPtychographyParameterGroup(
            object=self.object,
            probe=self.probe,
            probe_positions=self.probe_positions,
            opr_mode_weights=self.opr_mode_weights,
        )
        self.reconstructor = BlindMAGPIEReconstructor(
            parameter_group=par_group,
            dataset=self.dataset,
            options=self.reconstructor_options,
            multigrid_levels=self.multigrid_levels,
            probe_shift_tol=self.probe_shift_tol,
            assume_no_subpixel_shifts=self.assume_no_subpixel_shifts,
        )
        self.reconstructor.build()


def run_blind_magpie(
    dataset: SyntheticDataset,
    cfg: ExperimentConfig,
    device: api.Devices,
    multigrid_levels: int | None = None,
    assume_no_subpixel_shifts: bool = True,
    seed: int | None = None,
) -> ReconstructionResult:
    if cfg.object_step_size != 1.0 or cfg.probe_step_size != 1.0:
        raise ValueError(
            "Blind MAGPIE uses the surrogate minimizer directly, so "
            "object_step_size and probe_step_size must both be 1.0."
        )

    task = BlindMAGPIETask(
        build_rpie_options(dataset, cfg, device, seed=seed),
        multigrid_levels=multigrid_levels,
        assume_no_subpixel_shifts=assume_no_subpixel_shifts,
    )
    error_history = attach_reconstruction_error_tracker(task, dataset)
    print_final_residual = attach_residual_printer(
        task,
        "MAGPIE",
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
