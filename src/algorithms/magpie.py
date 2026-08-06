from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import ptychi.api as api
import ptychi.data_structures.parameter_group as paramgrp
import ptychi.image_proc as ip
from ptychi.api.task import PtychographyTask
from ptychi.reconstructors.pie import PIEReconstructor

from algorithms.geometric_mean import aligned_geom_mean_torch

if TYPE_CHECKING:
    import ptychi.data_structures.parameter_group as pg
    from utils.reconstruction import ProgressCallback, ReconstructionResult
    from utils.synthetic import ExperimentConfig, SyntheticDataset


class _MAGPIEMultigridMixin:
    """Shared single-factor MAGPIE V-cycle."""

    num_levels: int

    @staticmethod
    def _power(x: torch.Tensor) -> torch.Tensor:
        return x.real.square() + x.imag.square()

    @staticmethod
    def _divide(
        numerator: torch.Tensor,
        denominator: torch.Tensor,
    ) -> torch.Tensor:
        nonzero = denominator != 0
        return torch.where(
            nonzero,
            numerator / torch.where(nonzero, denominator, torch.ones_like(denominator)),
            torch.zeros_like(numerator),
        )

    @staticmethod
    def _downsample_2x(x: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(x):
            return torch.complex(
                F.avg_pool2d(x.real, 2, 2),
                F.avg_pool2d(x.imag, 2, 2),
            )
        return F.avg_pool2d(x, 2, 2)

    @staticmethod
    def _upsample_2x(x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        return (
            x.reshape(*shape[:-2], shape[-2], 1, shape[-1], 1)
            .expand(*shape[:-2], shape[-2], 2, shape[-1], 2)
            .reshape(*shape[:-2], shape[-2] * 2, shape[-1] * 2)
        )

    @staticmethod
    def _valid_num_levels(height: int, width: int) -> int:
        levels = 1
        while height % 2 == 0 and width % 2 == 0 and min(height, width) >= 2:
            height //= 2
            width //= 2
            levels += 1
        return levels

    def _proximal_step(
        self,
        z: torch.Tensor,
        q: torch.Tensor,
        target: torch.Tensor,
        q_power: torch.Tensor,
        regularizer: torch.Tensor,
    ) -> torch.Tensor:
        numerator = q.conj() * (target - q * z)
        return z + self._divide(numerator, q_power + regularizer)

    def _magpie_endpoint(
        self,
        factor_old: torch.Tensor,
        counterpart_old: torch.Tensor,
        psi_prime: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """Return a multigrid MAGPIE endpoint for one bilinear factor."""
        q_levels = [counterpart_old]
        for _ in range(1, self.num_levels):
            q_levels.append(self._downsample_2x(q_levels[-1]))

        q_power = [self._power(q) for q in q_levels]
        q_max = q_power[0].amax(dim=(-2, -1), keepdim=True)
        regularizer = [alpha * (q_max - q_power[0]).clamp_min(0)]
        downsampled_power = []
        for level in range(1, self.num_levels):
            power_down = self._downsample_2x(q_power[level - 1])
            downsampled_power.append(power_down)
            transfer = self._divide(q_power[level], power_down)
            transfer = torch.where(power_down > 0, transfer, torch.ones_like(transfer))
            regularizer.append(transfer * self._downsample_2x(regularizer[level - 1]))

        z_levels = [factor_old]
        target_levels = [psi_prime]
        for level in range(1, self.num_levels):
            power_down = downsampled_power[level - 1]
            weight_z = self._divide(
                q_power[level - 1],
                self._upsample_2x(power_down),
            )
            weight_target = self._divide(
                self._upsample_2x(q_levels[level]) * q_levels[level - 1].conj(),
                self._upsample_2x(power_down),
            )
            restricted = self._downsample_2x(
                torch.cat(
                    (
                        weight_z * z_levels[level - 1],
                        weight_target * target_levels[level - 1],
                    ),
                    dim=1,
                )
            )
            z_levels.append(restricted[:, :1])
            target_levels.append(restricted[:, 1:])

        level = self.num_levels - 1
        z_new = self._proximal_step(
            z_levels[level],
            q_levels[level],
            target_levels[level],
            q_power[level],
            regularizer[level],
        )
        for level in range(self.num_levels - 2, -1, -1):
            z_new = z_levels[level] + self._upsample_2x(z_new - z_levels[level + 1])
            z_new = self._proximal_step(
                z_new,
                q_levels[level],
                target_levels[level],
                q_power[level],
                regularizer[level],
            )
        return z_new


class BlindMAGPIEReconstructor(_MAGPIEMultigridMixin, PIEReconstructor):
    """Shared GM-rPIE and GM-MAGPIE pipeline.

    Pty-Chi supplies a fresh random permutation of all scan positions each
    epoch. Every minibatch performs exactly one simultaneous object/probe
    update followed by geometric averaging and counterpart-intensity-weighted
    synthesis. As in native rPIE, object patches are extracted at integer scan
    anchors and the fractional offsets are applied to the probe. The same global
    object/probe gauge normalization used by rPIE is applied after every epoch.
    """

    parameter_group: "pg.PlanarPtychographyParameterGroup"

    def __init__(
        self,
        parameter_group: "pg.PlanarPtychographyParameterGroup",
        dataset: Dataset,
        options: "api.options.pie.PIEReconstructorOptions | None" = None,
        multigrid_levels: int | None = None,
    ) -> None:
        self.requested_multigrid_levels = multigrid_levels
        super().__init__(parameter_group, dataset, options)

    def check_inputs(self) -> None:
        object_ = self.parameter_group.object
        probe = self.parameter_group.probe
        positions = self.parameter_group.probe_positions

        if object_.n_slices != 1:
            raise ValueError("MAGPIE requires exactly one object slice.")
        if tuple(probe.data.shape[:2]) != (1, 1):
            raise ValueError("MAGPIE requires one OPR mode and one probe mode.")
        if positions.optimizable:
            raise ValueError("MAGPIE requires fixed scan positions.")
        if self.options.batching_mode != api.BatchingModes.RANDOM:
            raise ValueError("MAGPIE requires random minibatches.")
        if object_.step_size != 1.0 or probe.step_size != 1.0:
            raise ValueError("MAGPIE requires unit object and probe step sizes.")
        for name, parameter in (("object", object_), ("probe", probe)):
            plan = parameter.optimization_plan
            if (
                not parameter.optimizable
                or plan.start != 0
                or plan.stop is not None
                or plan.stride != 1
            ):
                raise ValueError(f"MAGPIE must update the {name} in every minibatch.")
        max_levels = self._valid_num_levels(*probe.data.shape[-2:])
        if self.requested_multigrid_levels is None:
            self.num_levels = max_levels
        elif self.requested_multigrid_levels < 1:
            raise ValueError("multigrid_levels must be positive or None.")
        else:
            self.num_levels = min(self.requested_multigrid_levels, max_levels)

    def build(self) -> None:
        """Build only what the joint update uses."""
        self.check_inputs()
        self.build_dataloader()
        self.build_loss_tracker()
        self.build_counter()

    def step_all_step_size_schedulers(self) -> None:
        # Unit joint updates are fixed; no optimizer schedule is part of MAGPIE.
        pass

    def run_minibatch(self, input_data, y_true, *args, **kwargs) -> None:
        del args, kwargs
        indices = input_data[0].cpu()
        y_pred = self._joint_minibatch_update(indices, y_true)
        self.loss_tracker.update_batch_loss_with_metric_function(y_pred, y_true)

    def _joint_minibatch_update(
        self,
        indices: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        y_pred = self.forward_model.forward(indices)
        variables = self.forward_model.intermediate_variables
        object_old = variables["obj_patches"][:, 0:1]
        probe_old = variables.shifted_unique_probes[0]
        if probe_old.ndim == 3:
            probe_old = probe_old[None]

        psi_prime = self.replace_propagated_exit_wave_magnitude(
            variables["psi_far"],
            y_true,
            constrained_pixel_mask=self.get_constrained_pixel_mask(y_true),
        )
        psi_prime = self.forward_model.free_space_propagator.propagate_backward(
            psi_prime
        )
        delta_psi = psi_prime - variables["psi"]

        if self.num_levels == 1:
            object_plus = self._rpie_object_endpoint(
                object_old,
                probe_old,
                delta_psi,
            )
        else:
            # GM-MAGPIE solves a separate local objective at each scan, so its
            # object regularizer uses one spatial maximum per local probe.
            object_plus = self._magpie_endpoint(
                object_old,
                probe_old,
                psi_prime,
                float(self.parameter_group.object.options.alpha),
            )

        probe_old_batch = probe_old.expand(object_old.shape[0], -1, -1, -1)
        probe_plus = self._rpie_probe_endpoint(
            probe_old,
            object_old,
            delta_psi,
        )

        object_local = aligned_geom_mean_torch(object_old, object_plus)
        probe_local = aligned_geom_mean_torch(
            probe_old_batch,
            probe_plus,
        )
        self._synthesize_and_apply(
            indices,
            object_old,
            probe_old_batch,
            object_local,
            probe_local,
        )
        return y_pred

    def _rpie_object_endpoint(
        self,
        object_old: torch.Tensor,
        probe_old: torch.Tensor,
        delta_psi: torch.Tensor,
    ) -> torch.Tensor:
        alpha = float(self.parameter_group.object.options.alpha)
        probe_power = self._power(probe_old)
        denominator = (1.0 - alpha) * probe_power + alpha * probe_power.amax()
        return object_old + self._divide(probe_old.conj() * delta_psi, denominator)

    def _rpie_probe_endpoint(
        self,
        probe_old: torch.Tensor,
        object_old: torch.Tensor,
        delta_psi: torch.Tensor,
    ) -> torch.Tensor:
        alpha = float(self.parameter_group.probe.options.alpha)
        object_power = self._power(object_old)
        object_max = object_power.amax(dim=(-2, -1), keepdim=True)
        denominator = (1.0 - alpha) * object_power + alpha * object_max
        return probe_old + self._divide(object_old.conj() * delta_psi, denominator)

    def _synthesize_and_apply(
        self,
        indices: torch.Tensor,
        object_old: torch.Tensor,
        probe_old: torch.Tensor,
        object_local: torch.Tensor,
        probe_local: torch.Tensor,
    ) -> None:
        object_ = self.parameter_group.object
        probe = self.parameter_group.probe
        positions = self.parameter_group.probe_positions.tensor[indices]
        integer_positions = positions.round().int() + object_.pos_origin_coords

        # Minimize sum_k ||q'_k (P_k z - z'_k)||^2. P_k extracts an
        # integer object patch, so the normal equation is a weighted scatter.
        probe_weight = self._power(probe_local)
        object_field = object_.get_slice(0)
        numerator = ip.place_patches_integer(
            torch.zeros_like(object_field),
            integer_positions,
            (probe_weight * (object_local - object_old))[:, 0],
            op="add",
        )
        denominator = ip.place_patches_integer(
            torch.zeros_like(object_field.real),
            integer_positions,
            probe_weight[:, 0],
            op="add",
        )
        object_update = self._divide(numerator, denominator)
        object_.set_data(object_update, slicer=0, op="add")

        # Take a diagonal weighted-adjoint synthesis step for
        # sum_k ||z'_k (S_k Q - q'_k)||^2, using the same S_k as native rPIE.
        probe_delta = probe_local - probe_old
        if indices.numel() == 1:
            probe_update = self.adjoint_shift_probe_update_direction(
                indices,
                probe_delta,
                first_mode_only=True,
            )[0]
            probe.set_data(probe_update, slicer=0, op="add")
            return

        object_weight = self._power(object_local)
        probe_numerator = self.adjoint_shift_probe_update_direction(
            indices,
            object_weight * probe_delta,
            first_mode_only=True,
        ).sum(dim=0)
        probe_denominator = self.adjoint_shift_probe_update_direction(
            indices,
            object_weight,
            first_mode_only=True,
        ).sum(dim=0)
        probe_update = torch.where(
            probe_denominator > 0,
            self._divide(probe_numerator, probe_denominator),
            torch.zeros_like(probe_numerator),
        )
        probe.set_data(probe_update, slicer=0, op="add")


class BlindMAGPIETask(PtychographyTask):
    def __init__(
        self,
        options,
        multigrid_levels: int | None = None,
    ) -> None:
        self.multigrid_levels = multigrid_levels
        super().__init__(options)

    def build_reconstructor(self) -> None:
        parameter_group = paramgrp.PlanarPtychographyParameterGroup(
            object=self.object,
            probe=self.probe,
            probe_positions=self.probe_positions,
            opr_mode_weights=self.opr_mode_weights,
        )
        self.reconstructor = BlindMAGPIEReconstructor(
            parameter_group,
            self.dataset,
            self.reconstructor_options,
            self.multigrid_levels,
        )
        self.reconstructor.build()


def _run_synthetic_magpie_task(
    task: PtychographyTask,
    dataset: "SyntheticDataset",
    reconstruction_seed: int,
    error_stride: int | None,
    progress_callback: "ProgressCallback | None",
) -> "ReconstructionResult":
    from utils.reconstruction import run_reconstruction_task

    shuffle_generator = task.reconstructor.dataloader.generator
    if shuffle_generator is None:
        raise RuntimeError("MAGPIE random batching requires a shuffle generator.")
    shuffle_generator.manual_seed(reconstruction_seed)

    if error_stride is None:
        if progress_callback is not None:
            raise ValueError("progress_callback requires error_stride.")
        return run_reconstruction_task(task)

    from utils.synthetic import score_blind_reconstruction

    def score_errors(recon_object, recon_probe) -> dict[str, float]:
        metrics, _, _ = score_blind_reconstruction(
            recon_object,
            recon_probe,
            dataset,
        )
        return metrics

    return run_reconstruction_task(
        task,
        metric_function=score_errors,
        metric_stride=error_stride,
        progress_callback=progress_callback,
    )


def run_blind_magpie(
    dataset: "SyntheticDataset",
    cfg: "ExperimentConfig",
    device: api.Devices,
    multigrid_levels: int | None = None,
    seed: int | None = None,
    error_stride: int | None = None,
    progress_callback: "ProgressCallback | None" = None,
) -> "ReconstructionResult":
    from algorithms.rpie import build_synthetic_rpie_options

    if cfg.object_step_size != 1.0 or cfg.probe_step_size != 1.0:
        raise ValueError("MAGPIE requires unit object and probe step sizes.")

    reconstruction_seed = cfg.reconstruction_seed if seed is None else seed
    options = build_synthetic_rpie_options(
        dataset,
        cfg,
        device,
        seed=reconstruction_seed,
    )
    task = BlindMAGPIETask(options, multigrid_levels)
    return _run_synthetic_magpie_task(
        task,
        dataset,
        reconstruction_seed,
        error_stride,
        progress_callback,
    )


def run_gm_magpie(
    dataset: "SyntheticDataset",
    cfg: "ExperimentConfig",
    device: api.Devices,
    seed: int | None = None,
    error_stride: int | None = None,
    progress_callback: "ProgressCallback | None" = None,
) -> "ReconstructionResult":
    """Run GM-MAGPIE using every valid object multigrid level."""
    return run_blind_magpie(
        dataset,
        cfg,
        device,
        multigrid_levels=None,
        seed=seed,
        error_stride=error_stride,
        progress_callback=progress_callback,
    )
