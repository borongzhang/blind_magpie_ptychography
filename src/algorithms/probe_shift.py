from __future__ import annotations

import torch
import torch.nn.functional as F

import ptychi.forward_models as fm
import ptychi.image_proc as ip

__all__ = [
    "REPLICATE_PADDING_PIXELS",
    "ReplicatePaddedPlanarPtychographyForwardModel",
    "ReplicatePaddedProbeShiftMixin",
    "adjoint_shift_images_with_replicate_padding",
    "shift_images_with_replicate_padding",
]

REPLICATE_PADDING_PIXELS = 1


def shift_images_with_replicate_padding(
    images: torch.Tensor,
    shifts: torch.Tensor,
    *,
    method: str,
) -> torch.Tensor:
    """Apply the fixed one-pixel replicate-pad, shift, and crop operator."""
    pad = REPLICATE_PADDING_PIXELS
    padded = F.pad(images, (pad, pad, pad, pad), mode="replicate")
    shifted = ip.shift_images(
        padded,
        shifts,
        method=method,
        adjoint=False,
    )
    return shifted[..., pad:-pad, pad:-pad]


def adjoint_shift_images_with_replicate_padding(
    images: torch.Tensor,
    shifts: torch.Tensor,
    *,
    method: str,
) -> torch.Tensor:
    """Apply the exact adjoint of the fixed replicate-padded shift.

    If the forward operator is ``C U E``, where ``E`` replicate-pads by one
    pixel and ``C`` center-crops, its adjoint is ``E^H U^H C^H``. The
    cotangent is therefore zero-embedded, reverse-shifted on that enlarged
    grid, and its halo is folded onto the corresponding boundary pixels.
    """
    pad = REPLICATE_PADDING_PIXELS
    height, width = images.shape[-2:]

    # C^H: zero-embed the cropped cotangent. The reverse shift then acts on
    # this already enlarged grid, so no additional boundary extension is used.
    embedded = F.pad(images, (pad, pad, pad, pad))
    shifted = ip.shift_images(
        embedded,
        shifts,
        method=method,
        adjoint=True,
    )

    # E^H: crop the interior and fold the replicated halo onto its sources.
    result = shifted[..., pad : pad + height, pad : pad + width].clone()
    result[..., 0, :] += shifted[..., :pad, pad : pad + width].sum(dim=-2)
    result[..., -1, :] += shifted[..., pad + height :, pad : pad + width].sum(
        dim=-2
    )
    result[..., :, 0] += shifted[..., pad : pad + height, :pad].sum(dim=-1)
    result[..., :, -1] += shifted[..., pad : pad + height, pad + width :].sum(
        dim=-1
    )
    result[..., 0, 0] += shifted[..., :pad, :pad].sum(dim=(-2, -1))
    result[..., 0, -1] += shifted[..., :pad, pad + width :].sum(dim=(-2, -1))
    result[..., -1, 0] += shifted[..., pad + height :, :pad].sum(dim=(-2, -1))
    result[..., -1, -1] += shifted[..., pad + height :, pad + width :].sum(
        dim=(-2, -1)
    )
    return result


class ReplicatePaddedPlanarPtychographyForwardModel(
    fm.PlanarPtychographyForwardModel
):
    """Pty-Chi forward model with one fixed replicate-padded probe shift."""

    def __init__(self, *args, **kwargs) -> None:
        if "pad_for_shift" in kwargs:
            raise TypeError(
                "ReplicatePaddedPlanarPtychographyForwardModel fixes padding "
                "at one pixel."
            )
        super().__init__(
            *args,
            pad_for_shift=REPLICATE_PADDING_PIXELS,
            **kwargs,
        )

    def shift_unique_probes(
        self,
        indices: torch.Tensor,
        unique_probes: torch.Tensor,
        first_mode_only: bool = False,
    ) -> torch.Tensor:
        original_shape = unique_probes.shape
        positions = self.probe_positions.data[indices]
        fractional_shifts = positions - positions.round()

        if first_mode_only:
            probes_to_shift = unique_probes[..., 0, :, :]
            shifts = fractional_shifts
        else:
            mode_count = unique_probes.shape[1]
            probes_to_shift = unique_probes.reshape(
                unique_probes.shape[0] * mode_count,
                *unique_probes.shape[2:],
            )
            shifts = fractional_shifts.repeat_interleave(mode_count, dim=0)

        shifted = shift_images_with_replicate_padding(
            probes_to_shift,
            shifts,
            method=self.parameter_group.object.options.patch_interpolation_method,
        )
        if first_mode_only:
            return torch.cat(
                (shifted[..., None, :, :], unique_probes[..., 1:, :, :]),
                dim=-3,
            )
        return shifted.reshape(original_shape)


class ReplicatePaddedProbeShiftMixin:
    """Use the fixed replicate-padded probe shift and its exact adjoint."""

    def build_forward_model(self) -> None:
        options = self.options.forward_model_options
        if options.pad_for_shift != REPLICATE_PADDING_PIXELS:
            raise RuntimeError("Probe-shift padding must remain fixed at one pixel.")
        self.forward_model = ReplicatePaddedPlanarPtychographyForwardModel(
            parameter_group=self.parameter_group,
            retain_intermediates=True,
            detector_size=tuple(self.dataset.patterns.shape[-2:]),
            wavelength_m=self.dataset.wavelength_m,
            free_space_propagation_distance_m=(
                self.dataset.free_space_propagation_distance_m
            ),
            apply_subpixel_shifts_on_probe=True,
            low_memory_mode=options.low_memory_mode,
            diffraction_pattern_blur_sigma=options.diffraction_pattern_blur_sigma,
        )

    def adjoint_shift_probe_update_direction(
        self,
        indices: torch.Tensor,
        delta_p: torch.Tensor,
        first_mode_only: bool = False,
    ) -> torch.Tensor:
        positions = self.parameter_group.probe_positions.data[indices]
        fractional_shifts = positions - positions.round()
        original_shape = delta_p.shape

        if first_mode_only:
            updates_to_shift = delta_p[..., 0, :, :]
            shifts = fractional_shifts
        else:
            mode_count = delta_p.shape[1]
            updates_to_shift = delta_p.reshape(
                delta_p.shape[0] * mode_count,
                *delta_p.shape[2:],
            )
            shifts = fractional_shifts.repeat_interleave(mode_count, dim=0)

        shifted = adjoint_shift_images_with_replicate_padding(
            updates_to_shift,
            shifts,
            method=self.parameter_group.object.options.patch_interpolation_method,
        )
        if first_mode_only:
            return torch.cat(
                (shifted[..., None, :, :], delta_p[..., 1:, :, :]),
                dim=-3,
            )
        return shifted.reshape(original_shape)
