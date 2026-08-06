import torch


def aligned_geom_mean_torch(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Return the short-arc complex geometric mean closest to both inputs.

    Of the two values satisfying ``x**2 = a * b``, this independently chooses
    the branch on the shorter phase arc between ``a`` and ``b``. Numerically
    antipodal inputs take the positive relative-imaginary branch. Finite zero
    inputs return zero, while nonfinite inputs propagate.
    """
    dtype = torch.complex64
    if (not torch.is_complex(a)) or a.dtype != dtype:
        a = a.to(dtype)
    if (not torch.is_complex(b)) or b.dtype != dtype:
        b = b.to(dtype)

    a, b = torch.broadcast_tensors(a, b)
    phase_a = torch.sgn(a)
    phase_b = torch.sgn(b)
    relative_phase = phase_b * torch.conj(phase_a)
    relative_root = torch.sqrt(relative_phase)

    # Enforce a deterministic +i branch for numerically antipodal phases near
    # the square-root branch cut at phase ±π.
    tolerance = 2 * torch.finfo(relative_phase.real.dtype).eps
    flip_tie = (
        (relative_phase.real < 0)
        & (torch.abs(relative_phase.imag) <= tolerance)
        & (relative_root.imag < 0)
    )
    relative_root = torch.where(flip_tie, -relative_root, relative_root)

    magnitude = torch.sqrt(torch.abs(a)) * torch.sqrt(torch.abs(b))
    result = magnitude * phase_a * relative_root
    return result
