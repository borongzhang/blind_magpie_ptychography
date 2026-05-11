from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch


def aligned_geom_mean_torch(
    a: torch.Tensor,
    b: torch.Tensor,
    anchor: torch.Tensor,
    tol: float = 1e-14,
) -> torch.Tensor:
    """
    Phase-aligned complex geometric mean.

    Returns x satisfying x**2 = a * b, with the square-root branch chosen
    relative to anchor. In the MAGPIE-GM update, anchor is the current iterate.
    """
    if (
        a.dtype in (torch.float64, torch.complex128)
        or b.dtype in (torch.float64, torch.complex128)
        or anchor.dtype in (torch.float64, torch.complex128)
    ):
        dtype = torch.complex128
    else:
        dtype = torch.complex64

    if (not torch.is_complex(a)) or a.dtype != dtype:
        a = a.to(dtype)
    if (not torch.is_complex(b)) or b.dtype != dtype:
        b = b.to(dtype)
    if (not torch.is_complex(anchor)) or anchor.dtype != dtype:
        anchor = anchor.to(dtype)

    x = torch.sqrt(a * b)
    inner = x * torch.conj(anchor)
    re = inner.real
    im = inner.imag
    tol_tensor = torch.as_tensor(tol, dtype=re.dtype, device=re.device)
    flip = (re < -tol_tensor) | ((re.abs() <= tol_tensor) & (im < 0))
    return torch.where(flip, -x, x)
