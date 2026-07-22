"""The misspecified legacy regularizer: 1D smoothness (Tikhonov) plus a small ridge.

The legacy reconstructions are produced by a handcrafted regularizer that is not the true
prior -- one whose belief on the operator's blind (causal-shadow) subspace is TIGHTER than
the rough truth:

    R(x) = (alpha/2) ||G x||^2 + (rho/2) ||x||^2 = (1/2) x^T (alpha L + rho I) x,

with ``G`` the 1D first-difference gradient (so ``G^T G = L`` is the path-graph Laplacian) and
a small ridge ``rho`` that turns the otherwise singular ``L`` into a proper Gaussian
precision. It believes the field is smoother than the rough truth, so on the blind subspace,
where the data are silent, it confidently extrapolates a smooth continuation; its belief is
tightest exactly on the high-roughness shadow modes, which is where the rough truth genuinely
varies.

The strength ``alpha`` is chosen by a rule that never sees the truth -- Morozov's discrepancy
principle on the known noise level (see :mod:`.archive`). All matrices are float64.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from priorlaundermat.wave.blind import graph_laplacian_1d


@dataclass
class SmoothnessRegularizer:
    """Proper smoothness precision ``alpha * L + rho * I`` over ``R^n``.

    ``L`` is the 1D path-graph Laplacian; ``rho`` is a small ridge for properness.
    """

    L: Tensor          # (n, n) path-graph Laplacian
    alpha: float       # roughness strength (set by discrepancy)
    rho: float         # ridge (properness / marginal-variance anchor)
    n: int

    def value(self, x: Tensor) -> Tensor:
        """``R(x)`` for a batch ``x`` of shape ``(..., n)`` -> ``(...)``."""
        Lx = x @ self.L
        return 0.5 * self.alpha * (x * Lx).sum(dim=-1) + 0.5 * self.rho * x.square().sum(dim=-1)


def build_smoothness_regularizer(
    *, n: int, alpha: float, rho: float = 1e-2,
    dtype: torch.dtype = torch.float64, device=None,
) -> SmoothnessRegularizer:
    """Construct the misspecified 1D smoothness regularizer.

    Args:
        n: State dimension.
        alpha: Roughness strength (typically set by the discrepancy principle).
        rho: Ridge for properness; small, and it anchors the marginal variance so the frozen
            belief is well defined even where the regularizer is weak.
        dtype, device: Tensor placement.
    """
    L = graph_laplacian_1d(n, dtype=dtype)
    if device is not None:
        L = L.to(device)
    return SmoothnessRegularizer(L=L, alpha=float(alpha), rho=float(rho), n=int(n))
