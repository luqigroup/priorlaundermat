"""The true prior -- a rough random field -- and truth/observation synthesis.

The truth must genuinely vary in the causal shadow, otherwise there is nothing for the legacy
regularizer to launder. We use a zero-mean Gaussian field with an exponential
(Ornstein-Uhlenbeck) covariance ``C_ij = s_star^2 exp(-|i-j| / ell)``: a short correlation
length gives nowhere-smooth sample paths with substantial high-roughness content and marginal
standard deviation ``s_star`` on every node, including the shadow. This is exactly the field
the smoothness regularizer is wrong about.

:func:`synthesize_truths_and_data` draws ``(x_true, y)`` pairs from the joint ``p(x) p(y|x)``
with ``y = F(x) + sigma * noise``; the noise level is set as a fraction of the clean-signal
RMS, a known, truth-independent acquisition quantity. The returned truths are held out for
coverage: nothing that defines the split or the regularizer strength may look at them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from priorlaundermat.wave.solver import SemilinearWave1D


@dataclass
class RoughFieldPrior:
    """Zero-mean Gaussian rough-field prior with an exponential covariance.

    Args:
        n: Field dimension.
        s_star: Marginal standard deviation on every node (the true blind-subspace spread).
        ell: Correlation length in nodes (small = rough).
        dtype, device: Tensor placement.
    """

    n: int = 64
    s_star: float = 2.0
    ell: float = 2.0
    dtype: torch.dtype = torch.float64
    device: object = None

    def __post_init__(self) -> None:
        idx = torch.arange(self.n, dtype=self.dtype, device=self.device)
        cov = self.s_star ** 2 * torch.exp(-(idx[:, None] - idx[None, :]).abs() / self.ell)
        # jitter for a stable Cholesky factor
        self._chol = torch.linalg.cholesky(
            cov + 1e-9 * torch.eye(self.n, dtype=self.dtype, device=self.device))

    def sample(self, m: int, generator: torch.Generator) -> Tensor:
        """Draw ``m`` fields ``(m, n)`` from the prior."""
        z = torch.randn(self.n, m, generator=generator, dtype=self.dtype)
        if self.device is not None:
            z = z.to(self.device)
        return (self._chol @ z).T.contiguous()


def synthesize_truths_and_data(
    solver: SemilinearWave1D, prior: RoughFieldPrior, *,
    n: int, rel_noise: float, seed: int,
) -> tuple[Tensor, Tensor, float]:
    """Draw ``n`` ``(x_true, y)`` pairs from ``p(x) p(y|x)`` with known noise.

    ``y = F(x_true) + sigma * eps`` with ``sigma = rel_noise * RMS(F(x_true))`` over the
    batch -- a truth-independent acquisition quantity, since the recorded-signal level and
    the noise fraction are known to the operator. Reproducible from one CPU generator.

    Returns:
        ``(x_true [n, d_x], y [n, d_y], sigma)``.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    x_true = prior.sample(n, g)
    with torch.no_grad():
        clean = solver.forward(x_true)
    sigma = float(rel_noise * clean.std())
    eps = torch.randn(clean.shape, generator=g, dtype=solver.dtype)
    if solver.device is not None:
        eps = eps.to(solver.device)
    y = clean + sigma * eps
    return x_true, y, sigma
