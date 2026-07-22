"""Truncated Karhunen-Loeve prior for the Darcy log-permeability field.

The Gaussian prior of Beskos, Girolami, Lan, Farrell & Stuart (2016), Sec. 4.2, Eqs. (37)-(38): a
cosine eigenbasis on the unit square with a Whittle-Matern-type spectral decay.

Modes are the multi-index ``k = (i1, i2)`` with ``i1, i2 in {1, ..., K}`` (1-indexed, as in
Stuart), giving ``d = K^2`` modes:

    phi_k(x1, x2) = 2 cos(pi (i1 + 1/2) x1) cos(pi (i2 + 1/2) x2),               (Eq. 37)
    lambda_k^2    = sigma^2 ( alpha + pi^2 ((i1+1/2)^2 + (i2+1/2)^2) )^{-s}.     (Eq. 38)

``lambda_k^2`` is the covariance eigenvalue, so the KL amplitude is ``amp_k = lambda_k``. A
whitened standard-normal latent maps to the log-permeability field by

    u(x) = sum_k amp_k xi_k phi_k(x),   xi ~ N(0, I_d).

Stuart's groundwater hyper-parameters are ``alpha = 0, s = 1.1, sigma = 1, K = 10``, i.e. d = 100
unknowns. The basis is evaluated on the SAME vertex grid the solver uses (``x_i = i/(N-1)``,
endpoints included), so ``reconstruct(xi)`` returns field values at exactly the nodes the Darcy
operator reads.

Note that the paper's resolved/blind split is defined by the OPERATOR (the singular directions of
the linearized forward map), not by a cutoff on this mode index.
"""
from __future__ import annotations

import numpy as np


class StuartKLPrior:
    """Cosine-KL prior on the unit square with a whitened latent ``xi ~ N(0, I_d)``."""

    def __init__(self, grid_size: int, K: int = 10, alpha: float = 0.0,
                 s: float = 1.1, sigma: float = 1.0):
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K!r}")
        self.grid_size = int(grid_size)
        self.K = int(K)
        self.d = K * K
        self.alpha = float(alpha)
        self.s = float(s)
        self.sigma = float(sigma)

        # multi-index (i1, i2), 1-indexed on {1..K}^2
        i1 = np.arange(1, K + 1)
        I1, I2 = np.meshgrid(i1, i1, indexing="ij")
        self.i1 = I1.ravel().astype(np.int64)                                       # (d,)
        self.i2 = I2.ravel().astype(np.int64)

        # KL amplitudes amp_k = lambda_k = sigma (alpha + pi^2 |k+1/2|^2)^{-s/2}
        ksq = (self.i1 + 0.5) ** 2 + (self.i2 + 0.5) ** 2
        self.amp = np.sqrt(self.sigma ** 2 * (self.alpha + np.pi ** 2 * ksq) ** (-self.s))

        # eigenfunctions on the vertex grid x_i = i/(N-1)
        x = np.arange(self.grid_size) / (self.grid_size - 1)                        # [0,1]
        c1 = 2.0 ** 0.5 * np.cos(np.pi * (self.i1[:, None] + 0.5) * x[None, :])     # (d, N)
        c2 = 2.0 ** 0.5 * np.cos(np.pi * (self.i2[:, None] + 0.5) * x[None, :])     # (d, N)
        self.phi = np.einsum("ki,kj->kij", c1, c2)                                  # (d, N, N)

    def reconstruct(self, xi: np.ndarray) -> np.ndarray:
        """Whitened latent ``xi (..., d)`` -> log-permeability field ``(..., N, N)``."""
        xi = np.asarray(xi, dtype=np.float64)
        coef = self.amp * xi
        if xi.ndim == 1:
            return np.einsum("k,kij->ij", coef, self.phi)
        return np.einsum("nk,kij->nij", coef, self.phi)

    def stuart_true_xi(self) -> np.ndarray:
        """Stuart's fixed 'true' field in whitened coordinates.

        Stuart sets the true coordinate ``u_k = lambda_k sin((i1-1/2)^2 + (i2-1/2)^2)`` -- a
        deterministic field of typical prior magnitude -- so the whitened coefficient is the
        ``sin`` factor alone.
        """
        return np.sin((self.i1 - 0.5) ** 2 + (self.i2 - 0.5) ** 2)
