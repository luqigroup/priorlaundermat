"""Steady-state Darcy (groundwater) flow forward operator, its exact adjoint, and the sensor layout.

The elliptic inverse problem of Beskos, Girolami, Lan, Farrell & Stuart, "Geometric MCMC for
Infinite-Dimensional Inverse Problems" (2016), Sec. 4.2, Eq. (40):

    -div( exp(u(x)) grad p(x) ) = 0   on  D = [0,1]^2,
     p = x_1              on  x_2 = 0,
     p = 1 - x_1          on  x_2 = 1,
     dp/dx_1 = 0          on  x_1 = 0, 1   (zero flux).

The unknown ``u`` is the log-permeability, ``p`` the hydraulic head; the inverse problem recovers
``u`` from sparse noisy readings of ``p`` at 33 sensors.

Discretization: conservative finite differences on a VERTEX grid of size N (spacing
``h = 1/(N-1)``, nodes ``x_i = i*h`` including both endpoints). The system ``A(u) p = b`` is
symmetric positive definite (Dirichlet rows pinned to identity) and solved by a sparse LU
factorization; the SAME factor is reused for the adjoint solve because ``A`` is self-adjoint, so
the gradient costs one extra triangular solve. ``tests/test_groundwater.py`` checks the discrete
adjoint against finite differences.

numpy + scipy only -- no external PDE solver, no GPU. A single solve on the N=40 grid used
throughout the paper takes a few milliseconds.

Axis convention: array axis 0 is ``x_1`` (index ``i``), axis 1 is ``x_2`` (index ``j``); the
Dirichlet boundary is ``x_2 in {0,1}`` (axis-1 ends).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class DarcyForward:
    """Darcy forward solve and exact adjoint on an ``N x N`` vertex grid."""

    def __init__(self, N: int):
        self.N = int(N)
        self.h = 1.0 / (N - 1)
        # Dirichlet boundary = the two edges x_2 in {0, 1} (axis-1 ends).
        self.dirichlet = np.zeros((N, N), dtype=bool)
        self.dirichlet[:, 0] = True
        self.dirichlet[:, -1] = True
        x = np.arange(N) * self.h                     # x_1 coordinate along axis 0
        self.bc_value = np.zeros((N, N))
        self.bc_value[:, 0] = x                       # p = x_1        on x_2 = 0
        self.bc_value[:, -1] = 1.0 - x                # p = 1 - x_1    on x_2 = 1

    def _assemble(self, K: np.ndarray):
        """Assemble the SPD matrix ``A`` (Dirichlet rows pinned) and the right-hand side ``b``."""
        N, h2 = self.N, self.h * self.h
        idx = np.arange(N * N).reshape(N, N)
        rows, cols, vals = [], [], []

        Kx = 0.5 * (K[1:, :] + K[:-1, :]) / h2        # face conductance (i, i+1)
        Ky = 0.5 * (K[:, 1:] + K[:, :-1]) / h2        # face conductance (j, j+1)
        interior = ~self.dirichlet
        diag = np.zeros((N, N))

        def add(rmask, r_idx, c_idx, w):
            rows.append(r_idx[rmask]); cols.append(c_idx[rmask]); vals.append(w[rmask])

        m = interior.copy(); m[0, :] = False           # left face (i-1, i)
        w = np.zeros((N, N)); w[1:, :] = Kx
        add(m, idx, np.roll(idx, 1, axis=0), -w); diag += np.where(m, w, 0.0)
        m = interior.copy(); m[-1, :] = False          # right face (i, i+1)
        w = np.zeros((N, N)); w[:-1, :] = Kx
        add(m, idx, np.roll(idx, -1, axis=0), -w); diag += np.where(m, w, 0.0)
        m = interior.copy()                            # down face (j-1, j)
        w = np.zeros((N, N)); w[:, 1:] = Ky
        add(m, idx, np.roll(idx, 1, axis=1), -w); diag += np.where(m, w, 0.0)
        m = interior.copy()                            # up face (j, j+1)
        w = np.zeros((N, N)); w[:, :-1] = Ky
        add(m, idx, np.roll(idx, -1, axis=1), -w); diag += np.where(m, w, 0.0)

        add(interior, idx, idx, diag)                  # interior diagonal
        add(self.dirichlet, idx, idx, np.ones((N, N)))  # Dirichlet rows: identity

        rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
        A = sp.csc_matrix((vals, (rows, cols)), shape=(N * N, N * N))
        b = np.where(self.dirichlet, self.bc_value, 0.0).ravel()
        return A, b

    def solve(self, u_field: np.ndarray, return_factor: bool = False):
        """Return the head ``p`` (N, N). With ``return_factor``, also the LU factor and ``K``."""
        K = np.exp(np.asarray(u_field, dtype=np.float64))
        A, b = self._assemble(K)
        lu = spla.splu(A)
        p = lu.solve(b).reshape(self.N, self.N)
        if return_factor:
            return p, lu, K
        return p

    @staticmethod
    def observe(p: np.ndarray, sensors: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Sample the head at sensor grid indices ``(ix, iy)``."""
        ix, iy = sensors
        return p[ix, iy]

    def gradient_field(self, p: np.ndarray, lu, K: np.ndarray,
                       sensors: tuple[np.ndarray, np.ndarray],
                       y_obs: np.ndarray, sigma_y: float) -> np.ndarray:
        """Exact ``dJ/du`` (N, N) for ``J = 1/(2 sigma_y^2) ||S p - y||^2``.

        Solves the adjoint ``A lam = S^T (S p - y)/sigma_y^2`` -- reusing ``lu``, since ``A`` is
        self-adjoint -- and forms
        ``dJ/du_m = -1/2 e^{u_m}/h^2 sum_{(m,n) face} (lam_m - lam_n)(p_m - p_n)``.
        """
        N, h2 = self.N, self.h * self.h
        ix, iy = sensors
        resid = (p[ix, iy] - y_obs) / (sigma_y ** 2)
        r_adj = np.zeros((N, N))
        np.add.at(r_adj, (ix, iy), resid)
        r_adj[self.dirichlet] = 0.0
        lam = lu.solve(r_adj.ravel()).reshape(N, N)

        g = np.zeros((N, N))
        contr = (lam[1:, :] - lam[:-1, :]) * (p[1:, :] - p[:-1, :])
        g[:-1, :] += contr; g[1:, :] += contr
        contr = (lam[:, 1:] - lam[:, :-1]) * (p[:, 1:] - p[:, :-1])
        g[:, :-1] += contr; g[:, 1:] += contr
        g *= -0.5 * K / h2
        return g


def beskos_sensors(N: int, n_sensors: int = 33, radius: float | None = None):
    """Beskos/Stuart sensor layout: one at the centre, ``n_sensors-1`` on a near-boundary circle.

    Stuart (2016), Sec. 4.2 / Fig. 1: the 33 noisy head observations sit on a circle of radius
    ``(N-1)/(2N)`` about the domain centre ``(1/2, 1/2)``, plus one central sensor. Physical
    positions are mapped to the nearest node of the solver's vertex grid (``x_i = i/(N-1)``), so a
    reading is exactly ``p[ix, iy]`` -- no interpolation.

    Args:
        N: vertex-grid size the indices are computed for.
        n_sensors: total sensors (1 centre + ``n_sensors-1`` on the circle).
        radius: circle radius in ``[0,1]``; default ``(N-1)/(2N)``.

    Returns:
        ``((ix, iy), (sx, sy))``: grid indices and physical positions in ``[0,1]^2``.
    """
    if radius is None:
        radius = (N - 1) / (2.0 * N)
    ang = np.linspace(0.0, 2.0 * np.pi, n_sensors - 1, endpoint=False)
    sx = np.concatenate([[0.5], 0.5 + radius * np.cos(ang)])
    sy = np.concatenate([[0.5], 0.5 + radius * np.sin(ang)])
    ix = np.clip(np.round(sx * (N - 1)).astype(np.int64), 0, N - 1)
    iy = np.clip(np.round(sy * (N - 1)).astype(np.int64), 0, N - 1)
    return (ix, iy), (sx, sy)
