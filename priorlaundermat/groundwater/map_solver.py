"""Single-best (MAP) reconstruction for the Darcy inverse problem.

The legacy archive stores the maximum-a-posteriori estimate under the CORRECT Gaussian (KL) prior
-- the single-best curation the paper studies. In the whitened KL coordinates ``xi`` the negative
log-posterior is

    J(xi) = 1/(2 sigma_y^2) || S p(u(xi)) - y ||^2  +  1/2 || xi ||^2
            [ ------------- data misfit ------------- ]   [ N(0,I) prior ]

with ``u(xi) = sum_k amp_k xi_k phi_k`` the field and ``p`` the head. The data gradient is the
exact adjoint of :mod:`priorlaundermat.groundwater.darcy` pulled back to ``xi`` by the chain rule
``dJ/dxi_k = amp_k <dJ/du, phi_k>``. Minimized with L-BFGS-B; CPU, a few seconds per truth at the
committed iteration budget.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def _neg_log_post(xi, fwd, kl, sensors, y_obs, sigma_y):
    """Return ``(J, grad_xi)`` for the whitened-KL negative log-posterior."""
    u = kl.reconstruct(xi)
    p, lu, K = fwd.solve(u, return_factor=True)
    r = fwd.observe(p, sensors) - y_obs
    misfit = 0.5 * float(r @ r) / sigma_y ** 2
    J = misfit + 0.5 * float(xi @ xi)
    g_field = fwd.gradient_field(p, lu, K, sensors, y_obs, sigma_y)     # dJ_data/du
    g_data = kl.amp * np.einsum("ij,kij->k", g_field, kl.phi)           # -> dJ_data/dxi
    return J, g_data + xi


def map_reconstruct(fwd, kl, sensors, y_obs, sigma_y, *, xi0=None, maxiter: int = 200):
    """MAP estimate ``xi_hat`` under the correct KL prior.

    Args:
        fwd: :class:`~priorlaundermat.groundwater.darcy.DarcyForward` on the inversion grid.
        kl: :class:`~priorlaundermat.groundwater.kl_prior.StuartKLPrior` (gives ``amp``, ``phi``).
        sensors: ``(ix, iy)`` sensor grid indices.
        y_obs: (n_sensors,) noisy head observations.
        sigma_y: observation noise standard deviation.
        xi0: initial latent (default zeros = the prior mean).
        maxiter: L-BFGS-B iteration cap.

    Returns:
        ``xi_hat`` (d,), the MAP latent.
    """
    d = kl.d
    xi0 = np.zeros(d) if xi0 is None else np.asarray(xi0, dtype=np.float64)

    def fun(xi):
        return _neg_log_post(xi, fwd, kl, sensors, y_obs, sigma_y)

    res = minimize(fun, xi0, jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-10})
    return res.x
