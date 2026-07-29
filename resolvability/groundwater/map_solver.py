"""Single-best (MAP) reconstruction for the Darcy inverse problem.

The legacy archive stores the maximum-a-posteriori estimate under the CORRECT Gaussian (KL) prior
-- the single-best curation the paper studies. In the whitened KL coordinates ``xi`` the negative
log-posterior is

    J(xi) = 1/(2 sigma_y^2) || S p(u(xi)) - y ||^2  +  1/2 || xi ||^2
            [ ------------- data misfit ------------- ]   [ N(0,I) prior ]

with ``u(xi) = sum_k amp_k xi_k phi_k`` the field and ``p`` the head. The data gradient is the
exact adjoint of :mod:`resolvability.groundwater.darcy` pulled back to ``xi`` by the chain rule
``dJ/dxi_k = amp_k <dJ/du, phi_k>``. Minimized with L-BFGS-B; CPU, a few seconds per truth at the
committed iteration budget.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def _neg_log_post(xi, fwd, kl, sensors, y_obs, sigma_y, channel=None):
    """Return ``(J, grad_xi)`` for the whitened-KL negative log-posterior.

    ``channel`` : optional ``(C, z, sigma_u)`` added measurement, independent of the survey given
    the field -- the de-freezing corollary's extra channel. ``C`` is linear in the coefficients (a
    core sample reads ``u`` directly), so it contributes ``||C xi - z||^2 / (2 sigma_u^2)`` and its
    exact gradient, with no extra PDE solve.
    """
    u = kl.reconstruct(xi)
    try:
        p, lu, K = fwd.solve(u, return_factor=True)
    except RuntimeError:
        # A trial iterate wandered far enough that exp(u) overflowed and the stiffness matrix
        # factored as exactly singular. That is a step to reject, not a run to abort: report an
        # infinite objective with a gradient pointing back toward the prior mean, and the line
        # search backtracks. This cannot alter any iterate that solved successfully.
        return np.inf, np.asarray(xi, np.float64).copy()
    r = fwd.observe(p, sensors) - y_obs
    misfit = 0.5 * float(r @ r) / sigma_y ** 2
    J = misfit + 0.5 * float(xi @ xi)
    g_field = fwd.gradient_field(p, lu, K, sensors, y_obs, sigma_y)     # dJ_data/du
    g_data = kl.amp * np.einsum("ij,kij->k", g_field, kl.phi)           # -> dJ_data/dxi
    g = g_data + xi
    if channel is not None:
        C, z, sigma_u = channel
        rc = np.asarray(C, np.float64) @ xi - np.asarray(z, np.float64)
        J += 0.5 * float(rc @ rc) / sigma_u ** 2
        g = g + (np.asarray(C, np.float64).T @ rc) / sigma_u ** 2
    return J, g


def map_reconstruct(fwd, kl, sensors, y_obs, sigma_y, *, xi0=None, maxiter: int = 200,
                    channel=None):
    """MAP estimate ``xi_hat`` under the correct KL prior.

    Args:
        fwd: :class:`~resolvability.groundwater.darcy.DarcyForward` on the inversion grid.
        kl: :class:`~resolvability.groundwater.kl_prior.StuartKLPrior` (gives ``amp``, ``phi``).
        sensors: ``(ix, iy)`` sensor grid indices.
        y_obs: (n_sensors,) noisy head observations.
        sigma_y: observation noise standard deviation.
        xi0: initial latent (default zeros = the prior mean).
        maxiter: L-BFGS-B iteration cap.
        channel: optional ``(C, z, sigma_u)`` added measurement (the de-freezing corollary's extra
            channel). Curating JOINTLY means the archive itself is produced under the stacked
            operator, so this must be passed when building the joint archive -- reusing a
            survey-only archive silently invalidates the comparison. Note the augmented problem
            needs a larger ``maxiter`` than the survey-only one.

    Returns:
        ``xi_hat`` (d,), the MAP latent.
    """
    d = kl.d
    xi0 = np.zeros(d) if xi0 is None else np.asarray(xi0, dtype=np.float64)

    def fun(xi):
        return _neg_log_post(xi, fwd, kl, sensors, y_obs, sigma_y, channel=channel)

    res = minimize(fun, xi0, jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-10})
    return res.x
