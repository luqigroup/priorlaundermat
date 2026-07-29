"""Operator-defined resolved/blind subspaces for the Darcy inverse problem.

The paper defines the blind subspace from the OPERATOR, not from a cutoff on the KL mode index.
The 33 sparse head observations depend on the d = 100 KL coefficients through the (nonlinear)
forward map; its linearization at a reference field gives the Jacobian

    J[m, k] = d y_m / d xi_k        (33 x 100).

The singular value decomposition ``J = U S Vh`` ranks directions in xi-space by how strongly the
data see them: the RESOLVED subspace is the span of the top-r right singular vectors (singular
value above a noise-relative cutoff), the BLIND subspace its orthogonal complement (the operator's
near-null, dimension 100 - r). Coverage is read along the blind singular directions.

numpy only, on top of the darcy/kl_prior modules; a full exact Jacobian costs one sparse
factorization and 33 triangular solves, well under a second.
"""
from __future__ import annotations

import numpy as np


def exact_jacobian(fwd, kl, sensors, *, u_ref: np.ndarray | None = None) -> np.ndarray:
    """Jacobian of the sensor readings in whitened-KL coordinates, by the exact discrete adjoint.

    ``(n_sensors, d)``, computed with one factorization of the state matrix and one adjoint
    back-substitution per sensor -- no finite differences, no step-size choice.

    Prefer this to a finite-difference Jacobian whenever the SPLIT matters. A sensor that carries
    no information (the Darcy setup places two on the Dirichlet boundary, where the head is pinned)
    has an exactly-zero row here, around 1e-19, whereas finite differences give it a noise row near
    the step size, which a tight singular-value cutoff then mistakes for rank. That inflates the
    apparent rank and shrinks the exact null by one direction per dead sensor.

    The trick: ``gradient_field`` already performs the adjoint solve for the data misfit, so passing
    it a residual of one unit in sensor ``i`` returns exactly row ``i`` of the field Jacobian.
    """
    u_ref = np.zeros((kl.grid_size, kl.grid_size)) if u_ref is None else np.asarray(u_ref)
    p, lu, K = fwd.solve(u_ref, return_factor=True)
    y0 = fwd.observe(p, sensors)
    m = int(np.size(y0))

    J = np.empty((m, kl.d))
    for i in range(m):
        e = np.zeros(m); e[i] = 1.0
        # residual r = observe(p) - y_obs = e  =>  gradient is J_field^T e = row i (sigma_y = 1)
        g_field = fwd.gradient_field(p, lu, K, sensors, y0 - e, 1.0)
        J[i] = kl.amp * np.einsum("ij,kij->k", g_field, kl.phi)
    return J


def operator_subspace(fwd, kl, sensors, *, u_ref: np.ndarray | None = None,
                      eps: float = 1e-3, sv_cutoff: float | None = None,
                      sigma_y: float = 0.01, exact: bool = False):
    """Return the SVD of the linearized forward and the resolved/blind split.

    Args:
        fwd: :class:`~resolvability.groundwater.darcy.DarcyForward`.
        kl: :class:`~resolvability.groundwater.kl_prior.StuartKLPrior`.
        sensors: ``(ix, iy)`` sensor grid indices.
        u_ref: linearization field (default the prior mean, zeros).
        eps: finite-difference step in whitened-latent units.
        sv_cutoff: singular values below this are blind; default
            ``sigma_y`` (a direction seen below the noise is not resolved).
        sigma_y: observation noise std (used for the default cutoff).
        exact: build the Jacobian by the exact discrete adjoint instead of finite differences.
            Use this whenever the exact null matters -- see :func:`exact_jacobian`.

    Returns:
        dict with ``S`` (33,), ``Vh`` (100,100 right singular vectors), ``rank`` r,
        ``resolved`` (100,r) and ``blind`` (100,100-r) bases (columns are xi-space directions),
        the ``cutoff`` used, and the Jacobian ``J``.
    """
    d = kl.d
    u_ref = np.zeros((kl.grid_size, kl.grid_size)) if u_ref is None else u_ref
    if exact:
        J = exact_jacobian(fwd, kl, sensors, u_ref=u_ref)
    else:
        y0 = fwd.observe(fwd.solve(u_ref), sensors)
        J = np.empty((y0.size, d))
        for k in range(d):
            xi = np.zeros(d); xi[k] = eps
            u = u_ref + kl.reconstruct(xi)             # u_ref + eps*amp_k*phi_k
            J[:, k] = (fwd.observe(fwd.solve(u), sensors) - y0) / eps

    U, S, Vh = np.linalg.svd(J, full_matrices=True)    # U(33,33) S(33,) Vh(100,100)
    cutoff = sigma_y if sv_cutoff is None else sv_cutoff
    rank = int(np.sum(S > cutoff))
    resolved = Vh[:rank].T                             # (100, r)
    blind = Vh[rank:].T                                # (100, 100-r)
    return {"S": S, "Vh": Vh, "rank": rank, "resolved": resolved, "blind": blind,
            "cutoff": cutoff, "J": J}
