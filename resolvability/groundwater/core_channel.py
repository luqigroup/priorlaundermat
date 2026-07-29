"""The added measurement channel of the de-freezing corollary, for the Darcy operator: core
samples of the log-permeability ``u``.

The corollary needs a channel observed independently of the survey, resolving a subspace of the
operator's blind subspace. A core sample reads the log-permeability directly, so the channel row is
exact and linear in the KL coefficients, ``ker([A; C]) = ker A n ker C`` holds for the RAW reading,
and the de-frozen set is just the projection of ``C``'s rows onto ``ker A``. It is a different
measurement, not more of the same survey -- which is also the point the corollary makes: restricted
to the survey's null the head Jacobian tops out near the noise even with a sensor at every node, so
more head sensors cannot buy these directions back.

    from resolvability.groundwater.core_channel import select_core_nodes, build_core_channel
    nodes = select_core_nodes(kl, blind, n_cores=6)
    C = build_core_channel(kl, nodes)          # (n_cores, d), rows are exact and linear in xi

numpy only.
"""

from __future__ import annotations

import numpy as np

__all__ = ["build_core_channel", "full_field_rows", "select_core_nodes", "simulate_cores",
           "channel_subspaces"]


def full_field_rows(kl) -> np.ndarray:
    """Channel rows for a core sample at EVERY grid node: ``(N*N, d)``.

    Row ``i*N + j`` maps the whitened coefficients to the log-permeability at node ``(i, j)``,
    ``u(i,j) = sum_k amp_k xi_k phi_k(i,j)``, which is exact and linear -- no PDE solve, and no
    linearization, so unlike the head sensors these rows do not rotate with the field.
    """
    amp, phi = np.asarray(kl.amp, np.float64), np.asarray(kl.phi, np.float64)   # (d,), (d,N,N)
    d, N, _ = phi.shape
    return (amp[:, None] * phi.reshape(d, N * N)).T                             # (N*N, d)


def build_core_channel(kl, nodes) -> np.ndarray:
    """Channel matrix ``C`` for core samples at ``nodes`` (a sequence of ``(i, j)`` grid indices)."""
    N = int(np.asarray(kl.phi).shape[-1])
    rows = full_field_rows(kl)
    idx = [int(i) * N + int(j) for i, j in nodes]
    return rows[idx]


def select_core_nodes(kl, blind: np.ndarray, n_cores: int = 6, *, exclude_edges: bool = False,
                      candidates=None, objective: str = "logdet", sigma_u: float = 0.04
                      ) -> list[tuple[int, int]]:
    """Greedy core placement RESTRICTED to the blind subspace.

    Picks nodes whose channel rows, projected onto ``blind``, buy the most blind information.
    Restricting to the blind subspace is what makes the de-freeze measurable rather than nominal --
    an unrestricted or hand-placed design tends to resolve directions the survey already sees, or
    blind directions so weakly that they land under the noise.

    ``objective``:

    ``"logdet"`` (default) maximizes ``logdet(I + C_S C_S^T / sigma_u^2)``, the information the
        selected cores carry about the blind coordinates. This objective is monotone submodular, so
        the greedy solution is within ``1 - 1/e`` of the optimum -- a guarantee, not a hope.
    ``"emin"`` maximizes the smallest singular value of the selected projected rows. It is the right
        criterion when the report must hold for EVERY blind functional, since coverage is a minimum
        over directions, but it is **not submodular**, so greedy carries no guarantee and can land
        well short of the optimum. Use it only with the achieved value checked against a bound.

    ``blind`` : ``(d, b)`` basis of the blind subspace, columns in coefficient space. Pass the
    EXACT null of the Jacobian, not the noise-cutoff blind set, and beware that a finite-difference
    Jacobian gives the Dirichlet-pinned head sensors FD-noise rows that a tight cutoff mistakes for
    rank; the exact adjoint Jacobian (:func:`resolvability.groundwater.subspace.exact_jacobian`) is
    the reliable input here.

    ``exclude_edges`` is off by default: it guards against sensors on the Dirichlet boundary, where
    the HEAD is pinned, but a core sample reads the log-permeability, which is pinned nowhere, so
    boundary nodes are legitimate core locations.
    """
    B = np.asarray(blind, np.float64)
    R = full_field_rows(kl) @ B                                   # (N*N, b) candidates, in blind coords
    N = int(np.asarray(kl.phi).shape[-1])

    ok = np.ones(R.shape[0], bool)
    if exclude_edges:                       # the head is pinned on the Dirichlet edges
        ij = np.arange(R.shape[0])
        ok &= (ij % N != 0) & (ij % N != N - 1)
    if candidates is not None:
        mask = np.zeros(R.shape[0], bool)
        mask[[int(i) * N + int(j) for i, j in candidates]] = True
        ok &= mask

    if objective not in ("logdet", "emin"):
        raise ValueError(f"objective must be 'logdet' or 'emin', got {objective!r}")

    def score_of(idx: list[int]) -> float:
        M = R[idx]
        if objective == "logdet":
            return float(np.linalg.slogdet(np.eye(len(idx)) + M @ M.T / sigma_u ** 2)[1])
        sv = np.linalg.svd(M, compute_uv=False)
        return sv[-1] if len(idx) > 1 else sv[0]     # first pick: largest row; then max-min
    chosen: list[int] = []
    for _ in range(int(n_cores)):
        best, best_score = -1, -np.inf
        for p in np.flatnonzero(ok):
            if p in chosen:
                continue
            v = score_of(chosen + [p])
            if v > best_score:
                best, best_score = p, v
        if best < 0:
            break
        chosen.append(best)
    return [(p // N, p % N) for p in chosen]


def channel_subspaces(C: np.ndarray, blind: np.ndarray, tol: float = 1e-10):
    """Split the blind subspace into what the channel resolves and what it leaves frozen.

    Returns ``(defrozen, control)`` with columns in coefficient space: ``defrozen`` spans the blind
    directions the channel sees -- the ones joint curation must correct -- and ``control`` spans
    the rest of the blind subspace, which must NOT move. The control is the experiment's null.
    """
    B = np.asarray(blind, np.float64)
    M = np.asarray(C, np.float64) @ B                             # (m, b) channel in blind coords
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    k = int((s > tol * max(s.max(), 1.0)).sum())
    W = Vt[:k]                                                    # (k, b)
    b = B.shape[1]
    Wc = np.linalg.svd(np.eye(b) - W.T @ W)[0][:, : b - k].T if b > k else np.zeros((0, b))
    return B @ W.T, B @ Wc.T                                      # (d,k), (d,b-k)


def simulate_cores(xi_true: np.ndarray, C: np.ndarray, sigma_u: float,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """Synthesize the channel reading ``z = C xi + eps``, independent of the survey noise."""
    rng = np.random.default_rng() if rng is None else rng
    xi = np.asarray(xi_true, np.float64)
    z = xi @ np.asarray(C, np.float64).T
    return z + sigma_u * rng.standard_normal(z.shape)
