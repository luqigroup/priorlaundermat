"""The causal-shadow blind subspace, defined from the solver rather than from hand geometry.

The blind subspace is read off the forward map's exact sensitivity: at several random states
we form the Jacobian ``dF/dx`` and take the columns that are MACHINE-EXACTLY zero. Those
coordinates -- the causal shadow of the interior receivers over the short window -- are
invisible to ``F`` at every state, for both the linear and the nonlinear solver. The builder
verifies that the exactly-zero column set is identical across states (a fixed subspace, no
rotation with the field) and returns it.

Within the (large, coordinate-aligned) blind subspace we score a fixed orthonormal basis of
ROUGHNESS modes: the eigenvectors of the graph Laplacian restricted to the shadow block,
ordered smooth to rough. This is the basis in which the misspecified smoothness regularizer's
belief is most wrong -- it collapses the high-roughness shadow modes, where the rough truth
genuinely varies, while the smooth shadow modes track the resolved boundary. The resolved
subspace gets an illumination-ranked basis for the fairness control. Everything here is
truth-independent and read once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from priorlaundermat.wave.solver import SemilinearWave1D


def graph_laplacian_1d(n: int, dtype: torch.dtype = torch.float64) -> Tensor:
    """1D path-graph Laplacian ``L = G^T G`` (``G`` = first-difference gradient).

    ``x^T L x = sum_j (x_{j+1} - x_j)^2`` is the discrete roughness energy; PSD and
    tridiagonal, with the constant vector in its null space.
    """
    L = torch.zeros(n, n, dtype=dtype)
    for i in range(n - 1):
        L[i, i] += 1.0
        L[i + 1, i + 1] += 1.0
        L[i, i + 1] -= 1.0
        L[i + 1, i] -= 1.0
    return L


def _roughness_basis(block_idx: Tensor, n: int, dtype: torch.dtype, device=None
                     ) -> tuple[Tensor, Tensor]:
    """Roughness eigenbasis of the graph-Laplacian sub-block on ``block_idx``.

    Returns ``(Q, eigvals)`` where ``Q`` has ``len(block_idx)`` orthonormal ROWS (each a full
    ``n``-vector supported on the block) ordered smooth to rough, and ``eigvals`` the
    corresponding roughness eigenvalues (ascending).
    """
    L = graph_laplacian_1d(n, dtype=dtype).to(device)
    Lbb = L[block_idx][:, block_idx]
    evals, evecs = torch.linalg.eigh(Lbb)          # ascending
    Q = torch.zeros(block_idx.numel(), n, dtype=dtype, device=device)
    Q[:, block_idx] = evecs.T                      # rows = modes, embedded in R^n
    return Q, evals


@dataclass
class WaveBlindSubspace:
    """Fixed causal-shadow blind subspace with its blind and resolved scoring bases.

    Attributes:
        shadow_idx: Coordinate indices that are machine-exactly blind (the causal shadow).
            ``span{e_j : j in shadow_idx}`` is the blind subspace.
        resolved_idx: The complementary (resolved) coordinate indices.
        Q_blind: ``(n_blind, n)`` orthonormal rows -- the shadow ROUGHNESS modes, smooth to
            rough. Scoring basis for the blind functionals ``v^T x``; the misspecified
            smoothness regularizer is tightest on the rough end.
        Q_resolved: ``(n_resolved, n)`` orthonormal rows -- the resolved directions ranked by
            ILLUMINATION (mean-Gram eigenvalue), most to least illuminated. Ranking resolved
            directions by illumination rather than roughness matters: the weakly lit nodes
            just inside the shadow boundary behave near-blindly, and a roughness ranking
            would conflate them with the well-resolved core.
        rough_eigvals_blind: roughness (Laplacian) eigenvalue per blind mode (ascending).
        illum_eigvals_resolved: mean-Gram (illumination) eigenvalue per resolved mode
            (descending; large = strongly data-determined).
        n: State dimension.
        max_shadow_sensitivity: max ``|dF/dx_j|`` over shadow columns (should be 0).
        min_resolved_sensitivity: min over resolved columns of max ``|dF/dx_j|``.
        identical_across_states: the exact-zero set matched at every probe state.
        n_probe_states: number of states probed.
    """

    shadow_idx: Tensor
    resolved_idx: Tensor
    Q_blind: Tensor
    Q_resolved: Tensor
    rough_eigvals_blind: Tensor
    illum_eigvals_resolved: Tensor
    n: int
    max_shadow_sensitivity: float
    min_resolved_sensitivity: float
    identical_across_states: bool
    n_probe_states: int

    @property
    def n_blind(self) -> int:
        return int(self.shadow_idx.numel())

    @property
    def n_resolved(self) -> int:
        return int(self.resolved_idx.numel())

    def project_blind(self, X: Tensor) -> Tensor:
        """Blind roughness coordinates ``(..., n_blind)`` of fields ``X (..., n)``."""
        return X.to(self.Q_blind) @ self.Q_blind.T

    def project_resolved(self, X: Tensor) -> Tensor:
        """Resolved illumination coordinates ``(..., n_resolved)`` of ``X (..., n)``."""
        return X.to(self.Q_resolved) @ self.Q_resolved.T

    def rough_mask_blind(self, frac: float = 0.5) -> Tensor:
        """Boolean mask of the roughest ``frac`` of blind modes (the collapse regime)."""
        k = max(1, int(round(frac * self.n_blind)))
        mask = torch.zeros(self.n_blind, dtype=torch.bool)
        mask[self.n_blind - k:] = True
        return mask

    def well_illuminated_mask(self, frac: float = 0.5) -> Tensor:
        """Boolean mask of the best-illuminated ``frac`` of resolved modes (fairness set).

        ``Q_resolved`` is ordered most to least illuminated, so the fairness control is the
        LEADING ``frac`` of modes; the trailing modes are the weakly lit shadow-boundary
        transition band, which is near-blind and expected to tighten.
        """
        k = max(1, int(round(frac * self.n_resolved)))
        mask = torch.zeros(self.n_resolved, dtype=torch.bool)
        mask[:k] = True
        return mask


def build_wave_blind_subspace(
    solver: SemilinearWave1D,
    *,
    n_probe_states: int = 8,
    tol: float = 0.0,
    seed: int = 0,
    verify_eps: Optional[float] = 0.0,
) -> WaveBlindSubspace:
    """Extract the fixed causal-shadow blind subspace from the solver's Jacobian.

    Forms ``dF/dx`` at ``n_probe_states`` random states and takes the columns whose maximum
    magnitude is ``<= tol`` (``tol = 0`` is machine-exact zero). Verifies that the
    exactly-zero column set is identical across all probe states, and -- if ``verify_eps`` is
    given -- that a solver with the other nonlinearity yields the same blind set. Then builds
    the shadow and resolved scoring bases.

    Args:
        solver: The wave forward map (its ``eps`` defines the operator).
        n_probe_states: Random states at which to test exact-zero columns.
        tol: Blind threshold on ``|dF/dx_j|`` (0 = machine exact).
        seed: RNG seed for the probe states.
        verify_eps: If not None, also check the blind set against a solver with
            ``eps=verify_eps`` (default 0: confirm the linear operator shares the exact same
            shadow, so nonlinearity does not move it).
    """
    n = solver.n
    dev = solver.device
    g = torch.Generator(device=dev if dev is not None else "cpu").manual_seed(seed)

    def blind_cols(slv: SemilinearWave1D, gen: torch.Generator):
        masks = []
        max_shadow = 0.0
        min_resolved = float("inf")
        gram = torch.zeros(n, n, dtype=slv.dtype, device=dev)  # mean Gram E[J^T J]
        for _ in range(n_probe_states):
            x = torch.randn(n, generator=gen, dtype=slv.dtype, device=dev)
            J = slv.jacobian(x)
            gram += J.T @ J
            colmax = J.abs().max(dim=0).values
            m = colmax <= tol
            masks.append(m)
            if m.any():
                max_shadow = max(max_shadow, float(colmax[m].max()))
            if (~m).any():
                min_resolved = min(min_resolved, float(colmax[~m].min()))
        masks = torch.stack(masks)
        identical = bool((masks == masks[0]).all())
        return masks[0], max_shadow, min_resolved, identical, gram / n_probe_states

    mask0, max_shadow, min_resolved, identical, gram = blind_cols(solver, g)

    if verify_eps is not None and verify_eps != solver.eps:
        alt = SemilinearWave1D(
            n=solver.n, c=solver.c, cfl=solver.cfl, recv=solver.recv,
            nt=solver.nt, eps=verify_eps, domain_len=solver.domain_len,
            dtype=solver.dtype, device=solver.device,
        )
        g2 = torch.Generator(device=dev if dev is not None else "cpu").manual_seed(seed + 1)
        mask_alt, _, _, id_alt, _ = blind_cols(alt, g2)
        identical = identical and id_alt and bool((mask_alt == mask0).all())

    shadow_idx = torch.nonzero(mask0, as_tuple=False).squeeze(-1)
    resolved_idx = torch.nonzero(~mask0, as_tuple=False).squeeze(-1)
    if shadow_idx.numel() == 0:
        raise ValueError(
            "No exactly-blind coordinates found: shorten the window (nt) or move the "
            "receivers inward so a causal shadow exists.")

    # Blind subspace: roughness modes of the shadow block (smooth to rough).
    Q_blind, ev_blind = _roughness_basis(shadow_idx, n, solver.dtype, device=dev)
    # Resolved subspace: illumination modes -- eigenvectors of the mean Gram restricted to
    # the resolved block, ordered by descending illumination. The Gram is exactly block
    # diagonal (shadow columns of J are exactly zero), so the shadow directions are exact
    # zero eigenvectors and never mix into the resolved illumination modes.
    Grr = gram[resolved_idx][:, resolved_idx]
    ev_res, U = torch.linalg.eigh(0.5 * (Grr + Grr.T))     # ascending
    order = torch.argsort(ev_res, descending=True)         # most illuminated first
    illum_eigvals = ev_res[order]
    Q_resolved = torch.zeros(resolved_idx.numel(), n, dtype=solver.dtype, device=dev)
    Q_resolved[:, resolved_idx] = U[:, order].T

    return WaveBlindSubspace(
        shadow_idx=shadow_idx, resolved_idx=resolved_idx,
        Q_blind=Q_blind, Q_resolved=Q_resolved,
        rough_eigvals_blind=ev_blind, illum_eigvals_resolved=illum_eigvals,
        n=n, max_shadow_sensitivity=max_shadow,
        min_resolved_sensitivity=(0.0 if min_resolved == float("inf") else min_resolved),
        identical_across_states=identical, n_probe_states=n_probe_states,
    )
