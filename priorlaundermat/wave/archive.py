"""Legacy reconstruction archives: the MAP (single-best) archive and the blind-block freeze.

``select_alpha`` -- Morozov's discrepancy principle for the smoothness strength: pick
``alpha`` so the legacy MAP's data misfit chi^2 per degree of freedom matches its expectation.
It uses only the observations and the known noise level, never the truths.

``build_map_archive`` -- the single-best archive: one legacy MAP per observation,

    x_hat(y) = argmin_x ||F(x) - y||^2 / (2 sigma^2) + R(x),

with the misspecified smoothness ``R``. With no data in the shadow the smooth regularizer
extrapolates a nearly flat continuation, annihilating the rough shadow modes.

``build_freeze_archive`` -- posterior draws under the legacy model. Because the causal shadow
is an exact kernel of ``F`` (its Jacobian columns are machine zero there), the likelihood is
exactly flat on the blind subspace, so the legacy posterior restricted to the shadow is the
regularizer's own Gaussian conditional

    x_shadow | rest  ~  N(x_hat_shadow, (alpha L_SS + rho I)^{-1}),

an exact -- not Laplace-approximate -- picture of the frozen regularizer belief, valid for any
``eps`` and needing no ill-conditioned full-Hessian inversion. It is wider than the single MAP
(it retains the regularizer's belief spread) yet, when the regularizer is overconfident, still
tighter than the rough truth.

The MAP is solved by a batched, per-row-independent Adam: the objective is row separable, so a
batched solve equals ``n`` independent solves. All PDE and linear-algebra math is float64.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from priorlaundermat.wave.blind import WaveBlindSubspace
from priorlaundermat.wave.regularizers import (
    SmoothnessRegularizer,
    build_smoothness_regularizer,
)
from priorlaundermat.wave.solver import SemilinearWave1D


def map_objective(solver: SemilinearWave1D, x: Tensor, y: Tensor,
                  reg: SmoothnessRegularizer, sigma: float) -> Tensor:
    """Per-row negative log legacy posterior ``(n,)``: data misfit plus ``R``."""
    r = solver.forward(x) - y
    misfit = 0.5 * r.square().sum(dim=-1) / (sigma ** 2)
    return misfit + reg.value(x)


@dataclass
class MAPResult:
    x: Tensor                # (n, d_x) reconstructions
    chi2_per_dof: Tensor     # (n,) ||F(x)-y||^2 / (d_y sigma^2)


def solve_map_adam(
    solver: SemilinearWave1D, y: Tensor, reg: SmoothnessRegularizer, sigma: float,
    *, x0: Optional[Tensor] = None, n_iter: int = 1500, lr: float = 5e-2,
    lr_final: Optional[float] = 1e-3,
) -> MAPResult:
    """Batched, per-row-independent MAP by Adam (deterministic zero init by default)."""
    n = y.shape[0]
    if x0 is None:
        x0 = torch.zeros(n, solver.d_x, dtype=solver.dtype, device=solver.device)
    x = x0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter, eta_min=lr_final)
             if lr_final is not None else None)
    for _ in range(n_iter):
        opt.zero_grad()
        map_objective(solver, x, y, reg, sigma).sum().backward()
        opt.step()
        if sched is not None:
            sched.step()
    x = x.detach()
    with torch.no_grad():
        r = solver.forward(x) - y
        chi2 = r.square().sum(dim=-1) / (solver.d_y * sigma ** 2)
    return MAPResult(x=x, chi2_per_dof=chi2)


def select_alpha(
    solver: SemilinearWave1D, y_cal: Tensor, sigma: float, *, rho: float = 1e-2,
    target_chi2_per_dof: float = 1.0, log10_bracket: tuple[float, float] = (-2.0, 3.0),
    n_bisect: int = 8, map_iters: int = 3000, verbose: bool = False,
) -> tuple[float, dict]:
    """Morozov discrepancy: choose ``alpha`` so the mean chi^2 per dof matches ``target``.

    ``chi2/dof`` increases with ``alpha`` (more regularization means a worse data fit), so we
    bisect on ``log10(alpha)``. Only ``y_cal`` and the known ``sigma`` are used, never the
    truths.

    The inner MAP is solved to at least the archive's iteration budget (``map_iters``; pass
    the archive value) from the same zero initialization, so a still-descending MAP cannot
    report an inflated chi^2 and drive ``alpha`` to a spuriously low value. Because the inner
    solve mirrors the archive solve exactly, ``chi2_final`` predicts the archive MAP's chi^2.

    Returns:
        ``(alpha, info)``. ``info`` reports whether the discrepancy target was actually
        BRACKETED (``chi2_lo <= target <= chi2_hi``), whether the selected ``alpha`` is
        strictly INTERIOR to the log bracket, and the achieved ``chi2_final``.
    """
    lo0, hi0 = log10_bracket
    trace: list[tuple[float, float]] = []

    def chi2_at(log10a: float) -> float:
        reg = build_smoothness_regularizer(n=solver.d_x, alpha=10.0 ** log10a, rho=rho,
                                           dtype=solver.dtype, device=solver.device)
        res = solve_map_adam(solver, y_cal, reg, sigma, n_iter=map_iters)
        c = float(res.chi2_per_dof.mean())
        trace.append((10.0 ** log10a, c))
        if verbose:
            print(f"[alpha] log10a={log10a:+.3f}  chi2/dof={c:.3f}")
        return c

    # Bracket the target at the bracket ends (chi2 is increasing in alpha).
    c_lo = chi2_at(lo0)
    c_hi = chi2_at(hi0)
    bracketed = bool(c_lo <= target_chi2_per_dof <= c_hi)

    lo, hi = lo0, hi0
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        if chi2_at(mid) > target_chi2_per_dof:
            hi = mid
        else:
            lo = mid
    log10_alpha = 0.5 * (lo + hi)
    alpha = 10.0 ** log10_alpha
    chi2_final = chi2_at(log10_alpha)

    # Strictly interior: not floored within a couple of final-bracket widths of an end.
    margin = 2.0 * (hi0 - lo0) / (2 ** n_bisect)
    interior = bool((lo0 + margin) < log10_alpha < (hi0 - margin))
    info = {
        "alpha": alpha, "log10_alpha": log10_alpha,
        "chi2_lo": c_lo, "chi2_hi": c_hi, "chi2_final": chi2_final,
        "bracketed": bracketed, "interior": interior,
        "log10_bracket": (lo0, hi0), "map_iters": map_iters, "trace": trace,
    }
    return alpha, info


def build_map_archive(
    solver: SemilinearWave1D, y: Tensor, reg: SmoothnessRegularizer, sigma: float,
    *, map_iters: int = 1500, lr: float = 5e-2,
) -> MAPResult:
    """Single-best archive: one legacy MAP per observation, from a fixed zero init."""
    return solve_map_adam(solver, y, reg, sigma, n_iter=map_iters, lr=lr)


def build_freeze_archive(
    x_map: Tensor, split: WaveBlindSubspace, reg: SmoothnessRegularizer,
    *, n_per: int, seed: int, jitter: float = 1e-9,
) -> Tensor:
    """Freeze archive: exact regularizer-conditional posterior draws on the blind subspace.

    On the exact causal-shadow kernel the likelihood is flat, so
    ``x_shadow ~ N(x_map_shadow, (alpha L_SS + rho I)^{-1})`` is the exact legacy posterior
    there. Resolved coordinates are copied from the MAP: they are data determined, and the
    freeze statement is about the blind subspace. Returns a field archive
    ``(n_inst * n_per, d_x)``.

    Args:
        x_map: ``(n_inst, d_x)`` MAP reconstructions.
        split: the fixed blind subspace (gives the shadow indices).
        reg: the misspecified smoothness regularizer (gives ``alpha`` and ``rho``).
        n_per: draws per instance.
        seed: RNG seed for the draws.
    """
    shadow = split.shadow_idx
    nb = split.n_blind
    L = reg.L
    Lss = L[shadow][:, shadow]
    Hbb = reg.alpha * Lss + reg.rho * torch.eye(nb, dtype=L.dtype, device=L.device)
    Hbb = 0.5 * (Hbb + Hbb.T) + jitter * torch.eye(nb, dtype=L.dtype, device=L.device)
    Lh = torch.linalg.cholesky(Hbb)
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_inst = x_map.shape[0]
    out = []
    for i in range(n_inst):
        z = torch.randn(nb, n_per, generator=g, dtype=L.dtype)
        if L.device is not None and str(L.device) != "cpu":
            z = z.to(L.device)
        step = torch.linalg.solve_triangular(Lh.T, z, upper=True)   # (nb, n_per)
        base = x_map[i].unsqueeze(1).repeat(1, n_per)               # (d_x, n_per)
        base[shadow] = x_map[i, shadow].unsqueeze(1) + step
        out.append(base.T)
    return torch.cat(out, dim=0).contiguous()                       # (n_inst*n_per, d_x)
