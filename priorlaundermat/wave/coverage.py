"""Sampler-free blind-subspace coverage.

Coverage is measured on the PRIOR marginals of the projected coordinate ``c_j = v_j^T x`` in
the fixed shadow roughness basis ``V``, not on any posterior. Each generative prior (curated
or oracle) is sampled directly and its marginal along ``v_j`` compared with the held-out
truths' ``v_j^T x_true``. Keeping the posterior sampler out of the estimate isolates what the
prior believes and sidesteps multimodal non-mixing.

Per direction and credible level ``1 - alpha`` we compute the central interval
``[q_{a/2}, q_{1-a/2}]`` of the model marginal and the fraction of truths it contains.

The oracle is the control: :func:`oracle_control_checks` verifies that it sits at nominal on
the blind (rough) modes and that neither arm undercovers on the well-illuminated resolved
modes, before any curated shortfall is read.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from torch import Tensor


def central_coverage_1d(model: np.ndarray, truths: np.ndarray, level: float
                        ) -> tuple[float, tuple[float, float]]:
    """Central-interval coverage: the model's ``[q_{a/2}, q_{1-a/2}]`` against the truths."""
    alpha = 1.0 - level
    lo, hi = np.quantile(model, [alpha / 2, 1 - alpha / 2])
    covered = (truths >= lo) & (truths <= hi)
    return float(covered.mean()), (float(lo), float(hi))


@dataclass
class ArmCoverage:
    """Per-direction coverage for one arm (index = mode ``j``, smooth to rough)."""

    central: np.ndarray        # (d,) central-interval coverage
    level: float

    def mean_over(self, mask: np.ndarray) -> dict:
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() == 0:
            return {"central": float("nan"), "n": 0}
        return {"central": float(self.central[mask].mean()), "n": int(mask.sum())}


def arm_coverage(coords_model: Tensor, coords_true: Tensor, *, level: float) -> ArmCoverage:
    """Coverage of every projected direction for one arm.

    Args:
        coords_model: ``(S, d)`` projected prior samples ``V^T x_model``.
        coords_true: ``(T, d)`` projected held-out truths ``V^T x_true``.
        level: credible level ``1 - alpha`` (e.g. 0.9 for central-90%).
    """
    M = coords_model.detach().cpu().numpy()
    Tr = coords_true.detach().cpu().numpy()
    d = M.shape[1]
    cen = np.empty(d)
    for j in range(d):
        cen[j], _ = central_coverage_1d(M[:, j], Tr[:, j], level)
    return ArmCoverage(central=cen, level=level)


def oracle_control_checks(
    oracle_blind: ArmCoverage, curated_blind: ArmCoverage,
    oracle_resolved: ArmCoverage, curated_resolved: ArmCoverage, *,
    rough_mask: np.ndarray, resolved_mask: np.ndarray,
    level: float, tol: float = 0.1,
) -> dict:
    """Gate that must pass before any curated-shortfall claim is read.

    Passes iff (a) the ORACLE sits near nominal on the ROUGH blind modes, which is its control
    role, and (b) neither arm materially undercovers on the WELL-ILLUMINATED resolved modes.
    The weakly lit shadow-boundary transition band is excluded from (b): it is near-blind and
    expected to tighten.
    """
    o_blind = oracle_blind.mean_over(rough_mask)
    c_blind = curated_blind.mean_over(rough_mask)
    o_res = oracle_resolved.mean_over(resolved_mask)
    c_res = curated_resolved.mean_over(resolved_mask)
    oracle_nominal = o_blind["n"] > 0 and abs(o_blind["central"] - level) <= tol
    resolved_fair = (
        (o_res["n"] == 0 or o_res["central"] >= level - tol) and
        (c_res["n"] == 0 or c_res["central"] >= level - tol))
    return {
        "passed": bool(oracle_nominal and resolved_fair),
        "oracle_nominal_on_rough_blind": bool(oracle_nominal),
        "resolved_subspace_fair": bool(resolved_fair),
        "oracle_rough_blind": o_blind, "curated_rough_blind": c_blind,
        "oracle_resolved": o_res, "curated_resolved": c_res,
        "level": level, "tol": tol,
    }
