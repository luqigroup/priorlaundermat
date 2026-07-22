"""The exactly solvable linear--Gaussian model, where every claim is a matrix identity.

The truth is Gaussian, the regularizer is Gaussian, the operator is linear with a genuine kernel,
so the curated prior, the posterior, and the coverage all have closed forms and any miscalibration
is the prior's rather than the inference's. This module holds the pieces the closed-form checks
and the mechanism dial both read, so the two cannot drift apart.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import inv


def conditional_blind_variance(S: np.ndarray, Q: np.ndarray, r: int) -> np.ndarray:
    """Per-direction variance of the blind coordinates given the resolved ones.

    ``Q`` is the orthonormal split with the first ``r`` columns resolved and the rest blind. The
    conditional variance is the Schur complement of the resolved block, which is the quantity the
    theory freezes -- not the blind *marginal*, which stays positive even when every conditional
    interval has collapsed.
    """
    Sq = Q.T @ S @ Q
    RR, RB, BB = Sq[:r, :r], Sq[:r, r:], Sq[r:, r:]
    return np.diag(BB - RB.T @ inv(RR) @ RB)


def curated_covariances(S_rho: np.ndarray, S_star: np.ndarray, A: np.ndarray,
                        Gam: np.ndarray) -> dict[str, np.ndarray]:
    """The two curated priors and the posterior they come from.

    With regularizer covariance ``S_rho``, truth covariance ``S_star``, operator ``A`` and noise
    covariance ``Gam``, returns the posterior covariance ``S_post``, the gain ``G``, the data
    covariance ``S_y``, and the covariances of the two archives: ``S_qS`` for an archive of
    posterior *samples* and ``S_qM`` for one of single-best reconstructions. Their difference is
    exactly ``S_post`` -- what storing only the best reconstruction discards.
    """
    Gi = inv(Gam)
    S_post = inv(inv(S_rho) + A.T @ Gi @ A)
    G = S_post @ A.T @ Gi
    S_y = A @ S_star @ A.T + Gam
    return {"S_post": S_post, "G": G, "S_y": S_y,
            "S_qS": S_post + G @ S_y @ G.T, "S_qM": G @ S_y @ G.T}
