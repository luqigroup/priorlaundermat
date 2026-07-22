"""Blind-subspace report card: from the forward operator alone, name the directions in which a
learned prior's reported uncertainty cannot be verified from the measurements.

The failure this paper studies is confined to the operator's blind subspace ``ker A`` (more generally
the near-null where illumination vanishes). Where the likelihood is flat the posterior equals the
prior, so a learned prior's reported spread there is inherited from the prior, not earned from data,
and no measurement-side check --- held-out coverage or simulation-based calibration --- can detect a
shortfall. Crucially, that subspace is fixed by the forward operator and the noise level ALONE:
independent of the prior, of the training archive, and of the data actually collected. It can
therefore be named in advance and reported alongside any reconstruction, marking off the confidence
the measurements support from the belief the estimator supplied. This module computes it and prints
the "report card"; it is the concrete form of the paper's recommendation.

Operator-only core
------------------
    from priorlaundermat.report_card import blind_report
    card = blind_report(A)              # A: (m, n) dense forward matrix
    print(card.summary())

For an operator too large for a dense SVD, pass the ambient dimension and a precomputed orthonormal
blind basis (columns spanning ``ker A`` / the near-null, e.g. from illumination ranking; see
``priorlaundermat.seismic.blind``)::

    card = blind_report(n=n, N_blind=N_blind)

Optional helpers take user-supplied functionals or prior samples and flag which reported credible
intervals fall in the blind subspace (:meth:`ReportCard.classify`,
:meth:`ReportCard.inherited_variance_fraction`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReportCard:
    """Resolved / blind split of an operator, computed from the operator alone.

    Attributes
    ----------
    n : ambient (model) dimension.
    m : number of measurements, or ``None`` if only a blind basis was supplied.
    rank : numerical rank of ``A`` = dimension of the resolved subspace.
    blind_dim : dimension of the blind subspace, ``n - rank``.
    N_blind : ``(n, blind_dim)`` orthonormal columns spanning the blind subspace, or ``None``.
    sing : singular values of ``A`` in descending order, or ``None`` if not computed.
    tol : absolute singular-value threshold used for the rank cut.
    """

    n: int
    m: int | None
    rank: int
    blind_dim: int
    N_blind: np.ndarray | None = None
    sing: np.ndarray | None = None
    tol: float = 0.0

    @property
    def blind_fraction(self) -> float:
        """Fraction of the model that is blind (``blind_dim / n``)."""
        return self.blind_dim / self.n if self.n else 0.0

    def classify(self, V: np.ndarray, blind_thresh: float = 0.99,
                 resolved_thresh: float = 0.01) -> list[dict]:
        """Flag each functional/direction in ``V`` (rows, each length ``n``) as verifiable or not.

        For a direction ``v`` the blind energy fraction is ``||N_blind^T v||^2 / ||v||^2`` in
        ``[0, 1]``: a reported credible interval along ``v`` is data-verifiable only to the extent
        that fraction is small. Returns one dict per row with the blind fraction and a label in
        ``{"resolved", "mixed", "blind"}`` (``blind`` -> the reported uncertainty there is
        inherited, not earned from data).
        """
        if self.N_blind is None:
            raise ValueError("classify needs the blind basis; build the card from A or pass N_blind.")
        V = np.atleast_2d(np.asarray(V, np.float64))
        out = []
        for v in V:
            nrm2 = float(v @ v)
            if nrm2 <= 0:
                out.append({"blind_fraction": float("nan"), "label": "degenerate"})
                continue
            frac = float((self.N_blind.T @ v) @ (self.N_blind.T @ v) / nrm2)
            label = ("blind" if frac >= blind_thresh
                     else "resolved" if frac <= resolved_thresh else "mixed")
            out.append({"blind_fraction": frac, "label": label,
                        "verifiable": label == "resolved"})
        return out

    def inherited_variance_fraction(self, samples: np.ndarray) -> float:
        """Fraction of a prior's reported variance that lives on the blind subspace.

        ``samples`` : ``(k, n)`` draws from a learned prior (or its posterior). Returns
        ``tr(cov projected onto ker A) / tr(cov)`` in ``[0, 1]`` --- the share of the reported
        spread that is inherited from the prior rather than constrained by the data. Diagnostic
        only: it does not say whether that spread is right, which no measurement can.
        """
        if self.N_blind is None:
            raise ValueError("needs the blind basis; build the card from A or pass N_blind.")
        X = np.asarray(samples, np.float64)
        X = X - X.mean(0, keepdims=True)
        total = float((X * X).sum())
        if total <= 0:
            return float("nan")
        Cb = X @ self.N_blind                      # (k, blind_dim) blind coordinates
        return float((Cb * Cb).sum() / total)

    def summary(self) -> str:
        """Human-readable report card: the split and the verdict, in plain text."""
        pct = 100.0 * self.blind_fraction
        lines = [
            "Blind-subspace report card (computed from the operator alone)",
            "-" * 60,
            f"  ambient (model) dimension n : {self.n}",
            f"  measurements m              : {self.m if self.m is not None else 'n/a'}",
            f"  resolved rank               : {self.rank}",
            f"  BLIND dimension (n - rank)  : {self.blind_dim}  ({pct:.1f}% of the model)",
        ]
        if self.sing is not None and self.sing.size:
            smax = self.sing.max()
            lines.append(f"  singular-value cut          : {self.tol:.2e}  "
                         f"({self.tol / smax:.1e} x sigma_max)")
        lines += [
            "-" * 60,
            "VERDICT: along the blind directions a learned prior's reported uncertainty",
            "is NOT verifiable from measurements under this operator. There the likelihood",
            "is flat, so the posterior equals the prior and the reported spread is inherited",
            "from the prior, not earned from data; no held-out check or simulation-based",
            "calibration can detect a shortfall. Report these directions alongside any",
            "reconstruction and read their uncertainty as prior-supplied, not data-driven.",
        ]
        return "\n".join(lines)


def blind_report(A: np.ndarray | None = None, *, n: int | None = None,
                 N_blind: np.ndarray | None = None, rank: int | None = None,
                 rtol: float = 1e-9) -> ReportCard:
    """Build the :class:`ReportCard` for a linear forward operator, from the operator alone.

    Parameters
    ----------
    A : ``(m, n)`` dense forward matrix. If given, its blind subspace is the full-SVD null space
        ``V[:, rank:]`` (the exact ``ker A``, never a thin-SVD resolved artifact).
    n : ambient dimension, required when ``A`` is not passed.
    N_blind : ``(n, blind_dim)`` orthonormal columns spanning the blind subspace (or near-null),
        for operators too large for a dense SVD. Supply with ``n``.
    rank : override the resolved rank (else inferred from the singular values via ``rtol``).
    rtol : relative singular-value threshold for the rank cut, ``sigma > rtol * sigma_max``.
    """
    if A is not None:
        A = np.asarray(A, np.float64)
        m, nn = A.shape
        _, s, Vt = np.linalg.svd(A, full_matrices=True)
        tol = rtol * (s.max() if s.size else 1.0)
        r = int((s > tol).sum()) if rank is None else int(rank)
        Nb = Vt[r:].T.copy()                       # (n, n - r) exact ker A
        return ReportCard(n=nn, m=m, rank=r, blind_dim=nn - r, N_blind=Nb, sing=s, tol=tol)
    if N_blind is not None and n is not None:
        Nb = np.asarray(N_blind, np.float64)
        if Nb.shape[0] != n:                       # accept rows-as-basis too
            Nb = Nb.T
        bd = Nb.shape[1]
        return ReportCard(n=n, m=None, rank=n - bd, blind_dim=bd, N_blind=Nb, sing=None, tol=0.0)
    raise ValueError("pass either a dense operator A, or (n=..., N_blind=...).")
