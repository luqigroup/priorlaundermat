"""The operator's blind / resolved / scrambled image subspaces, from two cached probe files.

One definition, shared by every seismic script, so no two of them can drift on what "the blind
subspace" means.

The construction ranks a set of probe atoms by the scattered energy the operator produces from
them, ``||A v||^2 = sum_s ||born_one(v, s)||^2`` (the bare Born forward, no water mute):

* ``data/seismic/psf_illum_N256.npz`` holds two over-complete atom sets probed at 16
  point-spread-function sites, three atoms each: ``U_res`` (high ``||A v||^2``) and ``U_bl`` (the
  nominally blind candidates). Neither set is orthonormal -- the sites overlap -- so each is
  SVD-orthonormalized before any energy is measured.
* ``data/seismic/effect_existence_spectrum.npz`` holds the measured ``||A v||^2`` of every atom.
* A nominally blind atom is kept only if its ``||A v||^2`` falls below 1% of the resolved median.
  Point-spread-function leakage pushes a few candidates well above that; dropping them leaves a
  genuinely blind basis, rank 16 at N = 256.
* The scrambled null is a seeded set of random orthonormal image directions -- the control
  subspace, in which the inherited-overconfidence signature must not appear.

Reads the two cached ``.npz`` files only: no operator solve, no dataset access, no GPU.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from ..download import ensure

PSF_NPZ = "data/seismic/psf_illum_N256.npz"
SPECTRUM_NPZ = "data/seismic/effect_existence_spectrum.npz"

# A nominally blind atom is kept only if its bare ||A v||^2 is below this fraction of the
# resolved median -- a scale-free gap that cleanly separates the leakage-contaminated candidates.
REL_FLOOR_FRAC = 0.01
SCRAMBLE_SEED = 0
ORTHO_TOL = 1e-3


def orthonormalize(U: np.ndarray, tol: float = ORTHO_TOL) -> tuple[np.ndarray, int]:
    """SVD-orthonormalize overlapping unit atoms (rows).

    Returns an orthonormal-row basis ``Q`` (r, d) spanning the same row space, and its
    numerical rank ``r``.
    """
    U = np.asarray(U, np.float64)
    _, s, Vt = np.linalg.svd(U, full_matrices=False)
    r = int((s > tol * s.max()).sum())
    return Vt[:r].astype(np.float64), r


@dataclass
class BlindSubspace:
    """Orthonormal bases (rows) for the three image subspaces, plus their provenance.

    Attributes:
        Q_blind, Q_resolved, Q_scrambled: (r, N*N) orthonormal-row bases.
        r_blind, r_resolved, r_scrambled: their dimensions.
        n_blind_kept: nominally blind atoms surviving the relative-gap floor.
        rel_floor: the absolute ``||A v||^2`` floor used.
        res_median: median resolved ``||A v||^2``.
        N: image side length.
    """

    Q_blind: np.ndarray
    Q_resolved: np.ndarray
    Q_scrambled: np.ndarray
    r_blind: int
    r_resolved: int
    r_scrambled: int
    n_blind_kept: int
    rel_floor: float
    res_median: float
    N: int

    def project(self, X: np.ndarray, which: str) -> tuple[np.ndarray, np.ndarray]:
        """Project flattened images ``X`` (n, N*N) onto one subspace.

        Args:
            X: flattened images.
            which: ``"blind"``, ``"resolved"`` or ``"scrambled"``.

        Returns:
            Per-image projected energy (n,) and coordinates (n, r).
        """
        Q = {"blind": self.Q_blind, "resolved": self.Q_resolved,
             "scrambled": self.Q_scrambled}[which]
        C = np.asarray(X, np.float64) @ Q.T
        return (C ** 2).sum(axis=1), C


def build_blind_subspace(
    psf_npz: str | None = None,
    spectrum_npz: str | None = None,
    scramble_seed: int = SCRAMBLE_SEED,
    rel_floor_frac: float = REL_FLOOR_FRAC,
) -> BlindSubspace:
    """Build the blind / resolved / scrambled subspaces from the cached probe files.

    Reads ``data/seismic/psf_illum_N256.npz`` and
    ``data/seismic/effect_existence_spectrum.npz`` (both downloaded on first use). Writes
    nothing. A few seconds on CPU, dominated by the QR of the scrambled null.
    """
    psf_npz = ensure(PSF_NPZ) if psf_npz is None else psf_npz
    spectrum_npz = ensure(SPECTRUM_NPZ) if spectrum_npz is None else spectrum_npz
    for p in (psf_npz, spectrum_npz):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"required probe cache missing: {p}. Rebuild it with "
                f"scripts/seismic_probe_basis.py.")

    d = np.load(psf_npz)
    U_res = d["U_res"].astype(np.float64)
    U_bl = d["U_bl"].astype(np.float64)
    N = int(round(np.sqrt(U_res.shape[1])))

    z = np.load(spectrum_npz)
    res_bare = z["res_bare"].astype(np.float64)
    bl_bare = z["bl_bare"].astype(np.float64)

    res_median = float(np.median(res_bare))
    rel_floor = rel_floor_frac * res_median

    keep = bl_bare < rel_floor
    n_blind_kept = int(keep.sum())
    if n_blind_kept == 0:
        raise RuntimeError("no blind atoms survive the relative-gap floor; cannot build "
                           "a blind subspace.")
    Q_blind, r_blind = orthonormalize(U_bl[keep])
    Q_resolved, r_resolved = orthonormalize(U_res)

    rng = np.random.default_rng(scramble_seed)
    Gmat = rng.standard_normal((48, N * N))
    Qscr, _ = np.linalg.qr(Gmat.T)                 # (d, 48) orthonormal columns
    Q_scrambled, r_scrambled = orthonormalize(Qscr.T)

    return BlindSubspace(
        Q_blind=Q_blind, Q_resolved=Q_resolved, Q_scrambled=Q_scrambled,
        r_blind=r_blind, r_resolved=r_resolved, r_scrambled=r_scrambled,
        n_blind_kept=n_blind_kept, rel_floor=rel_floor, res_median=res_median, N=N)


def verify_blind_gap(sub: BlindSubspace, truth: np.ndarray, lsrtm: np.ndarray) -> dict:
    """Native truth-versus-archive energy ratio on each subspace.

    The prerequisite for the whole seismic experiment: the blind directions must carry content
    the legacy migration lacks, or there is nothing for a prior to launder.

    Args:
        sub: the subspaces.
        truth: (n, N*N) broadband reflectivity, on the same amplitude scale as ``lsrtm``.
        lsrtm: (n, N*N) legacy migration images.

    Returns:
        Per-subspace mean energies and their ratio, plus the three subspace dimensions.
    """
    out = {}
    for which in ("resolved", "blind", "scrambled"):
        Et, _ = sub.project(truth, which)
        El, _ = sub.project(lsrtm, which)
        out[f"{which}_native"] = float(Et.mean() / max(El.mean(), 1e-30))
        out[f"{which}_E_truth"] = float(Et.mean())
        out[f"{which}_E_lsrtm"] = float(El.mean())
    out["r_blind"] = sub.r_blind
    out["r_resolved"] = sub.r_resolved
    out["r_scrambled"] = sub.r_scrambled
    return out
