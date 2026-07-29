#!/usr/bin/env python
"""Score the de-freezing arm: what joint curation moves, and what it leaves alone.

Reports the archive-to-truth spread ratio on the three direction sets the corollary distinguishes --
the directions the channel resolves, the blind directions it does not, and the resolved subspace --
for the survey-only archive and the jointly-curated one.

Read the numbers carefully, and write them carefully:

  * This is an ARCHIVE-level marginal spread ratio. It is not coverage, and it is not a statement
    about a trained prior; the flows are what carry the prior-level claim. Nothing here should be
    called "calibrated".
  * The control does MOVE. It is not a statistical null, and saying so would be false. What makes
    the result stand is that the movement is tiny beside the de-frozen one, that exact
    linear-Gaussian algebra predicts it to be zero (so the residue is the nonlinearity), and the two
    controls run by ``scripts/darcy_augment_controls.py``: an information-free placebo channel
    moves it MORE, and it stays flat as the real channel's strength is swept. Leakage would scale
    with channel strength; this does not.

Reads the groundwater dataset, ``results/darcy_core_channel.npz``, and
``results/darcy_joint_archive.npz``; all download on first use. Prints only. CPU, seconds.

Run:  python scripts/darcy_score_augment.py [--n N] [--boot]
"""
from __future__ import annotations

import argparse

import h5py
import numpy as np

from resolvability.download import ensure

DATA = "data/darcy/darcy_laundering.h5"
CHAN = "results/darcy_core_channel.npz"
JOINT = "results/darcy_joint_archive.npz"


def ratio(X, T, V):
    """rms over directions of the per-direction spread ratio, archive against truth."""
    a, t = (X @ V).std(0), (T @ V).std(0)
    return float(np.sqrt((a ** 2).mean()) / np.sqrt((t ** 2).mean()))


def boot(X, T, V, n_boot=2000, seed=0):
    """Paired bootstrap CI for the ratio, resampling rows."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    vals = [ratio(X[i], T[i], V) for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--boot", action="store_true", help="bootstrap CIs (slower)")
    args = ap.parse_args()

    ch, jt = np.load(ensure(CHAN)), np.load(ensure(JOINT))
    DF, CT = ch["defrozen"], ch["control"]
    xi_joint = jt["xi_joint"]
    n = xi_joint.shape[0] if args.n is None else min(args.n, xi_joint.shape[0])
    with h5py.File(ensure(DATA), "r") as f:
        xi_true, xi_map = f["xi_true"][:n], f["xi_map"][:n]
        R = f["resolved"][:]
    xi_joint = xi_joint[:n]

    sets = [("de-frozen (%d)" % DF.shape[1], DF),
            ("control (%d)" % CT.shape[1], CT),
            ("resolved@noise (%d)" % R.shape[1], R)]
    print(f"archive-to-truth spread ratio, n={n}")
    print("  %-22s %-13s %-13s" % ("set", "survey-only", "+channel"))
    for name, V in sets:
        a, b = ratio(xi_map, xi_true, V), ratio(xi_joint, xi_true, V)
        line = "  %-22s %-13.4f %-13.4f" % (name, a, b)
        if args.boot:
            lo, hi = boot(xi_joint, xi_true, V)
            line += "  [%.4f, %.4f]" % (lo, hi)
        print(line)
    print("\nper-direction de-frozen (+channel): %s"
          % np.array2string((xi_joint @ DF).std(0) / (xi_true @ DF).std(0), precision=3))

    # The strongest evidence that the move is confined to the predicted span, and it is basis-free:
    # the change in covariance restricted to the exact null should have exactly as many large
    # eigenvalues as the channel has dimensions, and its top eigenspace should be the predicted one.
    B = ch["blind_exact"]
    d = np.cov((xi_joint @ B).T) - np.cov((xi_map @ B).T)
    w, Vv = np.linalg.eigh(d)
    top = w[::-1][:DF.shape[1] + 1]
    print("\nbasis-free check -- eigenvalues of the covariance CHANGE inside the exact null:")
    print("  top %d: %s   next: %.4f" % (DF.shape[1], np.array2string(top[:-1], precision=3), top[-1]))
    W = Vv[:, ::-1][:, :DF.shape[1]]
    cos = np.linalg.svd((B @ W).T @ DF, compute_uv=False)
    print("  cos(principal angles) to the predicted de-frozen span: %s"
          % np.array2string(cos, precision=6))
    inside = float((np.diag(DF.T @ B @ d @ B.T @ DF).sum()) / np.trace(d))
    print("  share of the covariance change inside the predicted span: %.4f" % inside)


if __name__ == "__main__":
    main()
