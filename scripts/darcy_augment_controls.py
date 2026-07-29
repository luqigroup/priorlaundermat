#!/usr/bin/env python
"""The two controls on the directions that must not move.

The de-freezing arm predicts that joint curation moves the directions the channel resolves and
leaves the rest of the operator's exact null alone. The control directions do rise slightly, and
these two runs establish that the rise is the estimator's rather than information leaking in:

  placebo  a channel carrying NO blind information at all -- the same core rows projected onto the
           resolved subspace, so ``C_placebo @ B = 0`` exactly -- moves the control FURTHER than the
           real channel does. An information channel cannot be beaten by an information-free one,
           so whatever moves the control is not the channel's content.

  sweep    varying the channel's noise level from sharp to nearly uninformative leaves the control
           flat while the de-frozen directions degrade smoothly. Leakage would scale with channel
           strength; this does not.

Both rebuild the archive the same way ``darcy_joint_archive.py`` does, on a subsample by default,
since the point is a comparison against the survey-only arm on the SAME rows rather than a
production archive. Single core, no GPU, and the arrays are all (n x 100) -- nothing here needs
more than a few MB.

Reads the groundwater dataset and ``results/darcy_core_channel.npz`` (downloaded on first use).
Writes ``results/darcy_augment_controls.npz``.

Run:  python scripts/darcy_augment_controls.py [--n 400] [--placebo] [--sweep]
      (with neither flag, runs both)
"""
from __future__ import annotations

import argparse
import os
import time

import h5py
import numpy as np

from resolvability.download import REPO, ensure
from resolvability.groundwater.core_channel import simulate_cores
from resolvability.groundwater.darcy import DarcyForward, beskos_sensors
from resolvability.groundwater.kl_prior import StuartKLPrior
from resolvability.groundwater.map_solver import map_reconstruct

DATA = "data/darcy/darcy_laundering.h5"
CHAN = "results/darcy_core_channel.npz"
OUT = os.path.join(REPO, "results/darcy_augment_controls.npz")


def ratio(X, T, V):
    """rms over directions of the per-direction spread ratio, archive against truth."""
    a, t = (X @ V).std(0), (T @ V).std(0)
    return float(np.sqrt((a ** 2).mean()) / np.sqrt((t ** 2).mean()))


def curate(fwd, kl, sensors, y_obs, sigma_y, C, xi_true, sigma_u, maxiter, seed, label):
    """Rebuild the archive under survey + the given channel. Mirrors darcy_joint_archive.py."""
    rng = np.random.default_rng([seed, 0xC0DE])          # NOT the dataset's stream (see that script)
    z = simulate_cores(xi_true, C, sigma_u, rng)
    out = np.empty((len(y_obs), kl.d), np.float64)
    t0 = time.time()
    for i in range(len(y_obs)):
        out[i] = map_reconstruct(fwd, kl, sensors, y_obs[i], sigma_y,
                                 maxiter=maxiter, channel=(C, z[i], sigma_u))
    print(f"  [{label}] {len(y_obs)} rows in {time.time()-t0:.0f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="rows to curate per arm")
    ap.add_argument("--maxiter", type=int, default=80)
    ap.add_argument("--sigma-u", type=float, default=0.04, help="the deployed channel noise")
    ap.add_argument("--sweep-sigma", type=float, nargs="+", default=[0.01, 0.04, 0.16, 0.64])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--placebo", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    run_placebo = args.placebo or not args.sweep         # neither flag -> both
    run_sweep = args.sweep or not args.placebo

    ch = np.load(ensure(CHAN))
    C, DF, CT, B = ch["C"], ch["defrozen"], ch["control"], ch["blind_exact"]
    with h5py.File(ensure(DATA), "r") as f:
        N, K = int(f.attrs["N"]), int(f.attrs["K"])
        alpha, s, sig = float(f.attrs["alpha"]), float(f.attrs["s"]), float(f.attrs["sigma"])
        sigma_y = float(f.attrs["sigma_y"])
        n = min(args.n, f["xi_true"].shape[0])
        xi_true, y_obs, xi_map = f["xi_true"][:n], f["y_obs"][:n], f["xi_map"][:n]
        R = f["resolved"][:]

    kl = StuartKLPrior(N, K=K, alpha=alpha, s=s, sigma=sig)
    fwd, sensors = DarcyForward(N), beskos_sensors(N, 33)[0]
    sets = [("de-frozen", DF), ("control", CT), ("resolved", R)]
    rows = {}

    def report(label, X):
        rows[label] = [ratio(X, xi_true, V) for _, V in sets]
        print("  %-22s %s" % (label, "  ".join("%-12.4f" % v for v in rows[label])))

    print(f"archive-to-truth spread ratio, n={n}")
    print("  %-22s %s" % ("arm", "  ".join("%-12s" % nm for nm, _ in sets)))
    report("survey only", xi_map)
    real = curate(fwd, kl, sensors, y_obs, sigma_y, C, xi_true,
                  args.sigma_u, args.maxiter, args.seed, "real")
    report("+ real channel", real)

    if run_placebo:
        # Project every core row onto the resolved subspace: the placebo sees exactly what the
        # survey already sees, so it carries no blind information whatsoever.
        C_pl = C - (C @ B) @ B.T
        leak = float(np.abs(C_pl @ B).max())
        assert leak < 1e-10, f"placebo still reaches the blind subspace ({leak:.2e})"
        X = curate(fwd, kl, sensors, y_obs, sigma_y, C_pl, xi_true,
                   args.sigma_u, args.maxiter, args.seed, "placebo")
        report("+ placebo channel", X)
        print("  -> placebo moves the control %s than the real channel"
              % ("MORE" if rows["+ placebo channel"][1] > rows["+ real channel"][1] else "LESS"))

    if run_sweep:
        print("\n  channel-strength sweep (control must stay flat; de-frozen must degrade)")
        for su in args.sweep_sigma:
            X = curate(fwd, kl, sensors, y_obs, sigma_y, C, xi_true,
                       su, args.maxiter, args.seed, f"sigma_u={su:g}")
            report("+ channel, sigma_u=%g" % su, X)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, n=n, labels=np.array(list(rows), dtype=object),
             ratios=np.array([rows[k] for k in rows]),
             sets=np.array([nm for nm, _ in sets], dtype=object))
    print("\nsaved", os.path.relpath(args.out, REPO))


if __name__ == "__main__":
    main()
