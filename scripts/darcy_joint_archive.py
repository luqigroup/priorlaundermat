#!/usr/bin/env python
"""Legacy MAP archive curated JOINTLY against the survey and an added core-sample channel.

This is the de-freezing corollary's archive. Joint curation de-freezes exactly the blind
directions the added channel resolves and nothing else, so the archive itself must be produced
under the STACKED operator -- reusing the survey-only archive would silently invalidate the whole
comparison.

The comparison is PAIRED by construction: every row reuses its stored survey observation
``y_obs[i]`` verbatim and only appends the channel reading ``z_i = C xi_true_i + sigma_u eps``, so
the two archives differ by the channel and by nothing else -- not by a different noise draw, not
by a different survey.

Reads the groundwater dataset and ``results/darcy_core_channel.npz`` (produced by
``scripts/darcy_build_channel.py``); both download on first use. Writes
``results/darcy_joint_archive.npz`` with the jointly-curated coefficients, alongside the channel
they were curated against. Single core, no GPU; L-BFGS at ``maxiter=80`` because the augmented
problem needs more iterations than the survey-only one (40 is binding once the channel is added).
About an hour for the full archive.

Run:  python scripts/darcy_joint_archive.py [--n 5000] [--sigma-u 0.04]
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
OUT = os.path.join(REPO, "results/darcy_joint_archive.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="rows to curate (default: all)")
    ap.add_argument("--sigma-u", type=float, default=0.04, help="core-sample noise, KL units")
    ap.add_argument("--maxiter", type=int, default=80, help="L-BFGS cap; 40 is binding once augmented")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    ch = np.load(ensure(CHAN))
    C = ch["C"]                                             # (6, d) exact, linear in the coefficients
    with h5py.File(ensure(DATA), "r") as f:
        N, K = int(f.attrs["N"]), int(f.attrs["K"])
        alpha, s, sig = float(f.attrs["alpha"]), float(f.attrs["s"]), float(f.attrs["sigma"])
        sigma_y = float(f.attrs["sigma_y"])
        n_all = f["xi_true"].shape[0]
        n = n_all if args.n is None else min(int(args.n), n_all)
        xi_true = f["xi_true"][:n]
        y_obs = f["y_obs"][:n]                              # reused VERBATIM -> paired comparison
        split = f["split"][:n]

    kl = StuartKLPrior(N, K=K, alpha=alpha, s=s, sigma=sig)
    fwd = DarcyForward(N)
    sensors, _ = beskos_sensors(N, 33)

    # Seed the channel noise from a SEPARATE stream. default_rng(0) is the stream the dataset
    # generator used, and its leading draws are xi_true itself -- reusing it would make the
    # "independent" channel noise a recycled copy of the truth, which contradicts the hypothesis the
    # channel is supposed to satisfy even where it does not move the measured numbers.
    rng = np.random.default_rng([args.seed, 0xC0DE])
    z_all = simulate_cores(xi_true, C, args.sigma_u, rng)   # (n, 6) channel readings

    xi_joint = np.empty((n, kl.d), np.float64)
    t0 = time.time()
    for i in range(n):
        xi_joint[i] = map_reconstruct(
            fwd, kl, sensors, y_obs[i], sigma_y,
            maxiter=args.maxiter, channel=(C, z_all[i], args.sigma_u))
        if (i + 1) % 250 == 0 or i + 1 == n:
            el = time.time() - t0
            print(f"  {i+1}/{n}  {el:.0f}s  ({el/(i+1):.3f} s/row, eta {el/(i+1)*(n-i-1):.0f}s)",
                  flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, xi_joint=xi_joint, z=z_all, C=C, sigma_u=args.sigma_u,
             maxiter=args.maxiter, seed=args.seed, split=split,
             nodes=ch["nodes"], defrozen=ch["defrozen"], control=ch["control"])
    print("saved", os.path.relpath(args.out, REPO))


if __name__ == "__main__":
    main()
