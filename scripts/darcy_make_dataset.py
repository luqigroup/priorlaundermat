#!/usr/bin/env python
"""Generate the groundwater legacy-curation dataset: truths, surveys, and the MAP archive.

For each of ``n_train + n_eval`` truths drawn from the KL prior (xi ~ N(0, I_d)) this forms the
true log-permeability field, solves the head, samples the 33 sensors with Gaussian noise, and
reconstructs the single-best (MAP) estimate under the CORRECT prior. It also computes the
operator-defined resolved/blind split by finite-differencing the forward map at the prior mean.

Writes ``data/darcy/darcy_laundering.h5``: the oracle targets ``xi_true``, the curated targets
``xi_map``, the surveys ``y_obs``, the train/eval ``split``, the operator's singular values ``sv``
with its ``resolved``/``blind`` bases, and the sensor indices.

CPU only, no external PDE solver. The full bank (5000 truths) takes about half an hour;
``--smoke`` runs 25 in ten seconds.
"""
from __future__ import annotations

import argparse
import os
import time

import h5py
import numpy as np

from resolvability.download import REPO
from resolvability.groundwater import (DarcyForward, StuartKLPrior, beskos_sensors,
                                         map_reconstruct)


def operator_subspace(fwd, kl, sensors, *, eps: float = 1e-3, sigma_y: float = 0.01):
    """Resolved/blind split of the whitened KL space, defined by the OPERATOR.

    The paper's blind subspace is not a hand-picked mode cutoff: it is the near-null of the
    linearized forward map. Finite-differencing the 33 sensor readings with respect to each of the
    d KL coefficients at the prior mean gives the Jacobian ``J[m, k] = dy_m / dxi_k``, whose right
    singular vectors rank directions in xi-space by how strongly the data see them. Directions
    with a singular value below the noise level ``sigma_y`` are not resolved -- this is the
    likelihood-informed subspace of Cui and Spantini, and coverage is read along its complement.

    Returns:
        ``(S, rank, resolved, blind)``: singular values (n_sensors,), the number above the noise
        floor, and the ``(d, rank)`` / ``(d, d-rank)`` bases whose columns are xi-space directions.
    """
    d = kl.d
    u_ref = np.zeros((kl.grid_size, kl.grid_size))          # the prior mean, permeability 1
    y0 = fwd.observe(fwd.solve(u_ref), sensors)

    J = np.empty((y0.size, d))
    for k in range(d):
        xi = np.zeros(d); xi[k] = eps
        J[:, k] = (fwd.observe(fwd.solve(u_ref + kl.reconstruct(xi)), sensors) - y0) / eps

    _, S, Vh = np.linalg.svd(J, full_matrices=True)
    rank = int(np.sum(S > sigma_y))
    return S, rank, Vh[:rank].T, Vh[rank:].T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=40)          # vertex grid
    ap.add_argument("--K", type=int, default=10)          # d = K^2 = 100 KL modes
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--s", type=float, default=1.1)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--sigma_y", type=float, default=0.01)
    ap.add_argument("--n_sensors", type=int, default=33)
    ap.add_argument("--maxiter", type=int, default=40)    # L-BFGS-B budget for one MAP solve
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_eval", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="tiny run, for checking the pipeline")
    args = ap.parse_args()
    if args.smoke:
        args.n_train, args.n_eval = 20, 5
    out = os.path.join(REPO, "data/darcy",
                       "darcy_laundering_smoke.h5" if args.smoke else "darcy_laundering.h5")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    kl = StuartKLPrior(args.N, K=args.K, alpha=args.alpha, s=args.s, sigma=args.sigma)
    fwd = DarcyForward(args.N)
    sensors, _ = beskos_sensors(args.N, args.n_sensors)
    rng = np.random.default_rng(args.seed)

    sv, rank, resolved, blind = operator_subspace(fwd, kl, sensors, sigma_y=args.sigma_y)
    print(f"[setup] N={args.N} d={kl.d}  resolved={rank} blind={kl.d - rank}  "
          f"S range [{sv.min():.2e}, {sv.max():.2e}]", flush=True)

    n = args.n_train + args.n_eval
    xi_true = rng.standard_normal((n, kl.d))
    xi_map = np.empty_like(xi_true)
    y_obs = np.empty((n, args.n_sensors))
    chi2 = np.empty(n)                          # fit quality, reported but not stored
    t0 = time.time()
    for i in range(n):
        u = kl.reconstruct(xi_true[i])
        y = fwd.observe(fwd.solve(u), sensors) + args.sigma_y * rng.standard_normal(args.n_sensors)
        y_obs[i] = y
        xi_map[i] = map_reconstruct(fwd, kl, sensors, y, args.sigma_y, maxiter=args.maxiter)
        r = fwd.observe(fwd.solve(kl.reconstruct(xi_map[i])), sensors) - y
        chi2[i] = float(r @ r) / args.sigma_y ** 2 / args.n_sensors
        if (i + 1) % max(1, n // 20) == 0:
            print(f"  {i+1}/{n}  chi2/dof={chi2[i]:.2f}  ({time.time()-t0:.0f}s)", flush=True)

    split = np.zeros(n, dtype=np.int8); split[args.n_train:] = 1     # 0 = train, 1 = eval
    with h5py.File(out, "w") as f:
        for k, v in dict(N=args.N, K=args.K, alpha=args.alpha, s=args.s, sigma=args.sigma,
                         sigma_y=args.sigma_y, n_sensors=args.n_sensors, maxiter=args.maxiter,
                         n_train=args.n_train, n_eval=args.n_eval, seed=args.seed,
                         rank=rank).items():
            f.attrs[k] = v
        f.create_dataset("xi_true", data=xi_true.astype(np.float32))
        f.create_dataset("xi_map", data=xi_map.astype(np.float32))
        f.create_dataset("y_obs", data=y_obs.astype(np.float32))
        f.create_dataset("split", data=split)
        f.create_dataset("sv", data=sv.astype(np.float32))
        f.create_dataset("resolved", data=resolved.astype(np.float32))
        f.create_dataset("blind", data=blind.astype(np.float32))
        f.create_dataset("sensor_ix", data=np.asarray(sensors[0]))
        f.create_dataset("sensor_iy", data=np.asarray(sensors[1]))
    print(f"[done] {n} samples  median chi2/dof={np.median(chi2):.2f}  "
          f"-> {os.path.relpath(out, REPO)}", flush=True)


if __name__ == "__main__":
    main()
