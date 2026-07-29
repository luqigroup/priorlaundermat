#!/usr/bin/env python
"""Build and cache the core-sample channel for the de-freezing arm.

Writes ``results/darcy_core_channel.npz``: the selected core nodes, the channel matrix ``C``, and
the de-frozen / control bases that split the operator's EXACT null. Everything here is computed
from the operator and the channel alone -- no truth, no archive, no data -- which is the point:
the split the experiment is scored on is fixed before any prior is trained.

The null is taken from the exact ADJOINT Jacobian, not finite differences. Two of the 33 sensors
sit on the Dirichlet boundary where the head is pinned and carry exactly no information; finite
differences give them step-size noise instead of zero, which a tight cutoff then counts as rank
and so understates the null by two directions.

Reads the groundwater dataset (downloaded on first use). CPU, under a minute.

Run:  python scripts/darcy_build_channel.py [--n-cores 6]
"""
from __future__ import annotations

import argparse
import os

import h5py
import numpy as np

from resolvability.download import REPO, ensure
from resolvability.groundwater.core_channel import (build_core_channel, channel_subspaces,
                                                   select_core_nodes)
from resolvability.groundwater.darcy import DarcyForward, beskos_sensors
from resolvability.groundwater.kl_prior import StuartKLPrior
from resolvability.groundwater.subspace import operator_subspace

DATA = "data/darcy/darcy_laundering.h5"
OUT = os.path.join(REPO, "results/darcy_core_channel.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cores", type=int, default=6)
    ap.add_argument("--sigma-u", type=float, default=0.04, help="only for the reported SNR")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    with h5py.File(ensure(DATA), "r") as f:
        N, K = int(f.attrs["N"]), int(f.attrs["K"])
        alpha, s, sig = float(f.attrs["alpha"]), float(f.attrs["s"]), float(f.attrs["sigma"])
    kl = StuartKLPrior(N, K=K, alpha=alpha, s=s, sigma=sig)
    fwd = DarcyForward(N)
    sensors, _ = beskos_sensors(N, 33)

    sub = operator_subspace(fwd, kl, sensors, sv_cutoff=1e-12, exact=True)
    B = sub["blind"]
    nodes = select_core_nodes(kl, B, n_cores=args.n_cores)
    C = build_core_channel(kl, nodes)
    DF, CT = channel_subspaces(C, B)
    sv = np.linalg.svd(C @ B, compute_uv=False)

    print(f"exact null {B.shape[1]}   de-frozen {DF.shape[1]}   control {CT.shape[1]}")
    print("cores:", [(int(i), int(j)) for i, j in nodes])
    print("channel SNR at sigma_u=%g: %s" % (args.sigma_u,
                                             np.array2string(sv / args.sigma_u, precision=2)))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, nodes=np.array(nodes), C=C, defrozen=DF, control=CT,
             blind_exact=B, sv=sv)
    print("saved", os.path.relpath(args.out, REPO))


if __name__ == "__main__":
    main()
