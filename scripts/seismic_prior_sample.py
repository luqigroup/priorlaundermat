#!/usr/bin/env python
"""Unconditional draws from both deployed seismic priors, on a common seed.

On the operator's blind subspace the likelihood is flat, so a deployed posterior's blind marginal
IS the prior's. These unconditional draws are therefore the deployed blind report, and they are
what the coverage and spread figures read.

Sampling is DDIM with eta = 0, i.e. deterministic: no per-step noise injection, so no residual
sampler noise leaks into the blind directions the samples are then measured on.

Writes results/seismic_prior_samples.npz (oracle, curated, nsamp), each (n, N*N) in physical
reflectivity units. One GPU, tens of minutes.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from resolvability.download import REPO
from resolvability.seismic.priors import find_all_ckpts, sample_prior

OUT = os.path.join(REPO, "results/seismic_prior_samples.npz")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=160, help="draws per prior")
    ap.add_argument("--steps", type=int, default=50, help="DDIM reverse steps")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--seed-index", type=int, default=0, help="which training seed of each arm")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    for tag in ("oracle", "curated"):
        path, base = find_all_ckpts(tag)[args.seed_index]
        torch.manual_seed(0)                       # same sampling seed for both arms
        out[tag] = sample_prior(path, base, args.n, args.chunk, args.steps, dev, "ddim")
        print(f"[sample] {tag}: {out[tag].shape}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, oracle=out["oracle"], curated=out["curated"], nsamp=args.n)
    print("saved", os.path.relpath(OUT, REPO), flush=True)


if __name__ == "__main__":
    main()
