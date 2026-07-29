#!/usr/bin/env python
"""Run the shipped resolvability statement on both deployed operators, and on the priors deployed
with them.

The statement (``resolvability/statement.py``) has three outputs: the split itself, the
classification of reported functionals, and the share of a deployed prior's advertised variance
that lives on the blind subspace -- the one number a practitioner can print without any ground
truth at all. This script produces all three on the paper's two operators.

The two operators exercise the statement's two paths, which report different things and say so:

  groundwater  small enough to factor, so the split is the exact kernel of the adjoint Jacobian and
               the statement is COMPLETE. Both cuts are printed: the exact kernel, and the wider
               near-null at the acquisition's noise level that the coverage experiments score on.
  seismic      too large to factor, so the statement takes a set of directions certified blind by
               probing (the illumination ranking). The blind subspace is not enumerated, so the
               statement marks itself incomplete and every share it reports is a LOWER bound.

With ``--latex`` it prints instead the verbatim block the paper exhibits, from the shipped API.

Reads the groundwater dataset, the certify-era flow checkpoints
(``data/checkpoints/darcy_flows/{arm}_s{seed}.pth``), the seismic probe caches, and the cached
unconditional seismic draws; all download on first use. Writes
``results/deployed_statement.npz``. Memory: the seismic path holds one (160 x 65536) sample block
and a (65536 x 16) basis, a little over 100 MB. CPU, a few minutes.

Run:  python scripts/deployed_statement.py [--latex]
"""
from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
import torch

from resolvability.download import REPO, ensure
from resolvability.groundwater.darcy import DarcyForward, beskos_sensors
from resolvability.groundwater.hint_flow import HINTFlow
from resolvability.groundwater.kl_prior import StuartKLPrior
from resolvability.groundwater.subspace import exact_jacobian
from resolvability.seismic.blind import build_blind_subspace
from resolvability.statement import blind_report
from resolvability.utils import Normalizer

DATA = "data/darcy/darcy_laundering.h5"
CKPT = "data/checkpoints/darcy_flows/{tag}_s{seed}.pth"
GALLERY = "results/seismic_gallery_samples.npz"
OUT = os.path.join(REPO, "results/deployed_statement.npz")
SEEDS = (10, 11, 12)


@torch.no_grad()
def flow_draws(tag: str, seed: int, n: int = 8000) -> np.ndarray:
    """``n`` unconditional draws from one trained groundwater flow, in KL coordinates."""
    ck = torch.load(ensure(CKPT.format(tag=tag, seed=seed)), map_location="cpu",
                    weights_only=False)
    flow = HINTFlow(ck["n_in"], 0, ck["n_hidden"], n_flow_layers=ck["n_layers"],
                    depth=ck.get("depth"), n_mlp_layers=3)
    flow.load_state_dict(ck["model"]); flow.eval()
    norm = Normalizer.from_stats(ck["norm_mean"], ck["norm_std"])
    torch.manual_seed(seed)
    return norm.unnormalize(flow.inverse(torch.randn(n, flow.n_in))).numpy()


def statement_block(card, title, V_blind, V_res):
    """The statement's own split lines plus two real classifications.

    Everything here comes from the shipped API -- ``summary(verdict=False)`` and ``classify`` --
    so the block is tool output that regenerates, not prose typed to look like it.
    """
    lines = [title, card.summary(verdict=False)]
    for name, v in (("leading blind mode   ", V_blind), ("leading resolved mode", V_res)):
        c = card.classify(v)[0]
        lines.append("  %s blind fraction %.2f  ->  %s"
                     % (name, c["blind_fraction"],
                        "UNVERIFIABLE (prior-supplied)" if c["label"] == "blind"
                        else "data-verifiable" if c["label"] == "resolved" else "mixed"))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true",
                    help="emit only the verbatim block the paper exhibits")
    args = ap.parse_args()
    out = {}

    # ---------------- groundwater: the complete path ------------------------------------------
    with h5py.File(ensure(DATA), "r") as f:
        N, K = int(f.attrs["N"]), int(f.attrs["K"])
        alpha, s, sig = float(f.attrs["alpha"]), float(f.attrs["s"]), float(f.attrs["sigma"])
        rank_noise = int(f.attrs["rank"])                 # rank at the acquisition's noise level
    kl = StuartKLPrior(N, K=K, alpha=alpha, s=s, sigma=sig)
    A = exact_jacobian(DarcyForward(N), kl, beskos_sensors(N, 33)[0])

    print("=" * 68)
    print("GROUNDWATER -- dense operator, exact kernel")
    print("=" * 68)
    card = blind_report(A)
    print(card.summary())

    card_noise = blind_report(A, rank=rank_noise)
    print(f"\nAt the acquisition's noise level the resolved rank is {rank_noise} instead, so the "
          f"blind subspace is {card_noise.blind_dim}-dimensional:\n"
          f"the exact kernel is the {card.blind_dim} directions no measurement reaches at any "
          f"noise level,\nand the rest of that {card_noise.blind_dim} is the tail the sensors see "
          f"but cannot separate from noise.")

    print("\n  share of each deployed prior's advertised variance lying on the blind subspace")
    print("  %-9s %-6s %-16s %s" % ("arm", "seed", "exact kernel", "near-null at noise"))
    for tag in ("curated", "oracle"):
        for sd in SEEDS:
            X = flow_draws(tag, sd)
            a = card.inherited_variance_fraction(X)
            b = card_noise.inherited_variance_fraction(X)
            out[f"gw_{tag}_s{sd}"] = (a, b)
            print("  %-9s %-6d %-16.3f %.3f" % (tag, sd, a, b))
            del X

    # ---------------- seismic: the probed, incomplete path --------------------------------------
    print("\n" + "=" * 68)
    print("SEISMIC -- operator too large to factor, blind directions certified by probing")
    print("=" * 68)
    sub = build_blind_subspace()
    Nb = np.ascontiguousarray(sub.Q_blind.T)              # (n, blind_dim) as the statement expects
    card_s = blind_report(n=Nb.shape[0], N_blind=Nb)      # complete=False -> everything a bound
    print(card_s.summary())

    if args.latex:
        # the exact listing the paper exhibits, both operators, from the shipped API
        Vr = np.linalg.svd(A, full_matrices=True)[2][0]      # leading right singular vector
        print("\n" + "=" * 68 + "\nLATEX BLOCK\n" + "=" * 68)
        print(statement_block(card_noise, "Groundwater (dense operator, cut at the noise level)",
                              card_noise.N_blind[:, 0], Vr))
        print()
        print(statement_block(card_s, "Seismic (operator given as code, blind directions probed)",
                              card_s.N_blind[:, 0], sub.Q_resolved[0]))
        return

    print("\n  share of each deployed prior's advertised variance certified blind (a LOWER bound)")
    z = np.load(ensure(GALLERY))
    for tag in ("curated", "oracle"):
        X = np.asarray(z[tag], np.float64)
        frac = card_s.inherited_variance_fraction(X)
        out[f"seis_{tag}"] = (frac, np.nan)
        print("  %-9s %.4f   (over %d samples)" % (tag, frac, len(X)))
        del X

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, keys=np.array(list(out), dtype=object),
             vals=np.array([out[k] for k in out], float),
             gw_exact=card.blind_dim, gw_noise=card_noise.blind_dim, seis_blind=Nb.shape[1])
    print("\nsaved", os.path.relpath(OUT, REPO))


if __name__ == "__main__":
    main()
