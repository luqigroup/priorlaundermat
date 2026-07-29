#!/usr/bin/env python
"""Run the certified blind band on the groundwater operator.

The paper's certified-band proposition says a split-conformal band built from the prior's own
standardized blind residual covers at the nominal level *whatever the prior believes* -- the prior
enters the band's width and never its validity. That is the paper's one constructive claim about a
deployed prior, and this script is what tests it.

Two arms, so the claim can be seen doing work:

  curated   trained on the legacy MAP archive, and overconfident on the blind subspace.
  oracle    trained on the truths, and calibrated there.

For a blind functional theta = v'x the prior supplies a conditional mean map m_rho(x_R) and a
conditional spread s_rho, both read off a Gaussian fit to its own draws. The score is
|theta - m_rho(x_R)| / s_rho; k held-out reference truths give the conformal quantile q; the band is
m_rho +- q s_rho. Two quantities come out:

  coverage      of the band on further held-out truths. Must reach nominal for BOTH arms. This is
                the validity claim, and it is the half that does not care about the prior.
  width ratio   q / z_{1-alpha/2}: the certified half-width divided by the one the prior advertises.
                Both are proportional to s_rho, so the ratio is exactly what a practitioner can
                report -- how much wider the truth is on this direction than the prior claims. It is
                near one for a calibrated prior and large for an overconfident one.

The references are drawn from the evaluation split, so they are held out of the archive every prior
was trained on -- the hypothesis the proposition needs in order to condition on the prior.

Reads the groundwater dataset, ``results/darcy_core_channel.npz`` (the operator's exact-null
split), and the flow checkpoints ``data/checkpoints/darcy_flows/{arm}_s{seed}.pth``; all download
on first use. Writes ``results/darcy_certify.npz``. CPU, a few minutes.

Run:  python scripts/darcy_certify.py [--alpha 0.1] [--k 9 19 49 99]
"""
from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
import torch
from scipy.stats import norm

from resolvability.download import REPO, ensure
from resolvability.groundwater.hint_flow import HINTFlow
from resolvability.utils import Normalizer

DATA = "data/darcy/darcy_laundering.h5"
CHAN = "results/darcy_core_channel.npz"
CKPT = "data/checkpoints/darcy_flows/{tag}_s{seed}.pth"
OUT = os.path.join(REPO, "results/darcy_certify.npz")


@torch.no_grad()
def prior_draws(tag, seed, n):
    ck = torch.load(ensure(CKPT.format(tag=tag, seed=seed)), map_location="cpu",
                    weights_only=False)
    flow = HINTFlow(ck["n_in"], 0, ck["n_hidden"], n_flow_layers=ck["n_layers"],
                    depth=ck.get("depth"), n_mlp_layers=3)
    flow.load_state_dict(ck["model"]); flow.eval()
    nrm = Normalizer.from_stats(ck["norm_mean"], ck["norm_std"])
    torch.manual_seed(seed)
    return nrm.unnormalize(flow.inverse(torch.randn(n, flow.n_in))).numpy()


def conditional_maps(X, R, V):
    """Gaussian fit of the prior; return the blind-fiber conditional mean map and spread.

    Returns ``(mu_theta, W, s, mu_a)`` with ``m(a) = mu_theta + (a - mu_a) @ W`` for resolved
    coords ``a = x @ R``, and ``s[j]`` the conditional spread of direction ``V[:, j]``.
    """
    mu, S = X.mean(0), np.cov(X.T)
    Saa = R.T @ S @ R                                             # (r, r) resolved block
    Sta = V.T @ S @ R                                             # (b, r) cross block
    Stt = np.einsum("ij,jk,ki->i", V.T, S, V)                     # (b,) blind variances
    sol = np.linalg.solve(Saa, Sta.T)                             # (r, b)
    s2 = Stt - np.einsum("ij,ji->i", Sta, sol)
    return mu @ V, sol, np.sqrt(np.maximum(s2, 1e-300)), mu @ R


def scores(xi, R, V, mu_t, W, s, mu_a):
    """Standardized blind residual |theta - m_rho(x_R)| / s_rho, one column per direction."""
    a, th = xi @ R, xi @ V                                        # (n, r), (n, b)
    return np.abs(th - (mu_t + (a - mu_a) @ W)) / s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.1, help="central-(1-alpha) band")
    ap.add_argument("--k", type=int, nargs="+", default=[9, 19, 49, 99], help="reference counts")
    ap.add_argument("--seeds", type=int, nargs="+", default=[10, 11, 12])
    ap.add_argument("--draws", type=int, default=8000, help="prior draws for the Gaussian fit")
    ap.add_argument("--reps", type=int, default=200, help="random reference/eval splits")
    ap.add_argument("--levels", type=float, nargs="+",
                    default=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
                    help="nominal levels for the certified-vs-uncertified curve")
    ap.add_argument("--k-curve", type=int, default=99, help="references for that curve")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    z = float(norm.ppf(1 - args.alpha / 2))                       # the width the prior advertises
    V = np.load(ensure(CHAN))["blind_exact"]                      # (d, b) the operator's exact null
    with h5py.File(ensure(DATA), "r") as f:
        xi_true, split, R = f["xi_true"][:], f["split"][:], f["resolved"][:]
    held = xi_true[split == split.max()]                          # evaluation split, unseen in training
    print(f"blind directions {V.shape[1]}   resolved {R.shape[1]}   held-out truths {len(held)}")
    print(f"central-{100*(1-args.alpha):.0f}% band; a finite band needs k >= {int(np.ceil(1/args.alpha))-1}")

    rng = np.random.default_rng(0)
    res = {}
    print("\n  %-9s %-6s %-8s %-14s %s" % ("arm", "seed", "k", "coverage", "width / advertised"))
    for tag in ("curated", "oracle"):
        for sd in args.seeds:
            X = prior_draws(tag, sd, args.draws)
            mu_t, W, s, mu_a = conditional_maps(X, R, V)
            S = scores(held, R, V, mu_t, W, s, mu_a)              # (n_held, b)
            for k in args.k:
                cov, wr = [], []
                for _ in range(args.reps):
                    p = rng.permutation(len(held))
                    ref, ev = S[p[:k]], S[p[k:]]
                    j = int(np.ceil((k + 1) * (1 - args.alpha))) - 1      # order statistic, 0-based
                    if j >= k:
                        cov.append(1.0); wr.append(np.inf); continue      # band is infinite, not wrong
                    q = np.sort(ref, axis=0)[j]                            # (b,) per direction
                    cov.append(float((ev <= q).mean()))
                    wr.append(float(np.median(q / z)))
                res[f"{tag}_s{sd}_k{k}"] = (float(np.mean(cov)), float(np.median(wr)))
                print("  %-9s %-6d %-8d %-14.3f %.2f"
                      % (tag, sd, k, np.mean(cov), np.median(wr)))

    # ---- certified vs uncertified, across nominal levels -------------------------------------
    # The uncertified band is what the prior itself advertises: m_rho +- z_{1-a/2} s_rho, the
    # quantity the reliability grid scores. The certified band replaces z by the conformal quantile.
    lev = np.asarray(args.levels, float)
    curve = {}
    print("\n  certified vs uncertified coverage, k=%d references" % args.k_curve)
    print("  %-9s %-8s %s" % ("arm", "band", "  ".join("%-7.2f" % L for L in lev)))
    for tag in ("curated", "oracle"):
        cert, raw = np.zeros((len(args.seeds), len(lev))), np.zeros((len(args.seeds), len(lev)))
        for si, sd in enumerate(args.seeds):
            X = prior_draws(tag, sd, args.draws)
            mu_t, W, s, mu_a = conditional_maps(X, R, V)
            S = scores(held, R, V, mu_t, W, s, mu_a)
            for li, L in enumerate(lev):
                raw[si, li] = float((S <= norm.ppf(1 - (1 - L) / 2)).mean())
                acc = []
                for _ in range(args.reps):
                    p = rng.permutation(len(held))
                    ref, ev = S[p[:args.k_curve]], S[p[args.k_curve:]]
                    j = int(np.ceil((args.k_curve + 1) * L)) - 1
                    if j >= args.k_curve:
                        acc.append(1.0); continue
                    acc.append(float((ev <= np.sort(ref, axis=0)[j]).mean()))
                cert[si, li] = float(np.mean(acc))
        curve[f"{tag}_cert"], curve[f"{tag}_raw"] = cert, raw
        for nm, arr in (("advertised", raw), ("certified", cert)):
            print("  %-9s %-8s %s" % (tag, nm, "  ".join("%-7.3f" % v for v in arr.mean(0))))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, alpha=args.alpha, z=z, ks=np.array(args.k), seeds=np.array(args.seeds),
             keys=np.array(list(res), dtype=object), levels=lev, k_curve=args.k_curve,
             vals=np.array([res[k] for k in res], float), b=V.shape[1], n_held=len(held),
             **curve)
    print("\nsaved", os.path.relpath(args.out, REPO))


if __name__ == "__main__":
    main()
