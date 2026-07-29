#!/usr/bin/env python
"""Posterior sampling for the groundwater example: pCN in the learned prior's latent.

Three reads of the same sampler, selected by ``--mode``:

``coverage``     one chain per held-out survey under each arm, scored as central-credible coverage
                 on the operator's blind and resolved subspaces.
                 -> ``results/darcy_pcn_{arm}_s{seed}.npz``
``single``       one survey in full, keeping the posterior mean and pointwise standard deviation
                 of the log-permeability field for both arms, next to the truth and the legacy MAP
                 reconstruction. -> ``results/darcy_pcn_single.npz``
``chainlength``  one long chain per truth, scored on prefixes of itself, so the whole chain-length
                 ladder plus the autocorrelation diagnostics cost a single run. The oracle arm is
                 the calibrated control, so the shortest prefix that sits on nominal is the
                 defensible production length. -> ``results/darcy_pcn_chainlength_{arm}.npz``

Reads ``data/darcy/darcy_laundering.h5`` and ``data/checkpoints/darcy_flow_*.pth``. CPU only; the
default coverage run is a bit over an hour per arm, ``single`` a couple of minutes.

The two arms share one random stream in the order oracle, curated, so a ``--mode coverage`` or
``--mode single`` run reproduces the released cache for both. The chain-length ladder is one arm
per invocation.

Examples:
    python scripts/darcy_pcn.py --mode coverage --seed 0
    python scripts/darcy_pcn.py --mode single
    python scripts/darcy_pcn.py --mode chainlength --arm oracle
    python scripts/darcy_pcn.py --mode chainlength --arm curated
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from resolvability.download import REPO
from resolvability.groundwater import (LEVELS, coverage, iat_sokal, load_flow, map_reconstruct,
                                         pcn_posterior, setup)

# Per-mode chain settings: the values that produced the released caches.
DEFAULTS = {
    "coverage":    dict(n_truths=24, n_steps=44000),
    "single":      dict(n_truths=1,  n_steps=11000),
    "chainlength": dict(n_truths=16, n_steps=88000),
}
LADDER = (11000, 22000, 44000)      # prefixes scored in chainlength mode, plus the full chain


def out_path(name: str) -> str:
    d = os.path.join(REPO, "results")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def run_coverage(args, st):
    """Blind- and resolved-subspace coverage over held-out surveys, one file per arm."""
    yt, xt = st.eval_truths(args.n_truths)
    print(f"[coverage] N={st.N} d={st.kl.d}  resolved={st.resolved.shape[1]} "
          f"blind={st.blind.shape[1]}  truths={len(yt)}  steps={args.n_steps} beta={args.beta}",
          flush=True)
    for arm in args.arms:
        flow, norm = load_flow(arm, args.seed)
        t0 = time.time()
        samp, acc = pcn_posterior(flow, norm, st, yt, n_steps=args.n_steps, n_burn=args.n_burn,
                                  beta=args.beta, thin=args.thin)
        cov_b, cov_b_pt = coverage(samp @ st.blind, xt @ st.blind)
        cov_r, cov_r_pt = coverage(samp @ st.resolved, xt @ st.resolved)
        p = out_path(f"darcy_pcn_{arm}_s{args.seed}.npz")
        np.savez(p, blind=cov_b, resolved=cov_r, blind_pt=cov_b_pt, resolved_pt=cov_r_pt,
                 acc=acc, levels=LEVELS, n_truths=len(yt), n_steps=args.n_steps, beta=args.beta)
        print(f"[{arm} s{args.seed}] acc={acc:.2f}  blind@90={cov_b[4]:.2f}  "
              f"resolved@90={cov_r[4]:.2f}  ({time.time()-t0:.0f}s)  "
              f"-> {os.path.relpath(p, REPO)}", flush=True)


def run_single(args, st):
    """One survey, both arms: the posterior mean and pointwise std fields the figure needs."""
    kl, fwd, sensors = st.kl, st.fwd, st.sensors
    n_sensors = st.y_obs.shape[1]
    if args.truth_ix is None:
        # Stuart's fixed ground-truth field, u*_k = lambda_k sin((i1-1/2)^2+(i2-1/2)^2), with a
        # freshly synthesized survey; its MAP reconstruction stands in for the legacy archive.
        rng = np.random.default_rng(args.noise_seed)
        xi_star = kl.stuart_true_xi()
        y = fwd.observe(fwd.solve(kl.reconstruct(xi_star)), sensors) \
            + st.sigma_y * rng.standard_normal(n_sensors)
        xi_map_star = map_reconstruct(fwd, kl, sensors, y, st.sigma_y)
        yt, map_field = y[None, :], kl.reconstruct(xi_map_star)
    else:
        ev = np.where(st.split == 1)[0][args.truth_ix]
        xi_star = st.xi_true[ev]
        yt, map_field = st.y_obs[ev:ev + 1], kl.reconstruct(st.xi_map[ev])
    truth_field = kl.reconstruct(xi_star)
    which = "Stuart fixed field" if args.truth_ix is None else f"eval index {args.truth_ix}"
    print(f"[single] {which}  steps={args.n_steps} beta={args.beta}", flush=True)

    out = dict(truth=truth_field, map_recon=map_field, N=st.N, stuart=args.truth_ix is None,
               sensor_ix=np.asarray(sensors[0]), sensor_iy=np.asarray(sensors[1]))
    for arm in args.arms:
        flow, norm = load_flow(arm, args.seed)
        t0 = time.time()
        samp, acc = pcn_posterior(flow, norm, st, yt, n_steps=args.n_steps, n_burn=args.n_burn,
                                  beta=args.beta, thin=args.thin)
        fields = np.stack([kl.reconstruct(x) for x in samp[:, 0, :]])
        out[f"{arm}_mean"], out[f"{arm}_std"] = fields.mean(0), fields.std(0)
        print(f"[{arm}] acc={acc:.2f}  mean posterior std={out[f'{arm}_std'].mean():.3f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
    p = out_path("darcy_pcn_single.npz")
    np.savez(p, **out)
    print(f"-> {os.path.relpath(p, REPO)}", flush=True)


def run_chainlength(args, st):
    """One long chain per truth, scored on its own prefixes, with mixing diagnostics."""
    yt, xt = st.eval_truths(args.n_truths)
    for arm in args.arms:
        flow, norm = load_flow(arm, args.seed)
        print(f"[chainlength] arm={arm} truths={len(yt)} steps={args.n_steps} beta={args.beta} "
              f"thin={args.thin}", flush=True)
        t0 = time.time()
        samp, acc = pcn_posterior(flow, norm, st, yt, n_steps=args.n_steps, n_burn=args.n_burn,
                                  beta=args.beta, thin=args.thin)
        print(f"[chainlength] done in {(time.time()-t0)/60:.1f} min  acc={acc:.3f}  "
              f"retained={samp.shape[0]} states/truth", flush=True)

        rows = []
        for steps in sorted({s for s in LADDER if s < args.n_steps} | {args.n_steps}):
            m = int((steps - args.n_burn) / args.thin)
            cb, _ = coverage(samp[:m] @ st.blind, xt @ st.blind)
            cr, _ = coverage(samp[:m] @ st.resolved, xt @ st.resolved)
            rows.append((steps, m, cb, cr))
            print(f"  steps={steps:6d} (m={m:5d})  blind=" + " ".join(f"{v:5.3f}" for v in cb)
                  + "   resolved=" + " ".join(f"{v:5.3f}" for v in cr), flush=True)

        pb, pr = samp @ st.blind, samp @ st.resolved                    # (m, nt, k)
        tau_b = np.array([[iat_sokal(pb[:, t, k]) for k in range(pb.shape[2])]
                          for t in range(pb.shape[1])])
        tau_r = np.array([[iat_sokal(pr[:, t, k]) for k in range(pr.shape[2])]
                          for t in range(pr.shape[1])])
        m = samp.shape[0]
        for nm, tau in (("blind", tau_b), ("resolved", tau_r)):
            tm = np.nanmedian(tau)
            print(f"  {nm:8s} IAT median={tm:7.1f} (retained units) -> ESS/truth "
                  f"median={m/tm:7.1f};  IAT p90={np.nanpercentile(tau, 90):7.1f}", flush=True)

        p = out_path(f"darcy_pcn_chainlength_{arm}.npz")
        np.savez(p, levels=LEVELS, acc=acc, n_truths=len(yt), thin=args.thin, beta=args.beta,
                 steps=np.array([r[0] for r in rows]), m_kept=np.array([r[1] for r in rows]),
                 cov_blind=np.stack([r[2] for r in rows]),
                 cov_resolved=np.stack([r[3] for r in rows]),
                 iat_blind=tau_b, iat_resolved=tau_r)
        print(f"-> {os.path.relpath(p, REPO)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", default="coverage", choices=sorted(DEFAULTS))
    ap.add_argument("--arm", default="both", choices=["oracle", "curated", "both"])
    ap.add_argument("--seed", type=int, default=0, choices=[0, 1, 2],
                    help="which trained prior to load; the chains themselves are always seeded 0")
    ap.add_argument("--n_truths", type=int, default=None, help="default depends on --mode")
    ap.add_argument("--n_steps", type=int, default=None, help="default depends on --mode")
    ap.add_argument("--n_burn", type=int, default=1000)
    ap.add_argument("--beta", type=float, default=0.08, help="pCN step size")
    ap.add_argument("--thin", type=int, default=5)
    ap.add_argument("--truth_ix", type=int, default=None,
                    help="single mode: use this held-out survey instead of Stuart's fixed field")
    ap.add_argument("--noise_seed", type=int, default=0,
                    help="single mode: noise on the synthesized Stuart survey")
    args = ap.parse_args()
    for k, v in DEFAULTS[args.mode].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    # The two arms share one pCN random stream, in this order, as the released caches were made.
    args.arms = ["oracle", "curated"] if args.arm == "both" else [args.arm]

    st = setup()
    {"coverage": run_coverage, "single": run_single, "chainlength": run_chainlength}[args.mode](
        args, st)


if __name__ == "__main__":
    main()
