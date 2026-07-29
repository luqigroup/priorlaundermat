#!/usr/bin/env python
"""Per-training-seed blind and resolved coverage of the deployed seismic priors.

For each arm and each training seed: draw a fresh sample set, put it on the common data-fixed
amplitude scale, project onto the operator's blind and resolved bases, and score how often the
held-out truths fall inside the samples' central band. Three seeds per arm is what turns a single
number into an error bar over training randomness.

The statistic is SPREAD and central coverage, never blind-band energy. The bridge to the deployed
setting is that on the blind subspace the likelihood is flat, so the deployed posterior's blind
marginal equals the prior's -- the prior's blind coverage IS the deployed report.

Common yardstick, with no ground truth in it: the data-energy calibration
(scripts/seismic_data_kappa.py) fixes each prior's target resolved standard deviation from the
observed survey energy over the same evaluation rows the truths come from, and every fresh seed's
draw is scalar-matched to that target through its own resolved standard deviation.

Reads results/seismic_data_kappa.npz, results/seismic_prior_samples.npz and the checkpoints.
Writes results/seismic_prior_seed_coords.npz: each seed's calibrated sample coordinates on the two
bases. One GPU, four to six hours for three seeds per arm.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from resolvability.download import REPO, ensure
from resolvability.seismic.blind import build_blind_subspace
from resolvability.seismic.priors import find_all_ckpts, load_eval, sample_prior

ALPHA = 0.90
OUT = os.path.join(REPO, "results/seismic_prior_seed_coords.npz")


def _device():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        if torch.cuda.mem_get_info()[0] / 1e9 < 3.0:
            print("[calib] under 3 GB free -> CPU", flush=True)
            return torch.device("cpu")
    except Exception:
        pass
    return torch.device("cuda")


def cov_trace(prior_coords, truth_coords):
    """Central-90% coverage of the truths by the prior's per-direction band, and the spread ratio.

    The trace ratio is var(prior) / var(truth) summed over directions. Both statistics use only
    the second moment, so this is a statement about SPREAD, not about energy.
    """
    lo = np.quantile(prior_coords, (1 - ALPHA) / 2, 0)
    hi = np.quantile(prior_coords, (1 + ALPHA) / 2, 0)
    cov = float(np.mean([((truth_coords[i] >= lo) & (truth_coords[i] <= hi)).mean()
                         for i in range(len(truth_coords))]))
    trr = float(prior_coords.var(0).sum() / (truth_coords.var(0).sum() + 1e-30))
    return cov, trr


def resolved_match(samples, Qr, truth_res_std):
    """The global scalar putting a sample set on the common, data-fixed resolved scale."""
    return float(truth_res_std / ((samples @ Qr.T).std(0).mean() + 1e-30))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-draw", type=int, default=384, help="prior draws per arm per seed")
    ap.add_argument("--steps", type=int, default=50, help="DDIM reverse steps")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--n-eval", type=int, default=160, help="held-out truths scored against")
    ap.add_argument("--seeds", type=int, default=3, help="training seeds per arm")
    ap.add_argument("--heldout-a", type=int, default=300, help="first evaluation row")
    args = ap.parse_args()

    dev = _device()
    sub = build_blind_subspace()
    bases = {"resolved": sub.Q_resolved, "blind": sub.Q_blind}
    print(f"[calib] dev={dev} blind={sub.r_blind} resolved={sub.r_resolved} | "
          f"draws={args.n_draw} steps={args.steps} n_eval={args.n_eval} seeds={args.seeds}",
          flush=True)

    a, b = args.heldout_a, args.heldout_a + args.n_eval
    truth = load_eval("broadband_dm", a, b)
    tp = {k: truth @ Q.T for k, Q in bases.items()}

    kz = np.load(ensure("results/seismic_data_kappa.npz"))
    gz = np.load(ensure("results/seismic_prior_samples.npz"))
    e_sig = kz["E_sig_rows"][a:b].mean()
    kap_data = {t: float(np.sqrt(e_sig / kz[f"E_{t}"].mean())) for t in ("oracle", "curated")}
    tgt_res_std = {t: kap_data[t] * (gz[t] @ sub.Q_resolved.T).std(0).mean()
                   for t in ("oracle", "curated")}

    agg = {tag: {k: [] for k in bases} for tag in ("oracle", "curated")}
    coords = {}
    t0 = time.time()
    for tag in ("oracle", "curated"):
        cks = find_all_ckpts(tag)[:args.seeds]
        print(f"[calib] {tag}: {len(cks)} training seed(s)", flush=True)
        for si, (path, base) in enumerate(cks):
            torch.manual_seed(0)                       # fix the sampling seed; vary only training
            S = sample_prior(path, base, args.n_draw, args.chunk, args.steps, dev, "ddim")
            kappa = resolved_match(S, sub.Q_resolved, tgt_res_std[tag])
            for k, Q in bases.items():
                P = (S @ Q.T) * kappa                  # project then scale, so float32 stays safe
                coords[f"{tag}_{'res' if k == 'resolved' else 'bl'}_s{si}"] = P.astype(np.float64)
                agg[tag][k].append(cov_trace(P, tp[k]))
            print(f"  [{tag} seed {si}] kappa={kappa:.3f}  "
                  + "  ".join(f"{k} cov={agg[tag][k][-1][0]:.3f} tr={agg[tag][k][-1][1]:.3f}"
                              for k in bases)
                  + f"  ({time.time() - t0:.0f}s)", flush=True)

    def ms(tag, k, i):
        arr = np.array([v[i] for v in agg[tag][k]]) if agg[tag][k] else np.array([np.nan])
        return arr.mean(), arr.std()

    print("\n===== coverage over training seeds (calibrated level 0.90) =====", flush=True)
    for k in ("resolved", "blind"):
        for tag in ("oracle", "curated"):
            print(f"     {tag:8s} {k:8s} cov={ms(tag, k, 0)[0]:.3f}+/-{ms(tag, k, 0)[1]:.3f}  "
                  f"trace={ms(tag, k, 1)[0]:.3f}+/-{ms(tag, k, 1)[1]:.3f}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, **coords, M=args.n_draw, NSTEPS=args.steps, NEVAL=args.n_eval,
             SEEDS=args.seeds, HELDOUT_A=args.heldout_a)
    print(f"[calib] saved {os.path.relpath(OUT, REPO)}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
