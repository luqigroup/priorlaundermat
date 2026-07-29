#!/usr/bin/env python
"""Blind coverage against the illumination cutoff that defines the seismic blind subspace.

The seismic operator's blind end is a low-illumination tail rather than an exact kernel, so the
resolved/blind split is set by a cutoff -- directions with ``||Av||^2`` below a fraction of the
resolved median. That fraction is a free knob, and a reading that only held at one value of it
would not be worth much. This sweeps it.

The sweep is cheap for one reason worth stating: the DDIM sampling does not depend on the cutoff,
only the projection does. So each (arm, training seed) batch is generated ONCE, with the same fixed
sampling seed ``seismic_calibrate.py`` uses, and then projected at every cutoff. The whole sweep
costs one calibration run.

Everything downstream of the projection is redone per cutoff, as it must be: the amplitude
calibration ``kappa`` is matched through the resolved subspace, and the resolved subspace moves
with the cutoff. No truth enters that calibration at any cutoff
(``results/seismic_data_kappa.npz``).

Reads ``results/seismic_prior_seed_coords.npz`` (for the shared held-out row window),
``results/seismic_data_kappa.npz``, ``results/seismic_gallery_samples.npz``, the evaluation
window, and -- unless ``--from-gallery`` -- the checkpoints; all download on first use. Writes
``results/seismic_cutoff_sweep.npz``. With ``--from-gallery`` the cached unconditional draws are
scored on CPU in minutes; regenerating per-seed batches needs one GPU for a few hours. Memory: one
(M x N*N) float32 sample block at a time, about 100 MB at the default M.

Run:  python scripts/seismic_cutoff_sweep.py [--fracs 0.003 0.01 0.03 0.1] [--from-gallery]
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

from seismic_calibrate import _device, resolved_match  # the sibling script in this directory

OUT = os.path.join(REPO, "results/seismic_cutoff_sweep.npz")
LEVELS = (0.5, 0.7, 0.8, 0.9, 0.95)
CHUNK = 8


def central_coverage(P, T, levels):
    """Per-direction central-(1-alpha) coverage of the truth by the prior's own quantiles."""
    out = []
    for L in levels:
        lo, hi = np.percentile(P, [50 * (1 - L), 50 * (1 + L)], axis=0)
        out.append(float(((T >= lo) & (T <= hi)).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.003, 0.01, 0.03, 0.1],
                    help="rel_floor_frac values; 0.01 is the deployed one")
    ap.add_argument("--M", type=int, default=384, help="prior draws per arm per training seed")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--neval", type=int, default=160)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--heldout", type=int, default=None)
    ap.add_argument("--from-gallery", action="store_true",
                    help="score cached prior draws on CPU instead of regenerating on GPU")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    z = np.load(ensure("results/seismic_prior_seed_coords.npz"))
    heldout = int(z["HELDOUT_A"]) if args.heldout is None else args.heldout
    dev = _device()

    # bases and truth coordinates, once per cutoff
    subs, truth = {}, load_eval("broadband_dm", heldout, heldout + args.neval)
    kz = np.load(ensure("results/seismic_data_kappa.npz"))
    gcache = np.load(ensure("results/seismic_gallery_samples.npz"))
    e_sig = kz["E_sig_rows"][heldout:heldout + args.neval].mean()
    kap_data = {t: float(np.sqrt(e_sig / kz[f"E_{t}"].mean())) for t in ("oracle", "curated")}
    for f in args.fracs:
        s = build_blind_subspace(rel_floor_frac=f)
        subs[f] = {"Qb": s.Q_blind, "Qr": s.Q_resolved, "rb": s.r_blind, "rr": s.r_resolved,
                   "tb": truth @ s.Q_blind.T, "tr": truth @ s.Q_resolved.T,
                   "tgt": {t: kap_data[t] * (gcache[t] @ s.Q_resolved.T).std(0).mean()
                           for t in ("oracle", "curated")}}
        print(f"[sweep] frac={f:<7g} blind={s.r_blind:<4d} resolved={s.r_resolved}", flush=True)

    res = {}
    t0 = time.time()
    for tag in ("oracle", "curated"):
        if args.from_gallery:
            # Cached unconditional draws, one checkpoint per arm. Enough for the cutoff question:
            # the SAME samples are scored at every cutoff, so the comparison across cutoffs is
            # paired and the sampling noise is common to all of them. Not a substitute for the
            # deployed per-seed read, and not reported as one.
            batches = [(np.asarray(gcache[tag], np.float64), -1)]
        else:
            batches = []
            for si, (path, base) in enumerate(find_all_ckpts(tag)[:args.seeds]):
                torch.manual_seed(0)                   # the sampling seed the calibration fixes
                batches.append((sample_prior(path, base, args.M, CHUNK, args.steps, dev, "ddim"), si))
        for S, si in batches:
            for f in args.fracs:                       # one sample block, reused at every cutoff
                d = subs[f]
                kap = resolved_match(S, d["Qr"], d["tgt"][tag])
                for key, Q, T in (("bl", d["Qb"], d["tb"]), ("res", d["Qr"], d["tr"])):
                    P = (S @ Q.T) * kap
                    res[f"{tag}_{key}_f{f}_s{si}"] = central_coverage(P, T, LEVELS)
            print(f"  [{tag} batch {si}] {len(S)} draws  {time.time()-t0:.0f}s", flush=True)
            del S

    print("\n  blind coverage against nominal, by cutoff (mean over training seeds)")
    print("  %-9s %-9s %-7s %s" % ("arm", "frac", "blind", "  ".join("%-6.2f" % L for L in LEVELS)))
    for tag in ("oracle", "curated"):
        for f in args.fracs:
            sis = [-1] if args.from_gallery else list(range(args.seeds))
            a = np.array([res[f"{tag}_bl_f{f}_s{si}"] for si in sis
                          if f"{tag}_bl_f{f}_s{si}" in res])
            if len(a):
                print("  %-9s %-9g %-7d %s"
                      % (tag, f, subs[f]["rb"], "  ".join("%-6.3f" % v for v in a.mean(0))))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, keys=np.array(list(res), dtype=object),
             vals=np.array([res[k] for k in res], float), levels=np.array(LEVELS),
             fracs=np.array(args.fracs), seeds=(1 if args.from_gallery else args.seeds),
             from_gallery=bool(args.from_gallery),
             blind_dim=np.array([subs[f]["rb"] for f in args.fracs]),
             resolved_dim=np.array([subs[f]["rr"] for f in args.fracs]))
    print("\nsaved", os.path.relpath(args.out, REPO))


if __name__ == "__main__":
    main()
