#!/usr/bin/env python
"""Diffusion posterior sampling on the seismic linearized-Born operator.

The deployed read: for one archived survey, draw a posterior ensemble under each prior by guiding
the prior's reverse diffusion with the data misfit. The resolved content is pinned by the shots;
the blind content is whatever the prior supplies. The two ensembles' pointwise spread is what the
teaser and crossover figures plot.

Guidance chain, normalized model space to data, with the score frozen (eps detached):

    eps  = UNet(x_t, t).detach()
    x0   = (x_t - sqrt(1 - abar) eps) / sqrt(abar)                     [normalized]
    dm   = x0 * (std + 1e-5) + mean;  dm <- mute(dm)                   [physical]
    r_s  = d_obs[s] - J_s(dm);  rnorm = sqrt(sum_s ||r_s||^2)          [per chain]
    x_t-1 = ancestral_step(eps, t, x_t) - preconditioned(grad rnorm)

The preconditioner is per-pixel AdaGrad, which is what tames the wild illumination imbalance of
``J^T r`` -- plain guidance cannot reach the noise floor here, its misfit plateauing well above it
for any step size. But a per-pixel rescale does not commute with the spatially extended blind
basis, and the bent component accumulates coherently over a run. So the applied physical update is
projected off the scored blind basis at every step. That projector is operator-derived and
truth-free: it enforces exactly the constraint the physics already imposes there, ``A N = 0``, and
leaves AdaGrad's equalization intact on the data-reachable complement.

Devito is CPU-only, so the guidance graph runs on CPU; the score network runs on the GPU if one is
free, chunked over chains, detached, and moved back. ``--unet-device cpu`` forces all-CPU.

Reads data/seismic/dataset_eval.h5 (the shot records) and the checkpoints.
Writes results/seismic_dps_recon.npz: the full posterior ensemble per arm, its mean and pointwise
standard deviation, and the truth of that survey. Hours; GPU strongly preferred for the score net.

    python scripts/seismic_dps_posterior.py --nchain 128
"""
from __future__ import annotations

import argparse
import os
import time

import h5py
import numpy as np
import torch
from diffusers import DDPMScheduler

from resolvability.download import REPO, ensure
from resolvability.seismic import MUTE_END, N, NSRC, TAPER
from resolvability.seismic.blind import build_blind_subspace
from resolvability.seismic.born import parihaka_imager
from resolvability.seismic.priors import find_all_ckpts, load_eval, make_unet

EVAL_H5 = "data/seismic/dataset_eval.h5"
OUT = os.path.join(REPO, "results/seismic_dps_recon.npz")


def load_prior(ckpt_path, base, unet_dev):
    """The score network plus the normalization statistics, shaped to broadcast over chains."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    unet = make_unet(base).to(unet_dev)
    unet.load_state_dict(ck["model"])
    unet.eval()
    mean, std = ck["norm_mean"].float(), ck["norm_std"].float()
    assert mean.shape == (N, N) and std.shape == (N, N), f"norm shape {tuple(mean.shape)}"
    return unet, mean.reshape(1, 1, N, N), std.reshape(1, 1, N, N)


def denorm(x0_norm, mean, std):
    """Normalized model space to physical reflectivity."""
    return x0_norm * (std + 1e-5) + mean


def seismic_rnorm(x0_norm, im, mean, std, d_obs, shot_ids):
    """Per-chain data-misfit norm, differentiable in ``x0_norm``.

    ``rnorm_c = sqrt( sum_s || d_obs[s] - J_s(mute(denorm(x0_c))) ||^2 )``. Everything is CPU:
    the wave solver is. The float64 accumulation matters -- summing 12 shot records of ~1e6
    magnitude in float32 loses the residual to cancellation.
    """
    dm = denorm(x0_norm, mean, std)
    dm = im.mute_op(dm, end=MUTE_END, length=TAPER)      # mute before the forward, as in generation
    out = []
    for c in range(dm.shape[0]):
        acc = torch.zeros((), dtype=torch.float64)
        dm_c = dm[c:c + 1]                               # keeps chain c's graph
        for s in shot_ids:
            r = d_obs[int(s)] - im.born_one(dm_c, int(s))
            acc = acc + (r.double() ** 2).sum()
        out.append(torch.sqrt(acc.clamp_min(1e-30)).float())
    return torch.stack(out)


def dps_posterior(unet, unet_dev, im, mean, std, d_obs, nchain, nrev, xi, nsub, guide_from,
                  uchunk, proj_Qb):
    """Guided ancestral sampling; returns the physical posterior ensemble (nchain, N*N).

    The likelihood is stochastic: ``nsub`` shots per reverse step, cycled through a seeded
    permutation. Guidance is skipped over the highest-noise ``guide_from`` fraction of the
    trajectory, where the predicted image is too crude for the misfit gradient to mean anything.
    """
    sched = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2",
                          clip_sample=False)
    sched.set_timesteps(nrev)
    abars = sched.alphas_cumprod
    x = torch.randn(nchain, 1, N, N)                      # CPU: this is the guidance leaf
    Gacc = torch.zeros(nchain, 1, N, N)                   # per-chain, per-pixel AdaGrad accumulator
    adg_eps = 1e-8
    steps = sched.timesteps
    n_skip = int(guide_from * len(steps))
    perm = np.random.permutation(NSRC)
    for k, t in enumerate(steps):
        ab = float(abars[t])
        eps = torch.empty(nchain, 1, N, N)
        with torch.no_grad():
            for i in range(0, nchain, uchunk):
                xb = x[i:i + uchunk].to(unet_dev)
                tb = t.expand(xb.shape[0]).to(unet_dev)
                eps[i:i + uchunk] = unet(xb, tb).sample.detach().cpu()
        if k < n_skip:
            with torch.no_grad():
                x = sched.step(eps, t, x).prev_sample
            continue
        x = x.detach().requires_grad_(True)
        x0 = (x - np.sqrt(1 - ab) * eps) / np.sqrt(ab)    # eps detached: the graph is linear in x
        j = (k - n_skip) * nsub
        shot_ids = [int(perm[(j + s) % NSRC]) for s in range(nsub)]
        rnorm = seismic_rnorm(x0, im, mean, std, d_obs, shot_ids)
        g = torch.autograd.grad(rnorm.sum(), x)[0]
        with torch.no_grad():
            xprev = sched.step(eps, t, x.detach()).prev_sample
            Gacc = Gacc + g ** 2
            upd = xi * g / (torch.sqrt(Gacc) + adg_eps)
            # Remove the applied physical update's component on the scored blind basis.
            dphys = ((std + 1e-5) * upd).reshape(upd.shape[0], -1)
            dphys = dphys - (dphys @ proj_Qb.T) @ proj_Qb
            upd = dphys.reshape(upd.shape) / (std + 1e-5)
            x = xprev - upd
    dm = denorm(x.detach(), mean, std).squeeze(1)
    return dm.reshape(nchain, N * N).numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unet-device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--nchain", type=int, default=128)
    ap.add_argument("--nrev", type=int, default=30)
    ap.add_argument("--xi", type=float, default=0.5, help="guidance step size")
    ap.add_argument("--nsub", type=int, default=1, help="shots per guided step")
    ap.add_argument("--guide-from", type=float, default=0.25)
    ap.add_argument("--uchunk", type=int, default=8, help="chains per score-network forward")
    ap.add_argument("--heldout-a", type=int, default=300, help="first evaluation row")
    ap.add_argument("--gallery-it", type=int, default=1, help="survey to sample, offset from it")
    ap.add_argument("--seed-index", type=int, default=0, help="which training seed of each arm")
    args = ap.parse_args()

    if args.unet_device == "auto":
        unet_dev = "cpu"
        if torch.cuda.is_available():
            try:
                unet_dev = "cuda" if torch.cuda.mem_get_info()[0] / 1e9 > 3.0 else "cpu"
            except Exception:
                unet_dev = "cpu"
    else:
        unet_dev = args.unet_device
    print(f"[dps] unet_device={unet_dev} (the Born solves are always CPU) | nchain={args.nchain} "
          f"nrev={args.nrev} xi={args.xi} nsub={args.nsub}", flush=True)

    im = parihaka_imager()
    sub = build_blind_subspace()
    pQb = torch.tensor(np.asarray(sub.Q_blind), dtype=torch.float32)

    row = args.heldout_a + args.gallery_it
    with h5py.File(ensure(EVAL_H5), "r") as f:
        d_obs = torch.tensor(np.asarray(f["shots"][row]), dtype=torch.float32)

    samp = {}
    t0 = time.time()
    for tag in ("oracle", "curated"):
        path, base = find_all_ckpts(tag)[args.seed_index]
        unet, mean, std = load_prior(path, base, unet_dev)
        torch.manual_seed(0)
        np.random.seed(0)                                # same init and shot order for both arms
        P = dps_posterior(unet, unet_dev, im, mean, std, d_obs, args.nchain, args.nrev,
                          args.xi, args.nsub, args.guide_from, args.uchunk, pQb)
        samp[tag] = np.asarray(P, dtype=np.float32)
        print(f"  [{tag}] {samp[tag].shape} ({time.time() - t0:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT,
             oracle_samples=samp["oracle"], curated_samples=samp["curated"],
             oracle=samp["oracle"].mean(0), curated=samp["curated"].mean(0),
             oracle_std=samp["oracle"].std(0), curated_std=samp["curated"].std(0),
             truth=load_eval("broadband_dm", row, row + 1)[0], gallery_it=args.gallery_it)
    print(f"[dps] saved {os.path.relpath(OUT, REPO)} (evaluation row {row})", flush=True)


if __name__ == "__main__":
    main()
