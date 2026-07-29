#!/usr/bin/env python
"""The legacy archive: simulate each survey, then image it the way the archive was imaged.

For every broadband reflectivity in the dataset file this writes

  shots       (n, 12, nt, 24)  the twelve shot records, ``J_s(M x) + coloured noise``
  rtm         (n, 256, 256)    the plain migration of those records, ``(2/nsrc) sum_s J_s^T d_s``
  lsrtm5      (n, 256, 256)    five epochs of stochastic single-shot AdaGrad least-squares
                               migration with an L1 prox -- the curated prior's training images
  objective   (n, 60)          that inversion's data-misfit history
  src_x, rec_x                 lateral source and receiver coordinates, metres
  recon_done  (n,)             1 once a row is fully written

The archive is a sparsity-promoting early-stopped least-squares migration, which is what a real
legacy archive is: an estimate, produced by somebody's regularizer, not a sample from a posterior.
That is the whole point of the experiment -- a prior trained on it inherits that regularizer's
belief about the directions the survey never constrained.

THE BROADBAND TRUTHS ARE SHIPPED, NOT REGENERATED. They already sit in
``data/seismic/dataset_train.h5`` and ``data/seismic/dataset_eval.h5`` under ``broadband_dm``,
alongside the tracked ``horizons`` they were built from. Rebuilding them from the raw Parihaka
volume is out of scope for this repository: it needs a Jython horizon tracker running under
Mines-JTK on JDK 8, and an 11.6 GB seismic volume that is not redistributed here. The convolutional
model that turns tracked horizons into reflectivity is
:func:`resolvability.seismic.pseudosynth.build_pseudosynth`.

Safe and resumable: opens the file, writes one row, closes -- so it is always valid on disk, and
re-running continues where it stopped. Each row's noise draw and shot ordering come from a
generator seeded by that row's global index, so a resumed row reproduces bit-for-bit under
single-threaded Devito. Run it on the shipped file and it reports there is nothing to do.

CPU only (Devito), and slow: 60 forward and 60 adjoint wave solves per row, minutes per row.

    OMP_NUM_THREADS=1 python scripts/seismic_make_dataset.py --split eval [--n 10]
"""
from __future__ import annotations

import argparse
import time

import h5py
import numpy as np
import torch

from resolvability.download import ensure
from resolvability.seismic import DATASET_SEED, MUTE_END, N, NREC, NSRC, N_TRAIN, SIGMA, TAPER
from resolvability.seismic.born import parihaka_imager

EPOCHS, LR = 5, 150.0
# Soft-threshold strength for the archive's sparsity-promoting arm, in the AdaGrad metric. Set at
# 20% of the 99th-percentile amplitude of the unregularized image per median cumulative step, so it
# bites on the noise and not on the reflectors.
L1_LAMBDA = 2615.0
SPLIT_H5 = {"train": "data/seismic/dataset_train.h5", "eval": "data/seismic/dataset_eval.h5"}


def fwd_shot(im, m_np, s):
    """One shot's record from a reflectivity, water-muted before the Born forward."""
    x = im.mute_op(torch.tensor(m_np.reshape(1, 1, N, N), dtype=torch.float32),
                   end=MUTE_END, length=TAPER)
    return im.born_one(x, s).detach().cpu().numpy()


def adj_shot(im, d_s, s):
    """One shot's migration ``J_s^T d_s``, taken as the gradient of the same forward."""
    x = torch.zeros(1, 1, N, N, dtype=torch.float32, requires_grad=True)
    (im.born_one(im.mute_op(x, end=MUTE_END, length=TAPER), s)
     * torch.tensor(d_s, dtype=torch.float32)).sum().backward()
    return x.grad.detach().cpu().numpy().reshape(N, N)


def adagrad_lsrtm(im, d_obs, rng, l1_lam=L1_LAMBDA):
    """Stochastic single-shot proximal-AdaGrad least-squares migration over all sources.

    Each shot update is an AdaGrad step on the data misfit followed by the L1 prox in the AdaGrad
    metric -- a per-pixel soft threshold at ``l1_lam * LR / sqrt(Gacc)``, which is the correct prox
    for that diagonal metric. With ``l1_lam = 0`` this is the plain early-stopped arm, whose
    regularization is implicit: zero initialization and a short run.

    Returns the image and the per-iteration misfit history.
    """
    m = np.zeros((N, N), np.float32)
    Gacc = np.zeros((N, N), np.float32)
    eps = 1e-6
    loss = []
    for _ in range(EPOCHS):
        for s in rng.permutation(NSRC):
            res = fwd_shot(im, m, int(s)) - d_obs[int(s)]
            loss.append(float(np.mean(res ** 2)))
            g = adj_shot(im, res, int(s))
            Gacc += g ** 2
            step = LR / (np.sqrt(Gacc) + eps)
            m = m - step * g
            if l1_lam > 0:
                m = np.sign(m) * np.maximum(np.abs(m) - l1_lam * step, 0.0)
    return m.astype(np.float32), np.array(loss, np.float32)


def rtm_from_dobs(im, d_obs):
    """Plain migration of the stored records, ``(2/nsrc) sum_s J_s^T d_s``, water-muted."""
    acc = np.zeros((N, N), np.float64)
    for s in range(NSRC):
        acc += adj_shot(im, d_obs[int(s)], int(s)).astype(np.float64)
    rtm = (2.0 / NSRC) * acc
    rtm = im.mute_op(torch.tensor(rtm.reshape(1, 1, N, N), dtype=torch.float32),
                     end=MUTE_END, length=TAPER).numpy().reshape(N, N)
    return rtm.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="train", choices=["train", "eval"])
    ap.add_argument("--n", type=int, default=None, help="stop after this many rows")
    args = ap.parse_args()

    torch.manual_seed(DATASET_SEED)
    np.random.seed(DATASET_SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    offset = 0 if args.split == "train" else N_TRAIN
    path = ensure(SPLIT_H5[args.split])
    t0 = time.time()

    im = parihaka_imager()
    ratio = im.adjoint_dot_ratio()
    assert abs(ratio - 1.0) < 1e-3, (
        f"the migration is not the transpose of the demigration (dot ratio {ratio:.6f}); "
        f"nothing written from this operator would be meaningful")
    print(f"[{time.time() - t0:.0f}s] operator built, adjoint dot ratio {ratio:.6f}, nt={im.nt}",
          flush=True)

    with h5py.File(path, "a") as f:
        n_total = f["broadband_dm"].shape[0]
        specs = {"rtm": (n_total, N, N), "lsrtm5": (n_total, N, N),
                 "objective": (n_total, EPOCHS * NSRC),
                 "shots": (n_total, NSRC, im.nt, NREC)}
        for name, shape in specs.items():
            if name not in f:
                f.create_dataset(name, shape=shape, dtype="f4", chunks=(1,) + shape[1:])
        if "recon_done" not in f:
            f.create_dataset("recon_done", shape=(n_total,), dtype="i1")
        if "src_x" not in f:
            f.create_dataset("src_x", data=np.asarray(im._src_x, np.float32))
            f.create_dataset("rec_x", data=np.linspace(
                0.0, im.model.domain_size[0], num=NREC).astype(np.float32))
        f.attrs["epochs"] = EPOCHS
        f.attrs["lr"] = LR
        f.attrs["l1_lambda"] = L1_LAMBDA
        ready = (f["done"][:] == 1) & (f["recon_done"][:] == 0)
        todo = np.where(ready)[0]
        if args.n:
            todo = todo[:args.n]
    print(f"[{time.time() - t0:.0f}s] {len(todo)} rows to image", flush=True)
    if len(todo) == 0:
        print("nothing to do: every row already has its reconstructions.")
        return

    for c, i in enumerate(todo):
        rng = np.random.default_rng(DATASET_SEED + offset + int(i))     # per global row index
        with h5py.File(path, "r") as f:
            truth = f["broadband_dm"][i].astype(np.float32)
        d_obs = np.stack([fwd_shot(im, truth, s) + im.sample_noise(SIGMA, rng)
                          for s in range(NSRC)])
        rtm = rtm_from_dobs(im, d_obs)
        lsrtm, obj = adagrad_lsrtm(im, d_obs, rng)
        with h5py.File(path, "a") as f:
            f["shots"][i] = d_obs
            f["rtm"][i] = rtm
            f["lsrtm5"][i] = lsrtm
            f["objective"][i] = obj
            f["recon_done"][i] = 1
        if c % 5 == 0 or c == len(todo) - 1:
            el = time.time() - t0
            print(f"    {c + 1}/{len(todo)} (row {i}); {el / (c + 1):.0f}s/row, "
                  f"ETA {el / (c + 1) * (len(todo) - c - 1) / 3600:.1f}h", flush=True)
    print(f"[{time.time() - t0:.0f}s] done: {SPLIT_H5[args.split]}", flush=True)


if __name__ == "__main__":
    main()
