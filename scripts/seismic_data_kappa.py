#!/usr/bin/env python
"""Measurement-only amplitude calibration for the deployed seismic priors.

A generative prior fixes the shape of its samples but not their absolute amplitude, and comparing
two priors' spreads means first putting them on a common scale. The scale here is the one a
deployed pipeline actually has: each sample set is placed at the amplitude at which its SIMULATED
surveys carry the same energy as the OBSERVED surveys,

    kappa(X)^2 = E_sig / mean_i ||J M X_i||^2,
    E_sig      = mean_n ||y_n||^2 - nsrc * nt * nrec * sigma^2,

with ``y_n`` the archived shot gathers, ``M`` the water mute, and the noise energy exact because
the record noise is renormalized to RMS sigma when it is drawn. No ground truth enters -- only the
shot records, the known noise level and the operator. Because both sides pass through the same
operator, illumination weighting and migration cross-talk cancel, and the estimator applied to the
truth ensemble itself returns one.

Two gates run before any sample forward:
  1. one archived shot row is reproduced bit-for-bit (forward convention and seeded noise);
  2. the estimator on 50 held-out truth rows returns kappa ~ 1.

Reads data/seismic/dataset_eval.h5 and results/seismic_prior_samples.npz.
Writes results/seismic_data_kappa.npz: per-row observed signal energy and per-sample forward
energies for the oracle, curated and archive sets. CPU only (Devito), around 20 minutes.
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")
import h5py                                       # noqa: E402
import numpy as np                                # noqa: E402
import torch                                      # noqa: E402

from resolvability.download import REPO, ensure                       # noqa: E402
from resolvability.seismic import (DATASET_SEED, MUTE_END, N, N_TRAIN, NREC, NSRC,  # noqa: E402
                                     SIGMA, TAPER)
from resolvability.seismic.born import parihaka_imager                # noqa: E402

torch.set_num_threads(1)
SEED_OFF = DATASET_SEED + N_TRAIN     # the evaluation rows continue the training split's seeds
ARCHIVE_A, ARCHIVE_B = 300, 460       # the archive window the sample figures use
VAL_ROWS = np.arange(0, 600, 12)      # 50 truth rows for the unbiasedness gate
OUT = os.path.join(REPO, "results/seismic_data_kappa.npz")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    t0 = time.time()
    im = parihaka_imager()
    eval_h5 = ensure("data/seismic/dataset_eval.h5")

    # Measurement side: observed survey energy, with the noise energy subtracted exactly.
    with h5py.File(eval_h5, "r") as f:
        n_rows, _, nt, _ = f["shots"].shape
        E_rows = np.empty(n_rows)
        for a in range(0, n_rows, 40):
            b = min(a + 40, n_rows)
            sh = f["shots"][a:b].astype(np.float64)
            E_rows[a:b] = (sh ** 2).sum(axis=(1, 2, 3))
    E_sig_rows = E_rows - NSRC * nt * NREC * SIGMA ** 2
    print(f"[kappa] E_sig = {E_sig_rows.mean():.4e} per row (energy SNR "
          f"{E_sig_rows.mean() / (NSRC * nt * NREC * SIGMA ** 2):.2f}, {time.time() - t0:.0f}s)",
          flush=True)

    def fwd_shot(m_np, s):
        x = im.mute_op(torch.tensor(np.asarray(m_np, np.float32).reshape(1, 1, N, N)),
                       end=MUTE_END, length=TAPER)
        return im.born_one(x, s).detach().cpu().numpy()

    def JX_energy(m_np):
        return float(sum((fwd_shot(m_np, s) ** 2).sum() for s in range(NSRC)))

    # Gate 1: reproduce one archived shot row.
    i0 = 300
    with h5py.File(eval_h5, "r") as f:
        truth0 = f["broadband_dm"][i0].astype(np.float32)
        cached = f["shots"][i0].astype(np.float64)
    rng = np.random.default_rng(SEED_OFF + i0)
    mine = np.stack([fwd_shot(truth0, s) + im.sample_noise(SIGMA, rng) for s in range(NSRC)])
    rel = float(np.abs(mine - cached).max() / np.abs(cached).max())
    assert rel < 1e-6, f"the forward convention has drifted from the archive: rel={rel:.3e}"
    print(f"[kappa] gate 1: archived row {i0} reproduced (rel {rel:.1e})", flush=True)

    # Gate 2: the estimator returns kappa ~ 1 on the truth ensemble.
    with h5py.File(eval_h5, "r") as f:
        E_truth = np.array([JX_energy(f["broadband_dm"][int(i)]) for i in VAL_ROWS])
    k_truth = float(np.sqrt(E_sig_rows[VAL_ROWS].mean() / E_truth.mean()))
    print(f"[kappa] gate 2: kappa(truth ensemble) = {k_truth:.4f} ({time.time() - t0:.0f}s)",
          flush=True)

    gz = np.load(ensure("results/seismic_prior_samples.npz"))
    sets = {"oracle": gz["oracle"], "curated": gz["curated"]}
    with h5py.File(eval_h5, "r") as f:
        sets["archive"] = f["lsrtm5"][ARCHIVE_A:ARCHIVE_B].reshape(-1, N * N)
    out = {"E_sig_rows": E_sig_rows, "val_rows": VAL_ROWS, "E_truth": E_truth,
           "kappa_truth_gate": k_truth, "sigma": SIGMA}
    for tag, X in sets.items():
        E = np.array([JX_energy(X[k]) for k in range(len(X))])
        out[f"E_{tag}"] = E
        print(f"[kappa] {tag:8s} mean||J X||^2 = {E.mean():.4e}  "
              f"kappa = {np.sqrt(E_sig_rows.mean() / E.mean()):.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, **out)
    print(f"[kappa] saved -> {os.path.relpath(OUT, REPO)}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
