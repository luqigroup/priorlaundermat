#!/usr/bin/env python
"""Deployed posterior samples across the causal shadow.

Diffusion posterior sampling on the wave operator under each prior, for one held-out
measurement, drawn over the 64 nodes with the causal shadow shaded. Outside the shadow the two
ensembles agree, because the data determine those nodes. Inside it the oracle's chains fan out
around the truth while the curated chains collapse to a thin band that excludes it -- the same
overconfidence the prior carries, now visible in a reconstruction.

Reads: results/wave_dps_recon.npz (or recomputes it from the eval archive and both priors)
Writes: figures/fig_wave_posterior.pdf
Runtime: seconds from the cache; about half a minute to recompute, CPU.

    python scripts/fig_wave_posterior.py
    python scripts/fig_wave_posterior.py --recompute
"""

from __future__ import annotations

import argparse
import json
import os

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from priorlaundermat.download import REPO, ensure  # noqa: E402
from priorlaundermat.style import LINEWIDTH_IN, PALETTE, apply_paper_style  # noqa: E402
from priorlaundermat.wave.blind import build_wave_blind_subspace  # noqa: E402
from priorlaundermat.wave.dps import dps_posterior  # noqa: E402
from priorlaundermat.wave.nets import load_prior  # noqa: E402
from priorlaundermat.wave.solver import SemilinearWave1D  # noqa: E402

apply_paper_style()
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
EM, MAD, TRU = PALETTE["em"], PALETTE["mad"], PALETTE["truth"]

CACHE = os.path.join(REPO, "results", "wave_dps_recon.npz")
OUT = os.path.join(REPO, "figures", "fig_wave_posterior.pdf")
N, XI, NREV, NCHAIN = 64, 0.5, 50, 96
SEED = 0


def compute() -> dict:
    """Posterior ensembles for both priors on the first eval measurement."""
    eval_h5 = ensure("data/wave/wave_dataset_eval.h5")
    with h5py.File(eval_h5, "r") as f:
        cfg = json.loads(f.attrs["config"])
        x_true = torch.tensor(f["x_true"][0], dtype=torch.float64)
        y = torch.tensor(f["y"][0], dtype=torch.float64)
    solver = SemilinearWave1D(n=cfg["n"], c=cfg["c"], cfl=cfg["cfl"], recv=tuple(cfg["recv"]),
                              nt=cfg["nt"], eps=cfg["eps"], domain_len=cfg["domain_len"])
    split = build_wave_blind_subspace(solver, n_probe_states=8, seed=cfg["seed"])
    shadow = np.sort(split.shadow_idx.numpy())

    D = {"truth": x_true.numpy(), "shadow": shadow}
    resolved = np.setdiff1d(np.arange(N), shadow)
    for arm in ("oracle", "curated"):
        unet, norm = load_prior(ensure(f"data/checkpoints/wave_prior_{arm}_seed0.pth"),
                                dtype=torch.float64)
        P = dps_posterior(unet, solver, norm, y, n_chain=NCHAIN, n_steps=NREV, xi=XI, seed=SEED)
        D[f"{arm}_samples"] = P
        print(f"[wave-post] {arm}: per-node std, shadow {P.std(0)[shadow].mean():.2f} "
              f"resolved {P.std(0)[resolved].mean():.2f}", flush=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, **D)
    return D


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true",
                    help="rerun posterior sampling instead of reading the cached ensembles")
    args = ap.parse_args()

    D = compute() if args.recompute else np.load(ensure("results/wave_dps_recon.npz"))
    shadow = D["shadow"]
    nodes = np.arange(N)

    # A single axis over 64 nodes needs the horizontal room, so it is authored at a little
    # under twice the text width; the paper scales it back to \linewidth.
    fig, ax = plt.subplots(figsize=(2 * LINEWIDTH_IN - 0.85, 3.4))
    fig.subplots_adjust(left=0.105, right=0.985, top=0.90, bottom=0.17)

    ax.axvspan(shadow.min() - 0.5, shadow.max() + 0.5, color=MAD, alpha=0.09, lw=0)
    OS, CS = D["oracle_samples"], D["curated_samples"]
    # 28 overlaid chains: enough alpha that a single chain reads on the page, low enough that
    # the oracle fan stays a fan, since the contrast with the curated band is the point
    for k in np.linspace(0, OS.shape[0] - 1, 28).astype(int):
        ax.plot(nodes, OS[k], color=EM, lw=0.8, alpha=0.22, zorder=2)
        ax.plot(nodes, CS[k], color=MAD, lw=0.8, alpha=0.22, zorder=3)
    ax.plot(nodes, D["truth"], color=TRU, lw=2.2, zorder=5)
    ax.plot([], [], color=EM, lw=1.6, label="oracle posterior samples")   # opaque legend proxies
    ax.plot([], [], color=MAD, lw=1.6, label="curated posterior samples")
    ax.plot([], [], color=TRU, lw=2.2, label="truth")
    ax.set_xlim(0, N - 1)
    ax.set_xlabel("node")
    ax.set_ylabel("field $x$")
    ax.set_title("truth and posterior samples across the causal shadow", fontsize=10)
    ax.legend(fontsize=7.2, loc="upper right", frameon=False, handlelength=1.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    print("wrote", os.path.relpath(OUT, REPO), flush=True)


if __name__ == "__main__":
    main()
