#!/usr/bin/env python
"""The de-freezing figure: what an added channel corrects, and what it leaves frozen.

(a) prior-to-truth spread ratio, direction by direction, for the three priors. The channel's own
    directions lift toward the truth; the blind directions it does not reach stay collapsed; the
    resolved control is untouched.
(b) the same statement without reference to any basis: the eigenvalues of the CHANGE in prior
    covariance restricted to the operator's blind subspace. Exactly as many large eigenvalues as
    the channel has dimensions, then a gap.

Both priors that were curated are drawn in the curated colour, solid for the survey alone and open
for the joint curation, since the channel is the only difference between them.

Reads the groundwater dataset, ``results/darcy_core_channel.npz``, and the flow checkpoints
``data/checkpoints/darcy_flows/{arm}_s10.pth`` for the oracle, curated and joint arms; all
download on first use. Writes ``figures/fig_darcy_augment.pdf``. CPU, under a minute.

Run:  python scripts/fig_darcy_augment.py
"""
from __future__ import annotations

import os

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from resolvability.download import REPO, ensure  # noqa: E402
from resolvability.groundwater.hint_flow import HINTFlow  # noqa: E402
from resolvability.style import PALETTE, apply_paper_style  # noqa: E402
from resolvability.utils import Normalizer  # noqa: E402
apply_paper_style(); plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
EM, MAD = PALETTE["em"], PALETTE["mad"]

DATA = "data/darcy/darcy_laundering.h5"
CHAN = "results/darcy_core_channel.npz"
CKPT = "data/checkpoints/darcy_flows/{tag}_s{seed}.pth"
OUT = os.path.join(REPO, "figures/fig_darcy_augment.pdf")
SEED, NDRAW = 10, 4000


@torch.no_grad()
def prior_draws(tag, seed=SEED, n=NDRAW):
    ck = torch.load(ensure(CKPT.format(tag=tag, seed=seed)), map_location="cpu",
                    weights_only=False)
    flow = HINTFlow(ck["n_in"], 0, ck["n_hidden"], n_flow_layers=ck["n_layers"],
                    depth=ck.get("depth"), n_mlp_layers=3)
    flow.load_state_dict(ck["model"]); flow.eval()
    nrm = Normalizer.from_stats(ck["norm_mean"], ck["norm_std"])
    torch.manual_seed(seed)
    return nrm.unnormalize(flow.inverse(torch.randn(n, flow.n_in))).numpy()


def main():
    ch = np.load(ensure(CHAN))
    DF, CT, B = ch["defrozen"], ch["control"], ch["blind_exact"]
    with h5py.File(ensure(DATA), "r") as f:
        xi_true, R = f["xi_true"][:], f["resolved"][:]

    X = {t: prior_draws(t) for t in ("oracle", "curated", "joint")}

    def per_dir(Xs, V):
        return (Xs @ V).std(0) / (xi_true @ V).std(0)

    fig, ax = plt.subplots(1, 2, figsize=(5.125, 2.25),
                           gridspec_kw={"width_ratios": [1.55, 1.0]})

    # ---- (a) direction by direction, the three sets laid side by side ----
    blocks = [("channel", DF), ("blind, unreached", CT), ("resolved", R)]
    style = [("oracle", EM, "o", "full", 1.0), ("curated", MAD, "s", "full", 1.0),
             ("joint", MAD, "s", "none", 1.0)]
    x0 = 0
    edges = []
    for name, V in blocks:
        k = V.shape[1]
        xs = np.arange(x0, x0 + k)
        for tag, col, mk, fill, a in style:
            y = per_dir(X[tag], V)
            ax[0].plot(xs, y, mk, ms=2.6, color=col, alpha=a,
                       markerfacecolor=("none" if fill == "none" else col),
                       markeredgewidth=0.7, linestyle="none")
        edges.append((x0, x0 + k, name))
        x0 += k + 3
    ax[0].axhline(1.0, color=PALETTE["truth"], lw=0.8, ls=":", zorder=0)   # the truth's own spread
    for a, b, name in edges:
        ax[0].text((a + b - 1) / 2, 1.30, name, fontsize=6.6, ha="center", color="0.3")
    ax[0].set_ylim(-0.05, 1.44)
    ax[0].set_xlim(-6, x0 - 1)                  # room for the first block label, which is centred
    ax[0].set_xticks([])
    ax[0].set_xlabel("direction", fontsize=8)
    ax[0].set_ylabel("spread, relative to the truth", fontsize=8)
    ax[0].tick_params(labelsize=7)
    ax[0].set_title("(a)  what the channel corrects", fontsize=8.4, loc="left", pad=3.0)
    h = [plt.Line2D([], [], color=EM, marker="o", ls="none", ms=3.2, label="oracle"),
         plt.Line2D([], [], color=MAD, marker="s", ls="none", ms=3.2, label="curated"),
         plt.Line2D([], [], color=MAD, marker="s", ls="none", ms=3.2,
                    markerfacecolor="none", label="curated, jointly")]
    ax[0].legend(handles=h, fontsize=6.2, frameon=False, loc="center left",
                 bbox_to_anchor=(0.005, 0.42), handletextpad=0.4, labelspacing=0.35)

    # ---- (b) basis-free: the spectrum of the covariance change inside the blind subspace ----
    d = np.cov((X["joint"] @ B).T) - np.cov((X["curated"] @ B).T)
    w = np.sort(np.linalg.eigvalsh(d))[::-1]
    k = DF.shape[1]
    ax[1].plot(np.arange(1, k + 1), w[:k], "s", ms=3.0, color=MAD, markerfacecolor="none",
               markeredgewidth=0.8)
    ax[1].plot(np.arange(k + 1, len(w) + 1), w[k:], "o", ms=1.8, color="0.55")
    ax[1].axhline(0.0, color="0.55", lw=0.7, ls=":", zorder=0)
    ax[1].set_xlabel("mode of the change", fontsize=8)
    ax[1].set_ylabel("eigenvalue", fontsize=8)
    ax[1].tick_params(labelsize=7)
    ax[1].set_title("(b)  and how many", fontsize=8.4, loc="left", pad=3.0)

    fig.tight_layout(w_pad=1.4)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02, dpi=300)
    print("saved", os.path.relpath(OUT, REPO))


if __name__ == "__main__":
    main()
