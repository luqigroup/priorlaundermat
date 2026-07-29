#!/usr/bin/env python
"""The certified blind band, on the groundwater operator.

(a) coverage against nominal, for the band the prior advertises (filled) and the certified band
    built from held-out references (open). The advertised band tracks nominal for the oracle and
    collapses far below it for the curated prior; the certified band sits on the diagonal for both.
    Validity does not depend on the prior, which is the proposition's claim.
(b) the certified half-width divided by the one the prior advertises, against the number of
    references. Both bands scale with the prior's own conditional spread, so the ratio is exactly
    the quantity a practitioner can report: how much wider the truth is on a blind direction than
    the prior claims. It is near one for the calibrated prior and far above it for the curated one,
    and it settles after a handful of references.

Reads ``results/darcy_certify.npz`` (produced by ``scripts/darcy_certify.py``); writes
``figures/fig_darcy_certify.pdf``. Seconds, CPU.

Run:  python scripts/fig_darcy_certify.py
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from resolvability.download import REPO, ensure  # noqa: E402
from resolvability.style import PALETTE, apply_paper_style  # noqa: E402
apply_paper_style(); plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
EM, MAD = PALETTE["em"], PALETTE["mad"]

CACHE = "results/darcy_certify.npz"
OUT = os.path.join(REPO, "figures/fig_darcy_certify.pdf")


def main():
    z = np.load(ensure(CACHE), allow_pickle=True)
    lev, ks, seeds = z["levels"], z["ks"], z["seeds"]
    keys = list(z["keys"]); vals = z["vals"]

    fig, ax = plt.subplots(1, 2, figsize=(5.125, 2.3))

    # ---- (a) coverage against nominal ----------------------------------------------------------
    ax[0].plot([0, 1], [0, 1], color="0.55", lw=0.7, ls=":", zorder=0)
    # Certified first, with large open markers, so the advertised curves stay visible on top of
    # them: all three land on the diagonal and would otherwise hide one another completely.
    for tag, col in (("oracle", EM), ("curated", MAD)):
        ax[0].plot(lev, z[f"{tag}_cert"].mean(0), "s--", ms=5.2, lw=1.0, color=col,
                   markerfacecolor="none", markeredgewidth=0.9, label=f"{tag}, certified")
    for tag, col in (("oracle", EM), ("curated", MAD)):
        ax[0].plot(lev, z[f"{tag}_raw"].mean(0), "o-", ms=2.6, lw=1.0, color=col,
                   label=f"{tag}, advertised")
    ax[0].set_xlabel("nominal credible level", fontsize=8)
    ax[0].set_ylabel("blind coverage", fontsize=8)
    ax[0].set_xlim(0.45, 1.0); ax[0].set_ylim(-0.03, 1.03)
    ax[0].tick_params(labelsize=7)
    ax[0].set_title("(a)  the band that distrusts the prior", fontsize=8.0, loc="left", pad=3.0)
    ax[0].legend(fontsize=5.9, frameon=False, loc="upper left", handletextpad=0.5,
                 labelspacing=0.3, borderpad=0.2)

    # ---- (b) certified width, relative to what the prior advertises -----------------------------
    for tag, col, mk in (("oracle", EM, "o"), ("curated", MAD, "s")):
        w = np.array([[dict(zip(keys, vals[:, 1]))[f"{tag}_s{s}_k{k}"] for k in ks] for s in seeds])
        for row in w:                                              # one line per training seed
            ax[1].plot(ks, row, mk + "-", ms=2.8, lw=0.9, color=col, alpha=0.85,
                       markerfacecolor=("none" if tag == "curated" else col), markeredgewidth=0.7)
    ax[1].axhline(1.0, color="0.55", lw=0.7, ls=":", zorder=0)
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xticks(list(ks)); ax[1].set_xticklabels([str(int(k)) for k in ks])
    ax[1].set_xlabel("reference truths", fontsize=8)
    ax[1].set_ylabel("certified half-width / advertised", fontsize=8)
    ax[1].tick_params(labelsize=7, which="both")
    ax[1].set_title("(b)  and what it costs to say so", fontsize=8.0, loc="left", pad=3.0)
    ax[1].set_ylim(0.72, 26)                       # room for the labels above each family
    for tag, col in (("curated", MAD), ("oracle", EM)):
        w = np.array([[dict(zip(keys, vals[:, 1]))[f"{tag}_s{s}_k{k}"] for k in ks] for s in seeds])
        ax[1].text(ks[0], w[:, 0].max() * 1.45, tag, fontsize=6.8, color=col, ha="left")

    # w_pad reserves horizontal room for panel (a)'s left-anchored title, which is wider than
    # the panel itself and otherwise runs into panel (b)'s y-label (tight_layout only reserves
    # vertical space for titles).
    fig.tight_layout(w_pad=3.4)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02, dpi=300)
    print("saved", os.path.relpath(OUT, REPO))


if __name__ == "__main__":
    main()
