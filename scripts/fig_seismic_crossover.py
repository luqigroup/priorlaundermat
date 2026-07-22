#!/usr/bin/env python
"""Deployed seismic crossover: posterior spread against the operator's illumination.

One panel. Pixels are binned into deciles of incident-wavefield illumination and each arm's mean
pointwise posterior standard deviation is plotted per decile. The oracle's spread falls as
illumination rises; the curated prior's rises. That reversal is the deployed seismic result: where
the wavefield does not reach, the prior alone decides the width of the interval, and the two priors
disagree about which way it should go.

Reads results/seismic_dps_recon.npz and results/seismic_illum_incident.npz.
Writes figures/fig_seismic_crossover.pdf. Seconds, CPU, no GPU.
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from priorlaundermat.download import REPO, ensure  # noqa: E402
from priorlaundermat.style import PALETTE, apply_paper_style  # noqa: E402

apply_paper_style()
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
EM, MAD, GREY = PALETTE["em"], PALETTE["mad"], PALETTE["memorized"]

N = 256
MUTE = 16          # water column, muted before the Born forward
NDEC = 10
OUT = os.path.join(REPO, "figures/fig_seismic_crossover.pdf")


def main() -> None:
    z = np.load(ensure("results/seismic_dps_recon.npz"))
    il = np.load(ensure("results/seismic_illum_incident.npz"))["illum"]      # (nlat, ndep)

    # Illumination deciles over the imaged region, water column excluded.
    depth = np.tile(np.arange(N), (N, 1)).reshape(-1)
    imaged = depth >= MUTE
    logil = np.log10(il.astype(float) + 1e-3).reshape(-1)
    q = np.quantile(logil[imaged], np.linspace(0, 1, NDEC + 1))
    q[0] -= 1e-6
    q[-1] += 1e-6
    binid = np.digitize(logil, q) - 1
    o_by, c_by, npx = [], [], []
    for b in range(NDEC):
        m = imaged & (binid == b)
        npx.append(int(m.sum()))
        o_by.append(np.nanmean(z["oracle_std"][m]))
        c_by.append(np.nanmean(z["curated_std"][m]))
    o_by, c_by = np.array(o_by), np.array(c_by)
    assert min(npx) > 1000, f"a decile is too sparse to average honestly: {npx}"

    fig, ax = plt.subplots(figsize=(3.5, 2.45))
    fig.subplots_adjust(left=0.175, right=0.985, top=0.965, bottom=0.185)

    dec = np.arange(NDEC)
    ax.axvspan(-0.5, 1.5, color=GREY, alpha=0.16, lw=0)
    ax.plot(dec, o_by, "o-", color=EM, lw=2.0, ms=4.6, label="oracle", zorder=3)
    ax.plot(dec, c_by, "s-", color=MAD, lw=2.0, ms=4.2, label="curated", zorder=3)

    ax.set_xlim(-0.5, NDEC - 0.5)
    ax.set_xticks([0, 3, 6, 9])
    ax.set_ylim(70.0, 1.20 * float(max(o_by.max(), c_by.max())))
    ax.set_xlabel("illumination decile  (dark $\\rightarrow$ illuminated)",
                  fontsize=8.6, labelpad=2.0)
    ax.set_ylabel("posterior standard deviation", fontsize=8.6, labelpad=2.0)
    ax.set_yticks([80, 140, 200])
    ax.tick_params(labelsize=7.6, length=2.4, pad=1.5)
    # "least illuminated", not "blind": the blind subspace is a relative-gap filter on probe atoms,
    # and the bottom two illumination deciles of PIXELS are a different set. The band is a reading
    # aid for the low-illumination end, so it says only that.
    ax.text(0.03, 0.955, "least illuminated", transform=ax.transAxes, fontsize=7.4,
            color="0.40", ha="left", va="top")
    # The wedge the diverging curves open on the right is the only region clear of both.
    ax.legend(fontsize=8.0, loc="center", bbox_to_anchor=(0.70, 0.44), frameon=False,
              handlelength=1.3, labelspacing=0.28, borderaxespad=0.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02, dpi=300)
    print("saved", os.path.relpath(OUT, REPO), flush=True)
    print(f"pixels/decile:     {npx}", flush=True)
    print(f"oracle  by decile: {np.round(o_by, 0)}", flush=True)
    print(f"curated by decile: {np.round(c_by, 0)}", flush=True)


if __name__ == "__main__":
    main()
