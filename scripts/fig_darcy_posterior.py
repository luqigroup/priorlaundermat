#!/usr/bin/env python
"""Groundwater single-survey posterior figure: the oracle's uncertainty against the curated one.

  Row 1 (shared field scale): truth | legacy MAP reconstruction | oracle posterior mean |
        curated posterior mean
  Row 2: oracle pointwise std | curated pointwise std (shared scale) | a mid-domain transect with
        central-90% bands | curated std against oracle std, pixel by pixel

Both priors reconstruct the survey plausibly, but the curated posterior's uncertainty collapses
relative to the oracle's -- overconfident where the sensors cannot constrain the field.

Reads ``results/darcy_pcn_single.npz`` (produced by ``scripts/darcy_pcn.py --mode single``);
writes ``figures/fig_darcy_posterior.pdf``. Seconds, CPU, renders from the cache alone.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from resolvability.download import REPO, ensure
from resolvability.style import PALETTE, apply_paper_style

apply_paper_style()
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
EM, MAD, GREEN = PALETTE["em"], PALETTE["mad"], PALETTE["truth"]
Z90 = 1.6448536269514722                     # central-90% half-width in standard deviations
OUT = os.path.join(REPO, "figures", "fig_darcy_posterior.pdf")


def plot_title(ax, s, color="k"):
    """Panel title above the axes, for the two line panels: an inset label would sit on the data."""
    ax.set_title(s, fontsize=12, fontweight="bold", color=color, pad=4.0, loc="left")


def label(ax, s, color="k"):
    """Inset boxed label, for the image panels: it sits on flat field background."""
    ax.text(0.04, 0.955, s, transform=ax.transAxes, fontsize=12, fontweight="bold",
            va="top", ha="left", color=color,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.82))


def main():
    z = np.load(ensure("results/darcy_pcn_single.npz"))
    N = int(z["N"])
    truth, mapr = z["truth"], z["map_recon"]
    om, cm, ostd, cstd = z["oracle_mean"], z["curated_mean"], z["oracle_std"], z["curated_std"]
    sx, sy = z["sensor_ix"] / (N - 1), z["sensor_iy"] / (N - 1)

    win = float(np.percentile(np.abs(np.stack([truth, mapr, om, cm])), 99))
    svmax = float(np.percentile(np.concatenate([ostd.ravel(), cstd.ravel()]), 99))

    fig = plt.figure(figsize=(9.6, 4.8))
    gs = gridspec.GridSpec(2, 4, figure=fig, wspace=0.34, hspace=0.20,
                           left=0.01, right=0.99, top=0.99, bottom=0.10)
    ax = np.empty((2, 4), dtype=object)
    for r in range(2):
        for c in range(4):
            ax[r, c] = fig.add_subplot(gs[r, c])

    def field(a, v, cmap, vmin, vmax, lab, labcol="k", sensors=False):
        im = a.imshow(v.T, cmap=cmap, origin="lower", extent=[0, 1, 0, 1], aspect="equal",
                      interpolation="bicubic", vmin=vmin, vmax=vmax)
        if sensors:
            a.scatter(sx, sy, s=8, facecolor="none", edgecolor="w", linewidth=0.6, zorder=3)
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_linewidth(0.7)
        label(a, lab, labcol)
        return im

    def hcbar(a, im, lab, ticks=None):
        cax = a.inset_axes([0.12, -0.085, 0.76, 0.04])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        if ticks is not None:
            cb.set_ticks(ticks)
        cb.ax.tick_params(labelsize=9, length=2.2, pad=1.5)
        cb.outline.set_linewidth(0.5); cb.set_label(lab, fontsize=10, labelpad=2.0)

    def vcbar(a, im, lab, ticks=None):
        # Row 1's shared bar lives in the right margin, which is free at every figure width;
        # hung below panel (d) it clips against row 2.
        cax = a.inset_axes([1.045, 0.06, 0.038, 0.88])
        cb = fig.colorbar(im, cax=cax, orientation="vertical")
        if ticks is not None:
            cb.set_ticks(ticks)
        cb.ax.tick_params(labelsize=9, length=2.2, pad=1.5)
        cb.outline.set_linewidth(0.5); cb.set_label(lab, fontsize=10, labelpad=3.0)

    # Row 1: reconstructions, on a shared field scale.
    field(ax[0, 0], truth, "RdBu_r", -win, win, "(a) truth")
    field(ax[0, 1], mapr, "RdBu_r", -win, win, "(b) legacy MAP")
    field(ax[0, 2], om, "RdBu_r", -win, win, "(c) oracle mean", EM)
    im1 = field(ax[0, 3], cm, "RdBu_r", -win, win, "(d) curated mean", MAD)
    vcbar(ax[0, 3], im1, "log-permeability", ticks=[-round(win, 1), 0, round(win, 1)])

    # Row 2: uncertainty, on a shared standard-deviation scale.
    field(ax[1, 0], ostd, "magma", 0, svmax, "(e) oracle std", EM, sensors=True)
    im2 = field(ax[1, 1], cstd, "magma", 0, svmax, "(f) curated std", MAD, sensors=True)
    hcbar(ax[1, 1], im2, "posterior std (shared)", ticks=[0, round(svmax, 2)])

    # (g) mid-domain transect. Built only from the means and stds already plotted, so it is not a
    # second computation of the posterior.
    axt = ax[1, 2]
    j = truth.shape[0] // 2
    xs = np.linspace(0.0, 1.0, truth.shape[1])
    for mu, sd, col in ((om[j], ostd[j], EM), (cm[j], cstd[j], MAD)):
        axt.fill_between(xs, mu - Z90 * sd, mu + Z90 * sd, color=col, alpha=0.28, lw=0)
        axt.plot(xs, mu, color=col, lw=1.3)
    axt.plot(xs, truth[j], color=GREEN, lw=1.6)
    axt.set_xlim(0, 1)
    # Limits from the truth's own range as well as the bands', so no excursion is clipped: the
    # truth leaving the curated band is the panel's point and has to be visible where it happens.
    lo = min(float(truth[j].min()), float((cm[j] - Z90 * cstd[j]).min()),
             float((om[j] - Z90 * ostd[j]).min()))
    hi = max(float(truth[j].max()), float((cm[j] + Z90 * cstd[j]).max()),
             float((om[j] + Z90 * ostd[j]).max()))
    pad = 0.08 * (hi - lo)
    axt.set_ylim(lo - pad, hi + pad)
    axt.set_xlabel("$x_2$ (transect at mid-domain)", fontsize=10.5)
    # The arm colours are the paper's convention and are named in the caption, so no legend.
    axt.set_ylabel("$u$", fontsize=10.5)
    axt.yaxis.set_label_coords(-0.21, 0.5)
    axt.yaxis.set_major_locator(plt.MaxNLocator(4))
    axt.tick_params(labelsize=9.5)
    for sp in ("top", "right"):
        axt.spines[sp].set_visible(False)
    plot_title(axt, "(g) transect")

    # (h) curated std against oracle std, one point per pixel.
    axh = ax[1, 3]
    o, c = ostd.ravel(), cstd.ravel()
    axh.scatter(o, c, s=2, c="0.35", alpha=0.30, edgecolors="none", rasterized=True)
    m = max(o.max(), c.max()) * 1.02
    axh.plot([0, m], [0, m], ls=(0, (4, 2)), c="0.4", lw=1.2)
    axh.set_xlim(0, m); axh.set_ylim(0, m)
    axh.set_xlabel("oracle std", fontsize=10.5); axh.set_ylabel("curated std", fontsize=10.5)
    axh.yaxis.set_label_coords(-0.15, 0.5)
    axh.tick_params(labelsize=9.5); axh.set_aspect("equal")
    for sp in ("top", "right"):
        axh.spines[sp].set_visible(False)
    plot_title(axh, "(h) per-pixel")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02, dpi=300)
    print(f"win={win:.2f} svmax={svmax:.3f} "
          f"curated tighter at {100 * float(np.mean(cstd < ostd)):.1f}% of pixels", flush=True)
    print("saved", os.path.relpath(OUT, REPO), flush=True)


if __name__ == "__main__":
    main()
