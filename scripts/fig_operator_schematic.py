#!/usr/bin/env python
"""What "the directions the data cannot see" means for the seismic operator, from the operator.

  (a) the illumination spectrum ||A v||^2 over illumination-ranked probe atoms. Resolved atoms sit
      high; the nominally blind candidates are split by a floor at 1% of the resolved median -- the
      ones point-spread-function leakage pushes above it are dropped, and the survivors span the
      blind basis. This panel is what the appendix cites when it defines the seismic split.
  (b) the radial wavenumber spectrum of a broadband truth against the legacy migration: the archive
      is band-limited and loses the high-|k| content the truth carries.

Purple and orange are reserved for the resolved and blind subspaces throughout the paper, so panel
(b)'s curves use truth green and archive red instead.

Reads data/seismic/effect_existence_spectrum.npz and the evaluation window.
Writes figures/operator_schematic.pdf. Seconds, CPU, no GPU.
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import gridspec  # noqa: E402

from priorlaundermat.download import REPO, ensure  # noqa: E402
from priorlaundermat.style import apply_paper_style  # noqa: E402
from priorlaundermat.seismic.priors import load_eval  # noqa: E402

apply_paper_style()
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
PURPLE, ORANGE = "#5e3c99", "#e66101"   # resolved / blind subspaces
GREEN, RED = "#2ca02c", "#d62728"       # broadband truth / legacy archive
N = 256
NRAD = 20                               # images averaged into each radial spectrum
KMAX = 118
OUT = os.path.join(REPO, "figures/operator_schematic.pdf")


def main() -> None:
    z = np.load(ensure("data/seismic/effect_existence_spectrum.npz"))
    res_ill = np.sort(z["res_bare"])[::-1]
    bl_ill = np.sort(z["bl_bare"])[::-1]
    floor = 0.01 * np.median(z["res_bare"])
    bl_keep = bl_ill < floor            # the same relative-gap filter that forms the blind basis

    truths = load_eval("broadband_dm", 300, 300 + NRAD)
    lsrtms = load_eval("lsrtm5", 300, 300 + NRAD)
    yy, xx = np.mgrid[0:N, 0:N]
    rad_idx = np.hypot(xx - N // 2, yy - N // 2).astype(int)

    def radial(im):
        F = np.abs(np.fft.fftshift(np.fft.fft2(im))) ** 2
        return np.array([F[rad_idx == k].mean() for k in range(1, KMAX)])

    rad_t = np.mean([radial(truths[i].reshape(N, N)) for i in range(NRAD)], 0)
    rad_l = np.mean([radial(lsrtms[i].reshape(N, N)) for i in range(NRAD)], 0)

    fig = plt.figure(figsize=(5.6, 2.35))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.42,
                           left=0.105, right=0.985, top=0.90, bottom=0.20)

    axd = fig.add_subplot(gs[0, 0])
    axd.semilogy(np.arange(len(res_ill)), res_ill, "o-", color=PURPLE, ms=3.2, lw=1.2,
                 label="resolved")
    ib = np.arange(len(bl_ill))
    axd.semilogy(ib[~bl_keep], bl_ill[~bl_keep], "s", mfc="none", mec="0.55", ms=3.2,
                 label="dropped")
    axd.semilogy(ib[bl_keep], bl_ill[bl_keep], "s-", color=ORANGE, ms=3.2, lw=1.2,
                 label="blind (kept)")
    axd.axhline(floor, color="0.3", ls="--", lw=1.0)
    axd.text(1, floor * 0.5, "blind-basis\ncut", fontsize=7.4, color="0.35", ha="left", va="top",
             linespacing=1.05)
    axd.set_xlabel("atom (illumination-ranked)", fontsize=8.6, labelpad=2.0)
    axd.set_ylabel("$\\|\\mathbf{A}\\mathbf{v}\\|^2$", fontsize=9.5, labelpad=2.0)
    axd.tick_params(labelsize=7.6, length=2.4, pad=1.5)
    axd.set_title("(a) illumination spectrum", fontsize=8.8, pad=3.0)
    axd.legend(fontsize=7.2, loc="lower left", frameon=True, framealpha=1.0, facecolor="white",
               edgecolor="0.85", handlelength=1.0, handletextpad=0.5, labelspacing=0.25,
               borderpad=0.3)
    for s in ("top", "right"):
        axd.spines[s].set_visible(False)
    # Pull the last atoms off the right spine so the (a)-(b) gutter breathes.
    dx0, dx1 = axd.get_xlim()
    axd.set_xlim(dx0, dx1 + 0.11 * (dx1 - dx0))

    axe = fig.add_subplot(gs[0, 1])
    kk = np.arange(1, KMAX)
    axe.loglog(kk, rad_t / rad_t.max(), color=GREEN, lw=1.8, label="broadband truth")
    axe.loglog(kk, rad_l / rad_t.max(), color=RED, lw=1.8, label="legacy migration")
    kb = kk[np.argmax(rad_l)]
    axe.axvspan(kk[0], kb, color=PURPLE, alpha=0.06)                # the resolved passband
    axe.set_xlabel("wavenumber $|k|$", fontsize=8.6, labelpad=2.0)
    axe.set_ylabel("power (normalized)", fontsize=8.6, labelpad=2.0)
    axe.tick_params(labelsize=7.6, length=2.4, pad=1.5)
    axe.set_title("(b) the archive is band-limited", fontsize=8.8, pad=3.0)
    # The curves are colour-coded inline in the caption, so no in-panel legend is drawn -- one here
    # would collide with the peak.
    for s in ("top", "right"):
        axe.spines[s].set_visible(False)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print("saved", os.path.relpath(OUT, REPO), flush=True)
    print(f"blind-basis floor {floor:.3g} | kept {int(bl_keep.sum())} of {len(bl_ill)} candidates",
          flush=True)


if __name__ == "__main__":
    main()
