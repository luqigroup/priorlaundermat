#!/usr/bin/env python
"""Groundwater operator-setup figure: what the 33 sensors can and cannot see.

  (a) a true log-permeability field u with the sensors overlaid
  (b) the hydraulic head H, the forward PDE solution the sensors sample
  (c) the operator's singular spectrum -- resolved above the noise floor, blind below, the
      near-null the 33-sensor operator cannot recover

Reads ``data/darcy/darcy_laundering.h5`` and re-solves the forward problem locally; writes
``figures/fig_darcy_setup.pdf``. Seconds, CPU.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from priorlaundermat.download import REPO
from priorlaundermat.groundwater import setup
from priorlaundermat.style import apply_paper_style

apply_paper_style()
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
BLIND, RESOLVED = "#e66101", "#5e3c99"      # the paper's blindcol / rescol
OUT = os.path.join(REPO, "figures", "fig_darcy_setup.pdf")


def main():
    st = setup()
    N, sv, sigma_y = st.N, st.sv, st.sigma_y
    rank = st.resolved.shape[1]

    u = st.kl.reconstruct(st.kl.stuart_true_xi())        # Stuart's fixed ground-truth field
    p = st.fwd.solve(u)
    sx, sy = st.sensors[0] / (N - 1), st.sensors[1] / (N - 1)      # sensor coords in [0,1]^2

    fig = plt.figure(figsize=(11.2, 3.15))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1.15], wspace=0.30,
                           left=0.02, right=0.985, top=0.88, bottom=0.17)
    axu, axp, axs = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

    um = np.abs(u).max()
    axu.imshow(u.T, cmap="RdBu_r", vmin=-um, vmax=um, origin="lower", extent=[0, 1, 0, 1],
               aspect="equal", interpolation="bicubic")
    axu.scatter(sx, sy, s=26, facecolor="none", edgecolor="k", linewidth=1.1, zorder=3)
    axu.set_xticks([0, 1]); axu.set_yticks([0, 1]); axu.tick_params(labelsize=8)
    axu.set_xlabel("$x_1$", fontsize=9); axu.set_ylabel("$x_2$", fontsize=9)
    axu.set_title(f"(a) log-permeability $u$ + {len(sx)} sensors", fontsize=9.5)

    im = axp.imshow(p.T, cmap="viridis", origin="lower", extent=[0, 1, 0, 1], aspect="equal",
                    interpolation="bicubic")
    axp.scatter(sx, sy, s=18, facecolor="none", edgecolor="w", linewidth=0.8, zorder=3)
    axp.set_xticks([0, 1]); axp.set_yticks([0, 1]); axp.tick_params(labelsize=8)
    axp.set_xlabel("$x_1$", fontsize=9)
    axp.set_title("(b) hydraulic head $H$", fontsize=9.5)
    cax = axp.inset_axes([1.03, 0.0, 0.045, 1.0])
    fig.colorbar(im, cax=cax).ax.tick_params(labelsize=7)

    idx = np.arange(1, len(sv) + 1)
    axs.semilogy(idx, sv, "-", color="0.4", lw=1.4, zorder=1)
    axs.scatter(idx[:rank], sv[:rank], s=16, color=RESOLVED, zorder=3, label="resolved")
    axs.scatter(idx[rank:], sv[rank:], s=16, color=BLIND, zorder=3, label="blind (near-null)")
    axs.axhline(sigma_y, color="0.5", ls=(0, (4, 2)), lw=1.1)
    axs.text(0.98, 0.06, r"noise floor $\sigma_y$", transform=axs.transAxes,
             ha="right", va="bottom", fontsize=7.0, color="0.45",
             bbox=dict(fc="w", ec="none", alpha=0.85, pad=1.2))
    axs.axvline(rank + 0.5, color="0.7", lw=0.9, zorder=0)
    axs.set_xlabel("singular index", fontsize=9); axs.set_ylabel("singular value", fontsize=9)
    axs.tick_params(labelsize=8); axs.grid(alpha=0.25, which="both")
    axs.legend(fontsize=8, loc="upper right", frameon=False)
    axs.set_title("(c) operator spectrum: resolved vs blind", fontsize=9.5)
    for sp in ("top", "right"):
        axs.spines[sp].set_visible(False)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02, dpi=300)
    print(f"rank={rank}  sv[0]={sv[0]:.2e} sv[-1]={sv[-1]:.2e}  sigma_y={sigma_y}", flush=True)
    print("saved", os.path.relpath(OUT, REPO), flush=True)


if __name__ == "__main__":
    main()
