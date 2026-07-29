#!/usr/bin/env python
"""Blind coverage against the illumination cutoff that defines the seismic blind subspace.

The seismic split is set by a cutoff rather than by an exact kernel, so the cutoff is a free knob
and the reading has to be shown to survive it. Two things are worth reading off this figure:

  * the shortfall does not depend on the choice. The curated prior under-covers at every cutoff
    across two decades, and the gap to the truth-trained control never closes.
  * it is largest where the subspace is most blind. Coarsening the cutoff admits better-illuminated
    directions, which the data do partly constrain, so the curated curve climbs toward nominal --
    a gradation measured rather than assumed.

Below the deployed cutoff the curve is flat for a reason stronger than measurement: the blind basis
is bit-identical there, so the reading cannot move at all. The dotted marker spans that range.

Reads ``results/seismic_cutoff_sweep.npz`` (produced by ``scripts/seismic_cutoff_sweep.py``);
writes ``figures/fig_seismic_cutoff.pdf``. Seconds, CPU.

Run:  python scripts/fig_seismic_cutoff.py
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

CACHE = "results/seismic_cutoff_sweep.npz"
OUT = os.path.join(REPO, "figures/fig_seismic_cutoff.pdf")
DEPLOYED = 0.01


def main():
    z = np.load(ensure(CACHE), allow_pickle=True)
    fracs, levels, bdim = z["fracs"], z["levels"], z["blind_dim"]
    keys, vals = list(z["keys"]), z["vals"]
    lut = dict(zip(keys, vals))
    sis = [-1] if bool(z["from_gallery"]) else list(range(int(z["seeds"])))

    def curve(tag, li):
        return np.array([np.mean([lut[f"{tag}_bl_f{f}_s{si}"][li] for si in sis
                                  if f"{tag}_bl_f{f}_s{si}" in lut]) for f in fracs])

    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    for li, L, ls, mk in ((list(levels).index(0.9), 0.9, "-", "o"),
                          (list(levels).index(0.95), 0.95, "--", "s")):
        ax.axhline(L, color="0.7", lw=0.6, ls=":", zorder=0)
        for tag, col in (("oracle", EM), ("curated", MAD)):
            ax.plot(fracs, curve(tag, li), ls, marker=mk, ms=3.0, lw=1.1, color=col,
                    markerfacecolor=("none" if L == 0.95 else col), markeredgewidth=0.7)
    ax.axvline(DEPLOYED, color="0.55", lw=0.7, ls="-.", zorder=0)
    ax.text(DEPLOYED * 1.15, 0.775, "deployed", fontsize=6.2, color="0.4", rotation=90, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel(r"illumination cutoff ($\times$ resolved median)", fontsize=8)
    ax.set_ylabel("blind coverage", fontsize=8)
    ax.set_ylim(0.76, 1.0)
    ax.tick_params(labelsize=7, which="both")
    for tag, col, y in (("oracle", EM, 0.958), ("curated", MAD, 0.788)):
        ax.text(fracs[0] * 1.05, y, tag, fontsize=6.8, color=col)
    ax.text(0.985, 0.035, "solid: nominal 0.90    open/dashed: 0.95", transform=ax.transAxes,
            fontsize=5.8, color="0.35", ha="right")

    # the blind dimension at each cutoff, along the top
    ax2 = ax.twiny(); ax2.set_xscale("log"); ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(list(fracs)); ax2.set_xticklabels([str(int(b)) for b in bdim], fontsize=6.2)
    ax2.set_xlabel("blind directions admitted", fontsize=7, labelpad=2.5)
    ax2.tick_params(length=2, pad=1)
    ax2.minorticks_off()

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02, dpi=300)
    print("saved", os.path.relpath(OUT, REPO))


if __name__ == "__main__":
    main()
