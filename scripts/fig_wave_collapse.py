#!/usr/bin/env python
"""The blind-subspace collapse deepens with shadow-mode roughness.

(a) blind spread ratio against roughness: the single-best MAP archive (red) collapses hardest,
    the freeze archive (grey) sits between, the oracle (blue) stays calibrated;
(b) central-90% coverage against roughness: the curated prior undercovers the rough shadow
    while the oracle hugs nominal;
(c) the nonlinear operator overlaid on its linear control -- nearly identical, so the collapse
    is structural, a consequence of regularizer misspecification on the exact kernel, and not a
    product of the nonlinearity.

Panels (a) and (b) read the deployed record. Both curves in panel (c) come from the
self-contained run, so the control varies the cubic coefficient and nothing else; pairing the
deployed record against one of them would conflate the two changes.

Reads: results/wave_collapse_deployed.npz, results/wave_collapse_eps40.npz,
       results/wave_collapse_eps0.npz
Writes: figures/wave_collapse.pdf
Runtime: seconds, CPU.

    python scripts/fig_wave_collapse.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from priorlaundermat.download import REPO, ensure  # noqa: E402
from priorlaundermat.style import LINEWIDTH_IN, PALETTE, apply_paper_style  # noqa: E402

apply_paper_style()

OUT = os.path.join(REPO, "figures", "wave_collapse.pdf")

BLUE = PALETTE["em"]       # oracle (truth-trained prior)
RED = PALETTE["mad"]       # curated (legacy-trained prior) / single-best MAP
GREY = "#7f7f7f"           # freeze (posterior-sample archive)
NOMINAL = "#444444"


def main() -> None:
    d = np.load(ensure("results/wave_collapse_deployed.npz"))
    dc = np.load(ensure("results/wave_collapse_eps40.npz"))
    d0 = np.load(ensure("results/wave_collapse_eps0.npz"))

    order = np.argsort(d["rough_eigvals_blind"])
    x = d["rough_eigvals_blind"][order]
    sm, sf, so = d["sr_map"][order], d["sr_freeze"][order], d["sr_oracle"][order]
    cc, co = d["cov_curated_blind_central"][order], d["cov_oracle_blind_central"][order]

    oc = np.argsort(dc["rough_eigvals_blind"])
    ccc, coc = dc["cov_curated_blind_central"][oc], dc["cov_oracle_blind_central"][oc]
    o0 = np.argsort(d0["rough_eigvals_blind"])
    cc0, co0 = d0["cov_curated_blind_central"][o0], d0["cov_oracle_blind_central"][o0]
    rough_half = slice(x.shape[0] // 2, x.shape[0])

    # bbox_inches="tight" trims the outer margins, so the three panels are authored a little
    # narrower than the text width and the cropped PDF lands at it.
    fig, ax = plt.subplots(1, 3, figsize=(LINEWIDTH_IN - 0.175, 2.45))
    for _a in ax:
        _a.set_xticks([0, 2, 4])

    # (a) spread ratio against roughness -----------------------------------------------
    a = ax[0]
    a.plot(x, so, "-", color=BLUE, lw=1.8, marker="o", ms=2.6, label="oracle")
    a.plot(x, sf, "-", color=GREY, lw=1.6, marker="s", ms=2.4, label="freeze")
    # red is named "curated" in every panel; the caption carries "single-best" as the
    # archive-type modifier, so one arm keeps one name
    a.plot(x, sm, "-", color=RED, lw=1.8, marker="^", ms=2.8, label="curated")
    a.axhline(1.0, ls=":", color=NOMINAL, lw=1.1)
    a.text(x[-1], 1.06, "calibrated", ha="right", va="bottom", color=NOMINAL, fontsize=8)
    a.set_yscale("log")
    a.set_xlabel("roughness")
    a.set_ylabel(r"blind spread ratio $s/s_\star$")
    a.set_title("(a) blind spread ratio", fontsize=8.5)
    a.legend(loc="lower left", fontsize=8, handlelength=1.2,
             handletextpad=0.5, labelspacing=0.35, borderpad=0.25)

    # (b) coverage against roughness ---------------------------------------------------
    b = ax[1]
    b.plot(x, co, "-", color=BLUE, lw=1.8, marker="o", ms=2.6, label="oracle")
    b.plot(x, cc, "-", color=RED, lw=1.8, marker="^", ms=2.8, label="curated")
    b.axhline(0.9, ls=":", color=NOMINAL, lw=1.1)
    b.text(x[-1], 0.93, "nominal", ha="right", va="bottom", color=NOMINAL, fontsize=8)
    b.set_ylim(-0.03, 1.03)
    b.set_xlabel("shadow-mode roughness")
    b.set_ylabel("central-90% coverage")
    b.set_title("(b) coverage collapse", fontsize=8.5)
    b.annotate(f"rough half:\ncurated {cc[rough_half].mean():.2f}\n"
               f"oracle {co[rough_half].mean():.2f}",
               xy=(0.97, 0.38), xycoords="axes fraction", fontsize=8,
               color=NOMINAL, ha="right", va="center")

    # (c) the linear control -----------------------------------------------------------
    c = ax[2]
    # Colour encodes the arm exactly as in (a) and (b); line style encodes the coefficient.
    # The legend keeps two short colour entries so its handles never reach the y-axis spine
    # or the steep red drop, and solid/dashed is noted separately in the open centre band.
    c.plot(x, coc, "-", color=BLUE, lw=1.7, label="oracle")
    c.plot(x, co0, "--", color=BLUE, lw=1.4, dashes=(3, 2))
    c.plot(x, ccc, "-", color=RED, lw=1.7, label="curated")
    c.plot(x, cc0, "--", color=RED, lw=1.4, dashes=(3, 2))
    c.axhline(0.9, ls=":", color=NOMINAL, lw=1.0)
    c.set_ylim(-0.03, 1.03)
    c.set_xlabel("roughness")
    c.set_ylabel("central-90% coverage")
    c.set_title("(c) linear control ($\\epsilon{=}0$)", fontsize=8.5)
    c.legend(loc="center", bbox_to_anchor=(0.60, 0.66), fontsize=8,
             handlelength=1.4, labelspacing=0.3)
    c.text(0.60, 0.36, "solid: $\\epsilon{=}40$\ndashed: $\\epsilon{=}0$",
           transform=c.transAxes, ha="center", va="center", fontsize=8,
           color=NOMINAL, linespacing=1.35)

    fig.tight_layout(w_pad=1.6, rect=(0, 0, 1, 0.98))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(OUT, REPO))


if __name__ == "__main__":
    main()
