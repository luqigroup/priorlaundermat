"""House plotting conventions, so every figure shows the same physics on the same colours.

One concept, one colour, matching the paper's own ``\\definecolor`` declarations; seismic sections
at true physical proportions; a colour bar on every quantitative map. Figures are authored at the
paper's true text width (:data:`LINEWIDTH_IN`) so labels print at the size they are set in.
"""

from __future__ import annotations

import matplotlib as mpl

LINEWIDTH_IN = 5.125          # the article class's \linewidth, in inches
DX, DZ = 20.0, 12.5           # seismic cell size, metres (lateral, depth)

PALETTE = {
    "em": "#1f77b4",         # oracle prior            (emcol)
    "mad": "#d62728",        # curated prior           (madcol)
    "truth": "#2ca02c",      # truth                   (healcol)
    "memorized": "#7f7f7f",  # archive / neutral grey
}


def apply_paper_style() -> None:
    """Set global rcParams for publication-grade vector figures."""
    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "figure.dpi": 150,
    })


def extent_km(n: int = 256) -> list[float]:
    """``[x0, x1, z1, z0]`` in km for ``imshow``, depth increasing downward.

    Drawn with ``aspect="equal"`` this puts the section at true proportions -- 5.12 km wide by
    3.2 km deep. Stretching it to the panel box would misrepresent dip and aperture.
    """
    return [0.0, n * DX / 1000.0, n * DZ / 1000.0, 0.0]


def attach_cbar(fig, im, ax, label="", *, fraction=0.046, pad=0.02, labelsize=6.6, ticks=None):
    """Colour bar for a quantitative map. Every standard-deviation or illumination panel gets one."""
    cb = fig.colorbar(im, ax=ax, fraction=fraction, pad=pad)
    cb.ax.tick_params(labelsize=labelsize, length=1.8, pad=1.0)
    if ticks is not None:
        cb.set_ticks(list(ticks))
    if label:
        cb.set_label(label, fontsize=labelsize + 0.6, labelpad=1.5)
    return cb
