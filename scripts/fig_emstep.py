#!/usr/bin/env python
"""The one-EM-step schematic: where an archive-trained prior's belief comes from.

In resolved-blind coordinates, an archive of legacy reconstructions is the handcrafted
regularizer (rho, dashed) advanced one population-EM step to the curated prior (q_rho, solid). On
the resolved direction the belief moves toward the truth -- a marginal-likelihood ascent, so it
does not reach it; on the blind direction it does not move at all, its mean and spread frozen at
the regularizer's, tighter than the truth. The "after" ellipse is one exact step of the
linear-Gaussian population-EM map applied to rho, not hand-drawn, so the geometry is faithful.

No inputs; a self-contained cartoon. Writes figures/ped_emstep.pdf. Instant, CPU, no GPU.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch  # noqa: E402

from priorlaundermat.download import REPO  # noqa: E402
from priorlaundermat.style import PALETTE, apply_paper_style  # noqa: E402

apply_paper_style()
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

RED, GREEN, GREY = PALETTE["mad"], PALETTE["truth"], PALETTE["memorized"]
RESOLVED, BLIND = "#5e3c99", "#e66101"          # paper convention: purple / orange

OUT = os.path.join(REPO, "figures/ped_emstep.pdf")


def em_step(mu, Sig, mu_star, Sig_star, A, gam2):
    """One population-EM / data-averaged-posterior step under the TRUE marginal (linear-Gaussian)."""
    Gi = np.array([[1.0 / gam2]])
    M = A.T @ Gi @ A
    Ct = np.linalg.inv(np.linalg.inv(Sig) + M)
    mu_n = Ct @ (np.linalg.inv(Sig) @ mu + M @ mu_star)
    S = A @ Sig_star @ A.T + gam2 * np.eye(1)
    g = Ct @ A.T @ Gi
    Sig_n = Ct + g @ S @ g.T
    return mu_n, Sig_n


def ellipse(ax, mu, Sig, color, lw=2.0, ls="-", z=3, fill=False, fa=0.0):
    vals, vecs = np.linalg.eigh(Sig)
    ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * np.sqrt(vals[0]), 2 * np.sqrt(vals[1])      # 1-sigma
    e = Ellipse(mu, w, h, angle=ang, edgecolor=color,
                facecolor=color if fill else "none", lw=lw, ls=ls, zorder=z)
    if fill:
        e.set_alpha(fa)
    ax.add_patch(e)


def main():
    A = np.array([[1.0, 0.0]])                              # x resolved, y blind
    gam2 = 0.15
    mu_star, Sig_star = np.array([0.0, 0.0]), np.diag([1.0, 1.0])
    mu_rho = np.array([2.8, 1.8])                           # off-centre start
    Sig_rho = np.diag([0.62 ** 2, 0.40 ** 2])               # over-confident on blind

    # ONE population-EM step from the regularizer: q_rho = T[rho]. The resolved
    # coordinate moves toward the truth (marginal-likelihood ascent -- it does NOT
    # reach it), and the blind coordinate is frozen exactly.
    mu_q, Sig_q = em_step(mu_rho, Sig_rho, mu_star, Sig_star, A, gam2)

    fig, ax = plt.subplots(figsize=(3.4, 2.7))

    # truth (green disc at the origin)
    ellipse(ax, mu_star, Sig_star, GREEN, lw=2.2, z=2)
    ellipse(ax, mu_star, Sig_star, GREEN, fill=True, fa=0.06, z=1)
    ax.plot(*mu_star, "o", color=GREEN, ms=6.5, zorder=4,
            markeredgecolor="white", markeredgewidth=0.6)

    # before: the regularizer rho (dashed)
    ellipse(ax, mu_rho, Sig_rho, RED, lw=2.0, ls=(0, (4, 2.4)), z=3)
    ax.plot(*mu_rho, "o", color=RED, ms=5.5, zorder=4,
            markeredgecolor="white", markeredgewidth=0.6)

    # after: the curated prior q_rho after one EM step (solid); blind coord frozen
    ellipse(ax, mu_q, Sig_q, RED, lw=2.4, z=3)
    ax.plot(*mu_q, "o", color=RED, ms=5.5, zorder=4,
            markeredgecolor="white", markeredgewidth=0.6)

    # the single EM-step arrow: purely horizontal (the blind coordinate never moves);
    # start/end at the ellipse edges in data coordinates so it never pierces a stroke
    x_tail = mu_rho[0] - np.sqrt(Sig_rho[0, 0]) - 0.14
    x_head = mu_q[0] + np.sqrt(Sig_q[0, 0]) + 0.14
    ax.add_patch(FancyArrowPatch((x_tail, mu_rho[1]), (x_head, mu_q[1]),
                                 arrowstyle="-|>", mutation_scale=15, lw=1.7,
                                 color=GREY, zorder=2, shrinkA=0, shrinkB=0))
    ax.text((mu_rho[0] + mu_q[0]) / 2, mu_rho[1] + 1.02,
            "resolved: moved toward the truth", color=RESOLVED, fontsize=8.5,
            ha="center", va="bottom")

    # the frozen blind offset: leadered double arrow between q_rho and the truth
    # (drawn in the blind orange so it parallels the purple resolved caption and its axis)
    xg = -1.55
    for yy in (mu_q[1], mu_star[1]):
        ax.plot([0.0, xg], [yy, yy], color=BLIND, lw=0.7, ls=":", alpha=0.5, zorder=1)
    ax.annotate("", xy=(xg, mu_q[1]), xytext=(xg, mu_star[1]),
                arrowprops=dict(arrowstyle="<->", color=BLIND, lw=1.9, mutation_scale=12))
    ax.text(xg - 0.45, (mu_q[1] + mu_star[1]) / 2, "blind:\nfrozen at\n$\\rho$'s belief",
            color=BLIND, fontsize=8.5, va="center", ha="right", linespacing=1.3)

    # direct labels on the three objects
    ax.text(mu_rho[0], mu_rho[1] - 0.58, r"$\rho$ (regularizer)", color=RED,
            fontsize=9, ha="center", va="top")
    ax.text(mu_q[0], mu_q[1] + 0.52, r"$q_\rho$ (curated prior)", color=RED,
            fontsize=9, ha="center", va="bottom")
    ax.text(mu_star[0], -1.22, r"$p_\star$ (truth)", color=GREEN,
            fontsize=9, ha="center", va="top")

    # schematic axes: directional arrows only, no ticks (this is a cartoon, not data)
    x0, y0 = -4.15, -1.75
    ax.set_xlim(-4.4, 3.95)
    ax.set_ylim(-2.1, 3.45)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.annotate("", xy=(3.85, y0), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color="0.4", lw=1.1))
    ax.annotate("", xy=(x0, 3.35), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color="0.4", lw=1.1))
    ax.text(2.3, y0 - 0.18, "resolved direction", color=RESOLVED, fontsize=8.5,
            ha="center", va="top")
    ax.text(x0 - 0.16, 1.75, "blind direction", color=BLIND, fontsize=8.5,
            rotation=90, ha="right", va="center")

    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
