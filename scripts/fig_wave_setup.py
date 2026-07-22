#!/usr/bin/env python
"""The semilinear-wave inverse problem and its causal shadow.

(a) the space-time wavefield for one truth, with the interior receivers and the numerical
    domain-of-dependence cone drawn: finite speed of propagation makes the causal shadow
    (orange) visible as the region no signal reaches within the short window;
(b) the operator's exact sensitivity per node -- resolved nodes (purple) stay order one down
    to the wavefront edge, shadow nodes (orange) are machine-exactly zero, and the linear
    control is overlaid as open rings so the kernel is seen to be the same set at both
    coefficients rather than merely asserted;
(c) the receiver trace at the nonlinear coefficient departs from the linear one, so the
    operator is genuinely nonlinear while the shadow stays dark in both.

Reads: results/wave_setup_bundle.npz
Writes: figures/wave_setup.pdf
Runtime: seconds, CPU.

    python scripts/fig_wave_setup.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import gridspec  # noqa: E402

from priorlaundermat.download import REPO, ensure  # noqa: E402
from priorlaundermat.style import LINEWIDTH_IN, apply_paper_style  # noqa: E402

apply_paper_style()

OUT = os.path.join(REPO, "figures", "wave_setup.pdf")

ORANGE = "#e66101"   # blind / causal shadow
PURPLE = "#5e3c99"   # resolved
DARK = "#222222"     # nonlinear
GREY = "#7f7f7f"     # linear control


def main() -> None:
    d = np.load(ensure("results/wave_setup_bundle.npz"))
    wf = d["wavefield_eps"]          # (nt, n)
    wf0 = d["wavefield_eps0"]
    sens = d["sensitivity_eps"]      # (n,)
    sens0 = d["sensitivity_eps0"]
    shadow = d["shadow_idx"]
    recv = d["recv"]
    nt, n = int(d["nt"]), int(d["n"])
    dt = float(d["dt"])
    T = (nt - 1) * dt
    shadow_lo = int(shadow.min())    # first blind node

    # bbox_inches="tight" trims the outer margins, so the figure is authored a touch wider
    # than the text width and the cropped PDF lands at it.
    fig = plt.figure(figsize=(LINEWIDTH_IN + 0.495, 3.0))
    # The gutter is generous enough that the (b) and (c) y-labels clear panel (a)'s colour-bar
    # tick labels at true print size.
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.32, 1.0], height_ratios=[1, 1],
                           hspace=0.66, wspace=0.54)

    # (a) space-time wavefield, causal cone, shadow ------------------------------------
    axa = fig.add_subplot(gs[:, 0])
    vmax = np.abs(wf).max()
    im = axa.imshow(wf, origin="lower", aspect="auto", cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax, extent=[0, n - 1, 0, T])
    axa.axvspan(shadow_lo - 0.5, n - 1 + 0.5, color=ORANGE, alpha=0.20, lw=0)
    # numerical domain of dependence of the binding (rightmost) receiver
    r = int(recv.max())
    tt = np.array([0.0, T])
    axa.plot(r + (tt / dt), tt, color="k", lw=1.4, ls="--")      # right characteristic
    axa.plot(r - (tt / dt), tt, color="k", lw=1.0, ls=":")       # left characteristic
    for rr in recv:
        axa.plot([rr], [0.012 * T], marker="v", color="k", ms=5, clip_on=False,
                 zorder=6, markeredgecolor="w", markeredgewidth=0.6)
    axa.text((shadow_lo + n) / 2, T * 0.5, "causal\nshadow", color=ORANGE,
             fontsize=8.5, ha="center", va="center", fontweight="bold", zorder=6,
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=ORANGE, lw=0.9, alpha=0.82))
    axa.annotate("domain of\ndependence",
                 xy=(r + 0.6 * (T / dt), 0.6 * T), xytext=(12.5, 0.72 * T),
                 color="k", fontsize=7.5, ha="center", va="top", zorder=6,
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", lw=0.8, alpha=1.0),
                 arrowprops=dict(arrowstyle="->", color="k", lw=0.9))
    axa.set_xlabel("node  $x$")
    axa.set_ylabel("time  $t$")
    axa.set_title("(a) wavefield $u(x,t)$:\nfinite speed hides the shadow", fontsize=8.5)
    axa.set_xlim(0, n - 1)
    axa.set_ylim(0, T)
    # cap the ticks below the data edge, so none lands at the very top next to the callout
    axa.set_yticks(np.arange(0.0, T, 0.02))
    cb = fig.colorbar(im, ax=axa, fraction=0.04, pad=0.015)
    cb.set_label("$u$", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # (b) exact sensitivity per node ---------------------------------------------------
    axb = fig.add_subplot(gs[0, 1])
    nodes = np.arange(n)
    res_mask = np.ones(n, bool)
    res_mask[shadow] = False
    floor = 1e-14
    axb.axvspan(shadow_lo - 0.5, n - 1 + 0.5, color=ORANGE, alpha=0.16, lw=0)
    axb.scatter(nodes[res_mask], np.clip(sens[res_mask], floor, None), s=9,
                color=PURPLE, label="resolved", zorder=3)
    axb.scatter(nodes[res_mask], np.clip(sens0[res_mask], floor, None), s=26,
                facecolors="none", edgecolors=PURPLE, linewidths=0.6, zorder=2)
    axb.scatter(shadow, np.full(shadow.shape, floor), s=9, color=ORANGE,
                marker="v", label=r"shadow ($\equiv 0$)", zorder=3)
    axb.set_yscale("log")
    axb.set_ylim(floor * 0.4, 3.0)
    axb.set_xlim(0, n - 1)
    axb.set_xlabel("node  $x$")
    axb.set_ylabel(r"$\max_t|\partial F/\partial x|$")
    axb.set_title("(b) exact kernel:\nshadow sensitivity $\\equiv 0$", fontsize=8.5)
    axb.legend(loc="upper right", fontsize=6.4, handlelength=1.0, borderpad=0.28,
               labelspacing=0.26, handletextpad=0.4, framealpha=0.95,
               bbox_to_anchor=(1.005, 0.99), frameon=True, edgecolor="0.75")

    # (c) genuine nonlinearity: the receiver trace at both coefficients -----------------
    axc = fig.add_subplot(gs[1, 1])
    rr = int(recv[-1])
    ts = np.arange(nt) * dt
    axc.plot(ts, wf[:, rr], "-", color=DARK, lw=1.8, label=r"$\epsilon=40$ (nonlinear)")
    axc.plot(ts, wf0[:, rr], "--", color=GREY, lw=1.6, dashes=(3, 2),
             label=r"$\epsilon=0$ (linear)")
    rel = np.linalg.norm(wf[:, recv] - wf0[:, recv]) / np.linalg.norm(wf0[:, recv])
    # headroom above for the legend and below for the misfit label, both off the trace
    tv = np.concatenate([wf[:, rr], wf0[:, rr]])
    vlo, vhi = float(tv.min()), float(tv.max())
    span = vhi - vlo
    axc.set_ylim(vlo - 0.28 * span, vhi + 0.34 * span)
    # keep the ticks inside the data band, so none floats alone in the headroom
    axc.set_yticks([t for t in axc.get_yticks()
                    if (vlo - 0.28 * span) <= t <= vhi + 0.15 * span])
    axc.set_xlabel("time  $t$")
    axc.set_ylabel(f"$u$ at receiver {rr}")
    axc.set_title("(c) genuinely nonlinear", fontsize=8.5)
    # the peak is on the right, so an upper-left legend clears the curve
    axc.legend(loc="upper left", fontsize=7.5, handlelength=1.6, ncol=1,
               labelspacing=0.3, borderpad=0.3, frameon=False)
    axc.text(0.03, 0.05, f"$\\|F_{{40}}-F_0\\|/\\|F_0\\|={rel:.2f}$",
             transform=axc.transAxes, fontsize=8, color=DARK, va="bottom", ha="left")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(OUT, REPO), "| nonlinearity rel =", round(float(rel), 3))


if __name__ == "__main__":
    main()
