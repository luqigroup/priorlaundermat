#!/usr/bin/env python
"""The teaser: the whole argument in one glance, on the deployed seismic Born arm.

Panel by panel: the defect is invisible in the reconstruction (a-c) even though the operator's
illumination varies strongly across the section (d); the two priors' posterior spread runs in
opposite directions against that illumination, the oracle widening where the wavefield does not
reach and the curated prior tightening exactly there (e, f); and only against ground truth does the
curated blind interval fall short (g).

The reveal is the standard-deviation maps rather than a single blind-mode marginal. A per-survey
marginal is hostage to one direction and one truth draw -- on this survey the truth sits far into
the tail of the truth population while both posteriors under-disperse on that mode, so it reads as
"both priors miss" rather than as the contrast. The std maps aggregate over every pixel, so the
sign flip cannot be spoiled by a single draw.

Reads results/seismic_dps_recon.npz, results/seismic_illum_incident.npz,
results/seismic_prior_samples.npz, results/seismic_data_kappa.npz, the evaluation window, and the
cached probe basis. Writes figures/hero.pdf. Under a minute, CPU, no GPU.
"""
from __future__ import annotations

import math
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

from resolvability.download import REPO, ensure  # noqa: E402
from resolvability.style import PALETTE, apply_paper_style, extent_km  # noqa: E402
from resolvability.seismic.blind import build_blind_subspace  # noqa: E402
from resolvability.seismic.priors import load_eval  # noqa: E402

apply_paper_style()
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
EM, MAD = PALETTE["em"], PALETTE["mad"]
N = 256
HELDOUT_A = 300
MUTE = 16              # water column: depths above this are muted before imaging
EXT = extent_km(N)     # km, 5.12 wide by 3.2 deep, depth downward
LEVELS = np.array([0.5, 0.7, 0.8, 0.9, 0.95])
RNG = np.random.default_rng(0)
NBOOT = 400
OUT = os.path.join(REPO, "figures/hero.pdf")


def _img(v: np.ndarray) -> np.ndarray:
    """Flat (N*N,) -> displayed image: reshape row-major, then transpose so depth runs downward."""
    return np.asarray(v, dtype=float).reshape(N, N).T


def load_survey() -> dict:
    """The single deployed survey: truth, posterior means, and pointwise posterior spread."""
    d = np.load(ensure("results/seismic_dps_recon.npz"))
    return {"truth": d["truth"], "row": HELDOUT_A + int(d["gallery_it"]),
            "mean": {"oracle": d["oracle"], "curated": d["curated"]},
            "std": {"oracle": d["oracle_std"], "curated": d["curated_std"]}}


def illumination() -> np.ndarray:
    """Incident-wavefield energy: dark where the operator is blind."""
    return np.load(ensure("results/seismic_illum_incident.npz"))["illum"]


def agg_cov(P: np.ndarray, Pt: np.ndarray) -> np.ndarray:
    """Coverage of truth coordinates ``Pt`` by the central band of ``P``, at each credible level."""
    return np.array([np.mean((Pt >= np.quantile(P, (1 - a) / 2, 0))
                             & (Pt <= np.quantile(P, (1 + a) / 2, 0)))
                     for a in LEVELS])


def blind_coverage(Qb: np.ndarray) -> dict:
    """Blind-subspace coverage of each prior, amplitude-calibrated from the data alone.

    The likelihood is flat on the blind subspace, so the deployed posterior's blind marginal is the
    prior's: reading the prior's unconditional draws there IS reading the deployed report.
    """
    z = np.load(ensure("results/seismic_prior_samples.npz"))
    lo, hi = HELDOUT_A, HELDOUT_A + len(z["oracle"])
    Pt = load_eval("broadband_dm", lo, hi) @ Qb.T
    k = np.load(ensure("results/seismic_data_kappa.npz"))
    e_sig = k["E_sig_rows"][lo:hi].mean()
    out = {}
    for tag in ("oracle", "curated"):
        P = (z[tag] @ Qb.T) * float(np.sqrt(e_sig / k[f"E_{tag}"].mean()))
        b = np.array([agg_cov(P[RNG.integers(0, len(P), len(P))],
                              Pt[RNG.integers(0, len(Pt), len(Pt))])
                      for _ in range(NBOOT)])
        out[tag] = (agg_cov(P, Pt), np.percentile(b, 2.5, 0), np.percentile(b, 97.5, 0))
    return out


def main() -> None:
    sv = load_survey()
    Qb = build_blind_subspace().Q_blind
    cov = blind_coverage(Qb)
    illum = illumination()

    # Three wide columns, each pairing a reconstruction with the map it must be read against --
    # the truth over illumination, and each prior's spread directly under its own reconstruction --
    # so the sign flip is a vertical glance rather than a comparison carried across the figure.
    # Nested grids so each gap is set on its own: the image rows sit tight (the upper row carries no
    # distance labels), while the strip below needs clearance for those labels and its own titles.
    fig = plt.figure(figsize=(6.7, 4.55))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[2.2, 0.58], hspace=0.52)
    gs = outer[0].subgridspec(2, 3, hspace=0.15, wspace=0.10)

    def _axes_km(ax, title, col="0.15", *, left=False, bottom=False):
        """Physical axes: depth labelled on the left column, distance on the bottom row."""
        ax.set_title(title, fontsize=8.6, color=col, pad=2.5)
        ax.set_xticks([0, 2, 4])
        ax.set_yticks([0, 1, 2, 3])
        ax.tick_params(labelsize=7.0, length=2.2, pad=1.5)
        if left:
            ax.set_ylabel("depth (km)", fontsize=8.0, labelpad=1.5)
        else:
            ax.set_yticklabels([])
        if bottom:
            ax.set_xlabel("distance (km)", fontsize=8.0, labelpad=1.5)
        else:
            ax.set_xticklabels([])

    # (a-c) the truth and the two posterior means, alike to the eye.
    imgs = [("(a) truth", sv["truth"], "0.15"), ("(b) oracle", sv["mean"]["oracle"], EM),
            ("(c) curated", sv["mean"]["curated"], MAD)]
    win = 2.3 * float(np.median([np.std(_img(v)[MUTE:]) for _, v, _ in imgs]))
    for j_, (name, v, col) in enumerate(imgs):
        ax = fig.add_subplot(gs[0, j_])
        ax.imshow(_img(v), cmap="gray", vmin=-win, vmax=win, aspect="equal", extent=EXT,
                  interpolation="bicubic")
        _axes_km(ax, name, col, left=(j_ == 0))

    # (d) beneath the truth: where the operator is blind. Transposed into the image frame so depth
    # runs downward like the reconstructions -- without that the illumination is rotated against the
    # maps it is read against and the sign flip vanishes into noise.
    ax = fig.add_subplot(gs[1, 0])
    li = np.log10(np.maximum(illum, illum[illum > 0].min())).T
    li[:MUTE] = np.nan
    li = np.ma.masked_invalid(li)
    ilo, ihi = np.percentile(li.compressed(), 5), np.percentile(li.compressed(), 98)
    im_il = ax.imshow(li, cmap="cividis", aspect="equal", extent=EXT, interpolation="bicubic",
                      vmin=ilo, vmax=ihi)
    _axes_km(ax, "(d) illumination (log$_{10}$)", left=True, bottom=True)
    # Horizontal colour bar below the panel: hung on the right it sat in the gutter and its labels
    # reached panel (e). Below, it has the panel's full width and collides with nothing.
    cax_il = ax.inset_axes([0.0, -0.36, 1.0, 0.050])
    cb = fig.colorbar(im_il, cax=cax_il, orientation="horizontal")
    cb.ax.tick_params(labelsize=6.4, length=1.8, pad=1.2)
    # Ticks must lie inside [vmin, vmax]: rounding the low end down puts it below vmin, and the bar
    # then prints a blank block labelled with a value the colormap never reaches.
    cb.set_ticks([math.ceil(float(ilo) * 100) / 100, math.floor(float(ihi) * 100) / 100])
    cb.outline.set_linewidth(0.5)

    # (e, f) each prior's spread directly beneath its own reconstruction.
    so, sc = _img(sv["std"]["oracle"]), _img(sv["std"]["curated"])
    so[:MUTE] = np.nan
    sc[:MUTE] = np.nan
    so, sc = np.ma.masked_invalid(so), np.ma.masked_invalid(sc)
    svmax = float(np.percentile(np.concatenate([so.compressed(), sc.compressed()]), 98))
    # Linear shared scale. A gamma stretch widens the display-space gap in the two maps' overall
    # LEVEL, while the claim is about the GRADIENT along illumination -- it misreads as "curated
    # wider". Linear keeps the one shared comparison honest.
    snorm = matplotlib.colors.Normalize(vmin=0.0, vmax=svmax)
    sp_axes = []
    for j_, (name, S, col) in enumerate((("(e) oracle spread", so, EM),
                                         ("(f) curated spread", sc, MAD))):
        ax = fig.add_subplot(gs[1, j_ + 1])
        im_s = ax.imshow(S, cmap="magma", aspect="equal", extent=EXT, interpolation="bicubic",
                         norm=snorm)
        _axes_km(ax, name, col, bottom=True)
        sp_axes.append(ax)
    # ONE bar spanning both panels, because they share one norm. A bar per panel would read as two
    # independent scales and quietly destroy the only comparison this row exists to make.
    cax_s = sp_axes[0].inset_axes([0.0, -0.36, 2.10, 0.050])
    cb_s = fig.colorbar(im_s, cax=cax_s, orientation="horizontal")
    cb_s.ax.tick_params(labelsize=6.4, length=1.8, pad=1.2)
    cb_s.set_label("posterior std (shared)", fontsize=6.8, labelpad=1.2)
    cb_s.outline.set_linewidth(0.5)

    # (g) against ground truth. Centred at half width so the panel keeps a conservative aspect.
    bot = outer[1].subgridspec(1, 3, wspace=0.26, width_ratios=[0.52, 1.0, 0.52])
    ax = fig.add_subplot(bot[0, 1])
    ax.plot([0.45, 1.0], [0.45, 1.0], color="0.55", ls="--", lw=1.0, zorder=1)
    for tag, col, mk in (("curated", MAD, "s"), ("oracle", EM, "o")):
        c, lo, hi = cov[tag]
        ax.fill_between(LEVELS, lo, hi, color=col, alpha=0.28, lw=0, zorder=2)
        ax.plot(LEVELS, c, marker=mk, color=col, lw=1.5, ms=3.0, zorder=3, label=tag)
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.45, 1.0)
    # Deliberately not square: the two axes span the same range, so widening the panel shallows the
    # diagonal and compresses the vertical gap -- it understates the separation rather than
    # flattering it.
    ax.set_xticks([0.5, 0.75, 1.0])
    ax.set_yticks([0.5, 0.75, 1.0])
    ax.tick_params(labelsize=7.0)
    ax.set_xlabel("credible level", fontsize=8.0, labelpad=1.5)
    ax.set_ylabel("coverage", fontsize=8.0, labelpad=1.5)
    ax.set_title("(g) vs. truth", fontsize=8.6, pad=2.5)
    ax.legend(fontsize=6.8, loc="upper left", frameon=False, handlelength=1.0,
              borderpad=0.1, labelspacing=0.2, handletextpad=0.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # No overall title: the caption's lead states the claim, and repeating it inside the figure is
    # the one place the paper would say the same sentence twice.
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"saved {os.path.relpath(OUT, REPO)} (survey row {sv['row']})", flush=True)


if __name__ == "__main__":
    main()
