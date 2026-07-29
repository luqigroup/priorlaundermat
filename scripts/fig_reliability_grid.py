#!/usr/bin/env python
"""The same priors scored on both subspaces, on both operators that report coverage against a
nominal credible level.

Columns: the resolved subspace, where calibration is checkable without the truth and a
measurement-side check passes, against the blind subspace, where only blind ground truth reveals
the shortfall. Rows: seismic linearized-Born imaging (diffusion priors) and groundwater flow
(normalizing-flow priors, pCN in the flow latent).


Each row keeps the coverage statistic its own example was scored with, rather than a re-derived
one, so the grid agrees with the standalone figures built from the same caches.

Reads ``results/seismic_prior_seed_coords.npz`` and ``results/darcy_pcn_{arm}_s{seed}.npz``.
Writes ``figures/reliability_grid.pdf``. CPU, a few seconds.
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from resolvability.download import REPO, ensure  # noqa: E402
from resolvability.style import PALETTE, apply_paper_style  # noqa: E402

apply_paper_style()
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
EM, MAD = PALETTE["em"], PALETTE["mad"]

SEIS_LEVELS = np.array([0.5, 0.7, 0.8, 0.9, 0.95])
# One generator per estimator. A shared generator would make each row's bootstrap depend on how
# many draws the other row burned first, so reordering the rows would move every band.
RNG_SEIS = np.random.default_rng(0)
RNG_DARCY = np.random.default_rng(0)
NBOOT_SEIS = 1000
NBOOT_DARCY = 2000
DARCY_SEEDS = (0, 1, 2)
OUT = os.path.join(REPO, "figures/reliability_grid.pdf")


def agg_cov(P: np.ndarray, Pt: np.ndarray, levs: np.ndarray) -> np.ndarray:
    """Central coverage of truths ``Pt`` by the sample band of ``P``, per credible level."""
    return np.array([
        np.mean((Pt >= np.quantile(P, (1 - a) / 2, 0)) & (Pt <= np.quantile(P, (1 + a) / 2, 0)))
        for a in levs
    ])


def agg_ci(P: np.ndarray, Pt: np.ndarray, levs: np.ndarray, B: int = NBOOT_SEIS):
    """Bootstrap jointly over the sample and truth axes.

    Resampling the sample axis carries the finite prior-draw budget into the interval, so this is
    wider than a pure evaluation-set bootstrap and must not be described as one.
    """
    c = np.empty((B, len(levs)))
    for b in range(B):
        c[b] = agg_cov(P[RNG_SEIS.integers(0, len(P), len(P))],
                       Pt[RNG_SEIS.integers(0, len(Pt), len(Pt))], levs)
    return np.percentile(c, 2.5, 0), np.percentile(c, 97.5, 0)


def boot_ci_truths(pt: np.ndarray, per_seed: list[int], nboot: int = NBOOT_DARCY):
    """Truth-clustered bootstrap of per-truth coverage ``pt``, shape ``(nL, sum(per_seed))``.

    ``pt`` pools the same truths across training seeds, so its columns are not independent:
    per-truth coverage correlates strongly across seeds. Resampling columns independently would
    understate the interval, so truth indices are resampled and all seeds of a drawn truth are
    carried together. ``per_seed`` is the actual per-seed truth count -- inferring it by division
    would accept seeds of unequal size whose block boundaries match no seed.
    """
    nL, M = pt.shape
    if len(set(per_seed)) != 1:
        raise ValueError(
            f"seeds cover different truth counts {per_seed}; they are not interchangeable draws of "
            f"one estimator and must not be pooled. Re-run the odd seeds at the common "
            f"configuration before rendering.")
    nt = per_seed[0]
    if nt * len(per_seed) != M:
        raise ValueError(f"pooled width {M} != {len(per_seed)} seeds x {nt} truths")
    blocks = pt.reshape(nL, len(per_seed), nt)                 # (nL, seeds, truths)
    means = np.empty((nboot, nL))
    for b in range(nboot):
        idx = RNG_DARCY.integers(0, nt, nt)
        means[b] = blocks[:, :, idx].mean(axis=(1, 2))
    return np.percentile(means, 2.5, axis=0), np.percentile(means, 97.5, axis=0)


def _panel(lev: np.ndarray, cov: dict, ci: dict, seeds: dict | None = None) -> dict:
    """Plot-ready panel, keyed by arm throughout: six same-typed positional arguments in
    oracle/curated order invite a silent transposition, and 'curated sits below oracle' is the
    claim the figure exists to make."""
    return {"lev": lev, "cov": cov, "ci": ci,
            "seeds": seeds or {"oracle": None, "curated": None}}


def seismic() -> dict:
    """Per-training-seed prior samples projected onto the illumination-defined subspaces.

    The amplitude calibration is baked into the cached coordinates and is measurement-only: no
    ground truth enters it. Reported as the mean curve over training seeds, per-seed open markers,
    and a joint samples-by-truths bootstrap band on the seed-pooled draws.
    """
    from resolvability.seismic.blind import build_blind_subspace
    from resolvability.seismic.priors import load_eval

    z = np.load(ensure("results/seismic_prior_seed_coords.npz"))
    nseed = int(z["SEEDS"])
    lo_row = int(z["HELDOUT_A"])
    truth = load_eval("broadband_dm", lo_row, lo_row + int(z["NEVAL"]))
    sub = build_blind_subspace()

    out = {}
    for key, Q, kk in (("resolved", sub.Q_resolved, "res"), ("blind", sub.Q_blind, "bl")):
        Pt = truth @ Q.T
        cov, ci, seeds = {}, {}, {}
        for arm in ("oracle", "curated"):
            per = [np.asarray(z[f"{arm}_{kk}_s{si}"], float) for si in range(nseed)]   # (M, r) each
            curves = np.stack([agg_cov(P, Pt, SEIS_LEVELS) for P in per])              # (K, nL)
            cov[arm] = curves.mean(0)
            ci[arm] = agg_ci(np.concatenate(per, axis=0), Pt, SEIS_LEVELS)
            seeds[arm] = curves
        out[key] = _panel(SEIS_LEVELS, cov, ci, seeds)
    return out


def groundwater() -> dict:
    """Per-seed pCN coverage curves, with a bootstrap over the pooled truths."""
    def load_arm(tag):
        # Explicit seeds rather than a glob: lexicographic order would read s0, s1, s10, s2 at ten
        # or more seeds, changing the pooling order and so the bootstrap draw.
        return [np.load(ensure(f"results/darcy_pcn_{tag}_s{s}.npz")) for s in DARCY_SEEDS]

    arms = {"oracle": load_arm("oracle"), "curated": load_arm("curated")}
    n_seeds = len(DARCY_SEEDS)
    lev = np.asarray(arms["oracle"][0]["levels"], dtype=float)
    # An asymmetric seed count or a level mismatch would compare a K-seed mean against a J-seed one
    # and read on the page as a real effect.
    cfg = {}
    for arm, S in arms.items():
        if len(S) != n_seeds:
            raise ValueError(f"{arm}: {len(S)} seeds, expected {n_seeds}")
        for i, s in enumerate(S):
            c = tuple(int(s[k]) for k in ("n_steps", "n_truths") if k in s.files)
            cfg.setdefault(c, []).append(f"{arm} s{DARCY_SEEDS[i]}")
            if not np.allclose(np.asarray(s["levels"], float), lev):
                raise ValueError(f"{arm} seed {DARCY_SEEDS[i]}: credible levels differ")
    if len(cfg) > 1:
        raise ValueError(
            "seeds were run at different configurations and cannot be pooled as one estimator:\n  "
            + "\n  ".join(f"n_steps/n_truths={k}: {', '.join(v)}" for k, v in cfg.items())
            + "\n pCN credible width depends on chain mixing, so a longer seed is a different "
              "estimator, not another draw of the same one. Re-run the odd seeds and try again.")

    out = {}
    for key in ("resolved", "blind"):
        cov, ci, seeds = {}, {}, {}
        for arm in ("oracle", "curated"):
            S = arms[arm]
            curves = np.stack([np.asarray(s[key], dtype=float) for s in S])            # (K, nL)
            per_seed = [int(np.asarray(s[f"{key}_pt"]).shape[1]) for s in S]
            pt = np.concatenate([np.asarray(s[f"{key}_pt"], dtype=float) for s in S], axis=1)
            cov[arm], ci[arm], seeds[arm] = curves.mean(0), boot_ci_truths(pt, per_seed), curves
        out[key] = _panel(lev, cov, ci, seeds)
    return out


ROWS = [("Seismic Born", seismic), ("Groundwater", groundwater)]
COLS = [("resolved", "resolved subspace", "measurement-side checks pass"),
        ("blind", "blind subspace", "only blind ground truth reveals it")]


def main() -> None:
    # Compute before rendering: a missing cache should abort before any figure state exists.
    panels = [(rname, loader()) for rname, loader in ROWS]

    fig, axes = plt.subplots(2, 2, figsize=(6.15, 3.30), sharex=True, sharey=True)
    for i, (rname, data) in enumerate(panels):
        for j, (ckey, ctitle, csub) in enumerate(COLS):
            ax, p = axes[i, j], data[ckey]
            lev = p["lev"]
            ax.plot([0.42, 1.02], [0.42, 1.02], color="0.55", ls="--", lw=1.1, zorder=1)
            for arm, col, mk in (("curated", MAD, "s"), ("oracle", EM, "o")):
                lo, hi = p["ci"][arm]
                ax.fill_between(lev, lo, hi, color=col, alpha=0.30, lw=0, zorder=3)
                ax.plot(lev, p["cov"][arm], marker=mk, ls="-", color=col, lw=1.9, ms=4.5,
                        zorder=5, label=arm)
                sd = p["seeds"][arm]
                if sd is not None:
                    for k in range(sd.shape[0]):
                        # Drawn above the mean marker and larger, else they hide underneath it.
                        ax.plot(lev, sd[k], marker=mk, ls="none", mfc="none", mec=col,
                                ms=6.2, mew=0.9, alpha=0.85, zorder=6)
            ax.tick_params(labelsize=9)
            if i == 0:
                hcol = "#5e3c99" if j == 0 else "#e66101"      # resolved purple / blind orange
                ax.annotate(ctitle, xy=(0.5, 1.08), xycoords="axes fraction", ha="center",
                            fontsize=10.5, fontweight="bold", color=hcol)
            if i == len(ROWS) - 1:
                ax.set_xlabel("nominal credible level $1-\\alpha$", fontsize=9.5)
            if j == 0:
                ax.set_ylabel(f"{rname}\ncoverage", fontsize=8.5)
            a = int(np.argmin(np.abs(lev - 0.9)))  # the anchor level, not a positional accident
            print(f"{rname:22s} {ckey:8s} oracle={p['cov']['oracle'][a]:.3f} "
                  f"curated={p['cov']['curated'][a]:.3f} (at level {lev[a]:.2f})", flush=True)

    axes[0, 0].legend(fontsize=9, loc="lower right", bbox_to_anchor=(1.0, 0.008),
                      labelspacing=0.22, handlelength=1.5, handletextpad=0.45, borderpad=0.36,
                      framealpha=0.95, edgecolor="0.85", frameon=True, facecolor="white")
    for ax in axes.ravel():
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.115, top=0.845, hspace=0.24, wspace=0.10)
    # WIDE, SHORT panels filling the text width. Both axes still span the same data range, set
    # once (sharex/sharey make per-panel calls global). The shallow diagonal COMPRESSES vertical
    # deviations, understating the coverage gap rather than flattering it -- the conservative
    # direction. The 'calibrated' label rotation is computed from the display transform below,
    # so it follows the panel shape.
    axes[0, 0].set_xlim(0.44, 1.01)
    axes[0, 0].set_ylim(0.0, 1.02)
    # No x tick at the shared edge, so the left column's last label and the right column's first
    # cannot run together across the narrow gap.
    axes[0, 0].set_xticks([0.5, 0.7, 0.9])
    axes[0, 0].set_yticks([0.0, 0.5, 1.0])
    # No overall title: the claim lives in the caption; panel labels only.
    fig.canvas.draw()
    p0 = axes[0, 0].transData.transform((0.5, 0.5))
    p1 = axes[0, 0].transData.transform((0.9, 0.9))
    cal_rot = float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])))
    for i in range(len(ROWS)):
        for j in range(len(COLS)):
            axes[i, j].text(0.62, 0.30, "calibrated", fontsize=8, color="0.5", rotation=cal_rot,
                            rotation_mode="anchor", ha="center", va="center", zorder=6)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", os.path.relpath(OUT, REPO), flush=True)


if __name__ == "__main__":
    main()
