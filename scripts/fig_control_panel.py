#!/usr/bin/env python
"""Three controls, displayed.

(a) The mechanism, in closed form: as the regularizer's blind spread is dialled, the curated
    prior's blind spread rides it, while the single-best archive collapses and the truth stays
    flat. One fixed aligned instance of the exactly solvable model; only the dial moves. No
    sampling and no training -- these are matrix identities.
(b) Robustness of the seismic blind coverage to where the resolved/blind cutoff is placed.
(c) Groundwater blind coverage against pCN chain length, answering whether the chain length was
    tuned on the outcome by showing the whole ladder.

Reads ``results/seismic_prior_samples.npz``, ``results/seismic_data_kappa.npz``,
``results/seismic_prior_seed_coords.npz`` and ``results/darcy_pcn_chainlength_{arm}.npz``.
Writes ``figures/fig_control_panel.pdf``. CPU, well under a minute.
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from priorlaundermat.closed_form import conditional_blind_variance, curated_covariances  # noqa: E402
from priorlaundermat.download import REPO, ensure  # noqa: E402
from priorlaundermat.style import LINEWIDTH_IN, PALETTE, apply_paper_style  # noqa: E402

apply_paper_style()
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
EM, MAD, TRU = PALETTE["em"], PALETTE["mad"], PALETTE["truth"]
OUT = os.path.join(REPO, "figures/fig_control_panel.pdf")
ALPHA = 0.90


def mechanism_dial():
    """Curated blind spread against the regularizer's, both relative to the truth's.

    A fixed aligned instance in ``d = 20`` with a resolved rank of 12; the regularizer's blind
    block is scaled by the dial and nothing else changes, so the curve isolates the mechanism.
    Returns ``(dial, posterior_sample_arm, single_best_arm)``.
    """
    rng = np.random.default_rng(0)
    d, r = 20, 12
    b = d - r
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Vr = Q[:, :r]
    A = (np.linspace(1.0, 0.3, r)[:, None]) * Vr.T          # r x d, kernel spanned by Q[:, r:]
    Gam = 0.05 * np.eye(r)
    diag_r = rng.uniform(.7, 1.3, r)
    diag_b = rng.uniform(.7, 1.3, b)
    S_star = Q @ np.diag(np.concatenate([diag_r, diag_b])) @ Q.T

    def spread(S):
        return np.sqrt(np.maximum(conditional_blind_variance(S, Q, r), 0.0))

    s_star = spread(S_star)
    dial = np.linspace(0.05, 2.0, 40)
    ps, sb = [], []
    for t in dial:
        S_rho = Q @ np.diag(np.concatenate([diag_r, (t ** 2) * diag_b])) @ Q.T
        c = curated_covariances(S_rho, S_star, A, Gam)
        ps.append((spread(c["S_qS"]) / s_star).mean())      # archive of posterior samples
        sb.append((spread(c["S_qM"]) / s_star).mean())      # archive of single-best reconstructions
    return dial, np.array(ps), np.array(sb)


def cutoff_sweep():
    """Seismic blind central-90% coverage as the resolved/blind cutoff is varied."""
    from priorlaundermat.seismic.blind import build_blind_subspace
    from priorlaundermat.seismic.priors import load_eval

    z = np.load(ensure("results/seismic_prior_seed_coords.npz"))
    lo = int(z["HELDOUT_A"])
    hi = lo + int(z["NEVAL"])
    truth = load_eval("broadband_dm", lo, hi)
    g = np.load(ensure("results/seismic_prior_samples.npz"))
    k = np.load(ensure("results/seismic_data_kappa.npz"))
    e_sig = k["E_sig_rows"][lo:hi].mean()
    kap = {t: float(np.sqrt(e_sig / k[f"E_{t}"].mean())) for t in ("oracle", "curated")}

    def cov(P, Pt):
        loq = np.quantile(P, (1 - ALPHA) / 2, 0)
        hiq = np.quantile(P, (1 + ALPHA) / 2, 0)
        return float(((Pt >= loq) & (Pt <= hiq)).mean())

    cuts = np.array([0.005, 0.01, 0.02, 0.05])
    out = {"oracle": [], "curated": []}
    for f in cuts:
        Qb = build_blind_subspace(rel_floor_frac=float(f)).Q_blind
        Pt = truth @ Qb.T
        for arm in out:
            out[arm].append(cov((g[arm] @ Qb.T) * kap[arm], Pt))
    return cuts, {a: np.array(v) for a, v in out.items()}


def chain_ladder():
    """Groundwater blind central-90% coverage at each chain-length rung, both arms."""
    out = {}
    for arm in ("oracle", "curated"):
        z = np.load(ensure(f"results/darcy_pcn_chainlength_{arm}.npz"))
        lev = np.asarray(z["levels"], float)
        j = int(np.argmin(np.abs(lev - ALPHA)))
        steps = np.asarray(z["steps"], int)
        out[arm] = (steps, np.asarray(z["cov_blind"], float)[:, j])
    return out


def main() -> None:
    dial, ps, sb = mechanism_dial()
    cuts, csw = cutoff_sweep()
    lad = chain_ladder()

    fig, axes = plt.subplots(1, 3, figsize=(LINEWIDTH_IN, 1.70))

    ax = axes[0]
    ax.plot(dial, dial, color="0.6", ls="--", lw=1.0, zorder=1)
    ax.plot(dial, ps, color=MAD, lw=1.8, zorder=3, label="posterior-sample")
    ax.plot(dial, sb, color=MAD, lw=1.6, ls=":", zorder=3, label="single-best")
    ax.axhline(1.0, color=TRU, lw=1.3, zorder=2, label="truth")
    ax.set_xlabel("regularizer / truth spread", fontsize=7.4)
    ax.set_ylabel("curated / truth spread", fontsize=7.4)
    ax.set_title("(a) closed-form mechanism", fontsize=8.2)
    # The dial rises from the origin, so upper-left is the only region clear of all three curves.
    ax.legend(fontsize=5.8, loc="lower right", bbox_to_anchor=(1.0, 0.06), frameon=True,
              framealpha=1.0, edgecolor="0.85", borderpad=0.24, labelspacing=0.20,
              handlelength=1.2)

    ax = axes[1]
    ax.axhline(ALPHA, color="0.6", ls="--", lw=1.0, zorder=1)
    for arm, col, mk in (("curated", MAD, "s"), ("oracle", EM, "o")):
        ax.plot(cuts * 100, csw[arm], marker=mk, color=col, lw=1.8, ms=4, zorder=3, label=arm)
    ax.axvline(1.0, color="0.75", lw=0.9, zorder=1)                 # the deployed cutoff
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 5]); ax.set_xticklabels(["0.5", "1", "2", "5"])
    ax.minorticks_off()                      # log minor labels collide with the custom ticks
    ax.set_xlabel("blind cutoff (% of median)", fontsize=7.4)
    ax.set_ylabel("blind central-90% coverage", fontsize=7.4)
    ax.set_title("(b) cutoff robustness", fontsize=8.2)
    ax.set_ylim(0.5, 1.02)

    ax = axes[2]
    ax.axhline(ALPHA, color="0.6", ls="--", lw=1.0, zorder=1)
    for arm, col, mk in (("curated", MAD, "s"), ("oracle", EM, "o")):
        st, cv = lad[arm]
        ax.plot(st / 1e3, cv, marker=mk, color=col, lw=1.8, ms=4, zorder=3, label=arm)
    ax.set_xscale("log")
    ax.set_xticks([11, 22, 44, 88]); ax.set_xticklabels(["11", "22", "44", "88"])
    ax.minorticks_off()
    ax.set_xlabel("chain length ($10^3$ steps)", fontsize=7.4)
    ax.set_ylabel("blind central-90% coverage", fontsize=7.4)
    ax.set_title("(c) chain-length ladder", fontsize=8.2)
    # The curated arm sits far below nominal at every length, which is the panel's point; panel
    # (b)'s floor of 0.5 would cut it off and show only the oracle's approach to nominal.
    ax.set_ylim(0.0, 1.02)
    ax.legend(fontsize=6.4, loc="center right", frameon=True, framealpha=0.9, edgecolor="0.85")

    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=6.8)
    fig.subplots_adjust(left=0.088, right=0.99, bottom=0.235, top=0.865, wspace=0.55)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", os.path.relpath(OUT, REPO), flush=True)


if __name__ == "__main__":
    main()
