#!/usr/bin/env python
"""The inherited belief in two prior families: blind-mode marginals for seismic and groundwater.

One row per example, identical panels, so the cross-family point is structural rather than
asserted: a diffusion prior on a band-limited migration archive and a normalizing flow on a
single-best MAP archive inherit their archive's blind belief the same way. Each row shows the
blind mode itself, the marginal density of the projection onto it, and the CDF; ``s`` is the
projection's sample standard deviation.

Reads ``results/seismic_prior_samples.npz``, ``results/seismic_data_kappa.npz``, the seismic
evaluation window and probe basis, the groundwater dataset, and both groundwater flows. Writes
``figures/fig_marginals.pdf``. CPU, well under a minute.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import gridspec  # noqa: E402

from priorlaundermat.download import REPO, ensure  # noqa: E402
from priorlaundermat.style import PALETTE, apply_paper_style, attach_cbar, extent_km  # noqa: E402

apply_paper_style()
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
EM, MAD, TRU, ARCH = PALETTE["em"], PALETTE["mad"], PALETTE["truth"], "0.25"

HELDOUT_A = 300              # first held-out evaluation row
N_FLOW_SAMPLES = 4000        # unconditional draws per groundwater flow
OUT = os.path.join(REPO, "figures/fig_marginals.pdf")


def top_blind_mode(coords_true: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Blind direction of greatest true variation, lifted back through the basis.

    The leading right singular vector of the centred blind coordinates. Identical construction for
    both examples, so the two rows are comparable.
    """
    C = coords_true @ basis.T if basis.shape[1] == coords_true.shape[1] else coords_true @ basis
    w = np.linalg.svd(C - C.mean(0), full_matrices=False)[2][0]
    return basis.T @ w if basis.shape[1] == coords_true.shape[1] else basis @ w


def seismic() -> dict:
    from priorlaundermat.seismic import N
    from priorlaundermat.seismic.blind import build_blind_subspace
    from priorlaundermat.seismic.priors import load_eval

    Qb = build_blind_subspace().Q_blind
    z = np.load(ensure("results/seismic_prior_samples.npz"))
    So, Sc = z["oracle"], z["curated"]
    nsamp = len(So)
    truth = load_eval("broadband_dm", HELDOUT_A, HELDOUT_A + nsamp)
    lsrtm = load_eval("lsrtm5", HELDOUT_A, HELDOUT_A + nsamp)

    # Amplitude calibration from the recorded data alone: the ratio of observed-survey energy to
    # the energy each sample set predicts. No ground truth enters it. Averaged over the same
    # evaluation rows the truths come from, since averaging over all rows would bias the ratio.
    k = np.load(ensure("results/seismic_data_kappa.npz"))
    e_sig = k["E_sig_rows"][HELDOUT_A:HELDOUT_A + nsamp].mean()

    def kap(t):
        return float(np.sqrt(e_sig / k[f"E_{t}"].mean()))

    phi = top_blind_mode(truth, Qb)
    proj = dict(truth=truth @ phi, oracle=(So * kap("oracle")) @ phi,
                curated=(Sc * kap("curated")) @ phi, archive=(lsrtm * kap("archive")) @ phi)
    return dict(mode=phi.reshape(N, N).T, proj=proj, kind="section",
                title="Seismic Born imaging, migration archive")


@torch.no_grad()
def _flow_samples(flow, norm, n: int, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    return norm.unnormalize(flow.inverse(torch.randn(n, flow.n_in))).numpy()


def groundwater() -> dict:
    from priorlaundermat.groundwater import load_flow, setup

    st = setup()
    of, on = load_flow("oracle")
    cf, cn = load_flow("curated")
    o_s = _flow_samples(of, on, N_FLOW_SAMPLES, 0)
    c_s = _flow_samples(cf, cn, N_FLOW_SAMPLES, 1)
    blind, xi_true, xi_map = st.blind, st.xi_true, st.xi_map
    v = blind @ np.linalg.svd((xi_true @ blind) - (xi_true @ blind).mean(0),
                              full_matrices=False)[2][0]
    proj = dict(truth=xi_true @ v, oracle=o_s @ v, curated=c_s @ v, archive=xi_map @ v)
    return dict(mode=st.kl.reconstruct(v), proj=proj, kind="field",
                title="Groundwater flow, single-best MAP archive")


def draw_row(fig, gs, r, D):
    axm, axh, axc = (fig.add_subplot(gs[r, c]) for c in range(3))
    p = D["proj"]
    mt = p["truth"].mean()
    s = {k: v.std() for k, v in p.items()}

    pm = float(np.abs(D["mode"]).max())
    if D["kind"] == "section":
        n = D["mode"].shape[0]
        im = axm.imshow(D["mode"], cmap="RdBu_r", vmin=-pm, vmax=pm, aspect="equal",
                        extent=extent_km(n), interpolation="bicubic")
        axm.set_xticks([0, 2, 4]); axm.set_yticks([0, 1, 2, 3])
        axm.set_xlabel("distance (km)", fontsize=7.6, labelpad=1.5)
        axm.set_ylabel("depth (km)", fontsize=7.6, labelpad=1.5)
    else:
        im = axm.imshow(D["mode"].T, cmap="RdBu_r", vmin=-pm, vmax=pm, origin="lower",
                        extent=[0, 1, 0, 1], aspect="equal", interpolation="bicubic")
        axm.set_xticks([0, 1]); axm.set_yticks([0, 1])
    axm.tick_params(labelsize=6.8, length=2.0, pad=1.2)
    attach_cbar(fig, im, axm, "")
    axm.set_title(f"({'ad'[r]}) blind mode $\\mathbf{{v}}$", fontsize=9)

    arms = [("truth", TRU, "-"), ("oracle", EM, "-"), ("curated", MAD, "-"), ("archive", ARCH, "--")]
    lo, hi = np.percentile(p["truth"] - mt, [0.5, 99.5]) * 1.15
    for name, col, ls in arms:
        x = p[name] - mt
        axh.hist(x, bins=45, range=(lo, hi), density=True, histtype="step", color=col, lw=1.5, ls=ls)
        xs = np.sort(x)
        axc.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=col, lw=1.6, ls=ls,
                 label=f"{name} ($s={s[name]:.3g}$)")
    # mathtext has no \bm, so the mode is bolded with \mathbf here.
    xlab = ("$\\langle\\mathbf{v},\\delta m\\rangle$ (centred)" if D["kind"] == "section"
            else "$\\langle\\mathbf{v},\\xi\\rangle$ (centred)")
    for ax, lab in ((axh, "density"), (axc, "CDF")):
        ax.set_xlim(lo, hi); ax.set_ylabel(lab, fontsize=8.4)
        ax.set_xlabel(xlab, fontsize=8.2, labelpad=1.5)
        ax.tick_params(labelsize=7.2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axh.set_yticks([])
    axh.set_title(f"({'be'[r]}) marginal", fontsize=9)
    axc.set_title(f"({'cf'[r]}) CDF", fontsize=9)
    axc.legend(fontsize=6.8, frameon=False, loc="upper left", handlelength=1.3, labelspacing=0.25)
    axm.text(-0.34, 0.5, D["title"], transform=axm.transAxes, rotation=90, va="center", ha="center",
             fontsize=8.2, color="0.3")


def main():
    fig = plt.figure(figsize=(10.4, 5.4))
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[0.84, 1.15, 1.65],
                           wspace=0.36, hspace=0.46, left=0.075, right=0.988, top=0.94, bottom=0.09)
    for r, D in enumerate((seismic(), groundwater())):
        draw_row(fig, gs, r, D)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02, dpi=300)
    print("saved", os.path.relpath(OUT, REPO), flush=True)


if __name__ == "__main__":
    main()
