"""Real training fields beside fresh unconditional flow samples for the two Darcy HINT priors.

Big field galleries only -- the train/val NLL curves live in fig_darcy_loss.py. Loads each arm's
seed-0 checkpoint (weights + normalizer), samples the flow, and reconstructs log-permeability
fields with the shared KL basis. CPU only.
Run:  python scripts/fig_darcy_training.py
"""
from __future__ import annotations

import os
import sys

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import gridspec  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from resolvability.download import REPO, ensure                 # noqa: E402
from resolvability.groundwater.kl_prior import StuartKLPrior    # noqa: E402
from resolvability.groundwater.hint_flow import HINTFlow        # noqa: E402
from resolvability.style import PALETTE, apply_paper_style      # noqa: E402
apply_paper_style(); plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
EM, MAD = PALETTE["em"], PALETTE["mad"]
OUT = os.path.join(REPO, "figures/fig_darcy_training.pdf")
NGAL = 4


def load_flow(tag):
    ck = torch.load(ensure(f"data/checkpoints/darcy_flow_{tag}_seed0.pth"),
                    map_location="cpu", weights_only=False)
    flow = HINTFlow(ck["n_in"], 0, ck["n_hidden"], n_flow_layers=ck["n_layers"],
                    depth=ck.get("depth"), n_mlp_layers=3)
    flow.load_state_dict(ck["model"]); flow.eval()
    return flow, ck["norm_mean"], ck["norm_std"]


@torch.no_grad()
def sample(flow, mean, std, n, seed):
    torch.manual_seed(seed)
    return (flow.inverse(torch.randn(n, flow.n_in)) * (std + 1e-5) + mean).numpy()


def main():
    with h5py.File(ensure("data/darcy/darcy_laundering.h5"), "r") as f:
        N, K = int(f.attrs["N"]), int(f.attrs["K"])
        alpha, s, sig = float(f.attrs["alpha"]), float(f.attrs["s"]), float(f.attrs["sigma"])
        xi_true, xi_map = f["xi_true"][:], f["xi_map"][:]
    kl = StuartKLPrior(N, K=K, alpha=alpha, s=s, sigma=sig)

    c_flow, c_m, c_s = load_flow("curated")
    o_flow, o_m, o_s = load_flow("oracle")
    rng = np.random.default_rng(1)
    it, im = rng.choice(len(xi_true), NGAL, replace=False), rng.choice(len(xi_map), NGAL, replace=False)
    real = {"curated": [kl.reconstruct(xi_map[i]) for i in im],
            "oracle":  [kl.reconstruct(xi_true[i]) for i in it]}
    samp = {"curated": [kl.reconstruct(x) for x in sample(c_flow, c_m, c_s, NGAL, seed=1)],
            "oracle":  [kl.reconstruct(x) for x in sample(o_flow, o_m, o_s, NGAL, seed=0)]}
    vmax = float(np.percentile(np.abs(np.stack(
        real["curated"] + real["oracle"] + samp["curated"] + samp["oracle"])), 99))

    fig = plt.figure(figsize=(5.125, 3.62))
    outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.23,
                              left=0.055, right=0.995, top=0.925, bottom=0.012)
    rows = [("curated", "curated prior  (legacy MAP archive)", MAD),
            ("oracle", "oracle prior  (KL truths)", EM)]
    for r, (tag, title, col) in enumerate(rows):
        inner = outer[r].subgridspec(2, NGAL, hspace=0.05, wspace=0.04)
        for rr, (imgs, lab) in enumerate([(real[tag], "real"), (samp[tag], "sample")]):
            for j in range(NGAL):
                ax = fig.add_subplot(inner[rr, j])
                ax.imshow(imgs[j], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_linewidth(0.5)
                if j == 0:
                    ax.set_ylabel(lab, fontsize=7.4, labelpad=2.0)
                if rr == 0 and j == 0:
                    ax.set_title(f"({'ab'[r]})  {title}", fontsize=8.2, color=col, loc="left", pad=3.0)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02, dpi=300)
    print("saved", os.path.relpath(OUT, REPO))


if __name__ == "__main__":
    main()
