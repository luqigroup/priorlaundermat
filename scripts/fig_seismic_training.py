"""Real training images beside fresh unconditional DDPM samples for the two seismic priors.

Big sample galleries only -- the train/val loss curves live in fig_seismic_loss.py. Draws
unconditional samples from each arm's seed-0 checkpoint and shows them beside real training
reflectivity on the shared seismic grayscale (2.3 x median pixel-std over the non-water part, the
window of fig_hero.py). Needs the evaluation archive (auto-downloaded) and is fastest on a GPU.
Run:  python scripts/fig_seismic_training.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import gridspec  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from resolvability.download import REPO                          # noqa: E402
from resolvability.seismic import MUTE_END, N                    # noqa: E402
from resolvability.seismic.priors import find_ckpt, sample_prior, load_eval  # noqa: E402
from resolvability.style import PALETTE, apply_paper_style, extent_km        # noqa: E402
apply_paper_style(); plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
EM, MAD = PALETTE["em"], PALETTE["mad"]
OUT = os.path.join(REPO, "figures/fig_seismic_training.pdf")
NGAL = 4
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KEY = {"curated": "lsrtm5", "oracle": "broadband_dm"}              # training target of each arm


def main():
    real, samp = {}, {}
    for tag in ("curated", "oracle"):
        path, base = find_ckpt(tag)
        real[tag] = load_eval(KEY[tag], 0, NGAL).reshape(NGAL, N, N)
        samp[tag] = sample_prior(path, base, NGAL, NGAL, 250, DEV, sampler="ddpm").reshape(NGAL, N, N)

    all_real = np.concatenate([real["curated"], real["oracle"]], axis=0)
    WIN = 2.3 * float(np.median([np.std(x[MUTE_END:]) for x in all_real]))

    fig = plt.figure(figsize=(5.125, 3.62))
    outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.23,
                              left=0.085, right=0.995, top=0.925, bottom=0.012)
    rows = [("curated", "curated prior  (legacy least-squares migration)", MAD),
            ("oracle", "oracle prior  (broadband reflectivity)", EM)]
    for r, (tag, title, col) in enumerate(rows):
        inner = outer[r].subgridspec(2, NGAL, hspace=0.05, wspace=0.04)
        for rr, (imgs, lab) in enumerate([(real[tag], "real"), (samp[tag], "sample")]):
            for j in range(NGAL):
                ax = fig.add_subplot(inner[rr, j])
                im = imgs[j] - float(imgs[j].mean())               # zero-mean contrast image
                ax.imshow(im.T, cmap="gray", vmin=-WIN, vmax=WIN, aspect="equal",
                          extent=extent_km(N), interpolation="bicubic")
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
