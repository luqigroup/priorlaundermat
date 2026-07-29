"""Train/validation NLL curves for the two Darcy HINT flows, side by side (1x2).

Companion to fig_darcy_training.py (the field galleries). Reads only the loss history stored in
each arm's seed-0 checkpoint. CPU only.
Run:  python scripts/fig_darcy_loss.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from resolvability.download import REPO, ensure                 # noqa: E402
from resolvability.style import PALETTE, apply_paper_style      # noqa: E402
apply_paper_style(); plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
EM, MAD = PALETTE["em"], PALETTE["mad"]
OUT = os.path.join(REPO, "figures/fig_darcy_loss.pdf")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(5.125, 1.95))
    panels = [("curated", "(a)  curated  (legacy MAP archive)", MAD),
              ("oracle", "(b)  oracle  (KL truths)", EM)]
    for ax, (tag, title, col) in zip(axes, panels):
        ck = torch.load(ensure(f"data/checkpoints/darcy_flow_{tag}_seed0.pth"),
                        map_location="cpu", weights_only=False)
        h = ck["hist"]
        st, tr, vl = np.asarray(h["step"]), np.asarray(h["train"]), np.asarray(h["val"])
        mt = np.isfinite(tr) & (tr < 1e3)      # mask the post-best-val divergence spike
        mv = np.isfinite(vl) & (vl < 1e3)
        ax.plot(st[mt], tr[mt], color="0.6", lw=1.0, label="train")
        ax.plot(st[mv], vl[mv], color=col, lw=1.6, label="val")
        ax.set_xlabel("training step", fontsize=8)
        ax.set_ylabel("NLL", fontsize=8)
        ax.set_title(title, fontsize=8.6, color=col, loc="left", pad=3.0)
        ax.tick_params(labelsize=7, length=2.5)
        ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout(w_pad=1.8)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02, dpi=300)
    print("saved", os.path.relpath(OUT, REPO))


if __name__ == "__main__":
    main()
