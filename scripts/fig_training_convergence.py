#!/usr/bin/env python
"""Validation loss for both priors on all three deployed operators.

Each prior is trained to a fitted target, so the blind-subspace shortfall the paper reports
belongs to the training target rather than to under-training. One panel per operator; the two
arms differ only in what they were trained on.

Reads the six checkpoints (each carries ``hist = {step, train, val}``) and nothing else.
Writes ``figures/fig_training_convergence.pdf``. CPU, a few seconds, no sampling.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from priorlaundermat.download import REPO, ensure  # noqa: E402
from priorlaundermat.style import LINEWIDTH_IN, PALETTE, apply_paper_style  # noqa: E402

apply_paper_style()
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
EM, MAD = PALETTE["em"], PALETTE["mad"]
OUT = os.path.join(REPO, "figures/fig_training_convergence.pdf")

# (panel title, checkpoint stem). Order follows the paper's presentation of the examples.
ROWS = [
    ("Semilinear wave", "wave_prior_{tag}_seed0"),
    ("Seismic Born", "seismic_prior_{tag}_seed0"),
    ("Groundwater flow", "darcy_flow_{tag}_seed0"),
]


def history(stem: str, tag: str) -> tuple[np.ndarray, np.ndarray] | None:
    """``(step, val)`` from a checkpoint's recorded history, or ``None`` if it carries none."""
    ck = torch.load(ensure(f"data/checkpoints/{stem.format(tag=tag)}.pth"),
                    map_location="cpu", weights_only=False)
    h = ck.get("hist")
    if not h:
        return None
    step = np.asarray(h["step"], dtype=float)
    val = np.asarray(h["val"], dtype=float)
    m = np.isfinite(val)
    return step[m], val[m]


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(LINEWIDTH_IN, 1.62))
    # The groundwater panel spans under a decade, so matplotlib labels a minor tick outside its
    # axis; the gutter is set wide enough for that label to sit clear of the panel to its left.
    fig.subplots_adjust(left=0.105, right=0.995, top=0.855, bottom=0.265, wspace=0.46)

    missing = []
    for ax, (title, stem) in zip(axes, ROWS):
        for tag, col in (("oracle", EM), ("curated", MAD)):
            hv = history(stem, tag)
            if hv is None:
                missing.append(f"{title}/{tag}")
                continue
            ax.semilogy(*hv, color=col, lw=1.5, label=tag)
        ax.set_title(title, fontsize=8.0, pad=3.0)
        ax.set_xlabel("training step", fontsize=7.6, labelpad=2.0)
        ax.tick_params(labelsize=6.8, length=2.4, pad=1.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("validation loss", fontsize=7.6, labelpad=2.0)
    axes[0].legend(fontsize=6.8, frameon=False, handlelength=1.3, labelspacing=0.25,
                   loc="upper right", borderaxespad=0.2)

    # A silently absent curve looks exactly like one that was never trained, and the figure's
    # claim is about all six priors -- so a partial render is refused rather than drawn.
    if missing:
        raise SystemExit("missing training histories: " + ", ".join(missing))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02, dpi=300)
    print("saved", os.path.relpath(OUT, REPO), flush=True)


if __name__ == "__main__":
    main()
