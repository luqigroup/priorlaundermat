#!/usr/bin/env python
"""Each deployed seismic prior's training data beside its own fresh unconditional samples.

  Row (a)  curated prior, trained on the legacy least-squares migration archive
  Row (b)  oracle prior, trained on the broadband reflectivity

The claim is only this: the samples are of a kind with the training data. So the deployed blind
reports carry the archive's belief about the unseen directions, not some artefact of a prior that
failed to fit. This is also the paper's only view of the legacy archive itself.

Reads results/seismic_training_gallery.npz. With ``--sample`` it instead redraws from
data/checkpoints/seismic_prior_{curated,oracle}_seed0.pth against training images from
data/seismic/dataset_train.h5 and rewrites the cache -- that path needs a GPU and pulls the 13 GB
training archive. Writes figures/fig_seismic_training.pdf. Seconds from cache, CPU.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import gridspec  # noqa: E402

from priorlaundermat.download import REPO, ensure  # noqa: E402
from priorlaundermat.style import LINEWIDTH_IN, PALETTE, apply_paper_style, extent_km  # noqa: E402

apply_paper_style()
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
EM, MAD = PALETTE["em"], PALETTE["mad"]

N, MUTE_END, TAPER, NGAL = 256, 16, 6, 4
NSTEPS = 250                       # reverse steps for the redrawn samples
CACHE = "results/seismic_training_gallery.npz"
TRAIN_H5 = "data/seismic/dataset_train.h5"
OUT = os.path.join(REPO, "figures/fig_seismic_training.pdf")
TAGS = ("curated", "oracle")


def load_real(data_key: str, n: int) -> np.ndarray:
    """The first ``n`` completed training images of ``data_key``, water-muted."""
    import h5py

    from priorlaundermat.seismic.pseudosynth import water_mute

    with h5py.File(ensure(TRAIN_H5), "r") as f:
        flag = "recon_done" if (data_key == "lsrtm5" and "recon_done" in f) else "done"
        sel = np.sort(np.where(f[flag][:] == 1)[0][:n])
        X = f[data_key][sel].astype(np.float32)
    return np.stack([water_mute(x, MUTE_END, TAPER) for x in X])


def draw(tag: str, n: int = NGAL) -> tuple[np.ndarray, np.ndarray]:
    """Training images and fresh unconditional samples for one prior, both in physical units."""
    import torch
    from diffusers import DDPMScheduler

    from priorlaundermat.seismic.priors import find_ckpt, make_unet
    from priorlaundermat.utils import Normalizer

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path, base = find_ckpt(tag)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    unet = make_unet(base).to(dev)
    unet.load_state_dict(ck["model"])
    unet.eval()
    norm = Normalizer.from_stats(ck["norm_mean"], ck["norm_std"])
    sched = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2",
                          clip_sample=False)
    sched.set_timesteps(NSTEPS, device=dev)
    torch.manual_seed(0)
    x = torch.randn(n, 1, N, N, device=dev)
    with torch.no_grad():
        for t in sched.timesteps:
            x = sched.step(unet(x, t.expand(n)).sample, t, x).prev_sample
    samp = norm.unnormalize(x.squeeze(1).cpu()).numpy()
    return load_real(ck["data_key"], n), samp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", action="store_true",
                    help="redraw the samples from the checkpoints and rewrite the cache (GPU)")
    args = ap.parse_args()

    if args.sample:
        D = {}
        for tag in TAGS:
            D[f"{tag}_real"], D[f"{tag}_samp"] = draw(tag)
        out = os.path.join(REPO, CACHE)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez(out, **D)
        print("saved", CACHE, flush=True)
    else:
        c = np.load(ensure(CACHE))
        D = {f"{tag}_{k}": c[f"{tag}_{k}"] for tag in TAGS for k in ("real", "samp")}

    # ONE shared symmetric grayscale window across both priors and their samples, at 2.3 times the
    # median pixel standard deviation -- the paper's seismic convention, so every seismic panel
    # reads on the same scale.
    all_real = np.concatenate([D[f"{t}_real"] for t in TAGS], axis=0)
    WIN = 2.3 * float(np.median([np.std(x) for x in all_real]))

    fig = plt.figure(figsize=(LINEWIDTH_IN, 3.62))
    outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.23,
                              left=0.085, right=0.995, top=0.925, bottom=0.012)
    rows = [("curated", "curated prior  (legacy LSRTM archive)", MAD),
            ("oracle", "oracle prior  (broadband reflectivity)", EM)]
    for r, (tag, title, col) in enumerate(rows):
        real, samp = D[f"{tag}_real"], D[f"{tag}_samp"]
        inner = outer[r].subgridspec(2, NGAL, hspace=0.05, wspace=0.04)
        for rr, (imgs, lab) in enumerate([(real, "real"), (samp, "sample")]):
            for j in range(NGAL):
                ax = fig.add_subplot(inner[rr, j])
                # Reflectivity is a zero-mean contrast image -- its DC is physically meaningless and
                # sits in the operator's null space -- so each panel is mean-centred. Otherwise
                # a sampler's harmless DC drift renders some samples uniformly dark.
                im = imgs[j] - float(imgs[j].mean())
                # A 5.12 x 3.2 km section at true proportions. These panels carry the "alike in
                # kind" judgement, so they must not be stretched. Ticks stay off: sixteen panels.
                ax.imshow(im.T, cmap="gray", vmin=-WIN, vmax=WIN, aspect="equal",
                          extent=extent_km(N), interpolation="bicubic")
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_linewidth(0.5)
                if j == 0:
                    ax.set_ylabel(lab, fontsize=7.4, labelpad=2.0)
                if rr == 0 and j == 0:
                    ax.set_title(f"({'ab'[r]})  {title}", fontsize=8.2, color=col, loc="left",
                                 pad=3.0)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02, dpi=300)
    print("saved", os.path.relpath(OUT, REPO), flush=True)


if __name__ == "__main__":
    main()
