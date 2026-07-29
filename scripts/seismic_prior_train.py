#!/usr/bin/env python
"""Train one seismic generative prior: an unconditional DDPM over 256 x 256 reflectivity.

The arm is chosen by which images the prior sees, and nothing else about the two runs differs:

    --arm oracle   trains on the broadband reflectivity  (data key ``broadband_dm``)
    --arm curated  trains on the legacy migration archive (data key ``lsrtm5``)

Images are per-pixel standardized before training, and the statistics are stored in the checkpoint
so samples come back in physical units. The scheduler runs with ``clip_sample=False``: the default
clamps the predicted image to [-1, 1] every reverse step, which is right for [-1, 1] pictures and
wrong for standardized data, where it would flatten every reflector.

Reads data/seismic/dataset_train.h5. Writes
data/checkpoints/seismic_prior_<arm>_seed<seed>.pth with the weights, the normalization statistics
and the loss history. One GPU, about four hours per seed.

    python scripts/seismic_prior_train.py --arm oracle --seed 0
"""
from __future__ import annotations

import argparse
import os
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler

from resolvability.download import REPO, ensure
from resolvability.seismic import MUTE_END, TAPER
from resolvability.seismic.priors import BASE, CKPT, make_unet
from resolvability.seismic.pseudosynth import water_mute
from resolvability.utils import Normalizer

TRAIN_H5 = "data/seismic/dataset_train.h5"
DATA_KEY = {"oracle": "broadband_dm", "curated": "lsrtm5"}


def load(key: str, a: int, b: int, device) -> torch.Tensor:
    """Rows ``[a, b)`` of the completed training images for ``key``, water-muted."""
    with h5py.File(ensure(TRAIN_H5), "r") as f:
        # lsrtm5 needs recon_done, since the imaging pass may be partial; the truths need only done.
        flag = "recon_done" if (key == "lsrtm5" and "recon_done" in f) else "done"
        idx = np.where(f[flag][:] == 1)[0] if flag in f else np.arange(f[key].shape[0])
        sel = np.sort(idx[a:b])
        X = f[key][sel].astype(np.float32)
    return torch.from_numpy(np.stack([water_mute(x, MUTE_END, TAPER) for x in X])).to(device)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["oracle", "curated"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=3800)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--base", type=int, default=BASE)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=3000)
    ap.add_argument("--num-train-timesteps", type=int, default=1000)
    ap.add_argument("--beta-schedule", default="squaredcos_cap_v2")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    key = DATA_KEY[args.arm]

    Xtr = load(key, 0, args.n_train, "cpu")
    Xva = load(key, args.n_train, args.n_train + args.n_val, "cpu")
    normalizer = Normalizer(Xtr)                                  # per-pixel, fit on train only
    Xtr = normalizer.normalize(Xtr).unsqueeze(1).to(device)
    Xva = normalizer.normalize(Xva).unsqueeze(1).to(device)
    print(f"[{args.arm} seed {args.seed}] train {tuple(Xtr.shape)} | per-pixel std in "
          f"[{float(normalizer.std.min()):.1f}, {float(normalizer.std.max()):.1f}] | normalized "
          f"range [{float(Xtr.min()):.1f}, {float(Xtr.max()):.1f}]", flush=True)

    unet = make_unet(args.base).to(device)
    sched = DDPMScheduler(num_train_timesteps=args.num_train_timesteps,
                          beta_schedule=args.beta_schedule, clip_sample=False)
    opt = torch.optim.AdamW(unet.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.n_steps,
                                              pct_start=0.05)
    nT = sched.config.num_train_timesteps
    hist = {"step": [], "train": [], "val": []}
    t0 = time.time()

    @torch.no_grad()
    def val_loss():
        """Denoising MSE on the held-out images, at a fixed noise and timestep draw."""
        tot, cnt = 0.0, 0
        g = torch.Generator(device=device).manual_seed(0)
        for s in range(0, Xva.shape[0], 8):
            x0 = Xva[s:s + 16]
            noise = torch.randn(x0.shape, generator=g, device=device)
            t = torch.randint(0, nT, (x0.shape[0],), generator=g, device=device)
            tot += F.mse_loss(unet(sched.add_noise(x0, noise, t), t).sample, noise,
                              reduction="sum").item()
            cnt += noise.numel()
        return tot / max(cnt, 1)

    unet.train()
    for step in range(args.n_steps):
        idx = torch.randint(0, Xtr.shape[0], (args.batch_size,), device=device)
        x0 = Xtr[idx]
        noise = torch.randn_like(x0)
        t = torch.randint(0, nT, (args.batch_size,), device=device)
        loss = F.mse_loss(unet(sched.add_noise(x0, noise, t), t).sample, noise)   # epsilon target
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        opt.step()
        sch.step()
        if step % 200 == 0 or step == args.n_steps - 1:
            unet.eval()
            vl = val_loss()
            unet.train()
            hist["step"].append(step)
            hist["train"].append(float(loss.item()))
            hist["val"].append(vl)
            print(f"  step {step:5d}/{args.n_steps}  train {loss.item():.4f}  val {vl:.4f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    out = os.path.join(REPO, CKPT.format(tag=args.arm, seed=args.seed))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"model": unet.state_dict(), "norm_mean": normalizer.mean,
                "norm_std": normalizer.std, "hist": hist, "data_key": key}, out)
    print(f"saved {os.path.relpath(out, REPO)}", flush=True)


if __name__ == "__main__":
    main()
