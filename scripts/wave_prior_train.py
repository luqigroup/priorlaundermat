#!/usr/bin/env python
"""Train one wave generative prior -- an unconditional UNet DDPM over the 64-node field.

The two arms differ only in their training target: ``--data-key x_true`` gives the oracle
prior, ``--data-key x_map`` the curated prior trained on the legacy MAP archive. Recipe:
per-node standardization (no clamping, not [-1, 1]), a cosine beta schedule with
``clip_sample=False``, epsilon prediction, AdamW with a one-cycle schedule, gradient clipping,
and no EMA. The weights shipped are the validation minimum, not the final step.

Reads: data/wave/wave_dataset_train.h5
Writes: data/checkpoints/wave_prior_{oracle,curated}_seed{seed}.pth
Runtime: a few minutes on one GPU, longer on CPU.

    python scripts/wave_prior_train.py --data-key x_true --seed 0
    python scripts/wave_prior_train.py --data-key x_map  --seed 0
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

from priorlaundermat.download import REPO, ensure
from priorlaundermat.utils import Normalizer
from priorlaundermat.wave.nets import NNODE, SIDE, make_unet

ARM = {"x_true": "oracle", "x_map": "curated"}


def load(path: str, key: str, a: int, b: int) -> torch.Tensor:
    with h5py.File(path, "r") as f:
        idx = np.where(f["done"][:] == 1)[0]
        X = f[key][np.sort(idx[a:b])].astype(np.float32)
    return torch.from_numpy(X)                                     # (n, 64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-key", choices=["x_true", "x_map"], default="x_true")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--n-steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-train-timesteps", type=int, default=1000)
    ap.add_argument("--beta-schedule", default="squaredcos_cap_v2")
    ap.add_argument("--n-train", type=int, default=3800)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--n-sample", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    arm = ARM[args.data_key]
    tag = f"wave_prior_{arm}_seed{args.seed}"
    train_h5 = ensure("data/wave/wave_dataset_train.h5")
    print(f"[{tag}] device={dev} data_key={args.data_key} n_steps={args.n_steps} "
          f"batch={args.batch_size}", flush=True)

    torch.manual_seed(args.seed)
    Xtr = load(train_h5, args.data_key, 0, args.n_train)
    Xva = load(train_h5, args.data_key, args.n_train, args.n_train + args.n_val)
    normalizer = Normalizer(Xtr)                                   # per-node, fit on train
    x_range = (float(Xtr.min()), float(Xtr.max()))
    Xtr = normalizer.normalize(Xtr).reshape(-1, 1, SIDE, SIDE).to(dev)
    Xva = normalizer.normalize(Xva).reshape(-1, 1, SIDE, SIDE).to(dev)
    print(f"[{tag}] train {tuple(Xtr.shape)} | per-node std in "
          f"[{float(normalizer.std.min()):.2f},{float(normalizer.std.max()):.2f}] | "
          f"physical range {x_range}", flush=True)

    unet = make_unet().to(dev)
    sched = DDPMScheduler(num_train_timesteps=args.num_train_timesteps,
                          beta_schedule=args.beta_schedule, clip_sample=False)
    opt = torch.optim.AdamW(unet.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                              total_steps=args.n_steps + 1, pct_start=0.05)
    nT = sched.config.num_train_timesteps

    @torch.no_grad()
    def val_loss() -> float:
        g = torch.Generator(device=dev).manual_seed(0)             # same noise every eval
        tot, nb = 0.0, 0
        for i in range(0, Xva.shape[0], args.batch_size):
            x0 = Xva[i:i + args.batch_size]
            noise = torch.randn(x0.shape, generator=g, device=dev)
            t = torch.randint(0, nT, (x0.shape[0],), generator=g, device=dev)
            tot += F.mse_loss(unet(sched.add_noise(x0, noise, t), t).sample, noise,
                              reduction="sum").item()
            nb += x0.numel()
        return tot / nb

    hist = {"step": [], "train": [], "val": []}
    best_val, best_state, best_step = float("inf"), None, 0
    unet.train()
    t0 = time.time()
    for step in range(args.n_steps + 1):
        idx = torch.randint(0, Xtr.shape[0], (args.batch_size,), device=dev)
        x0 = Xtr[idx]
        noise = torch.randn_like(x0)
        t = torch.randint(0, nT, (args.batch_size,), device=dev)
        loss = F.mse_loss(unet(sched.add_noise(x0, noise, t), t).sample, noise)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        opt.step()
        sch.step()
        if step % max(1, args.n_steps // 20) == 0:
            unet.eval()
            vl = val_loss()
            unet.train()
            hist["step"].append(step)
            hist["train"].append(float(loss.item()))
            hist["val"].append(vl)
            if vl < best_val:                                      # keep the val-minimum weights
                best_val, best_step = vl, step
                best_state = {k: v.detach().cpu().clone() for k, v in unet.state_dict().items()}
            print(f"  step {step:6d}/{args.n_steps}  train {loss.item():.4f}  val {vl:.4f}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
    if best_state is not None:
        unet.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
    print(f"[{tag}] best val {best_val:.4f} at step {best_step}", flush=True)

    ckdir = os.path.join(REPO, "data", "checkpoints")
    os.makedirs(ckdir, exist_ok=True)
    ckpt = os.path.join(ckdir, f"{tag}.pth")
    torch.save({"model": unet.state_dict(), "norm_mean": normalizer.mean,
                "norm_std": normalizer.std, "hist": hist, "data_key": args.data_key,
                "phys_range": x_range, "best_val": best_val, "best_step": best_step}, ckpt)
    print(f"[{tag}] saved {os.path.relpath(ckpt, REPO)}", flush=True)

    # Amplitude check: with clip_sample=False and a per-node normalizer, sampled fields must
    # span the training range rather than being squeezed towards [-1, 1].
    unet.eval()
    sched.set_timesteps(250)
    with torch.no_grad():
        x = torch.randn(args.n_sample, 1, SIDE, SIDE, device=dev)
        for t in sched.timesteps:
            x = sched.step(unet(x, t.expand(args.n_sample)).sample, t, x).prev_sample
    samp = normalizer.unnormalize(x.reshape(args.n_sample, NNODE).cpu()).numpy()
    real = normalizer.unnormalize(Xtr[:args.n_sample].reshape(args.n_sample, NNODE).cpu()).numpy()
    print(f"[{tag}] amplitude check: train range [{real.min():.2f},{real.max():.2f}]  "
          f"sample range [{samp.min():.2f},{samp.max():.2f}]  "
          f"(sample/train std ratio {samp.std() / real.std():.2f})", flush=True)


if __name__ == "__main__":
    main()
