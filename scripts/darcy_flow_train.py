#!/usr/bin/env python
"""Train the two learned groundwater priors over Darcy KL coefficients: oracle vs curated.

The ORACLE flow is fit to the true-field coefficients ``xi_true`` -- the truth's own law -- and the
CURATED flow to the MAP-archive coefficients ``xi_map``, the legacy reconstructions.
Both are HINT normalizing flows over the d=100 whitened KL coefficients, trained by maximum
likelihood with save-best-validation and early stopping.

Reads ``data/darcy/darcy_laundering.h5``; writes
``data/checkpoints/darcy_flow_{oracle,curated}_seed{seed}.pth``, each holding the best-validation
weights, the loss history, and the normalizer statistics. CPU only; both arms take a few minutes.
"""
from __future__ import annotations

import argparse
import os
import time

import h5py
import numpy as np
import torch

from resolvability.download import REPO, ensure
from resolvability.groundwater import HINTFlow
from resolvability.utils import Normalizer, PolynomialDecayLR

DATA = "data/darcy/darcy_laundering.h5"


def train_one(X_tr, X_va, args, device, tag, ck_path):
    torch.manual_seed(args.seed)
    flow = HINTFlow(X_tr.shape[1], n_cond=0, n_hidden=args.n_hidden,
                    n_flow_layers=args.n_layers, depth=args.depth, n_mlp_layers=3).to(device)
    opt = torch.optim.Adam(flow.parameters(), lr=args.lr)
    sched = PolynomialDecayLR(opt, args.lr, args.lr_final, args.n_steps)
    normalizer = Normalizer(torch.tensor(X_tr, dtype=torch.float32))
    Xtr = normalizer.normalize(torch.tensor(X_tr, dtype=torch.float32)).to(device)
    Xva = normalizer.normalize(torch.tensor(X_va, dtype=torch.float32)).to(device)
    meta = dict(n_in=X_tr.shape[1], n_hidden=args.n_hidden, n_layers=args.n_layers,
                depth=args.depth, tag=tag, norm_mean=normalizer.mean, norm_std=normalizer.std)

    best_val, best_state, hist = np.inf, None, {"step": [], "train": [], "val": []}
    since_improved = 0
    t0 = time.time()
    for step in range(args.n_steps):
        idx = torch.randint(0, Xtr.shape[0], (args.batch_size,), device=device)
        batch = Xtr[idx]
        if args.deq_noise > 0:
            # Dequantization noise. The curated coefficients concentrate on a low-dimensional
            # manifold; without it the flow over-concentrates and its inverse explodes.
            batch = batch + args.deq_noise * torch.randn_like(batch)
        loss = -flow.log_prob(batch).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), args.grad_clip); opt.step(); sched.step()
        if step % 100 == 0 or step == args.n_steps - 1:
            flow.eval()
            with torch.no_grad():
                vl = -flow.log_prob(Xva).mean().item()
            flow.train()
            hist["step"].append(step)
            hist["train"].append(float(loss.item())); hist["val"].append(vl)
            if vl < best_val:
                best_val = vl; since_improved = 0
                best_state = {k: v.detach().cpu().clone() for k, v in flow.state_dict().items()}
            else:
                since_improved += 1
            torch.save({"model": best_state, "hist": hist, "best_val": best_val,
                        "step": step, **meta}, ck_path)
            if step % 500 == 0:
                print(f"  [{tag}] step {step:5d}  train {loss.item():.3f}  val {vl:.3f} "
                      f"(best {best_val:.3f}, {time.time()-t0:.0f}s)", flush=True)
            if not np.isfinite(vl) or vl > 1e3:
                print(f"  [{tag}] val diverged ({vl:.3g}) at step {step}; stopping, "
                      f"best {best_val:.3f} kept", flush=True)
                break
            if args.patience > 0 and since_improved >= args.patience:
                print(f"  [{tag}] no val improvement for {args.patience} evals; stopping at step "
                      f"{step} (best {best_val:.3f})", flush=True)
                break
    return best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_hidden", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=4)      # HINT layers
    ap.add_argument("--depth", type=int, default=2)         # tree depth per HINT layer
    ap.add_argument("--n_steps", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_final", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=10, help="val evals without improvement")
    ap.add_argument("--grad_clip", type=float, default=10.0)
    ap.add_argument("--deq_noise", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="both", choices=["oracle", "curated", "both"])
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    args = ap.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = torch.device("cpu")

    with h5py.File(ensure(DATA), "r") as f:
        split = f["split"][:]
        xi_true, xi_map = f["xi_true"][:], f["xi_map"][:]
    tr, va = split == 0, split == 1
    print(f"[data] train {tr.sum()} eval {va.sum()}  d={xi_true.shape[1]}  "
          f"xi_true std={xi_true[tr].std():.3f}  xi_map std={xi_map[tr].std():.3f}", flush=True)

    ck_dir = os.path.join(REPO, "data/checkpoints"); os.makedirs(ck_dir, exist_ok=True)
    arms = [("oracle", xi_true), ("curated", xi_map)]
    if args.tag != "both":
        arms = [a for a in arms if a[0] == args.tag]
    for tag, X in arms:
        ck_path = os.path.join(ck_dir, f"darcy_flow_{tag}_seed{args.seed}.pth")
        bv = train_one(X[tr], X[va], args, device, tag, ck_path)
        print(f"[{tag}] done. best val NLL {bv:.3f}  -> {os.path.relpath(ck_path, REPO)}",
              flush=True)


if __name__ == "__main__":
    main()
