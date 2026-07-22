#!/usr/bin/env python
"""The nonlinearity control: run the whole wave experiment at one value of ``eps``.

Panel (c) of fig_wave_collapse compares the nonlinear operator against its linear counterpart,
so both curves must come from a run in which nothing but the cubic coefficient changes. This
driver generates truths, data, the legacy MAP archive, the freeze archive and both priors in
one process from the configuration, so ``--eps 40`` and ``--eps 0`` differ in exactly that one
number. The shipped ``results/wave_collapse_eps40.npz`` and ``results/wave_collapse_eps0.npz``
are the records the paper used.

Reads: nothing on disk.
Writes: results/wave_collapse_{tag}.npz
Runtime: a few hours per run at the default scale on one GPU (the MAP archive dominates on
CPU); the scale flags below shorten it for a smoke test.

    python scripts/wave_eps_control.py --eps 40 --tag eps40
    python scripts/wave_eps_control.py --eps 0  --tag eps0
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from priorlaundermat.download import REPO
from priorlaundermat.wave.config import WaveLaunderingConfig
from priorlaundermat.wave.pipeline import run_pipeline


def main() -> None:
    cfg = WaveLaunderingConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, required=True,
                    help="cubic coefficient; 40 is the nonlinear operator, 0 the linear control")
    ap.add_argument("--tag", required=True, help="names the output results/wave_collapse_{tag}.npz")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=cfg.seed)
    # scale knobs, so a short run can exercise the whole path
    ap.add_argument("--n-truths", type=int, default=cfg.n_truths)
    ap.add_argument("--n-archive", type=int, default=cfg.n_archive)
    ap.add_argument("--n-oracle", type=int, default=cfg.n_oracle)
    ap.add_argument("--map-iters", type=int, default=cfg.map_iters)
    ap.add_argument("--train-steps", type=int, default=cfg.ddpm_train_steps)
    ap.add_argument("--n-sample", type=int, default=cfg.ddpm_sample)
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" \
        else args.device
    cfg = WaveLaunderingConfig(
        eps=args.eps, seed=args.seed, device=device,
        n_truths=args.n_truths, n_archive=args.n_archive, n_oracle=args.n_oracle,
        map_iters=args.map_iters, ddpm_train_steps=args.train_steps,
        ddpm_sample=args.n_sample)
    print(f"[eps-control] eps={cfg.eps} tag={args.tag} device={cfg.device}", flush=True)

    res = run_pipeline(cfg)
    rough = res["rough_mask"]
    print(f"[eps-control] ordering single-best < freeze < oracle on the rough shadow: "
          f"{res['ordering_ok']}", flush=True)

    out = os.path.join(REPO, "results", f"wave_collapse_{args.tag}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(
        out,
        sr_map=res["sr_map"], sr_freeze=res["sr_freeze"], sr_oracle=res["sr_oracle"],
        rough_eigvals_blind=res["rough_eigvals_blind"], rough_mask=rough,
        cov_curated_blind_central=res["cov_curated_blind"].central,
        cov_oracle_blind_central=res["cov_oracle_blind"].central,
        cov_curated_resolved_central=res["cov_curated_resolved"].central,
        cov_oracle_resolved_central=res["cov_oracle_resolved"].central,
        well_lit_mask=res["well_lit_mask"],
        eps=float(cfg.eps), alpha=res["alpha"], sigma=res["sigma"],
        map_chi2_per_dof=res["map_chi2_per_dof"], n_blind=res["split"].n_blind,
        ordering_ok=res["ordering_ok"], gate_passed=bool(res["gate"]["passed"]),
    )
    print(f"[out] wrote {os.path.relpath(out, REPO)}", flush=True)


if __name__ == "__main__":
    main()
