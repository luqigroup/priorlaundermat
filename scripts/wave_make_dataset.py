#!/usr/bin/env python
"""Build the wave train and eval archives.

Per row: the rough-field truth (the oracle prior's target), the short-window observed data,
and the legacy smoothness-MAP reconstruction (the curated prior's target). The regularizer
strength is fixed once by Morozov's discrepancy principle on a train calibration set, which
never sees a truth, and reused for every MAP solve in both splits.

Reads: nothing on disk.
Writes: data/wave/wave_dataset_train.h5, data/wave/wave_dataset_eval.h5
Runtime: under an hour, CPU (batched Adam through the leapfrog solver).

    python scripts/wave_make_dataset.py
"""

from __future__ import annotations

import json
import os
import time

import h5py
import numpy as np

from priorlaundermat.download import REPO
from priorlaundermat.wave.archive import build_map_archive, select_alpha
from priorlaundermat.wave.blind import build_wave_blind_subspace
from priorlaundermat.wave.config import WaveLaunderingConfig
from priorlaundermat.wave.priors import RoughFieldPrior, synthesize_truths_and_data
from priorlaundermat.wave.regularizers import build_smoothness_regularizer
from priorlaundermat.wave.solver import SemilinearWave1D

OUT_DIR = os.path.join(REPO, "data", "wave")
N_TRAIN, N_EVAL = 4000, 600
N_CALIBRATION = 256          # rows used to set the regularizer strength


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    cfg = WaveLaunderingConfig()
    solver = SemilinearWave1D(n=cfg.n, c=cfg.c, cfl=cfg.cfl, recv=cfg.recv, nt=cfg.nt,
                              eps=cfg.eps, domain_len=cfg.domain_len)
    split = build_wave_blind_subspace(solver, n_probe_states=cfg.blind_probe_states,
                                      tol=cfg.blind_tol, seed=cfg.seed,
                                      verify_eps=cfg.verify_eps)
    shadow = np.sort(split.shadow_idx.numpy()).astype(np.int32)
    resolved = np.sort(split.resolved_idx.numpy()).astype(np.int32)
    print(f"[wave-data] shadow(blind)={shadow.min()}..{shadow.max()} "
          f"({split.n_blind}/{cfg.n}) eps={cfg.eps} "
          f"fixed_across_states={split.identical_across_states}", flush=True)

    # Train and eval draw from disjoint seed offsets.
    prior = RoughFieldPrior(n=cfg.n, s_star=cfg.s_star, ell=cfg.ell)
    xt_tr, y_tr, sigma = synthesize_truths_and_data(
        solver, prior, n=N_TRAIN, rel_noise=cfg.rel_noise, seed=cfg.seed + 1)
    xt_ev, y_ev, _ = synthesize_truths_and_data(
        solver, prior, n=N_EVAL, rel_noise=cfg.rel_noise, seed=cfg.seed + 2)
    print(f"[wave-data] truths+data: train {tuple(xt_tr.shape)} eval {tuple(xt_ev.shape)} | "
          f"sigma={sigma:.5g} | data dim={y_tr.shape[1]}", flush=True)

    alpha, _ = select_alpha(solver, y_tr[:N_CALIBRATION], sigma, rho=cfg.reg_ridge,
                            map_iters=cfg.map_iters)
    reg = build_smoothness_regularizer(n=cfg.n, alpha=alpha, rho=cfg.reg_ridge)
    print(f"[wave-data] smoothness Tikhonov, alpha={alpha:.5g} (Morozov discrepancy) "
          f"rho={cfg.reg_ridge}", flush=True)

    print(f"[wave-data] solving the legacy MAP ({cfg.map_iters} iterations) ...", flush=True)
    xm_tr = build_map_archive(solver, y_tr, reg, sigma, map_iters=cfg.map_iters, lr=cfg.map_lr)
    xm_ev = build_map_archive(solver, y_ev, reg, sigma, map_iters=cfg.map_iters, lr=cfg.map_lr)
    print(f"[wave-data] MAP chi2/dof median: train {float(xm_tr.chi2_per_dof.median()):.3f} "
          f"eval {float(xm_ev.chi2_per_dof.median()):.3f}  ({time.time() - t0:.0f}s)", flush=True)

    meta = {"n": cfg.n, "c": cfg.c, "cfl": cfg.cfl, "recv": list(cfg.recv), "nt": cfg.nt,
            "eps": cfg.eps, "domain_len": cfg.domain_len, "s_star": cfg.s_star,
            "ell": cfg.ell, "rel_noise": cfg.rel_noise, "reg_ridge": cfg.reg_ridge,
            "map_iters": cfg.map_iters, "map_lr": cfg.map_lr, "seed": cfg.seed,
            "regularizer": "smoothness_tikhonov_discrepancy"}

    def write(name, xt, y, mapres):
        path = os.path.join(OUT_DIR, f"wave_dataset_{name}.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("x_true", data=xt.numpy().astype(np.float32))
            f.create_dataset("y", data=y.numpy().astype(np.float32))
            f.create_dataset("x_map", data=mapres.x.numpy().astype(np.float32))
            f.create_dataset("chi2_per_dof",
                             data=mapres.chi2_per_dof.numpy().astype(np.float32))
            f.create_dataset("done", data=np.ones(xt.shape[0], np.int8))
            f.create_dataset("shadow_idx", data=shadow)
            f.create_dataset("resolved_idx", data=resolved)
            f.attrs["sigma"] = float(sigma)
            f.attrs["alpha"] = float(alpha)
            f.attrs["rho"] = float(cfg.reg_ridge)
            f.attrs["config"] = json.dumps(meta)
        print(f"[wave-data] wrote {os.path.relpath(path, REPO)} ({xt.shape[0]} rows)",
              flush=True)

    write("train", xt_tr, y_tr, xm_tr)
    write("eval", xt_ev, y_ev, xm_ev)
    print(f"[wave-data] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
