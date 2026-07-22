#!/usr/bin/env python
"""Record blind-subspace spread and coverage for the deployed wave priors.

Three arms, scored on the held-out eval truths in the fixed shadow roughness basis:

  * oracle and curated: unconditional DDIM draws from the trained UNet checkpoints,
    denormalized with each checkpoint's own per-node statistics;
  * freeze: the exact blind-block Gaussian archive around the eval records' legacy MAPs, an
    archive-level curve by construction, with no prior trained on it.

Per shadow roughness mode it writes the spread ratio (sample standard deviation over the
truths') and the central-90% coverage, plus the resolved-subspace fairness read and the
oracle-control gate. Both priors live in the same physical units as the truths, so no
resolved-scale rescaling is applied.

Reads: data/wave/wave_dataset_eval.h5, data/checkpoints/wave_prior_{oracle,curated}_seed0.pth
Writes: results/wave_collapse_deployed.npz
Runtime: about a minute on GPU, a few minutes on CPU.

    python scripts/wave_collapse_record.py
"""

from __future__ import annotations

import json
import os

import h5py
import numpy as np
import torch

from priorlaundermat.download import REPO, ensure
from priorlaundermat.wave.archive import build_freeze_archive
from priorlaundermat.wave.blind import build_wave_blind_subspace
from priorlaundermat.wave.coverage import arm_coverage, oracle_control_checks
from priorlaundermat.wave.nets import load_prior, sample_prior_ddim
from priorlaundermat.wave.regularizers import build_smoothness_regularizer
from priorlaundermat.wave.solver import SemilinearWave1D

OUT = os.path.join(REPO, "results", "wave_collapse_deployed.npz")
N_SAMPLES = 384          # DDIM draws per prior
DDIM_STEPS = 50
FREEZE_PER = 8           # freeze draws per eval record
LEVEL = 0.90
ROUGH_FRAC = 0.5
GATE_TOL = 0.1
SEED = 0


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    eval_h5 = ensure("data/wave/wave_dataset_eval.h5")
    with h5py.File(eval_h5, "r") as f:
        x_true = torch.from_numpy(f["x_true"][:]).to(dtype)
        x_map = torch.from_numpy(f["x_map"][:]).to(dtype)
        shadow_h5 = np.sort(f["shadow_idx"][:])
        alpha, rho = float(f.attrs["alpha"]), float(f.attrs["rho"])
        sigma = float(f.attrs["sigma"])
        cfg = json.loads(f.attrs["config"])
    n_truths = x_true.shape[0]

    solver = SemilinearWave1D(n=cfg["n"], c=cfg["c"], cfl=cfg["cfl"], recv=tuple(cfg["recv"]),
                              nt=cfg["nt"], eps=cfg["eps"], domain_len=cfg["domain_len"],
                              dtype=dtype)
    split = build_wave_blind_subspace(solver, n_probe_states=8, tol=0.0,
                                      seed=cfg["seed"], verify_eps=0.0)
    assert np.array_equal(np.sort(split.shadow_idx.numpy()), shadow_h5), \
        "blind subspace does not match the dataset's shadow_idx"
    print(f"[blind] n_blind={split.n_blind}/{cfg['n']} "
          f"shadow=[{int(split.shadow_idx.min())}..{int(split.shadow_idx.max())}] "
          f"fixed_across_states/eps={split.identical_across_states}", flush=True)

    samples = {}
    for arm in ("oracle", "curated"):
        ck = ensure(f"data/checkpoints/wave_prior_{arm}_seed0.pth")
        unet, norm = load_prior(ck, device=device)
        s = sample_prior_ddim(unet, norm, N_SAMPLES, n_steps=DDIM_STEPS,
                              device=device, seed=SEED).cpu().numpy()
        samples[arm] = s
        print(f"[ddim] {arm}: {s.shape[0]} draws, {DDIM_STEPS} steps, "
              f"range [{s.min():.2f},{s.max():.2f}], per-draw std median "
              f"{np.median(s.std(1)):.2f}", flush=True)

    reg = build_smoothness_regularizer(n=cfg["n"], alpha=alpha, rho=rho, dtype=dtype)
    x_freeze = build_freeze_archive(x_map, split, reg, n_per=FREEZE_PER, seed=cfg["seed"] + 4)
    print(f"[freeze] archive {tuple(x_freeze.shape)} around the {n_truths} eval MAPs "
          f"(alpha={alpha:.4g} rho={rho})", flush=True)

    b_true = split.project_blind(x_true)
    s_star = b_true.numpy().std(axis=0)
    x_ora = torch.from_numpy(samples["oracle"]).to(dtype)
    x_cur = torch.from_numpy(samples["curated"]).to(dtype)

    def spread_ratio(X):
        return split.project_blind(X).numpy().std(axis=0) / np.clip(s_star, 1e-12, None)

    sr_oracle, sr_map, sr_freeze = spread_ratio(x_ora), spread_ratio(x_cur), spread_ratio(x_freeze)

    cov_cur_blind = arm_coverage(split.project_blind(x_cur), b_true, level=LEVEL)
    cov_ora_blind = arm_coverage(split.project_blind(x_ora), b_true, level=LEVEL)
    r_true = split.project_resolved(x_true)
    cov_cur_res = arm_coverage(split.project_resolved(x_cur), r_true, level=LEVEL)
    cov_ora_res = arm_coverage(split.project_resolved(x_ora), r_true, level=LEVEL)
    rough = split.rough_mask_blind(ROUGH_FRAC).numpy()
    well_lit = split.well_illuminated_mask(ROUGH_FRAC).numpy()
    gate = oracle_control_checks(cov_ora_blind, cov_cur_blind, cov_ora_res, cov_cur_res,
                                 rough_mask=rough, resolved_mask=well_lit,
                                 level=LEVEL, tol=GATE_TOL)

    def m(a, mask=rough):
        return float(np.asarray(a)[mask].mean())

    print(f"[spread s/s* rough half] curated={m(sr_map):.3f}  freeze={m(sr_freeze):.3f}  "
          f"oracle={m(sr_oracle):.3f}", flush=True)
    print(f"[spread s/s* smooth half] curated={m(sr_map, ~rough):.3f}  "
          f"freeze={m(sr_freeze, ~rough):.3f}  oracle={m(sr_oracle, ~rough):.3f}", flush=True)
    print(f"[coverage central-90 rough blind] curated={m(cov_cur_blind.central):.3f}  "
          f"oracle={m(cov_ora_blind.central):.3f}", flush=True)
    print(f"[coverage central-90 well-lit resolved] "
          f"curated={m(cov_cur_res.central, well_lit):.3f}  "
          f"oracle={m(cov_ora_res.central, well_lit):.3f}", flush=True)
    print(f"[gate] passed={gate['passed']} oracle_nominal_on_rough_blind="
          f"{gate['oracle_nominal_on_rough_blind']} "
          f"resolved_fair={gate['resolved_subspace_fair']}", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(
        OUT,
        sr_map=sr_map, sr_freeze=sr_freeze, sr_oracle=sr_oracle,
        rough_eigvals_blind=split.rough_eigvals_blind.numpy(),
        cov_curated_blind_central=cov_cur_blind.central,
        cov_oracle_blind_central=cov_ora_blind.central,
        cov_curated_resolved_central=cov_cur_res.central,
        cov_oracle_resolved_central=cov_ora_res.central,
        rough_mask=rough, well_lit_mask=well_lit,
        gate_passed=bool(gate["passed"]),
        alpha=alpha, sigma=sigma, n_blind=split.n_blind,
        n_samples=N_SAMPLES, n_truths=n_truths, ddim_steps=DDIM_STEPS,
        freeze_per=FREEZE_PER, freeze_size=int(x_freeze.shape[0]),
    )
    print(f"[out] wrote {os.path.relpath(OUT, REPO)}", flush=True)


if __name__ == "__main__":
    main()
