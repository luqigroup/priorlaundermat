"""Self-contained wave run: truths -> archive -> priors -> coverage.

Every quantity is generated inside one process from the configuration, so a single knob --
the cubic coefficient ``eps`` -- can be varied with nothing else changing. Stages:

  1. build the wave operator and read the fixed causal-shadow blind subspace from its
     exact-zero Jacobian columns, verifying it does not rotate across states or with ``eps``;
  2. synthesize rough-field truths and short-window data with known noise;
  3. select the smoothness strength by the truth-free discrepancy principle;
  4. build the legacy MAP archive and the exact blind-block freeze archive;
  5. read the raw blind-mode spreads of the MAP, freeze and truth archives -- the mechanism,
     exact and independent of any trained prior;
  6. train the curated prior on the MAP archive and the oracle prior on the truths, sample
     both, and measure sampler-free coverage on the shadow roughness modes (the headline) and
     on the resolved modes (the fairness control);
  7. run the oracle-control gate.
"""

from __future__ import annotations

import numpy as np
import torch

from priorlaundermat.wave.archive import (
    build_freeze_archive,
    build_map_archive,
    select_alpha,
)
from priorlaundermat.wave.blind import build_wave_blind_subspace
from priorlaundermat.wave.config import WaveLaunderingConfig
from priorlaundermat.wave.coverage import arm_coverage, oracle_control_checks
from priorlaundermat.wave.ddpm import VectorDDPM, train_vector_ddpm
from priorlaundermat.wave.priors import RoughFieldPrior, synthesize_truths_and_data
from priorlaundermat.wave.regularizers import build_smoothness_regularizer
from priorlaundermat.wave.solver import SemilinearWave1D


def _raw_spread_ratio(coords_model: torch.Tensor, coords_true: torch.Tensor) -> np.ndarray:
    """Per-direction standard-deviation ratio, exact and prior-independent."""
    m = coords_model.detach().cpu().numpy().std(axis=0)
    t = coords_true.detach().cpu().numpy().std(axis=0)
    return m / np.clip(t, 1e-12, None)


def run_pipeline(cfg: WaveLaunderingConfig, *, verbose: bool = True) -> dict:
    """Run the whole experiment for one configuration and return the results."""
    torch.manual_seed(cfg.seed)
    dtype = torch.float64
    device = torch.device(cfg.device)

    def log(*a):
        if verbose:
            print(*a, flush=True)

    # --- 1. operator and fixed blind subspace ---------------------------------------- #
    solver = SemilinearWave1D(
        n=cfg.n, c=cfg.c, cfl=cfg.cfl, recv=cfg.recv, nt=cfg.nt, eps=cfg.eps,
        domain_len=cfg.domain_len, dtype=dtype, device=device)
    split = build_wave_blind_subspace(
        solver, n_probe_states=cfg.blind_probe_states, tol=cfg.blind_tol,
        seed=cfg.seed, verify_eps=cfg.verify_eps)
    log(f"[blind] n_blind={split.n_blind}/{cfg.n} shadow=[{int(split.shadow_idx.min())}.."
        f"{int(split.shadow_idx.max())}]  max_shadow_sens={split.max_shadow_sensitivity:.1e}"
        f"  min_resolved_sens={split.min_resolved_sensitivity:.1e}"
        f"  fixed_across_states/eps={split.identical_across_states}")

    # --- 2. truths and data ---------------------------------------------------------- #
    prior = RoughFieldPrior(n=cfg.n, s_star=cfg.s_star, ell=cfg.ell,
                            dtype=dtype, device=device)
    x_true, _, _ = synthesize_truths_and_data(
        solver, prior, n=cfg.n_truths, rel_noise=cfg.rel_noise, seed=cfg.seed + 1)
    _, y_arch, sigma = synthesize_truths_and_data(
        solver, prior, n=cfg.n_archive, rel_noise=cfg.rel_noise, seed=cfg.seed + 2)
    # oracle-arm training fields: independent draws from the true prior
    x_oracle = prior.sample(cfg.n_oracle,
                            torch.Generator(device="cpu").manual_seed(cfg.seed + 3)).to(device)
    log(f"[data] sigma={sigma:.4g}  n_truths={cfg.n_truths} n_archive={cfg.n_archive} "
        f"n_oracle={cfg.n_oracle}")

    # --- 3. smoothness strength, chosen without the truths --------------------------- #
    n_cal = min(64, cfg.n_archive)
    alpha, alpha_info = select_alpha(
        solver, y_arch[:n_cal], sigma, rho=cfg.reg_ridge, map_iters=cfg.map_iters)
    # A floored or non-converged alpha silently inverts the freeze arm: a weak regularizer
    # gives a wide frozen belief, which would reverse the ordering the experiment reads. Refuse
    # to proceed unless the discrepancy target was bracketed, the selected alpha is strictly
    # interior to the log bracket, and the inner MAP hit chi2/dof near 1.
    c2 = alpha_info["chi2_final"]
    if not (alpha_info["bracketed"] and alpha_info["interior"] and 0.8 <= c2 <= 1.25):
        raise RuntimeError(
            "discrepancy alpha selection failed its pre-run guard: "
            f"alpha={alpha:.4g} log10={alpha_info['log10_alpha']:.3f} "
            f"bracket={alpha_info['log10_bracket']} bracketed={alpha_info['bracketed']} "
            f"interior={alpha_info['interior']} chi2_lo={alpha_info['chi2_lo']:.3f} "
            f"chi2_hi={alpha_info['chi2_hi']:.3f} chi2_final={c2:.3f}. Widen the bracket, "
            "raise map_iters, or raise rel_noise so the regularizer is genuinely "
            "overconfident.")
    reg = build_smoothness_regularizer(n=cfg.n, alpha=alpha, rho=cfg.reg_ridge,
                                       dtype=dtype, device=device)
    log(f"[reg] alpha={alpha:.4g} ridge={cfg.reg_ridge} "
        f"(bracketed={alpha_info['bracketed']} interior={alpha_info['interior']} "
        f"chi2_final={alpha_info['chi2_final']:.3f})")

    # --- 4. archives ----------------------------------------------------------------- #
    map_res = build_map_archive(solver, y_arch, reg, sigma,
                                map_iters=cfg.map_iters, lr=cfg.map_lr)
    x_map = map_res.x
    x_freeze = build_freeze_archive(x_map, split, reg, n_per=cfg.freeze_per, seed=cfg.seed + 4)
    log(f"[archive] MAP chi2/dof mean={float(map_res.chi2_per_dof.mean()):.3f}  "
        f"freeze_size={x_freeze.shape[0]}")

    # --- 5. raw blind-mode spreads (the mechanism) ----------------------------------- #
    b_true = split.project_blind(x_true)
    rough = split.rough_mask_blind(cfg.rough_frac).numpy()
    sr_map = _raw_spread_ratio(split.project_blind(x_map), b_true)
    sr_freeze = _raw_spread_ratio(split.project_blind(x_freeze), b_true)
    sr_oracle = _raw_spread_ratio(split.project_blind(x_oracle), b_true)
    ordering_ok = bool(sr_map[rough].mean() < sr_freeze[rough].mean()
                       < sr_oracle[rough].mean() * 1.15)
    log(f"[raw rough-mode spread s/s*]  MAP={sr_map[rough].mean():.3f}  "
        f"freeze={sr_freeze[rough].mean():.3f}  oracle={sr_oracle[rough].mean():.3f}  "
        f"ordering_ok={ordering_ok}")

    # --- 6. curated and oracle priors ------------------------------------------------ #
    def make_ddpm() -> VectorDDPM:
        return VectorDDPM(d_x=cfg.n, hidden=cfg.ddpm_hidden, depth=cfg.ddpm_depth,
                          n_timesteps=cfg.ddpm_timesteps).to(device)

    curated = make_ddpm()
    l_cur = train_vector_ddpm(curated, x_map.float(), batch_size=cfg.ddpm_batch,
                              n_steps=cfg.ddpm_train_steps, lr=cfg.ddpm_lr,
                              lr_final=cfg.ddpm_lr * 1e-2, seed=cfg.seed + 5)
    oracle = make_ddpm()
    l_ora = train_vector_ddpm(oracle, x_oracle.float(), batch_size=cfg.ddpm_batch,
                              n_steps=cfg.ddpm_train_steps, lr=cfg.ddpm_lr,
                              lr_final=cfg.ddpm_lr * 1e-2, seed=cfg.seed + 6)
    log(f"[ddpm] curated final loss={l_cur[-1]:.4f}  oracle final loss={l_ora[-1]:.4f}  "
        f"params={curated.count_parameters()}")

    # the sampling generator must live on the same device as the draws
    x_cur = curated.sample(cfg.ddpm_sample, device=device,
                           generator=torch.Generator(device=device).manual_seed(cfg.seed + 7)
                           ).double()
    x_ora = oracle.sample(cfg.ddpm_sample, device=device,
                          generator=torch.Generator(device=device).manual_seed(cfg.seed + 8)
                          ).double()

    # --- 7. coverage: blind (headline) and resolved (fairness control) --------------- #
    lvl = cfg.level
    cov_cur_blind = arm_coverage(split.project_blind(x_cur), b_true, level=lvl)
    cov_ora_blind = arm_coverage(split.project_blind(x_ora), b_true, level=lvl)
    r_true = split.project_resolved(x_true)
    cov_cur_res = arm_coverage(split.project_resolved(x_cur), r_true, level=lvl)
    cov_ora_res = arm_coverage(split.project_resolved(x_ora), r_true, level=lvl)
    well_lit = split.well_illuminated_mask(cfg.rough_frac).numpy()

    gate = oracle_control_checks(
        cov_ora_blind, cov_cur_blind, cov_ora_res, cov_cur_res,
        rough_mask=rough, resolved_mask=well_lit, level=lvl, tol=cfg.gate_tol)

    log(f"[coverage central-{int(lvl * 100)}% rough blind]  "
        f"curated={cov_cur_blind.central[rough].mean():.3f}  "
        f"oracle={cov_ora_blind.central[rough].mean():.3f}")
    log(f"[coverage well-lit resolved (fairness)]  "
        f"curated={cov_cur_res.central[well_lit].mean():.3f}  "
        f"oracle={cov_ora_res.central[well_lit].mean():.3f}")
    log(f"[gate] passed={gate['passed']}  oracle_nominal_on_rough_blind="
        f"{gate['oracle_nominal_on_rough_blind']}  resolved_fair={gate['resolved_subspace_fair']}")

    return {
        "config": cfg, "sigma": sigma, "alpha": alpha, "alpha_info": alpha_info,
        "map_chi2_per_dof": float(map_res.chi2_per_dof.mean()),
        "split": split, "rough_mask": rough, "well_lit_mask": well_lit,
        "rough_eigvals_blind": split.rough_eigvals_blind.cpu().numpy(),
        "sr_map": sr_map, "sr_freeze": sr_freeze, "sr_oracle": sr_oracle,
        "ordering_ok": ordering_ok,
        "ddpm_loss_curated": l_cur, "ddpm_loss_oracle": l_ora,
        "cov_curated_blind": cov_cur_blind, "cov_oracle_blind": cov_ora_blind,
        "cov_curated_resolved": cov_cur_res, "cov_oracle_resolved": cov_ora_res,
        "gate": gate,
    }
