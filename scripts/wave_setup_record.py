#!/usr/bin/env python
"""Record the operator quantities that fig_wave_setup.py draws.

Everything here is a property of the forward map alone: the space-time wavefield of one
reference field at both the nonlinear and the linear coefficient, the exact per-node
sensitivity at both, and the causal-shadow geometry. No prior is trained and no
reconstruction is solved.

The reference field is the first draw of the archive's truth batch, so the panel shows the
same field the rest of the experiment uses.

Reads: nothing on disk.
Writes: results/wave_setup_bundle.npz
Runtime: a few seconds, CPU.

    python scripts/wave_setup_record.py
"""

from __future__ import annotations

import os

import numpy as np
import torch

from priorlaundermat.download import REPO
from priorlaundermat.wave.blind import build_wave_blind_subspace
from priorlaundermat.wave.config import WaveLaunderingConfig
from priorlaundermat.wave.priors import RoughFieldPrior
from priorlaundermat.wave.solver import SemilinearWave1D

OUT = os.path.join(REPO, "results", "wave_setup_bundle.npz")


def main() -> None:
    cfg = WaveLaunderingConfig()
    solver = SemilinearWave1D(n=cfg.n, c=cfg.c, cfl=cfg.cfl, recv=cfg.recv, nt=cfg.nt,
                              eps=cfg.eps, domain_len=cfg.domain_len)
    linear = SemilinearWave1D(n=cfg.n, c=cfg.c, cfl=cfg.cfl, recv=cfg.recv, nt=cfg.nt,
                              eps=0.0, domain_len=cfg.domain_len)

    split = build_wave_blind_subspace(solver, n_probe_states=cfg.blind_probe_states,
                                      tol=cfg.blind_tol, seed=cfg.seed,
                                      verify_eps=cfg.verify_eps)
    print(f"[blind] n_blind={split.n_blind}/{cfg.n} "
          f"shadow=[{int(split.shadow_idx.min())}..{int(split.shadow_idx.max())}] "
          f"max_shadow_sens={split.max_shadow_sensitivity:.1e} "
          f"fixed_across_states/eps={split.identical_across_states}", flush=True)

    # The archive's truth batch, first row; the batch size enters the draw, so it is the
    # configured n_archive and not a smaller convenience number.
    prior = RoughFieldPrior(n=cfg.n, s_star=cfg.s_star, ell=cfg.ell)
    g = torch.Generator(device="cpu").manual_seed(cfg.seed + 2)
    x_ref = prior.sample(cfg.n_archive, g)[0]

    bundle = {
        "wavefield_eps": solver.wavefield(x_ref).numpy(),
        "wavefield_eps0": linear.wavefield(x_ref).numpy(),
        "sensitivity_eps": solver.jacobian(x_ref).abs().amax(dim=0).numpy(),
        "sensitivity_eps0": linear.jacobian(x_ref).abs().amax(dim=0).numpy(),
        "shadow_idx": split.shadow_idx.numpy(),
        "resolved_idx": split.resolved_idx.numpy(),
        "recv": np.asarray(list(cfg.recv)),
        "eps": float(cfg.eps), "s_star": float(cfg.s_star),
        "dx": float(solver.dx), "dt": float(solver.dt),
        "nt": int(cfg.nt), "n": int(cfg.n),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, **bundle)
    sens = bundle["sensitivity_eps"]
    print(f"[out] wrote {os.path.relpath(OUT, REPO)} | shadow sensitivity max "
          f"{sens[bundle['shadow_idx']].max():.1e}, resolved sensitivity min "
          f"{sens[bundle['resolved_idx']].min():.2e}", flush=True)


if __name__ == "__main__":
    main()
