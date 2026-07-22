"""Configuration for the self-contained wave run behind the epsilon control.

The defaults are the settled design: 64 nodes, interior receivers at nodes 1 and 3, a
20-sample recording window (which leaves 41 exactly blind nodes), a defocusing cubic with
``eps = 40``, a rough truth prior, 5% relative noise, and a discrepancy-tuned smoothness
regularizer with a small ridge. Only ``eps`` is varied to produce the linear control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class WaveLaunderingConfig:
    # --- forward operator ---
    n: int = 64
    c: float = 1.0
    cfl: float = 0.5
    recv: Sequence[int] = (1, 3)
    nt: int = 20
    eps: float = 40.0                  # nonlinear operator; 0.0 is the linear control
    domain_len: float = 1.0

    # --- truth prior (rough field) ---
    s_star: float = 2.0
    ell: float = 2.0

    # --- acquisition and legacy regularizer ---
    rel_noise: float = 0.05            # sigma = rel_noise * RMS(clean signal)
    reg_ridge: float = 1e-2

    # --- blind subspace extraction ---
    blind_probe_states: int = 8
    blind_tol: float = 0.0             # machine exact-zero columns
    verify_eps: float = 0.0            # cross-check the linear operator's shadow

    # --- archive sizes ---
    n_truths: int = 2000               # held-out coverage truths
    n_archive: int = 4000              # legacy observations -> MAP archive
    n_oracle: int = 4000               # true fields for the oracle arm
    freeze_per: int = 8                # freeze draws per MAP instance
    map_iters: int = 3000
    map_lr: float = 5e-2

    # --- the two priors ---
    ddpm_hidden: int = 256
    ddpm_depth: int = 4
    ddpm_timesteps: int = 1000
    ddpm_train_steps: int = 40000
    ddpm_batch: int = 256
    ddpm_lr: float = 1e-3
    ddpm_sample: int = 4000            # prior draws for coverage

    # --- coverage metric ---
    level: float = 0.90                # central-90%
    rough_frac: float = 0.5            # roughest fraction of blind modes
    gate_tol: float = 0.1

    # --- run ---
    seed: int = 0
    device: str = "cpu"
