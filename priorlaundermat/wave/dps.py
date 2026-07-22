"""Diffusion posterior sampling on the wave operator.

An ancestral DDPM reverse loop with a data-misfit guidance step: at each guided step the
current Tweedie estimate ``x0`` is pushed downhill on ``||y - F(x0)||`` with an AdaGrad
preconditioner accumulated over the trajectory. The leapfrog solver is exactly autograd
differentiable, so the guidance gradient is machine exact -- no adjoint approximation enters.

Everything here runs in float64: the network is cast to double so the gradient through the
solver keeps the operator's precision. Guidance is switched on only after the first
``guide_from`` fraction of the reverse trajectory, while the iterate is still mostly noise and
a misfit gradient would only fight the prior.
"""

from __future__ import annotations

import numpy as np
import torch
from diffusers import DDPMScheduler
from torch import Tensor

from priorlaundermat.utils import Normalizer
from priorlaundermat.wave.nets import NNODE, SIDE
from priorlaundermat.wave.solver import SemilinearWave1D


def data_misfit_norm(x0_img: Tensor, solver: SemilinearWave1D, norm: Normalizer,
                     y: Tensor) -> Tensor:
    """Per-chain ``||y - F(x)||`` for normalized images ``x0_img``, differentiable in them."""
    x = norm.unnormalize(x0_img.reshape(x0_img.shape[0], NNODE))
    r = solver.forward(x) - y.reshape(1, -1)
    return torch.sqrt((r ** 2).sum(dim=1).clamp_min(1e-30))


def dps_posterior(unet, solver: SemilinearWave1D, norm: Normalizer, y: Tensor, *,
                  n_chain: int, n_steps: int, xi: float = 0.5,
                  guide_from: float = 0.25, seed: int = 0) -> np.ndarray:
    """Posterior samples for one measurement ``y``; ``(n_chain, 64)`` physical fields.

    Args:
        unet: a prior loaded in float64 (see :func:`~priorlaundermat.wave.nets.load_prior`).
        solver: the wave forward map the measurement came from.
        norm: the prior's per-node normalizer.
        y: one measurement, shape ``(d_y,)``.
        n_chain: independent chains run in parallel (they are the posterior ensemble).
        n_steps: reverse diffusion steps.
        xi: guidance strength.
        guide_from: fraction of the trajectory run unguided before guidance starts.
        seed: RNG seed for the initial noise.
    """
    sched = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2",
                          clip_sample=False)
    sched.set_timesteps(n_steps)
    abars = sched.alphas_cumprod
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n_chain, 1, SIDE, SIDE, dtype=torch.float64, generator=g)
    grad_sq = torch.zeros(n_chain, 1, SIDE, SIDE, dtype=torch.float64)
    adagrad_eps = 1e-8
    steps = sched.timesteps
    n_skip = int(guide_from * len(steps))
    for k, t in enumerate(steps):
        ab = float(abars[t])
        with torch.no_grad():
            eps = unet(x, t.expand(n_chain)).sample
        if k < n_skip:
            with torch.no_grad():
                x = sched.step(eps, t, x, generator=g).prev_sample
            continue
        x = x.detach().requires_grad_(True)
        # eps is detached, so the graph from x to x0 is linear and the gradient is the exact
        # derivative of the misfit through the solver.
        x0 = (x - np.sqrt(1 - ab) * eps) / np.sqrt(ab)
        rnorm = data_misfit_norm(x0, solver, norm, y)
        grad = torch.autograd.grad(rnorm.sum(), x)[0]
        with torch.no_grad():
            x_prev = sched.step(eps, t, x.detach(), generator=g).prev_sample
            grad_sq = grad_sq + grad ** 2
            x = x_prev - xi * grad / (torch.sqrt(grad_sq) + adagrad_eps)
    return norm.unnormalize(x.detach().reshape(n_chain, NNODE)).numpy()
