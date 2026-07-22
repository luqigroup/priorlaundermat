"""Differentiable explicit leapfrog solver for a 1D semilinear (defocusing) wave.

    d_tt u - c^2 Lap u + eps u^3 = 0,   u(.,0) = x,   d_t u(.,0) = 0,   u = 0 at both ends.

The unknown is the initial displacement field ``x`` on ``n`` grid nodes; a few INTERIOR
receivers record ``u`` at their nodes over a SHORT window ``t in [0, T]``. The forward map
``F : x -> {u(r_k, t_m)}`` is genuinely nonlinear in ``x`` for ``eps > 0`` (the cubic is
defocusing, hence energy-stable: no blow-up).

By finite speed of propagation the record depends on ``x`` only through its restriction to
the receivers' domain of dependence over the window, so ``F(x) = G(P x)`` for a FIXED linear
projection ``P``. Discretely the leapfrog stencil couples one cell per step, so ``u(r, step
m)`` depends only on ``x_j`` with ``min_r |j - r| <= m``; nodes with ``min_r |j - r| > nt-1``
are machine-exactly blind, at both ``eps = 0`` and ``eps > 0``.

The solver is batched over a leading dimension and differentiable end to end (autograd
through the time stepping), so it backs the MAP archive, the blind-subspace extraction, and
the posterior-sampling guidance gradient. Everything is float64.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor


@dataclass
class SemilinearWave1D:
    """1D semilinear-wave forward map with an interior, short-window acquisition.

    Args:
        n: Number of grid nodes (state dimension ``d_x``).
        c: Wave speed.
        cfl: Courant number ``c*dt/dx`` (``<= 1`` for stability).
        recv: Interior receiver node indices.
        nt: Number of recorded time samples (steps ``0 .. nt-1``); short, so the causal
            shadow is nonempty and reflections do not return within the window.
        eps: Cubic coefficient. ``0`` is the linear control; ``40`` is genuinely nonlinear
            at the truth-field scale.
        domain_len: Physical length ``L`` (grid spacing ``dx = L/(n-1)``).
        dtype, device: Tensor placement (float64 throughout).
    """

    n: int = 64
    c: float = 1.0
    cfl: float = 0.5
    recv: Sequence[int] = (1, 3)
    nt: int = 20
    eps: float = 40.0
    domain_len: float = 1.0
    dtype: torch.dtype = torch.float64
    device: Optional[torch.device] = None

    def __post_init__(self) -> None:
        self.dx = self.domain_len / (self.n - 1)
        self.dt = self.cfl * self.dx / self.c
        self._recv = torch.as_tensor(list(self.recv), dtype=torch.long, device=self.device)
        # Interior mask (0 at the Dirichlet endpoints) applied by multiplication, so the
        # boundary condition is imposed WITHOUT in-place ops on the gradient path.
        m = torch.ones(self.n, dtype=self.dtype, device=self.device)
        m[0] = 0.0
        m[-1] = 0.0
        self._interior = m

    @property
    def d_x(self) -> int:
        return self.n

    @property
    def d_y(self) -> int:
        return self.nt * len(self.recv)

    def _laplacian(self, u: Tensor) -> Tensor:
        """Second difference with Dirichlet (zero) ghosts, shape-preserving."""
        lap = torch.zeros_like(u)
        lap[..., 1:-1] = (u[..., 2:] - 2.0 * u[..., 1:-1] + u[..., :-2]) / self.dx ** 2
        return lap

    def forward(self, x: Tensor) -> Tensor:
        """Record ``u`` at the receivers over ``nt`` steps.

        Args:
            x: Initial displacement field, shape ``(..., n)``.

        Returns:
            ``(..., nt * n_recv)`` receiver trace (time-major within each row).
        """
        x = x.to(self.dtype)
        c2 = self.c ** 2
        dt2 = self.dt ** 2
        eps = self.eps
        interior = self._interior
        recv = self._recv

        u = x
        # u_t(0) = 0  =>  u^1 = u^0 + 0.5 dt^2 (c^2 Lap u^0 - eps (u^0)^3)
        u_next = u + 0.5 * dt2 * (c2 * self._laplacian(u) - eps * u ** 3)
        u_next = u_next * interior
        trace = [u[..., recv], u_next[..., recv]]
        u_prev, u = u, u_next
        for _ in range(self.nt - 2):
            u_next = 2.0 * u - u_prev + dt2 * (c2 * self._laplacian(u) - eps * u ** 3)
            u_next = u_next * interior
            trace.append(u_next[..., recv])
            u_prev, u = u, u_next
        return torch.stack(trace, dim=-2).reshape(*x.shape[:-1], self.d_y)

    __call__ = forward

    def wavefield(self, x: Tensor) -> Tensor:
        """Full displacement field over every node and recorded step.

        Same leapfrog as :meth:`forward`, but returns the whole space-time field rather than
        the receiver samples only.

        Args:
            x: Initial displacement field, shape ``(..., n)``.

        Returns:
            ``(..., nt, n)`` displacement field (time index major over space).
        """
        x = x.to(self.dtype)
        c2 = self.c ** 2
        dt2 = self.dt ** 2
        eps = self.eps
        interior = self._interior
        u = x
        u_next = u + 0.5 * dt2 * (c2 * self._laplacian(u) - eps * u ** 3)
        u_next = u_next * interior
        frames = [u, u_next]
        u_prev, u = u, u_next
        for _ in range(self.nt - 2):
            u_next = 2.0 * u - u_prev + dt2 * (c2 * self._laplacian(u) - eps * u ** 3)
            u_next = u_next * interior
            frames.append(u_next)
            u_prev, u = u, u_next
        return torch.stack(frames, dim=-2)

    def jacobian(self, x: Tensor) -> Tensor:
        """Exact ``dF/dx`` at a single state ``x``, shape ``(d_y, d_x)``.

        Reverse-mode (``d_y <= d_x`` here) vmapped Jacobian through the leapfrog. The columns
        for causal-shadow nodes are machine-exactly zero at every ``x``; :mod:`.blind` reads
        them to define the blind set.
        """
        x = x.detach().to(self.dtype)

        def f(v: Tensor) -> Tensor:
            return self.forward(v.unsqueeze(0)).squeeze(0)

        return torch.autograd.functional.jacobian(f, x, vectorize=True)
