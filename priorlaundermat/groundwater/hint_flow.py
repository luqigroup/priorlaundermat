"""HINT (Hierarchical Invertible Neural Transport) normalizing flow.

A stack of binary-tree affine-coupling layers with a fixed random permutation between them, with
an exact ``log_prob`` and an ``inverse`` (latent -> sample) map, after Kruse et al., "HINT:
Hierarchical Invertible Neural Transport" (arXiv:1905.10687).

Used here as the learned prior over the d=100 whitened KL coefficients of the Darcy
log-permeability: the oracle flow is trained on true-field coefficients, the curated flow on the
MAP-archive coefficients. ``n_cond = 0`` gives an unconditional prior; posterior sampling is then
pCN in the flow's Gaussian latent, ``xi = flow.inverse(w)`` with ``w ~ N(0, I)``.

Imports only ``math`` and ``torch``; the d=100 model used here trains and samples on CPU.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class _NFTree(nn.Module):
    """One HINT binary tree of affine couplings (forward encode / inverse sample)."""

    def __init__(self, n_in: int, n_cond: int, n_hidden: int, depth: int,
                 n_mlp_layers: int = 3, activation: nn.Module | None = None):
        super().__init__()
        if activation is None:
            activation = nn.ReLU()
        self.n_in = n_in
        self.n_cond = n_cond
        self.depth = depth

        if depth <= 0 or n_in <= 1:
            self.is_leaf = True
            return
        self.is_leaf = False
        self.split_idx = n_in // 2
        upper_dim, lower_dim = self.split_idx, n_in - self.split_idx

        layers = [nn.Linear(upper_dim + n_cond, n_hidden), activation]
        for _ in range(n_mlp_layers - 2):
            layers += [nn.Linear(n_hidden, n_hidden), activation]
        final = nn.Linear(n_hidden, 2 * lower_dim)
        nn.init.normal_(final.weight, std=0.01); nn.init.zeros_(final.bias)
        layers.append(final)
        self.coupling_mlp = nn.Sequential(*layers)

        self.upper_tree = _NFTree(upper_dim, n_cond, n_hidden, depth - 1,
                                  n_mlp_layers=n_mlp_layers, activation=activation)
        self.lower_tree = _NFTree(lower_dim, n_cond, n_hidden, depth - 1,
                                  n_mlp_layers=n_mlp_layers, activation=activation)

    def _coupling(self, z_upper, cond):
        out = self.coupling_mlp(torch.cat([z_upper, cond], dim=1))
        lower_dim = self.n_in - self.split_idx
        log_S = torch.clamp(out[:, :lower_dim], -5.0, 5.0)     # exp -> S > 0, so the map inverts
        T = out[:, lower_dim:]
        return torch.exp(log_S), T, log_S

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.is_leaf:
            return x, torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        x_upper, x_lower = x[:, :self.split_idx], x[:, self.split_idx:]
        z_upper, ld_upper = self.upper_tree(x_upper, cond)
        S, T, log_S = self._coupling(z_upper, cond)
        z_lower, ld_lower = self.lower_tree(S * x_lower + T, cond)
        z = torch.cat([z_upper, z_lower], dim=1)
        return z, ld_upper + log_S.sum(dim=1) + ld_lower

    def inverse(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if self.is_leaf:
            return z
        z_upper, z_lower = z[:, :self.split_idx], z[:, self.split_idx:]
        x_upper = self.upper_tree.inverse(z_upper, cond)
        S, T, _ = self._coupling(z_upper, cond)
        x_lower = (self.lower_tree.inverse(z_lower, cond) - T) / S
        return torch.cat([x_upper, x_lower], dim=1)


class HINTFlow(nn.Module):
    """``n_flow_layers`` stacked HINT trees with random permutations between them."""

    def __init__(self, n_in: int, n_cond: int = 0, n_hidden: int = 256,
                 n_flow_layers: int = 8, depth: Optional[int] = None, n_mlp_layers: int = 3):
        super().__init__()
        self.n_in = n_in
        self.n_cond = n_cond
        depth = (max(1, int(math.log2(n_in))) if n_in > 1 else 0) if depth is None else depth
        self.trees = nn.ModuleList(
            [_NFTree(n_in, n_cond, n_hidden, depth, n_mlp_layers=n_mlp_layers)
             for _ in range(n_flow_layers)])
        perms = [torch.randperm(n_in) for _ in range(n_flow_layers)]
        self.register_buffer("perms", torch.stack(perms))
        self.register_buffer("inv_perms", torch.stack([torch.argsort(p) for p in perms]))

    def _cond(self, batch: int, cond, device, dtype):
        if self.n_cond == 0:
            return torch.zeros(batch, 0, device=device, dtype=dtype)
        return cond

    def forward(self, x, cond=None):
        """Sample -> latent, with the log-determinant of the Jacobian."""
        cond = self._cond(x.shape[0], cond, x.device, x.dtype)
        z = x
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for k, tree in enumerate(self.trees):
            z, ld = tree(z[:, self.perms[k]], cond)
            log_det = log_det + ld
        return z, log_det

    def inverse(self, z, cond=None):
        """Latent -> sample."""
        cond = self._cond(z.shape[0], cond, z.device, z.dtype)
        x = z
        for k in reversed(range(len(self.trees))):
            x = self.trees[k].inverse(x, cond)[:, self.inv_perms[k]]
        return x

    def log_prob(self, x, cond=None):
        z, log_det = self.forward(x, cond)
        log_pz = -0.5 * z.pow(2).sum(dim=1) - 0.5 * self.n_in * math.log(2 * math.pi)
        return log_pz + log_det
