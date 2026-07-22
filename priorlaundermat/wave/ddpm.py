"""A minimal unconditional DDPM over a fixed-length vector.

The epsilon control needs an unconditional sampler over the ``d_x``-dimensional field: one
trained on the legacy MAP archive (the curated prior) and one trained on the true fields (the
oracle prior), with an IDENTICAL architecture, so the only difference between the arms is the
training data. This is the smallest thing that samples ``p(x)`` correctly:

  * epsilon prediction, ``x_t = sqrt(abar_t) x0 + sqrt(1 - abar_t) eps``;
  * a linear beta schedule over ``T`` steps;
  * a sinusoidal-time-embedding GELU MLP denoiser;
  * per-coordinate input standardization (kept in buffers), so the same architecture trains
    stably both on the calibrated truths and on the collapsed archive, whose rough shadow
    modes are much tighter; a standard-deviation floor keeps a near-collapsed coordinate from
    blowing up the scaling. Samples are returned in PHYSICAL space, where the blind-subspace
    projection is measured.

Ancestral DDPM sampling only. The network runs in float32; the wave operator stays float64.
Reference: Ho, Jain and Abbeel, NeurIPS 2020, arXiv:2006.11239.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn


def _sinusoidal_time_embedding(t: Tensor, dim: int) -> Tensor:
    """Transformer-style sinusoidal embedding of integer timesteps ``(B,) -> (B, dim)``."""
    if dim % 2 != 0:
        raise ValueError(f"time-embedding dim must be even, got {dim}")
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32)
        / max(half - 1, 1))
    ang = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class VectorDDPM(nn.Module):
    """Unconditional DDPM prior over ``R^{d_x}`` with an MLP epsilon predictor."""

    def __init__(
        self, d_x: int = 64, hidden: int = 256, depth: int = 3,
        n_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2,
        time_embed_dim: int = 64, std_floor: float = 1e-3,
    ):
        super().__init__()
        self.d_x = int(d_x)
        self.n_timesteps = int(n_timesteps)
        self.time_embed_dim = int(time_embed_dim)
        self.std_floor = float(std_floor)

        betas = torch.linspace(beta_start, beta_end, self.n_timesteps)
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphabars", torch.cumprod(alphas, dim=0))
        self.register_buffer("data_mean", torch.zeros(self.d_x))
        self.register_buffer("data_std", torch.ones(self.d_x))

        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden))
        layers: list[nn.Module] = [nn.Linear(self.d_x + hidden, hidden), nn.GELU()]
        for _ in range(max(depth - 1, 0)):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, self.d_x)]
        self.net = nn.Sequential(*layers)

    def fit_standardizer(self, x0: Tensor) -> None:
        with torch.no_grad():
            self.data_mean.copy_(x0.mean(dim=0))
            self.data_std.copy_(x0.std(dim=0).clamp_min(self.std_floor))

    def _standardize(self, x0: Tensor) -> Tensor:
        return (x0 - self.data_mean) / self.data_std

    def _unstandardize(self, z: Tensor) -> Tensor:
        return z * self.data_std + self.data_mean

    def _pdtype(self) -> torch.dtype:
        return self.net[0].weight.dtype

    def eps_pred(self, x_t: Tensor, t: Tensor) -> Tensor:
        dt = self._pdtype()
        emb = _sinusoidal_time_embedding(t, self.time_embed_dim).to(dt)
        return self.net(torch.cat([x_t.to(dt), self.time_mlp(emb)], dim=-1))

    def loss(self, x0_phys: Tensor, generator: Optional[torch.Generator] = None) -> Tensor:
        x0 = self._standardize(x0_phys.to(self._pdtype()))
        b = x0.shape[0]
        dev = x0.device
        t = torch.randint(0, self.n_timesteps, (b,), device=dev, generator=generator)
        eps = torch.randn(x0.shape, device=dev, dtype=x0.dtype, generator=generator)
        abar = self.alphabars[t].unsqueeze(-1)
        x_t = abar.sqrt() * x0 + (1.0 - abar).sqrt() * eps
        return (self.eps_pred(x_t, t) - eps).square().mean()

    @torch.no_grad()
    def sample(self, n: int, generator: Optional[torch.Generator] = None,
               device: Optional[torch.device] = None) -> Tensor:
        """Ancestral DDPM sampling; ``n`` draws returned in physical space."""
        dev = device if device is not None else self.data_mean.device
        dt = self._pdtype()
        x = torch.randn(n, self.d_x, device=dev, dtype=dt, generator=generator)
        for i in reversed(range(self.n_timesteps)):
            t = torch.full((n,), i, device=dev, dtype=torch.long)
            eps_hat = self.eps_pred(x, t)
            alpha = self.alphas[i]
            abar = self.alphabars[i]
            mean = (x - (1.0 - alpha) / (1.0 - abar).sqrt() * eps_hat) / alpha.sqrt()
            if i > 0:
                noise = torch.randn(n, self.d_x, device=dev, dtype=dt, generator=generator)
                x = mean + self.betas[i].sqrt() * noise
            else:
                x = mean
        return self._unstandardize(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_vector_ddpm(
    model: VectorDDPM, x0_phys: Tensor, *, batch_size: int, n_steps: int,
    lr: float = 1e-3, lr_final: Optional[float] = None, grad_clip: float = 1.0,
    seed: int = 0, progress: bool = False,
) -> list[float]:
    """Train ``model`` on physical samples by DDPM epsilon matching; return the loss trace."""
    device = next(model.parameters()).device
    x0_phys = x0_phys.to(device)
    model.fit_standardizer(x0_phys)
    n = x0_phys.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=lr_final)
             if lr_final is not None else None)
    gen = torch.Generator(device=device).manual_seed(seed)
    losses: list[float] = []
    model.train()
    it = range(n_steps)
    if progress:
        try:
            from tqdm import trange
            it = trange(n_steps, desc="ddpm")
        except ImportError:
            pass
    for _ in it:
        idx = torch.randint(0, n, (min(batch_size, n),), device=device, generator=gen)
        loss = model.loss(x0_phys[idx], generator=gen)
        opt.zero_grad()
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        if sched is not None:
            sched.step()
        losses.append(float(loss.item()))
    model.eval()
    return losses
