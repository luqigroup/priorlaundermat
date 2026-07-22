"""Small shared pieces of the training loops."""

from __future__ import annotations

import math

import torch


class Normalizer:
    """Per-pixel standardization by the training set's mean and standard deviation.

    A prior is trained on normalized images, so its samples must be returned through
    :meth:`unnormalize` before they are compared with anything physical.
    """

    def __init__(self, dataset: torch.Tensor, eps: float = 1e-5):
        self.mean = torch.mean(dataset, 0)
        self.std = torch.std(dataset, 0)
        self.eps = eps

    @classmethod
    def from_stats(cls, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-5) -> "Normalizer":
        """Rebuild from a checkpoint's stored statistics, without the training set."""
        obj = cls.__new__(cls)
        obj.mean, obj.std, obj.eps = mean, std, eps
        return obj

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.std + self.eps) + self.mean


class PolynomialDecayLR:
    """``lr(k) = a (b + k)^gamma``, with ``a`` and ``b`` fixed by the endpoints.

    ``a`` and ``b`` are solved so that ``lr(0) = initial_lr`` and ``lr(max_step) = final_lr``.
    With the default ``gamma = -1/3`` the rate falls steeply within the first few steps and then
    flattens, which is what the groundwater flows were trained under; a cosine schedule over the
    same endpoints fits the archive far more tightly and does not reproduce those checkpoints.
    """

    def __init__(self, optimizer, initial_lr: float, final_lr: float, max_step: int,
                 gamma: float = -1 / 3):
        if final_lr > initial_lr:
            raise ValueError("final_lr must be <= initial_lr")
        if gamma > 0.0:
            raise ValueError("gamma must be negative for a decaying schedule")
        self.opt, self.initial_lr, self.final_lr, self.gamma = optimizer, initial_lr, final_lr, gamma
        if initial_lr != final_lr:
            self.b = max_step / ((final_lr / initial_lr) ** (1 / gamma) - 1.0)
            self.a = initial_lr / (self.b ** gamma)
        self.count = 0

    def step(self) -> float:
        lr = (self.initial_lr if self.initial_lr == self.final_lr
              else self.a * (self.b + self.count) ** self.gamma)
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.count += 1
        return lr


class CosineWarmupLR:
    """Linear warmup to ``lr``, then cosine decay to ``lr_final``, stepped per optimizer step."""

    def __init__(self, optimizer, lr: float, n_steps: int, lr_final: float = 0.0,
                 warmup: int = 0):
        self.opt, self.lr, self.n_steps = optimizer, lr, max(1, n_steps)
        self.lr_final, self.warmup, self.step_i = lr_final, warmup, 0

    def step(self) -> float:
        self.step_i += 1
        if self.warmup and self.step_i <= self.warmup:
            cur = self.lr * self.step_i / self.warmup
        else:
            t = (self.step_i - self.warmup) / max(1, self.n_steps - self.warmup)
            t = min(1.0, max(0.0, t))
            cur = self.lr_final + 0.5 * (self.lr - self.lr_final) * (1.0 + math.cos(math.pi * t))
        for g in self.opt.param_groups:
            g["lr"] = cur
        return cur
