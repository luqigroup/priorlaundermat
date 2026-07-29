"""Preconditioned Crank-Nicolson posterior sampling under a learned Darcy prior.

The posterior is sampled by pCN IN THE FLOW'S GAUSSIAN LATENT: reparameterize
``xi = flow.inverse(w)`` with ``w ~ N(0, I)`` and run pCN on ``w`` targeting

    pi(w | y)  propto  exp( -|| S p(u(xi(w))) - y ||^2 / (2 sigma_y^2) )  N(w; 0, I).

Because the latent reference measure is Gaussian, Stuart's dimension-robust pCN proposal applies
unchanged to a learned nonlinear latent -- no diffusion guidance, no likelihood approximation. The
oracle flow (trained on truths) is the calibrated control; the curated flow (trained on the MAP
archive) inherits the collapsed belief on the operator's blind subspace.

Everything here is CPU: the cost is one sparse LU solve per truth per pCN step.
"""
from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np
import torch

from resolvability.download import ensure
from resolvability.groundwater.darcy import DarcyForward, beskos_sensors
from resolvability.groundwater.hint_flow import HINTFlow
from resolvability.groundwater.kl_prior import StuartKLPrior
from resolvability.utils import Normalizer

DATA = "data/darcy/darcy_laundering.h5"
LEVELS = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])


@dataclass
class DarcySetup:
    """Everything a pCN run needs: the problem, the archive, and the operator's subspaces."""

    kl: StuartKLPrior
    fwd: DarcyForward
    sensors: tuple[np.ndarray, np.ndarray]
    sigma_y: float
    N: int
    split: np.ndarray            # (n,) 0 = train, 1 = eval
    xi_true: np.ndarray          # (n, d) truth coefficients
    xi_map: np.ndarray           # (n, d) legacy MAP-archive coefficients
    y_obs: np.ndarray            # (n, n_sensors) noisy head observations
    resolved: np.ndarray         # (d, r)     operator-resolved directions in xi-space
    blind: np.ndarray            # (d, d - r) the operator's near-null
    sv: np.ndarray               # (n_sensors,) singular values of the linearized forward map

    def eval_truths(self, n_truths: int):
        """First ``n_truths`` held-out surveys: ``(y_obs, xi_true)``, in dataset order."""
        ev = np.where(self.split == 1)[0][:n_truths]
        return self.y_obs[ev], self.xi_true[ev]


def setup(seed: int = 0) -> DarcySetup:
    """Read the dataset and rebuild the forward problem it was generated with.

    Seeds the global numpy and torch generators, which drive the pCN proposal and acceptance, so
    every mode below is reproducible from its command line alone.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    with h5py.File(ensure(DATA), "r") as f:
        N, K = int(f.attrs["N"]), int(f.attrs["K"])
        alpha, s, sig = float(f.attrs["alpha"]), float(f.attrs["s"]), float(f.attrs["sigma"])
        sigma_y = float(f.attrs["sigma_y"])
        split, xi_true, xi_map = f["split"][:], f["xi_true"][:], f["xi_map"][:]
        y_obs, resolved, blind = f["y_obs"][:], f["resolved"][:], f["blind"][:]
        sv = f["sv"][:]
    sensors, _ = beskos_sensors(N, y_obs.shape[1])
    return DarcySetup(kl=StuartKLPrior(N, K=K, alpha=alpha, s=s, sigma=sig),
                      fwd=DarcyForward(N), sensors=sensors, sigma_y=sigma_y, N=N,
                      split=split, xi_true=xi_true, xi_map=xi_map, y_obs=y_obs,
                      resolved=resolved, blind=blind, sv=sv)


def load_flow(arm: str, seed: int = 0, device: str | torch.device = "cpu"):
    """Load a trained HINT prior. Returns ``(flow, normalizer)``.

    Args:
        arm: ``"oracle"`` (trained on truths) or ``"curated"`` (trained on the MAP archive).
        seed: training seed, 0-2.
        device: torch device; the d=100 flow runs comfortably on CPU.
    """
    ck = torch.load(ensure(f"data/checkpoints/darcy_flow_{arm}_seed{seed}.pth"),
                    map_location=device, weights_only=False)
    flow = HINTFlow(ck["n_in"], n_cond=0, n_hidden=ck["n_hidden"],
                    n_flow_layers=ck["n_layers"], depth=ck.get("depth"))
    flow.load_state_dict(ck["model"])
    flow.to(device).eval()
    norm = Normalizer.from_stats(ck["norm_mean"].to(device), ck["norm_std"].to(device))
    return flow, norm


def pcn_posterior(flow, norm, st: DarcySetup, y_obs, *, n_steps, n_burn, beta, thin,
                  device="cpu"):
    """pCN in the flow latent, run for all truths at once.

    Args:
        flow, norm: a prior from :func:`load_flow`.
        st: the problem from :func:`setup`.
        y_obs: (nt, n_sensors) surveys, one chain per row.
        n_steps: total pCN iterations; ``n_burn`` discarded, then every ``thin``-th kept.
        beta: pCN step size (0.08 gives ~63% acceptance here).

    Returns:
        ``(samples, acceptance)`` with ``samples`` of shape ``(n_kept, nt, d)`` in KL coordinates.
    """
    kl, fwd, sensors, sigma_y = st.kl, st.fwd, st.sensors, st.sigma_y
    nt, d = y_obs.shape[0], flow.n_in

    def misfit(w):
        with torch.no_grad():
            xi = norm.unnormalize(flow.inverse(w)).cpu().numpy()      # -> KL scale
        m = np.empty(nt)
        for t in range(nt):
            r = fwd.observe(fwd.solve(kl.reconstruct(xi[t])), sensors) - y_obs[t]
            m[t] = 0.5 * float(r @ r) / sigma_y ** 2
        return m, xi

    w = torch.randn(nt, d, device=device)
    Phi, xi = misfit(w)
    samples, n_acc = [], 0
    for step in range(n_steps):
        wp = np.sqrt(1 - beta ** 2) * w + beta * torch.randn(nt, d, device=device)
        Phip, xip = misfit(wp)
        acc = np.random.random(nt) < np.exp(np.minimum(0.0, Phi - Phip))
        n_acc += acc.mean()
        a = torch.tensor(acc, device=device)
        w = torch.where(a[:, None], wp, w)
        Phi = np.where(acc, Phip, Phi); xi = np.where(acc[:, None], xip, xi)
        if step >= n_burn and (step - n_burn) % thin == 0:
            samples.append(xi.copy())
    return np.stack(samples), n_acc / n_steps


def coverage(samples_proj, truth_proj, levels=LEVELS):
    """Central-``L`` credible coverage per level.

    Args:
        samples_proj: (n_kept, nt, k) posterior samples projected on a subspace.
        truth_proj: (nt, k) the truths on the same subspace.

    Returns:
        ``(aggregate, per_truth)`` of shapes ``(n_levels,)`` and ``(n_levels, nt)``.
    """
    agg, per_truth = [], []
    for L in levels:
        lo = np.quantile(samples_proj, (1 - L) / 2, axis=0)
        hi = np.quantile(samples_proj, (1 + L) / 2, axis=0)
        inside = (truth_proj >= lo) & (truth_proj <= hi)
        agg.append(float(inside.mean())); per_truth.append(inside.mean(axis=1))
    return np.array(agg), np.array(per_truth)


def iat_sokal(x: np.ndarray, c: float = 5.0) -> float:
    """Integrated autocorrelation time of a 1-D chain, by Sokal's automatic windowing."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = x.size
    if n < 8 or np.allclose(x, 0.0):
        return np.nan
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conjugate(f))[:n].real
    if acf[0] <= 0:
        return np.nan
    acf /= acf[0]
    tau = 2.0 * np.cumsum(acf) - 1.0
    idx = np.arange(n)
    win = idx < c * tau
    m = int(np.argmin(win)) if not win.all() else n - 1
    return float(tau[m])
