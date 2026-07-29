"""Unit tests for the groundwater (Darcy) example. Fast, CPU, no downloads."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from resolvability.groundwater import (DarcyForward, HINTFlow, StuartKLPrior, beskos_sensors)
from resolvability.groundwater.map_solver import _neg_log_post

N = 16
SIGMA_Y = 0.05


def _problem(n_sensors=9):
    fwd = DarcyForward(N)
    sensors, _ = beskos_sensors(N, n_sensors)
    kl = StuartKLPrior(N, K=4)
    return fwd, sensors, kl


# --------------------------------------------------------------------------- forward + adjoint
def test_solve_matches_boundary_conditions():
    fwd = DarcyForward(N)
    p = fwd.solve(np.zeros((N, N)))
    x = np.arange(N) / (N - 1)
    assert np.allclose(p[:, 0], x)
    assert np.allclose(p[:, -1], 1.0 - x)


def test_discrete_adjoint_matches_finite_differences():
    """The exact adjoint gradient of the misfit against a central difference in the field.

    This is the one piece of nontrivial numerics in the Darcy operator, and the whole MAP archive
    is built on it, so it is checked directly rather than through a solver.
    """
    fwd, sensors, _ = _problem()
    rng = np.random.default_rng(0)
    u = 0.3 * rng.standard_normal((N, N))
    y_obs = fwd.observe(fwd.solve(u), sensors) + SIGMA_Y * rng.standard_normal(len(sensors[0]))

    p, lu, K = fwd.solve(u, return_factor=True)
    g = fwd.gradient_field(p, lu, K, sensors, y_obs, SIGMA_Y)

    def J(field):
        r = fwd.observe(fwd.solve(field), sensors) - y_obs
        return 0.5 * float(r @ r) / SIGMA_Y ** 2

    # Each evaluation refactorizes, so its rounding does not cancel between the two sides; a step
    # this large keeps the difference well above that noise and still well inside the quadratic
    # truncation regime.
    eps = 1e-4
    probes = [(0, 0), (1, 5), (N // 2, N // 2), (N - 2, 3), (N - 1, N - 1), (4, N - 1)]
    for i, j in probes:
        up, um = u.copy(), u.copy()
        up[i, j] += eps; um[i, j] -= eps
        fd = (J(up) - J(um)) / (2 * eps)
        assert fd == pytest.approx(g[i, j], rel=1e-4, abs=1e-8), f"node ({i}, {j})"

    # One directional derivative exercises every node at once, boundary rows included.
    v = rng.standard_normal((N, N)); v /= np.linalg.norm(v)
    fd = (J(u + eps * v) - J(u - eps * v)) / (2 * eps)
    assert fd == pytest.approx(float((g * v).sum()), rel=1e-4)


def test_kl_pullback_gradient_matches_finite_differences():
    """The chain rule from field gradient to KL coefficients, as the MAP solver uses it."""
    fwd, sensors, kl = _problem()
    rng = np.random.default_rng(1)
    xi_true = rng.standard_normal(kl.d)
    y_obs = fwd.observe(fwd.solve(kl.reconstruct(xi_true)), sensors)

    xi = 0.5 * rng.standard_normal(kl.d)
    _, grad = _neg_log_post(xi, fwd, kl, sensors, y_obs, SIGMA_Y)

    eps = 1e-4
    for k in (0, 3, kl.d // 2, kl.d - 1):
        e = np.zeros(kl.d); e[k] = eps
        fd = (_neg_log_post(xi + e, fwd, kl, sensors, y_obs, SIGMA_Y)[0]
              - _neg_log_post(xi - e, fwd, kl, sensors, y_obs, SIGMA_Y)[0]) / (2 * eps)
        assert fd == pytest.approx(grad[k], rel=1e-4, abs=1e-8), f"mode {k}"


# ------------------------------------------------------------------------------------- KL prior
def test_kl_reconstruct_coefficient_round_trip():
    """The eigenbasis is orthonormal on the domain, so projecting a field recovers its latent."""
    kl = StuartKLPrior(129, K=6)
    rng = np.random.default_rng(2)
    xi = rng.standard_normal(kl.d)
    u = kl.reconstruct(xi)

    h = 1.0 / (kl.grid_size - 1)
    w = np.ones(kl.grid_size); w[0] = w[-1] = 0.5             # trapezoid over [0,1]^2
    quad = h * h * np.outer(w, w)
    xi_back = np.einsum("ij,kij->k", u * quad, kl.phi) / kl.amp
    assert np.allclose(xi_back, xi, rtol=2e-3, atol=2e-3)


def test_kl_reconstruct_is_linear_and_batched():
    kl = StuartKLPrior(N, K=4)
    rng = np.random.default_rng(3)
    xi = rng.standard_normal((5, kl.d))
    batched = kl.reconstruct(xi)
    assert batched.shape == (5, N, N)
    for i in range(5):
        assert np.allclose(batched[i], kl.reconstruct(xi[i]))
    assert np.allclose(kl.reconstruct(xi[0] + 2 * xi[1]),
                       kl.reconstruct(xi[0]) + 2 * kl.reconstruct(xi[1]))


def test_kl_amplitudes_follow_stuart_spectral_decay():
    kl = StuartKLPrior(N, K=4, alpha=0.0, s=1.1, sigma=1.0)
    ksq = (kl.i1 + 0.5) ** 2 + (kl.i2 + 0.5) ** 2
    assert np.allclose(kl.amp, (np.pi ** 2 * ksq) ** (-1.1 / 2))
    assert kl.stuart_true_xi().shape == (kl.d,)


# ------------------------------------------------------------------------------------ HINT flow
def _flow(n_in=8, seed=0):
    torch.manual_seed(seed)
    return HINTFlow(n_in, n_cond=0, n_hidden=16, n_flow_layers=3, depth=2).double().eval()


def test_flow_forward_and_inverse_are_mutual_inverses():
    flow = _flow()
    torch.manual_seed(1)
    # Untrained couplings are near-identity, so perturb them into a genuinely nonlinear map.
    with torch.no_grad():
        for p in flow.parameters():
            p.add_(0.3 * torch.randn_like(p))
    x = torch.randn(7, flow.n_in, dtype=torch.float64)
    z, _ = flow(x)
    assert torch.allclose(flow.inverse(z), x, atol=1e-10)
    w = torch.randn(7, flow.n_in, dtype=torch.float64)
    assert torch.allclose(flow(flow.inverse(w))[0], w, atol=1e-10)


def test_flow_log_prob_matches_change_of_variables():
    """log p(x) = log N(z; 0, I) + log|det df/dx|, with the determinant taken from autograd."""
    flow = _flow(n_in=6, seed=4)
    with torch.no_grad():
        for p in flow.parameters():
            p.add_(0.3 * torch.randn_like(p))
    torch.manual_seed(5)
    x = torch.randn(3, flow.n_in, dtype=torch.float64)

    lp = flow.log_prob(x).detach()
    for i in range(x.shape[0]):
        jac = torch.autograd.functional.jacobian(lambda v: flow(v[None])[0][0], x[i])
        z = flow(x[i][None])[0][0]
        expect = (-0.5 * z.pow(2).sum() - 0.5 * flow.n_in * math.log(2 * math.pi)
                  + torch.log(torch.abs(torch.det(jac))))
        assert float(lp[i]) == pytest.approx(float(expect.detach()), rel=1e-9, abs=1e-9)


# --------------------------------------------------------------------------------------- sensors
def test_beskos_sensors_are_distinct_and_in_domain():
    (ix, iy), (sx, sy) = beskos_sensors(40, 33)
    assert ix.shape == iy.shape == (33,)
    assert len({(int(a), int(b)) for a, b in zip(ix, iy)}) == 33
    assert ix.min() >= 0 and ix.max() <= 39 and iy.min() >= 0 and iy.max() <= 39
    assert np.all((sx >= 0) & (sx <= 1) & (sy >= 0) & (sy <= 1))
    assert (sx[0], sy[0]) == (0.5, 0.5)                       # one sensor at the centre
    r = np.hypot(sx[1:] - 0.5, sy[1:] - 0.5)                  # the rest on one circle
    assert np.allclose(r, 39 / 80.0)
