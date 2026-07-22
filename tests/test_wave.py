"""Correctness checks for the semilinear-wave example. CPU, seconds, no downloads.

Everything here is generated in the test, so nothing depends on a released artifact.
"""

from __future__ import annotations

import numpy as np
import torch

from priorlaundermat.wave.blind import build_wave_blind_subspace, graph_laplacian_1d
from priorlaundermat.wave.coverage import arm_coverage
from priorlaundermat.wave.priors import RoughFieldPrior, synthesize_truths_and_data
from priorlaundermat.wave.regularizers import build_smoothness_regularizer
from priorlaundermat.wave.solver import SemilinearWave1D

N, NT, RECV = 32, 8, (1, 3)


def _solver(eps: float = 40.0, n: int = N, nt: int = NT) -> SemilinearWave1D:
    return SemilinearWave1D(n=n, recv=RECV, nt=nt, eps=eps)


def test_jacobian_matches_finite_differences():
    """The autograd Jacobian is the real derivative of the nonlinear forward map."""
    solver = _solver()
    g = torch.Generator().manual_seed(0)
    x = torch.randn(N, generator=g, dtype=torch.float64)
    v = torch.randn(N, generator=g, dtype=torch.float64)

    J = solver.jacobian(x)
    h = 1e-6
    fd = (solver.forward((x + h * v).unsqueeze(0)).squeeze(0)
          - solver.forward((x - h * v).unsqueeze(0)).squeeze(0)) / (2 * h)
    rel = float((J @ v - fd).norm() / fd.norm())
    assert rel < 1e-7, f"directional derivative mismatch, relative error {rel:.2e}"


def test_linearized_adjoint_identity():
    """<J v, w> == <v, J^T w> for the linearization of the nonlinear map."""
    solver = _solver()
    g = torch.Generator().manual_seed(1)
    x = torch.randn(N, generator=g, dtype=torch.float64)
    J = solver.jacobian(x)
    v = torch.randn(N, generator=g, dtype=torch.float64)
    w = torch.randn(solver.d_y, generator=g, dtype=torch.float64)
    lhs = float((J @ v) @ w)
    rhs = float(v @ (J.T @ w))
    assert abs(lhs - rhs) <= 1e-10 * max(abs(lhs), 1.0)


def test_shadow_is_exactly_blind_and_outside_the_causal_cone():
    """The blind set is machine-exactly zero and matches the receivers' causal cone."""
    solver = _solver()
    split = build_wave_blind_subspace(solver, n_probe_states=3, seed=0, verify_eps=0.0)

    assert split.max_shadow_sensitivity == 0.0
    assert split.min_resolved_sensitivity > 0.0
    assert split.identical_across_states, "the blind set moved across states or with eps"

    # A node is reachable within nt-1 leapfrog steps iff some receiver is within nt-1 cells.
    nodes = np.arange(N)
    reach = np.min(np.abs(nodes[:, None] - np.asarray(RECV)[None, :]), axis=1)
    expected_shadow = nodes[reach > NT - 1]
    assert np.array_equal(np.sort(split.shadow_idx.numpy()), expected_shadow)
    assert split.n_blind + split.n_resolved == N


def test_blind_and_resolved_bases_are_orthonormal_and_disjoint():
    solver = _solver()
    split = build_wave_blind_subspace(solver, n_probe_states=3, seed=0, verify_eps=None)

    for Q in (split.Q_blind, split.Q_resolved):
        gram = Q @ Q.T
        assert torch.allclose(gram, torch.eye(Q.shape[0], dtype=Q.dtype), atol=1e-12)
    # the two bases live on disjoint coordinate blocks, so they are mutually orthogonal
    cross = split.Q_blind @ split.Q_resolved.T
    assert float(cross.abs().max()) == 0.0
    # blind modes are ordered smooth to rough
    ev = split.rough_eigvals_blind.numpy()
    assert np.all(np.diff(ev) >= -1e-12)


def test_forward_ignores_the_shadow():
    """Perturbing the shadow leaves the record bit-identical, at both coefficients."""
    g = torch.Generator().manual_seed(2)
    x = torch.randn(N, generator=g, dtype=torch.float64)
    for eps in (0.0, 40.0):
        solver = _solver(eps=eps)
        split = build_wave_blind_subspace(solver, n_probe_states=2, seed=0, verify_eps=None)
        pert = x.clone()
        pert[split.shadow_idx] += 7.0 * torch.randn(split.n_blind, generator=g,
                                                    dtype=torch.float64)
        assert torch.equal(solver.forward(x.unsqueeze(0)), solver.forward(pert.unsqueeze(0)))


def test_graph_laplacian_is_the_roughness_energy():
    n = 9
    L = graph_laplacian_1d(n)
    g = torch.Generator().manual_seed(3)
    x = torch.randn(n, generator=g, dtype=torch.float64)
    assert torch.allclose(x @ L @ x, (x[1:] - x[:-1]).square().sum())
    assert torch.allclose(L @ torch.ones(n, dtype=torch.float64),
                          torch.zeros(n, dtype=torch.float64))


def test_regularizer_value_matches_its_quadratic_form():
    reg = build_smoothness_regularizer(n=N, alpha=3.0, rho=0.25)
    g = torch.Generator().manual_seed(4)
    x = torch.randn(2, N, generator=g, dtype=torch.float64)
    P = reg.alpha * reg.L + reg.rho * torch.eye(N, dtype=torch.float64)
    assert torch.allclose(reg.value(x), 0.5 * torch.einsum("bi,ij,bj->b", x, P, x))


def test_coverage_is_nominal_on_matched_gaussians():
    """A calibrated model covers at its nominal level; a shrunken one does not."""
    g = torch.Generator().manual_seed(5)
    truths = torch.randn(4000, 3, generator=g, dtype=torch.float64)
    model = torch.randn(20000, 3, generator=g, dtype=torch.float64)

    cov = arm_coverage(model, truths, level=0.9)
    assert np.all(np.abs(cov.central - 0.9) < 0.02)

    shrunk = arm_coverage(0.1 * model, truths, level=0.9)
    assert np.all(shrunk.central < 0.3)


def test_noise_level_is_a_known_fraction_of_the_signal():
    solver = _solver()
    prior = RoughFieldPrior(n=N, s_star=2.0, ell=2.0)
    x_true, y, sigma = synthesize_truths_and_data(solver, prior, n=64, rel_noise=0.05, seed=0)
    clean = solver.forward(x_true)
    assert x_true.shape == (64, N) and y.shape == (64, solver.d_y)
    assert abs(sigma - 0.05 * float(clean.std())) < 1e-12
    assert abs(float((y - clean).std()) / sigma - 1.0) < 0.15
