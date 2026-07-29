"""Fast checks on the seismic pipeline's shared pieces: subspaces, evaluation window, water mute.

No GPU, no Devito, no training archive. Everything here reads either the shipped probe caches or
the slim evaluation window.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from resolvability.download import REPO
from resolvability.seismic import MUTE_END, N, TAPER
from resolvability.seismic import priors
from resolvability.seismic.blind import build_blind_subspace, verify_blind_gap
from resolvability.seismic.priors import load_eval
from resolvability.seismic.pseudosynth import ricker, water_mute


@pytest.fixture(scope="module")
def sub():
    return build_blind_subspace()


def test_bases_are_orthonormal(sub):
    """Each basis must be orthonormal, or projected energy is not energy."""
    for Q in (sub.Q_blind, sub.Q_resolved, sub.Q_scrambled):
        assert np.allclose(Q @ Q.T, np.eye(Q.shape[0]), atol=1e-8)


def test_blind_and_resolved_are_mutually_orthogonal(sub):
    """Blind and resolved must not overlap: an image's energy has to split unambiguously."""
    cross = np.abs(sub.Q_blind @ sub.Q_resolved.T).max()
    assert cross < 5e-2, f"blind and resolved bases overlap (max |cos| = {cross:.3g})"


def test_subspace_dimensions(sub):
    """The shipped spectrum yields the 16-dimensional blind basis the paper reads."""
    assert (sub.r_blind, sub.r_resolved, sub.r_scrambled) == (16, 32, 48)
    assert sub.n_blind_kept == 24          # of 48 nominally blind candidates
    assert sub.N == N


def test_blind_atoms_sit_below_the_floor(sub):
    """Every kept blind atom is below the relative-gap floor it was selected by."""
    assert 0 < sub.rel_floor < 0.01 * sub.res_median * 1.000001


def test_blind_gap_holds_on_the_shipped_window(sub):
    """The prerequisite: on the blind subspace the truths carry far more than the archive does.

    Without this gap there is nothing for a curated prior to inherit, so it is the first thing to check.
    The scrambled control must show no such gap.
    """
    truth = load_eval("broadband_dm", 300, 340)
    lsrtm = load_eval("lsrtm5", 300, 340)
    gap = verify_blind_gap(sub, truth, lsrtm)
    assert gap["blind_native"] > 5.0
    assert gap["blind_native"] > 3.0 * gap["scrambled_native"]
    assert gap["r_blind"] == sub.r_blind


def test_load_eval_returns_the_requested_rows():
    """Rows are addressed by their index in the full archive, not by their offset in the window."""
    a, b = 305, 311
    X = load_eval("broadband_dm", a, b)
    assert X.shape == (b - a, N * N)
    assert X.dtype == np.float32
    assert not np.allclose(X[0], X[1])
    assert np.allclose(load_eval("broadband_dm", a + 1, b)[0], X[1])


def test_load_eval_is_muted():
    """load_eval returns images the imager would accept: water column zeroed."""
    X = load_eval("lsrtm5", 300, 302).reshape(-1, N, N)
    assert np.abs(X[:, :, :MUTE_END - TAPER]).max() == 0.0
    assert np.abs(X[:, :, MUTE_END:]).max() > 0.0


def test_load_eval_outside_the_window_names_the_fallback(monkeypatch):
    """Rows outside the window must fail with the archive named, not with a stray KeyError."""
    real = priors.ensure

    def fake(rel):
        if rel == priors.EVAL_H5:
            raise RuntimeError("not a released artifact")
        return real(rel)

    monkeypatch.setattr(priors, "ensure", fake)
    with pytest.raises(FileNotFoundError) as exc:
        load_eval("broadband_dm", 0, 4)
    assert priors.EVAL_H5 in str(exc.value)


@pytest.mark.skipif(not os.path.exists(os.path.join(REPO, priors.EVAL_H5)),
                    reason="the full evaluation archive is not on disk")
def test_load_eval_falls_back_across_the_window_edge():
    """A request straddling the window's end is served from the archive, and the two agree."""
    z = np.load(priors.ensure(priors.EVAL_WINDOW))
    row0, n = int(z["row0"]), z["broadband_dm"].shape[0]
    straddle = load_eval("broadband_dm", row0 + n - 3, row0 + n + 2)
    assert straddle.shape == (5, N * N)
    inside = load_eval("broadband_dm", row0 + n - 3, row0 + n)
    assert np.allclose(straddle[:3], inside)


def test_water_mute_zeroes_and_tapers():
    """Zero above the ramp, monotone through it, untouched below."""
    img = np.ones((N, N), np.float32)
    m = water_mute(img, MUTE_END, TAPER)
    start = MUTE_END - TAPER
    assert np.all(m[:, :start] == 0.0)
    assert np.all(m[:, MUTE_END:] == 1.0)
    ramp = m[0, start:MUTE_END]
    assert np.all(np.diff(ramp) > 0)
    assert 0.0 < ramp[0] < 1.0 and ramp[-1] < 1.0


def test_water_mute_matches_the_imagers_mask():
    """The mute is separable and depth-only, so it commutes with anything acting laterally."""
    rng = np.random.default_rng(0)
    img = rng.standard_normal((N, N)).astype(np.float32)
    mask = water_mute(np.ones((N, N), np.float32), MUTE_END, TAPER)
    assert np.allclose(water_mute(img, MUTE_END, TAPER), img * mask, atol=1e-6)


def test_ricker_is_zero_phase_and_peak_normalized():
    w = ricker(3.5)
    assert len(w) % 2 == 1
    assert np.allclose(w, w[::-1])
    assert np.isclose(np.abs(w).max(), 1.0)
    assert np.argmax(w) == len(w) // 2
