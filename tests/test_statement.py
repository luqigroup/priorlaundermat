"""The resolvability statement: the split it computes, and the honesty of its accounting."""
import numpy as np
import pytest

from resolvability.statement import blind_report


def _band_limited(n=64, keep=16):
    """Keep the lowest `keep` DCT frequencies of an n-vector: an exact kernel of dimension n-keep."""
    k = np.arange(n)
    B = np.cos(np.pi * (k[:, None] + 0.5) * np.arange(n)[None, :] / n)
    return B[:keep]


def test_dense_operator_recovers_the_exact_kernel():
    A = _band_limited(64, 16)
    card = blind_report(A)
    assert card.complete
    assert card.rank == 16
    assert card.blind_dim == 64 - 16
    assert card.blind_fraction == pytest.approx((64 - 16) / 64)
    # the recovered basis really is annihilated by A
    assert np.abs(A @ card.N_blind).max() < 1e-8


def test_classify_separates_resolved_from_blind():
    A = _band_limited(64, 16)
    card = blind_report(A)
    v_blind = card.N_blind[:, 0]
    v_res = np.linalg.svd(A, full_matrices=True)[2][0]      # a leading right-singular vector
    labels = card.classify(np.stack([v_blind, v_res]))
    assert labels[0]["label"] == "blind" and not labels[0]["verifiable"]
    assert labels[1]["label"] == "resolved" and labels[1]["verifiable"]


def test_inherited_variance_fraction_matches_a_known_split():
    A = _band_limited(64, 16)
    card = blind_report(A)
    rng = np.random.default_rng(0)
    # samples supported ONLY on the blind subspace: the whole reported spread is inherited
    X = rng.normal(size=(500, card.blind_dim)) @ card.N_blind.T
    assert card.inherited_variance_fraction(X) == pytest.approx(1.0, abs=1e-8)


def test_probed_near_null_is_reported_as_a_lower_bound_not_a_full_split():
    """A certified SUBSET of blind directions must not be presented as the whole blind subspace."""
    A = _band_limited(64, 16)
    full = blind_report(A)
    probed = blind_report(n=64, N_blind=full.N_blind[:, :4])   # only 4 of the 48 blind directions
    assert not probed.complete
    assert probed.blind_dim == 4
    assert np.isnan(probed.blind_fraction)                     # refuses to quote a share it cannot know
    assert "LOWER bound" in probed.summary()
    assert "resolved rank" not in probed.summary()             # no bogus rank claim
    # and it still classifies the directions it does certify
    assert probed.classify(full.N_blind[:, 0])[0]["label"] == "blind"


def test_complete_can_be_asserted_explicitly_for_a_supplied_basis():
    A = _band_limited(64, 16)
    full = blind_report(A)
    card = blind_report(n=64, N_blind=full.N_blind, complete=True)
    assert card.complete and card.rank == 16
    assert card.blind_fraction == pytest.approx(48 / 64)
