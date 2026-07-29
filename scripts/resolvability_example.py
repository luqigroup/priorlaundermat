#!/usr/bin/env python
"""Worked example of the resolvability statement (resolvability.statement).

Computes, from the forward operator ALONE --- no prior, no training archive, no data --- the
directions in which a learned prior's reported uncertainty cannot be checked against the
measurements, and prints the resolvability statement the paper recommends shipping alongside any
reconstruction.

The example operator is a band-limited acquisition (keep the lowest spatial frequencies of a 1D
signal), the clean analogue of the seismic band limit in the paper: its blind subspace is exactly
the high-frequency complement the measurements never see. The same call works on any dense
operator, and on large operators via a precomputed blind basis (see
``resolvability/seismic/blind.py`` and ``scripts/deployed_statement.py``).

Run:  python scripts/resolvability_example.py      # numpy only, no downloads
"""
import numpy as np

from resolvability.statement import blind_report


def band_limited_operator(n: int, keep: int) -> np.ndarray:
    """(m, n) real operator that keeps the lowest ``keep`` spatial frequencies of a length-n
    signal (cosine + sine rows). Blind subspace = the high-frequency directions it drops."""
    x = np.arange(n)
    rows = [np.ones(n) / np.sqrt(n)]                         # DC
    for k in range(1, keep):
        rows.append(np.sqrt(2.0 / n) * np.cos(2 * np.pi * k * x / n))
        rows.append(np.sqrt(2.0 / n) * np.sin(2 * np.pi * k * x / n))
    return np.array(rows)


def main() -> None:
    n, keep = 128, 16
    A = band_limited_operator(n, keep)
    card = blind_report(A)
    print(card.summary())

    # Flag two reported quantities: a smooth (resolved) functional and a high-frequency (blind) one.
    x = np.arange(n)
    v_smooth = np.cos(2 * np.pi * 3 * x / n)                 # inside the pass band -> resolved
    v_fine = np.cos(2 * np.pi * 40 * x / n)                  # above the band       -> blind
    print("\nReported-interval check (does the data constrain this direction?):")
    for name, v, res in zip(("smooth feature", "fine detail"), (v_smooth, v_fine),
                            card.classify(np.vstack([v_smooth, v_fine]))):
        verdict = "data-verifiable" if res["label"] == "resolved" else \
                  "UNVERIFIABLE (prior-supplied)" if res["label"] == "blind" else "partly verifiable"
        print(f"  {name:14s}: blind fraction {res['blind_fraction']:.2f}  ->  {verdict}")


if __name__ == "__main__":
    main()
