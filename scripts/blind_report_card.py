#!/usr/bin/env python
"""Worked example of the blind-subspace report card.

Computes, from the forward operator alone -- no prior, no training archive, no data -- the
directions in which a learned prior's reported uncertainty cannot be checked against the
measurements, and prints the report card the paper recommends shipping alongside any
reconstruction.

The example operator is a band-limited acquisition (keep the lowest spatial frequencies of a 1D
signal), the clean analogue of the seismic band limit: its blind subspace is exactly the
high-frequency complement the measurements never see. The same call works on any dense operator,
and on large operators via a precomputed blind basis (see :mod:`priorlaundermat.seismic.blind`).

Reads nothing, writes nothing. Runs in under a second on numpy alone.
"""

import numpy as np

from priorlaundermat.report_card import blind_report


def band_limited_operator(n: int, keep: int) -> np.ndarray:
    """``(m, n)`` operator keeping the lowest ``keep`` spatial frequencies of a length-``n`` signal.

    Rows are the orthonormal cosine/sine pairs up to ``keep``; the blind subspace is the
    high-frequency complement the operator drops.
    """
    x = np.arange(n)
    rows = [np.ones(n) / np.sqrt(n)]                         # DC
    for k in range(1, keep):
        rows.append(np.sqrt(2.0 / n) * np.cos(2 * np.pi * k * x / n))
        rows.append(np.sqrt(2.0 / n) * np.sin(2 * np.pi * k * x / n))
    return np.array(rows)


def main() -> None:
    n, keep = 128, 16
    card = blind_report(band_limited_operator(n, keep))
    print(card.summary())

    # Two reported quantities: one inside the pass band, one above it.
    x = np.arange(n)
    v_smooth = np.cos(2 * np.pi * 3 * x / n)
    v_fine = np.cos(2 * np.pi * 40 * x / n)
    print("\nReported-interval check (does the data constrain this direction?):")
    for name, res in zip(("smooth feature", "fine detail"),
                         card.classify(np.vstack([v_smooth, v_fine]))):
        verdict = ("data-verifiable" if res["label"] == "resolved" else
                   "UNVERIFIABLE (prior-supplied)" if res["label"] == "blind" else
                   "partly verifiable")
        print(f"  {name:14s}: blind fraction {res['blind_fraction']:.2f}  ->  {verdict}")


if __name__ == "__main__":
    main()
