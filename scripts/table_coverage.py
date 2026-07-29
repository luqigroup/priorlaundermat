#!/usr/bin/env python
"""The paper's coverage table, printed as LaTeX from the reliability grid's own loaders.

Achieved central-(1-alpha) coverage at the levels the two operators share, averaged over the
training seeds -- the same numbers the reliability grid draws, read from the same caches through
the SAME loaders (``fig_reliability_grid.seismic`` / ``.groundwater``), so the table cannot drift
from the figure. The per-seed markers and bootstrap bands stay on the figure; the table carries
only the seed means.

Reads the same caches as ``scripts/fig_reliability_grid.py`` (downloaded on first use). Prints the
tabular to stdout. CPU, seconds.

Run:  python scripts/table_coverage.py
"""
from __future__ import annotations

import numpy as np

from fig_reliability_grid import groundwater, seismic  # the sibling script in this directory

# The levels both operators were scored at (the groundwater cache carries 0.60 as well; the
# seismic one does not, so the table keeps the shared grid).
SHARED = (0.50, 0.70, 0.80, 0.90, 0.95)
ARMS = (("oracle", "emcol"), ("curated", "madcol"))


def fmt(v: float) -> str:
    """Two decimals, leading zero dropped, as the paper sets them: 0.69 -> $.69$."""
    return "$" + ("%.2f" % v).lstrip("0") + "$"


def block(op_name: str, data: dict) -> list[str]:
    """The four rows of one operator: {resolved, blind} x {oracle, curated}."""
    out = []
    for sub in ("resolved", "blind"):
        p = data[sub]
        lev = np.asarray(p["lev"], float)
        idx = [int(np.argmin(np.abs(lev - L))) for L in SHARED]
        for L, i in zip(SHARED, idx):
            if abs(lev[i] - L) > 1e-9:
                raise ValueError(f"{op_name}/{sub}: level {L} not in the cache ({lev})")
        for arm, colour in ARMS:
            cells = [fmt(float(p["cov"][arm][i])) for i in idx]
            lead = op_name if (sub == "resolved" and arm == "oracle") else ""
            mid = sub if arm == "oracle" else ""
            out.append(f"{lead} & {mid} & \\textcolor{{{colour}}}{{{arm}}} & "
                       + " & ".join(cells) + r" \\")
    return out


def main() -> None:
    rows = block("Seismic Born", seismic()) + block("Groundwater", groundwater())
    head = [
        r"\begin{tabular}{lllccccc}",
        r"\toprule",
        r" & & & \multicolumn{5}{c}{nominal central-$(1-\alpha)$ level} \\",
        r"\cmidrule(lr){4-8}",
        "operator & subspace & prior & "
        + " & ".join("$" + ("%.2f" % L).lstrip("0") + "$" for L in SHARED) + r" \\",
        r"\midrule",
    ]
    print("\n".join(head + rows + [r"\bottomrule", r"\end{tabular}"]))


if __name__ == "__main__":
    main()
