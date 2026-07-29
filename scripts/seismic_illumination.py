#!/usr/bin/env python
"""Incident-wavefield energy: the source-side illumination of the Born operator, in image space.

For each source, propagate the Ricker through the background model and accumulate the
time-integrated squared wavefield, then crop the damping layer. Low energy means the survey's
wavefield barely reaches that part of the section, so the operator is close to blind there and the
prior alone decides what goes in the reconstruction. This is the physical, per-pixel illumination
proxy; the probe-atom spectrum is the subspace-level one.

Writes results/seismic_illum_incident.npz (illum, grid size, spacing).
CPU only (Devito), twelve wave solves, a few minutes. Needs the wavefield saved over all time
steps, so it wants a few GB of memory.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from resolvability.download import REPO
from resolvability.seismic.born import parihaka_imager

OUT = os.path.join(REPO, "results/seismic_illum_incident.npz")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    im = parihaka_imager()
    nb = im.nbl
    E = None
    for s in range(im.nsrc):
        im._src.coordinates.data[:, 0] = im._src_x[s]
        u0 = im.solver.forward(src=im._src, save=True)[1]      # background wavefield over all time
        e = (np.asarray(u0.data) ** 2).sum(axis=0)             # (nx_pad, nz_pad)
        e = e[nb:-nb, nb:-nb]                                  # crop the damping layer
        E = e if E is None else E + e
        print(f"  source {s + 1}/{im.nsrc} at x={im._src_x[s]:.0f} m  "
              f"cumulative energy max={E.max():.3e}", flush=True)
        del u0
    E = np.asarray(E, dtype=np.float32)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, illum=E, nlat=im.nlat, ndep=im.ndep, spacing=np.asarray(im.spacing))
    print(f"saved {os.path.relpath(OUT, REPO)}  shape={E.shape}  "
          f"min={E.min():.3e} max={E.max():.3e}", flush=True)


if __name__ == "__main__":
    main()
