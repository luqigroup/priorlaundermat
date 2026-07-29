#!/usr/bin/env python
"""The probe basis and the illumination spectrum that define the operator's blind subspace.

The normal operator ``A^T A`` is pseudodifferential, so its point-spread functions have a local
Fourier transform that is the illumination symbol: large where the survey constrains a wavenumber
at a location, small where it does not. Reading the symbol off a handful of point-spread functions
costs a few operator applications, instead of the hundreds of matrix-vector products a Krylov
decomposition of a 65536-dimensional operator would need.

Stage 1 -- the atoms. Fire a lattice of 16 well-separated deltas through ``A^T A`` in one apply,
take each tile's local Fourier symbol, and build Gabor atoms at the three highest and three lowest
in-band wavenumbers of each tile: 48 resolved and 48 nominally blind unit atoms.

Stage 2 -- the spectrum. Measure ``||A v||^2`` for every atom, with and without the water mute, and
for a seeded set of 48 random orthonormal directions as a control. Every claimed direction is
therefore validated by an actual forward solve, not by the symbol that proposed it.

The blind basis itself is not stored: :func:`resolvability.seismic.blind.build_blind_subspace`
derives it from these two files by dropping the nominally blind atoms whose measured ``||A v||^2``
leakage pushes above 1% of the resolved median, then orthonormalizing the survivors.

Writes data/seismic/psf_illum_N256.npz (the atoms and the symbol) and
data/seismic/effect_existence_spectrum.npz (the measured spectra).
CPU only (Devito). Stage 2 is roughly 240 forward evaluations of 12 shots each: many hours.
Run it single-threaded if the spectrum must be reproduced exactly.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from resolvability.download import REPO
from resolvability.seismic import MUTE_END, N, TAPER
from resolvability.seismic.born import parihaka_imager

SCRAMBLE_SEED = 0
KP = 3                       # atoms per tile per class
OUT_ATOMS = os.path.join(REPO, "data/seismic/psf_illum_N256.npz")
OUT_SPECTRUM = os.path.join(REPO, "data/seismic/effect_existence_spectrum.npz")


def Ju2(im, v, mute: bool = False):
    """``||A v||^2``, optionally through the water mute the archived data passes through."""
    x = torch.tensor(np.asarray(v, np.float32).reshape(1, 1, N, N), dtype=torch.float32)
    if mute:
        x = im.mute_op(x, end=MUTE_END, length=TAPER)
    return float(sum((im.born_one(x, s) ** 2).sum() for s in range(im.nsrc)))


def normal_apply(im, v):
    """``A^T A v``, via autograd of the squared record."""
    x = torch.tensor(np.asarray(v, np.float32).reshape(1, 1, N, N), requires_grad=True)
    d = torch.cat([im.born_one(x, s).reshape(-1) for s in range(im.nsrc)])
    (d.detach() * d).sum().backward()                       # gradient is 2 A^T A v
    return 0.5 * x.grad.detach().cpu().numpy().ravel().astype(np.float64)


def hann2d(p):
    w = np.hanning(p)
    return np.outer(w, w)


def build_atoms(im, t0):
    """Stage 1: the resolved and nominally blind Gabor atoms, and the global illumination symbol."""
    P = 64                                                  # tile size
    g = max(2, (N - P) // P + 1)
    cs = np.linspace(P // 2, N - P // 2, g).astype(int)
    probes = [(a, b) for a in cs for b in cs]
    print(f"[{time.time() - t0:.0f}s] probe lattice {g}x{g}={len(probes)}, tile {P}, spacing "
          f"{cs[1] - cs[0]} cells (at least one tile apart, so the tiles do not overlap)",
          flush=True)

    R0 = np.zeros(N * N, np.float32)
    for (a, b) in probes:
        R0[a * N + b] = 1.0
    Nr0 = normal_apply(im, R0).reshape(N, N)                # all 16 point-spread functions at once
    print(f"[{time.time() - t0:.0f}s] point-spread stack done", flush=True)

    symbols = []
    h = P // 2
    for (a, b) in probes:
        a0, b0 = np.clip(a - h, 0, N - P), np.clip(b - h, 0, N - P)
        tile = Nr0[a0:a0 + P, b0:b0 + P].copy()
        tile -= tile.mean()
        symbols.append(np.abs(np.fft.fftshift(np.fft.fft2(hann2d(P) * tile))) ** 2)
    symbols = np.stack(symbols)
    # How far the local symbols wander from their mean. Small means one global symbol describes the
    # whole section; large would mean the split has to be made per depth zone instead.
    drift = float(np.max(np.linalg.norm(symbols - symbols.mean(0), axis=(1, 2)))
                  / (np.linalg.norm(symbols.mean(0)) + 1e-30))
    a_sym = symbols.mean(0)
    a_sym /= a_sym.max() + 1e-30
    print(f"[{time.time() - t0:.0f}s] symbol built; stationarity drift {drift:.2f}", flush=True)

    kk = np.fft.fftshift(np.fft.fftfreq(P))
    KX, KZ = np.meshgrid(kk, kk, indexing="ij")
    krad = np.sqrt(KX ** 2 + KZ ** 2)
    band = (krad > 0.06) & (krad < 0.5)                     # drop DC and the evanescent corner
    idx_band = np.where(band.ravel())[0]
    xx, zz = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    sig = P / 4.0

    def gabor(a, b, ki):
        """A unit atom localized at (a, b) with wavenumber index ``ki``."""
        kx, kz = KX.ravel()[ki], KZ.ravel()[ki]
        win = np.exp(-((xx - a) ** 2 + (zz - b) ** 2) / (2 * sig ** 2))
        u = win * np.cos(2 * np.pi * (kx * (xx - a) + kz * (zz - b)))
        return (u / (np.linalg.norm(u) + 1e-30)).ravel()

    res_atoms, bl_atoms = [], []
    for mi, (a, b) in enumerate(probes):
        sm = symbols[mi].ravel()
        order = idx_band[np.argsort(sm[idx_band])]          # ranked by this tile's own symbol
        for ki in order[:KP]:
            bl_atoms.append(gabor(a, b, ki))
        for ki in order[-KP:]:
            res_atoms.append(gabor(a, b, ki))
    U_res = np.stack(res_atoms).astype(np.float32)
    U_bl = np.stack(bl_atoms).astype(np.float32)

    # Spot-check the symbol's proposal against the operator: five atoms of each class.
    res_val = [Ju2(im, U_res[j]) for j in range(0, len(U_res), max(1, len(U_res) // 5))]
    bl_val = [Ju2(im, U_bl[j]) for j in range(0, len(U_bl), max(1, len(U_bl) // 5))]
    print(f"[{time.time() - t0:.0f}s] spot check ||A v||^2: resolved {np.mean(res_val):.3e}  "
          f"blind {np.mean(bl_val):.3e}  ratio "
          f"{np.mean(res_val) / max(np.mean(bl_val), 1e-30):.1e}", flush=True)
    os.makedirs(os.path.dirname(OUT_ATOMS), exist_ok=True)
    np.savez(OUT_ATOMS, a_sym=a_sym, drift=drift, U_res=U_res, U_bl=U_bl,
             res_val=np.array(res_val), bl_val=np.array(bl_val), band=band, krad=krad)
    print(f"[{time.time() - t0:.0f}s] saved {os.path.relpath(OUT_ATOMS, REPO)}", flush=True)
    return U_res, U_bl


def build_spectrum(im, U_res, U_bl, t0):
    """Stage 2: every atom's measured ``||A v||^2``, plus the seeded random control."""
    k = U_res.shape[0]
    res_bare = np.array([Ju2(im, U_res[j]) for j in range(k)])
    bl_bare = np.array([Ju2(im, U_bl[j]) for j in range(k)])
    print(f"[{time.time() - t0:.0f}s] bare spectrum done; resolved median "
          f"{np.median(res_bare):.4g}, blind range [{bl_bare.min():.4g}, {bl_bare.max():.4g}]",
          flush=True)
    res_muted = np.array([Ju2(im, U_res[j], mute=True) for j in range(k)])
    bl_muted = np.array([Ju2(im, U_bl[j], mute=True) for j in range(k)])

    rng = np.random.default_rng(SCRAMBLE_SEED)
    Gmat = rng.standard_normal((k, N * N))
    Qscr, _ = np.linalg.qr(Gmat.T)
    scr_bare = np.array([Ju2(im, Qscr.T[j]) for j in range(k)])
    print(f"[{time.time() - t0:.0f}s] scrambled control median {np.median(scr_bare):.4g}",
          flush=True)

    os.makedirs(os.path.dirname(OUT_SPECTRUM), exist_ok=True)
    np.savez(OUT_SPECTRUM, res_bare=res_bare, bl_bare=bl_bare, res_muted=res_muted,
             bl_muted=bl_muted, scr_bare=scr_bare)
    print(f"[{time.time() - t0:.0f}s] saved {os.path.relpath(OUT_SPECTRUM, REPO)}", flush=True)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    t0 = time.time()
    im = parihaka_imager()
    ratio = im.adjoint_dot_ratio()
    assert abs(ratio - 1.0) < 1e-3, (
        f"the migration is not the transpose of the demigration (dot ratio {ratio:.6f}); "
        f"the normal operator this basis is read from would not be symmetric")
    print(f"[{time.time() - t0:.0f}s] operator built, adjoint dot ratio {ratio:.6f}, nt={im.nt}",
          flush=True)
    U_res, U_bl = build_atoms(im, t0)
    build_spectrum(im, U_res, U_bl, t0)


if __name__ == "__main__":
    main()
