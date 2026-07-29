"""Broadband ground-truth reflectivity from tracked Parihaka horizons, and the water mute.

Convolutional model: reflection coefficients placed on the (laterally smoothed, sub-pixel)
horizon lines, convolved down each column with a Ricker of dominant vertical wavelength ``lam``
in samples. The horizons give real geology; the wavelet sets a frequency content that extends
past the band the 12-shot survey can resolve, so the truth carries content the operator is blind
to.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


def ricker(lam: float, half: int | None = None) -> np.ndarray:
    """Zero-phase Ricker of dominant vertical wavelength ``lam`` (samples), peak-normalized."""
    half = half or int(round(3 * lam))
    k = np.arange(-half, half + 1)
    a = (np.pi * k / lam) ** 2
    w = (1 - 2 * a) * np.exp(-a)
    return w / np.abs(w).max()


def water_mute(img: np.ndarray, end: int, length: int) -> np.ndarray:
    """Tapered water mute along the depth axis (the last one).

    Zero above ``end - length``, a raised half-sine ramp on ``[end - length, end)``, and one from
    ``end`` on. This is the same mask the imager applies, so a muted image and the operator agree
    on where the water column is.
    """
    damp = np.zeros(img.shape[-1], np.float32)
    damp[end:] = 1.0
    damp[end - length:end] = (1.0 + np.sin((np.pi / 2.0 * np.arange(length)) / length)) / 2.0
    return (img * damp).astype(np.float32)


def build_pseudosynth(dm: np.ndarray, horizons: np.ndarray, lam: float,
                      sig_lat: float, rc_smooth: float = 1.0) -> np.ndarray:
    """Broadband reflectivity (N, N) from a Parihaka image and its tracked horizons.

    The reflection coefficient on each horizon, per column, is the original Parihaka amplitude at
    that horizon location, read before any lateral smoothing, and the wavelet is peak-normalized.
    So the broadband truth inherits the real column-wise horizon amplitudes -- deep reflectors
    stay as weak as they are in the field data, and the inverse problem stays as hard.

    Args:
        dm: (N, N) Parihaka image.
        horizons: (n_h, N) horizon depths, in fractional depth samples.
        lam: wavelet dominant wavelength, in depth samples.
        sig_lat: lateral smoothing of the horizon depth and amplitude, in columns.
        rc_smooth: unused placeholder kept for the dataset config's parameter set.

    Returns:
        (N, N) float32 reflectivity, amplitude-range matched to ``dm``.
    """
    N = dm.shape[0]
    ax = np.arange(N)
    refl = np.zeros((N, N), np.float32)
    for h in horizons:
        hi = np.clip(np.round(h).astype(int), 0, N - 1)
        amp = dm[ax, hi].astype(np.float32)
        di = h.astype(float)
        if sig_lat > 0:
            di = gaussian_filter1d(di, sig_lat, mode="nearest")
            amp = gaussian_filter1d(amp, sig_lat, mode="nearest")
        di = np.clip(di, 0, N - 1.001)
        d0 = np.floor(di).astype(int)
        fr = (di - d0).astype(np.float32)          # sub-pixel placement, so horizons are not jagged
        np.add.at(refl, (ax, d0), amp * (1 - fr))
        np.add.at(refl, (ax, d0 + 1), amp * fr)
    wav = ricker(lam)
    synth = np.stack([np.convolve(refl[j], wav, mode="same") for j in range(N)]).astype(np.float32)
    return (synth * (dm.std() / (synth.std() + 1e-9))).astype(np.float32)
