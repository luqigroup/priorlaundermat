r"""The sparse linearized-Born imager: one shot's record :math:`J_s\,\delta m` and its adjoint.

Devito builds the acoustic model, the acquisition geometry and the Ricker source once for a fixed
background squared slowness; only the reflectivity and the source's lateral position change per
call, so no kernel is recompiled between shots. The forward and the adjoint are wrapped in a
``torch.autograd.Function``, which is what lets the posterior sampler differentiate the data
misfit straight through the wave solver.

The boundary pad is a zero pad, and that is load-bearing. The adjoint crops the ``nbl``-cell
damping layer, ``g[nbl:-nbl, nbl:-nbl]``, and cropping is the exact transpose of zero-padding but
not of replication-padding -- which would fabricate a boundary reflectivity slab and leave the
pair off by an order-one factor. Zero is also the physically right value: there is no reflectivity
inside an absorbing layer. :meth:`SparseBornImager.adjoint_dot_ratio` certifies the pair at
``~0.99999``; anything else means the operator is not a transpose pair and nothing downstream is
trustworthy.

Devito's threaded C kernels are not bit-reproducible across thread counts, so any run that must
reproduce an archived record sets ``OMP_NUM_THREADS=1`` before importing this module.

CPU only, and slow: one shot is one wave-equation solve.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import signal

from devito import configuration
from examples.seismic import AcquisitionGeometry, Model, RickerSource, TimeAxis
from examples.seismic.acoustic import AcousticWaveSolver

from . import DX, DZ, F0_KHZ, N, NBL, NREC, NSRC, SPACE_ORDER, TN_MS, WATER_ZERO_DEPTH

configuration["log-level"] = "WARNING"


class _ForwardBorn(torch.autograd.Function):
    """Linearized Born ``J dm`` for one source, with the migration as its backward.

    Forward: zero-pad the reflectivity into the damping layer, then one demigration solve.
    Backward: write the upstream residual into the geometry receiver, recompute the background
    wavefield, migrate, and crop the damping layer (the transpose of the forward's zero pad).

    The caller repositions the shared source before the forward and does not move it before the
    paired backward.
    """

    @staticmethod
    def forward(ctx, dm, pad, src, model, geometry, solver, device):
        ctx.model, ctx.geometry, ctx.solver = model, geometry, solver
        ctx.device, ctx.src = device, src
        padded = pad(dm).detach().cpu().numpy()
        d_lin = solver.jacobian(padded[0, 0, :, :], src=src)[0].data
        return torch.from_numpy(np.asarray(d_lin)).to(device)

    @staticmethod
    def backward(ctx, grad_output):
        rec = ctx.geometry.rec
        rec.data[:] = grad_output.detach().cpu().numpy()[:]
        u0 = ctx.solver.forward(src=ctx.src, save=True)[1]
        g = ctx.solver.jacobian_adjoint(rec, u0)[0].data
        nb = ctx.model.nbl
        g = torch.from_numpy(np.asarray(g[nb:-nb, nb:-nb])).to(ctx.device)
        return (g.view(1, 1, g.shape[0], g.shape[1]), None, None, None, None, None, None)


class SparseBornImager:
    r"""Build-once linearized-Born imager over a fixed background and a sparse acquisition.

    Args:
        m0: (nlat, ndep) background squared slowness in s^2/km^2; the background velocity is
            ``1/sqrt(m0)`` km/s. Depth is axis 1.
        spacing: (lateral, depth) grid spacing in metres.
        nsrc: sources on the shallow line.
        nrec: receivers on the shallow line.
        tn: record length in milliseconds.
        f0: Ricker peak frequency in kHz.
        space_order: spatial finite-difference order.
        nbl: width of the damping boundary, in cells.
        src_depth_cells, rec_depth_cells: source and receiver depth, in grid cells.
        device: torch device for the returned records.
    """

    def __init__(
        self,
        m0: np.ndarray,
        spacing: tuple[float, float],
        *,
        nsrc: int,
        nrec: int,
        tn: float,
        f0: float,
        space_order: int = 16,
        nbl: int = 40,
        src_depth_cells: float = 2.0,
        rec_depth_cells: float = 2.0,
        device: str = "cpu",
    ) -> None:
        m0 = np.ascontiguousarray(m0, dtype=np.float32)
        if m0.ndim != 2:
            raise ValueError(f"m0 must be 2-D (nlat, ndep); got {m0.shape!r}")
        if int(nsrc) < 1 or int(nrec) < 1:
            raise ValueError(f"nsrc and nrec must be >= 1; got {nsrc!r}, {nrec!r}")
        self.nlat, self.ndep = int(m0.shape[0]), int(m0.shape[1])
        self.spacing = (float(spacing[0]), float(spacing[1]))
        self.nsrc, self.nrec = int(nsrc), int(nrec)
        self.tn, self.f0 = float(tn), float(f0)
        self.space_order, self.nbl = int(space_order), int(nbl)
        self.device = torch.device(device)
        self._pad = torch.nn.ZeroPad2d(self.nbl)

        self.model = Model(
            space_order=self.space_order, vp=1.0 / np.sqrt(m0),
            origin=(0.0, 0.0), shape=(self.nlat, self.ndep), dtype=np.float32,
            spacing=self.spacing, nbl=self.nbl, bcs="damp",
        )
        dt = float(self.model.critical_dt)
        time_range = TimeAxis(start=0.0, stop=self.tn, step=dt)
        self.nt = int(time_range.num)

        rec_coordinates = np.empty((self.nrec, 2), dtype=np.float32)
        rec_coordinates[:, 0] = np.linspace(0.0, self.model.domain_size[0], num=self.nrec)
        rec_coordinates[:, 1] = rec_depth_cells * self.spacing[1]

        # One physical source at a time, repositioned per shot -- runtime data, so no recompile.
        self._src = RickerSource(name="src", grid=self.model.grid, f0=self.f0,
                                 time_range=time_range, npoint=1)
        self.wavelet = np.array(self._src.data[:, 0:1])
        self._src.coordinates.data[:, 0] = 0.5 * self.model.domain_size[0]
        self._src.coordinates.data[:, -1] = src_depth_cells * self.spacing[1]
        self._src_x = np.linspace(0.0, self.model.domain_size[0], num=self.nsrc)

        self.geometry = AcquisitionGeometry(
            self.model, rec_coordinates, self._src.coordinates.data,
            t0=0.0, tn=self.tn, src_type="Ricker", f0=self.f0,
        )
        self.solver = AcousticWaveSolver(self.model, self.geometry,
                                         space_order=self.space_order)

    @property
    def record_shape(self) -> tuple[int, int]:
        """Single-shot record shape ``(nt, nrec)``."""
        return (self.nt, self.nrec)

    def mute_op(self, dm: torch.Tensor, end: int = 10, length: int = 1) -> torch.Tensor:
        """Zero the water column: a raised half-sine ramp over ``[end - length, end)`` in depth.

        Diagonal and self-adjoint, so a misfit gradient taken through it needs no special care.
        """
        start = end - length
        damp = torch.zeros(dm.shape[-2], dm.shape[-1])
        damp[..., end:] = 1.0
        damp[..., start:end] = (
            1.0 + torch.sin((np.pi / 2.0 * torch.arange(0, length)) / length)) / 2.0
        return damp.to(dm) * dm

    def sample_noise(self, sigma: float, rng: np.random.Generator) -> np.ndarray:
        """Wavelet-coloured Gaussian record noise of exactly RMS ``sigma``, shape ``(nt, nrec)``.

        White noise convolved along time with the first 50 samples of the source wavelet, then
        renormalized -- so the noise has the bandwidth of the data, and its total energy is
        deterministic, which is what lets the calibration subtract it exactly. Drawn from the
        supplied generator, so a resumed row reproduces its record.
        """
        shape = (self.nt, self.nrec)
        kernel = self.wavelet[:50].reshape(-1, 1)
        e = sigma * rng.standard_normal(shape)
        e = signal.fftconvolve(e, kernel, mode="same")
        e *= 1.0 / np.linalg.norm(e)
        e *= np.sqrt(np.prod(e.shape)) * sigma
        return e

    def born_one(self, dm_1x1: torch.Tensor, src_idx: int) -> torch.Tensor:
        """One source's record ``J_s dm``, shape ``(nt, nrec)``, differentiable in ``dm_1x1``.

        Args:
            dm_1x1: (1, 1, nlat, ndep) reflectivity.
            src_idx: which source on the shallow line.
        """
        self._src.coordinates.data[:, 0] = self._src_x[src_idx]
        return _ForwardBorn.apply(dm_1x1, self._pad, self._src, self.model,
                                  self.geometry, self.solver, self.device)

    def adjoint_dot_ratio(self, seed: int = 0) -> float:
        r"""The transpose gate: :math:`\langle Jx, y\rangle / \langle x, J^\top y\rangle`, i.e. 1.

        Two solves. Run it before writing anything: a ratio away from 1 means the migration is not
        the transpose of the demigration, and every image, gradient and coverage number built on it
        is meaningless.
        """
        g_t = torch.Generator().manual_seed(int(seed))
        x = torch.randn(1, 1, self.nlat, self.ndep, generator=g_t,
                        dtype=torch.float32).requires_grad_(True)
        y = torch.randn(self.nt, self.nrec, generator=g_t, dtype=torch.float32)
        d = self.born_one(x, 0)
        jt_y = torch.autograd.grad(outputs=d, inputs=x, grad_outputs=y)[0]
        lhs = float(torch.dot(d.detach().reshape(-1), y.reshape(-1)))
        rhs = float(torch.dot(x.detach().reshape(-1), jt_y.reshape(-1)))
        return lhs / (rhs + 1e-30)


def parihaka_background_m0(nlat: int = 256, ndep: int = 256,
                           water_zero_depth: int = 10) -> np.ndarray:
    r"""The fixed background squared slowness: a water layer over a v(z) gradient.

    2.5 km/s scaled by a linear depth gradient to 4.5 km/s at the bottom, with the top
    ``water_zero_depth`` cells set to 1.5 km/s water. Returned as ``m0 = (1/v)^2`` in
    (lateral, depth) orientation. The same background for every image in the archive, so the
    operator is one fixed linear map.
    """
    v0 = np.ones((int(nlat), int(ndep)), dtype=np.float32) * 2.5
    v0 *= np.linspace(1.0, 4.5 / v0.max(), int(ndep)).astype(np.float32)
    s = 1.0 / v0
    s[:, : int(water_zero_depth)] = 1.0 / 1.5
    return (s ** 2).astype(np.float32)


def parihaka_imager() -> SparseBornImager:
    """The one operator every seismic script uses, at the acquisition declared in this subpackage.

    Building it takes a Devito compile, tens of seconds; every script that needs a forward solve
    builds it exactly this way, so the archived records, the illumination spectrum, the calibration
    and the posterior sampler all speak about the same linear map.
    """
    n = N
    m0 = parihaka_background_m0(nlat=n, ndep=n, water_zero_depth=WATER_ZERO_DEPTH)
    return SparseBornImager(m0, spacing=(DX, DZ), nsrc=NSRC, nrec=NREC, tn=TN_MS, f0=F0_KHZ,
                            space_order=SPACE_ORDER, nbl=NBL, device="cpu")
