"""Groundwater flow: the elliptic Darcy inverse problem of Beskos et al. (2016).

A self-contained (numpy/scipy/torch) example of a prior curated from legacy reconstructions. The forward operator is a
conservative finite-difference Darcy solve with an exact adjoint; the prior over log-permeability
is Stuart's truncated Karhunen-Loeve expansion; the legacy archive is the single-best (MAP)
reconstruction under that prior; the learned priors are HINT normalizing flows over the KL
coefficients; and the posterior read is preconditioned Crank-Nicolson in the flow's latent.

Nothing here needs a GPU or an external PDE solver.
"""
from __future__ import annotations

from resolvability.groundwater.darcy import DarcyForward, beskos_sensors
from resolvability.groundwater.hint_flow import HINTFlow
from resolvability.groundwater.kl_prior import StuartKLPrior
from resolvability.groundwater.map_solver import map_reconstruct
from resolvability.groundwater.pcn import (LEVELS, DarcySetup, coverage, iat_sokal, load_flow,
                                             pcn_posterior, setup)

__all__ = ["LEVELS", "DarcyForward", "DarcySetup", "HINTFlow", "StuartKLPrior", "beskos_sensors",
           "coverage", "iat_sokal", "load_flow", "map_reconstruct", "pcn_posterior", "setup"]
