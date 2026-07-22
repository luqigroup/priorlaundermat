"""Code for *Prior laundering: priors trained on legacy reconstructions inherit undetectable
overconfidence*.

Subpackages are one per example -- :mod:`priorlaundermat.seismic` (linearized Born),
:mod:`priorlaundermat.wave` (semilinear wave), :mod:`priorlaundermat.groundwater` (Darcy flow) --
and are deliberately not imported here, so that using one costs nothing from the others.
"""

__version__ = "1.0.0"
