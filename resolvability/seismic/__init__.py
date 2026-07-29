"""The seismic example: a sparse linearized-Born survey over a Parihaka-derived reflectivity.

Modules are imported individually, not from here, so that reading the blind subspace costs no
torch and no Devito:

``blind``       the operator's blind / resolved / scrambled image subspaces (numpy only).
``pseudosynth`` the water mute and the broadband-reflectivity model (numpy only).
``priors``      the deployed DDPM priors: architecture, checkpoint lookup, sampling (torch).
``born``        the linearized-Born forward and its adjoint (Devito; regeneration only).

The grid and the acquisition below are fixed for the whole example -- one operator, one background,
one survey geometry -- and are declared here so no two modules can state them differently. Twelve
sources and twenty-four receivers on a shallow line is a deliberately sparse survey: it is what
leaves a blind subspace large enough to measure.
"""

N = 256                              # image side, cells
DX, DZ = 20.0, 12.5                  # cell size in metres, lateral and depth
NSRC, NREC = 12, 24                  # sources and receivers on the shallow line
TN_MS = 2000.0                       # record length, milliseconds
F0_KHZ = 0.030                       # Ricker peak frequency
SIGMA = 4000.0                       # record noise, RMS per sample
SPACE_ORDER, NBL = 16, 40            # finite-difference order, damping-boundary width in cells
WATER_ZERO_DEPTH = 10                # cells of water column
TAPER = 6                            # cells of half-sine ramp below the water column
MUTE_END = WATER_ZERO_DEPTH + TAPER  # first fully unmuted depth cell

N_TRAIN, N_EVAL = 4000, 600          # rows in the training and evaluation archives
DATASET_SEED = 0                     # each row's noise and shot order come from DATASET_SEED + row
