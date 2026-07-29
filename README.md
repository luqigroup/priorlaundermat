# resolvability

Code, data, and machine-checked proofs for

> **Priors learned from legacy reconstructions inherit undetectable overconfidence.**
> Ali Siahkoohi and Sina Alemohammad. Preprint, 2026.

## Overview

Training a generative prior for an ill-posed inverse problem needs truths, which seismic and
medical imaging do not have. The recourse is an archive of legacy reconstructions. Where the
measurements are uninformative the posterior reverts to the prior, so the confidence reported
there is the archive's — and nothing in deployment reveals it: truths differing only on those
directions induce identical data laws, and self-consistency diagnostics pass whatever the prior
believes.

This repository reproduces the paper's two examples and its theory:

| example | operator | prior | what it shows |
|---|---|---|---|
| seismic | linearized Born, Parihaka | DDPM | under-coverage where the wavefield never reaches |
| groundwater | Darcy flow, 33 sensors | HINT flow | the same, with a single-best archive |

The theoretical results are machine-checked in Lean 4 (see [Formal verification](#formal-verification)).

## Installation

```bash
git clone https://github.com/luqigroup/resolvability
cd resolvability
pip install -e .
```

Python 3.10+ with PyTorch. A GPU speeds up prior training and sampling but is not needed to
reproduce any figure from the released caches.

Regenerating the seismic dataset additionally needs [Devito](https://www.devitoproject.org)
(`pip install -e ".[seismic]"`); nothing else in the repository requires an external PDE solver.

## Data and checkpoints

The datasets and trained checkpoints each example needs are hosted publicly and **downloaded
automatically on first use** — each script calls `resolvability.download.ensure` on what it
needs, which is a no-op once the file is on disk. There is nothing to fetch by hand, and no
figures to download: the figures are produced by the scripts below.

To pre-fetch a tier instead of letting it stream in:

```python
from resolvability.download import ensure_tier
ensure_tier("checkpoints")     # the trained priors, three seeds per arm
ensure_tier("datasets")        # the full seismic training and evaluation archives
```

The full path→URL table is `resolvability/download.py`.

## Reproducing the paper's figures

Most figures render from a cache in seconds on a CPU; inputs download on first use. (`fig_seismic_training` draws fresh unconditional samples and is fastest on a GPU.)

```bash
python scripts/fig_hero.py               # figures/hero.pdf — the seismic teaser
python scripts/fig_emstep.py             # figures/ped_emstep.pdf — the one-EM-step schematic
python scripts/fig_reliability_grid.py   # figures/reliability_grid.pdf — coverage on both operators
python scripts/fig_darcy_posterior.py    # figures/fig_darcy_posterior.pdf — the groundwater posterior
python scripts/fig_seismic_training.py   # figures/fig_seismic_training.pdf — seismic prior samples vs training data
python scripts/fig_seismic_loss.py       # figures/fig_seismic_loss.pdf — seismic prior train/val loss
python scripts/fig_darcy_training.py     # figures/fig_darcy_training.pdf — groundwater flow samples vs training fields
python scripts/fig_darcy_loss.py         # figures/fig_darcy_loss.pdf — groundwater flow train/val NLL
```

Output goes to `figures/`.

## Regenerating from scratch

Each example is a chain of producer scripts, with the figure script at the end. The seismic
pipeline (`scripts/seismic_*.py`) covers survey simulation and the migration archive, the probe
basis and illumination spectrum, prior training and sampling, the measurement-only amplitude
calibration, and diffusion posterior sampling on the Born operator. The groundwater pipeline
(`scripts/darcy_*.py`) covers the dataset and its MAP archive, flow training, and pCN posterior
sampling. Each stage's defaults are the configuration that produced the shipped caches.

## Formal verification

`formal/` is a Lean 4 development that machine-checks the paper's theoretical results against
`mathlib`. Every theorem is kernel-verified with **no `sorry`**: `#print axioms` on each result
lists only Lean's three standard axioms (`propext`, `Classical.choice`, `Quot.sound`). Nothing
domain-specific is axiomatized — where an analytic step is not in `mathlib` it is either built
here or appears as an explicit hypothesis, never as an axiom.

```bash
cd formal
lake exe cache get     # prebuilt mathlib oleans for the pinned toolchain
lake build
```

`formal/README.md` maps each paper result to the file that proves it, and states precisely what
is and is not covered.

## Tests

```bash
pytest tests/ -v
```

Fast, CPU-only, no downloads: the discrete adjoints against finite differences, the blind/resolved
bases orthonormal and mutually orthogonal, and the flow's forward and inverse mutually inverting
with the change-of-variables log-density.

## License

MIT — see [LICENSE](LICENSE).

## Contact

Ali Siahkoohi — <alisk@ucf.edu>
