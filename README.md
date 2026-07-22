# Prior laundering

Code, data, and machine-checked proofs for

> **Prior laundering: priors trained on legacy reconstructions inherit undetectable overconfidence.**
> Ali Siahkoohi and Sina Alemohammad. Submitted to *SIAM/ASA Journal on Uncertainty Quantification*.

## Overview

Training a generative prior for an ill-posed inverse problem needs truths, which seismic and
medical imaging do not have. The recourse is an archive of legacy reconstructions. Where the
measurements are uninformative the posterior reverts to the prior, so the confidence reported
there is the archive's — and nothing in deployment reveals it: truths differing only on those
directions induce identical data laws, and self-consistency diagnostics pass whatever the prior
believes.

This repository reproduces every experiment in the paper:

| example | operator | prior | what it shows |
|---|---|---|---|
| closed-form | linear–Gaussian, exact | closed-form | the theory at machine precision |
| semilinear wave | nonlinear, causal shadow | DDPM | blind under-coverage, and the ε control |
| seismic | linearized Born, Parihaka | DDPM | under-coverage where the wavefield never reaches |
| groundwater | Darcy flow, 33 sensors | HINT flow | the same, with a single-best archive |

and ships the paper's recommendation as a standalone
[blind-subspace report card](#the-blind-subspace-report-card).

## Installation

```bash
git clone https://github.com/luqigroup/priorlaundermat
cd priorlaundermat
pip install -e .
```

Python 3.10+ with PyTorch. A GPU speeds up prior training and sampling but is not needed to
reproduce any figure from the released caches.

Regenerating the seismic dataset additionally needs [Devito](https://www.devitoproject.org)
(`pip install -e ".[seismic]"`); nothing else in the repository requires an external PDE solver.

## Data

Every dataset, checkpoint, and result cache is hosted publicly and **downloaded automatically on
first use** — each script calls `priorlaundermat.download.ensure` on what it needs, which is a
no-op once the file is on disk. There is nothing to fetch by hand.

| tier | size | what it is | needed for |
|---|---|---|---|
| `figures` | ~260 MB | result caches, the seismic probe basis and illumination spectra, the slim evaluation window, the groundwater and wave datasets | redrawing every figure |
| `checkpoints` | ~712 MB | the trained priors (3 seeds per arm for seismic and groundwater, 1 for wave) | re-running sampling |
| `datasets` | ~15 GB | the full seismic training and evaluation archives, the wave training set | retraining a prior |

To pre-fetch a tier instead of letting it stream in:

```python
from priorlaundermat.download import ensure_tier
ensure_tier("figures")
```

The full path→URL table is `priorlaundermat/download.py`.

## Reproducing the paper's figures

Every figure renders from a cache in seconds on a CPU:

```bash
python scripts/fig_hero.py                  # the seismic deployment
python scripts/verify_closed_form.py        # the closed-form checks, all nine
```

Output goes to `figures/`. See [FIGURES.md](FIGURES.md) for the figure→script→input table.

## Regenerating from scratch

Each example's pipeline is a chain of producer scripts; the figure scripts sit at the end. The
stage tables — inputs, outputs, runtimes, and which stages need a GPU — are in
[FIGURES.md](FIGURES.md).

## The blind-subspace report card

The paper's recommendation, shipped as code: from the forward operator **alone** — no prior, no
training archive, no data — name the directions along which a reported credible interval cannot
be checked against the measurements.

```bash
python scripts/blind_report_card.py
```

```python
from priorlaundermat.report_card import blind_report
card = blind_report(A)            # A: (m, n) dense forward operator
print(card.summary())
card.classify(V)                  # per-direction: resolved / mixed / blind
```

For operators too large for a dense SVD, pass a precomputed orthonormal blind basis:
`blind_report(n=n, N_blind=N_blind)`.

## Formal verification

`formal/` is a Lean 4 development that machine-checks the paper's theoretical results against
`mathlib`. Every theorem is kernel-verified with **no `sorry`**: `#print axioms` on each result
lists only Lean's three standard axioms (`propext`, `Classical.choice`, `Quot.sound`). Nothing
domain-specific is axiomatized — where an analytic step is not in `mathlib` it is either built
here or appears as an explicit hypothesis, never as an axiom.

```bash
cd formal
lake exe cache get     # prebuilt mathlib oleans for the pinned toolchain
lake build             # 8639 jobs
```

`formal/README.md` maps each paper result to the file that proves it, and states precisely what
is and is not covered.

## Tests

```bash
pytest tests/ -v
```

Fast, CPU-only, no downloads beyond the small inputs: the discrete adjoints against finite
differences, the blind/resolved bases orthonormal and mutually orthogonal, the flow's forward and
inverse mutually inverting with the change-of-variables log-density, and coverage hitting nominal
on exactly-Gaussian input.

## Conventions

- **Colours are semantic and fixed** (`priorlaundermat.style.PALETTE`): blue = oracle prior,
  red = curated prior, green = truth. They match the paper's own colour definitions.
- **Seismic sections are drawn at true physical proportions** (5.12 km × 3.2 km), never stretched
  to the panel box, and every quantitative map carries a colour bar.
- **Figures are vector PDFs** with embedded fonts, authored at the size they are printed at rather
  than scaled up from a larger canvas, and checked for label collisions at that size.
- **Seeds are explicit**: three training seeds for the seismic and groundwater priors, one for the
  wave example, seed 0 for the operators and evaluation sets. No ordering depends on file
  modification times or glob order.

## License

MIT — see [LICENSE](LICENSE).

## Contact

Ali Siahkoohi — <alisk@ucf.edu>
