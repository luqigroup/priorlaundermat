# Reproducing the figures

Every figure renders from a cache in seconds on a CPU. The inputs download on first use, so a
fresh clone needs nothing but `pip install -e .`:

```bash
python scripts/fig_hero.py
python scripts/fig_reliability_grid.py
python scripts/fig_darcy_posterior.py
```

Output lands in `figures/`.

## Figure → script

| paper figure | script | reads |
|---|---|---|
| the seismic teaser | `fig_hero.py` | seismic DPS reconstructions, prior samples, illumination, κ, evaluation window |
| reliability grid | `fig_reliability_grid.py` | seismic per-seed coordinates; groundwater pCN coverage, 3 seeds per arm |
| groundwater single-survey posterior | `fig_darcy_posterior.py` | groundwater single-survey pCN record |

## Regenerating the caches

The stages below produce what the figures read. Each is independent of the others unless listed
as a dependency. Runtimes are on one modern GPU where marked, otherwise one CPU core.

### Seismic (linearized Born)

| # | stage | script | time | GPU |
|---|---|---|---|---|
| 1 | survey simulation and the least-squares migration archive | `seismic_make_dataset.py` | days | no (Devito) |
| 2 | probe basis and illumination spectrum | `seismic_probe_basis.py` | ~2 h | no (Devito) |
| 3 | train one prior; run for 2 arms × 3 seeds | `seismic_prior_train.py` | ~4 h each | yes |
| 4 | unconditional prior draws | `seismic_prior_sample.py` | ~40 min | yes |
| 5 | measurement-only amplitude calibration κ | `seismic_data_kappa.py` | ~5 h | no (Devito) |
| 6 | incident-wavefield illumination | `seismic_illumination.py` | ~10 min | no (Devito) |
| 7 | per-seed blind/resolved coverage coordinates | `seismic_calibrate.py` | ~6 h | yes |
| 8 | diffusion posterior sampling on the Born operator | `seismic_dps_posterior.py` | ~4 h | yes |

### Groundwater (Darcy flow)

| # | stage | script | time | GPU |
|---|---|---|---|---|
| 1 | truths, surveys, and the MAP archive | `darcy_make_dataset.py` | ~18 min | no |
| 2 | train both flows; run for seeds 0–2 | `darcy_flow_train.py --seed {0,1,2}` | ~1.5 min each | no |
| 3 | coverage over held-out surveys | `darcy_pcn.py --mode coverage --arm {oracle,curated} --seed {0,1,2}` | ~75 min each | no |
| 4 | single-survey posterior fields | `darcy_pcn.py --mode single` | ~45 s per arm | no |

Nothing here needs a GPU or an external PDE solver: the Darcy forward map is a conservative
finite-difference assembly with an exact adjoint, checked against finite differences in
`tests/test_groundwater.py`.

Each mode's defaults are the configuration that produced the shipped caches, so the commands
above reproduce them without extra flags.
