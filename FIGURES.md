# Reproducing the figures

Every figure renders from a cache in seconds on a CPU. The inputs download on first use, so a
fresh clone needs nothing but `pip install -e .`:

```bash
for f in scripts/fig_*.py; do python "$f"; done
python scripts/verify_closed_form.py
```

Output lands in `figures/`.

## Figure → script

| paper figure | script | reads |
|---|---|---|
| the seismic teaser | `fig_hero.py` | seismic DPS reconstructions, prior samples, illumination, κ, evaluation window |
| seismic spread vs illumination | `fig_seismic_crossover.py` | seismic DPS reconstructions, illumination |
| blind-mode marginals, both examples | `fig_marginals.py` | seismic prior samples + κ + evaluation window; groundwater dataset + both flows |
| reliability grid | `fig_reliability_grid.py` | seismic per-seed coordinates; groundwater pCN coverage, 3 seeds per arm |
| wave setup and causal shadow | `fig_wave_setup.py` | wave setup bundle |
| wave collapse vs roughness | `fig_wave_collapse.py` | wave deployed record; the two ε-control records |
| wave posterior across the shadow | `fig_wave_posterior.py` | wave evaluation set, both wave priors |
| groundwater single-survey posterior | `fig_darcy_posterior.py` | groundwater single-survey pCN record |
| groundwater operator setup | `fig_darcy_setup.py` | groundwater dataset |
| seismic operator and blind subspace | `fig_operator_schematic.py` | probe basis, illumination spectrum, evaluation window |
| seismic archive and samples | `fig_seismic_training.py` | seismic training gallery cache |
| training convergence, all six priors | `fig_training_convergence.py` | the six checkpoints |
| the three controls | `fig_control_panel.py` | closed form; seismic prior samples + κ; groundwater chain-length ladder |
| closed-form residual table | `verify_closed_form.py` | nothing — pure matrix identities |

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

### Semilinear wave

| # | stage | script | time | GPU |
|---|---|---|---|---|
| 1 | truths, surveys, and the MAP archive | `wave_make_dataset.py` | ~12 min | no |
| 2 | train one prior; run for both arms | `wave_prior_train.py` | ~21 min each | optional |
| 3 | deployed blind spread and coverage record | `wave_collapse_record.py` | seconds | optional |
| 4 | operator quantities for the setup figure | `wave_setup_record.py` | seconds | no |
| 5 | the ε control, run once per ε | `wave_eps_control.py --eps {40,0}` | ~20 min each | yes |

### Groundwater (Darcy flow)

| # | stage | script | time | GPU |
|---|---|---|---|---|
| 1 | truths, surveys, and the MAP archive | `darcy_make_dataset.py` | ~18 min | no |
| 2 | train both flows; run for seeds 0–2 | `darcy_flow_train.py --seed {0,1,2}` | ~1.5 min each | no |
| 3 | coverage over held-out surveys | `darcy_pcn.py --mode coverage --arm {oracle,curated} --seed {0,1,2}` | ~75 min each | no |
| 4 | chain-length ladder | `darcy_pcn.py --mode chainlength --arm {oracle,curated}` | ~2.7 h each | no |
| 5 | single-survey posterior fields | `darcy_pcn.py --mode single` | ~45 s per arm | no |

Nothing here needs a GPU or an external PDE solver: the Darcy forward map is a conservative
finite-difference assembly with an exact adjoint, checked against finite differences in
`tests/test_groundwater.py`.

Each mode's defaults are the configuration that produced the shipped caches, so the commands
above reproduce them without extra flags.
