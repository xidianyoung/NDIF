# NDIF — Neural Dynamics Identifiability Framework

Software and stored analyses for the manuscript *Assessing dynamical identifiability in neural recordings*.  NDIF takes a segmented state series, its preprocessing metadata and an estimator,
and reports, for each claim class, whether a claim is admitted, refused or indeterminate, the first failed gate, the
gate margins and a model-conditional detection threshold.  Claims are of three classes: state geometry (L1), state variability (L2) and
intrinsic dynamics (L3).  Admission means that the configured gates pass; it is not an accuracy certificate.

This repository is Supplementary Software 1 of the manuscript: the `ndif` package (gates G0–G8, the two assessed
estimators, the five comparison methods, the calibration generators and the minimum-action solver); the drivers,
settings and stored outputs of the assessments under the three analysis configurations (Sleep-EDF, HCP MEG, COG-BCI,
the non-neural series, the known-rate benchmark and the extended simulation evaluation); the analysis scripts behind
the other reported results (state construction of the five recording cohorts, landscape readouts, separation floors,
trained recurrent networks, rate-identifiability diagnostics, sleep-stage and telemetry contrasts, task location,
solver validation) with their stored outputs; and the stored summaries from which the figures are drawn.  The analyses start from the published, preprocessed recordings (the ds003838 derivative and the HCP fMRI
parcel series were produced by the preprocessing described in the Methods, which is not part of this repository).

## System requirements

* **Package (`ndif`)**: Python 3.9 or later with `numpy >= 1.21` and `scipy >= 1.7`; pure Python, no compiled
  extensions and no platform-specific code.  Tested with Python 3.13.9, NumPy 2.4.4 and SciPy 1.17.1 on Windows 11
  Pro (x86-64), the environment of every reported analysis.
* **Analyses and figures**: the pinned versions in `requirements.txt` (package, tests, non-neural series, COG-BCI
  state construction, figures) and `requirements-analyses.txt` (the additional libraries of the analyses of the
  recordings and of the trained networks).
* **Hardware**: a desktop computer.  No special hardware is needed for the package, the tests, the examples or the
  figures.  Long analyses use several worker processes (set by their `--workers` option); the analyses of the trained
  recurrent networks use a CUDA GPU when one is available and otherwise run on the CPU.

## Installation guide

```bash
pip install -e .                       # the package (under one minute)
pip install -e ".[test]"               # + pytest
pip install -r requirements.txt        # the versions used for the reported analyses and figures (a few minutes)
pip install -r requirements-analyses.txt   # only for re-running the analyses of the recordings and networks
python -m pytest tests -q              # the test suite (several minutes on one core)
```

## Demo

```bash
python examples/example_01_quickstart.py
```

The demo needs no data.  It prints a certificate for a simulated two-condition recording (6,000 samples, 3
dimensions, 40 segments; L1, L2 and L3 admitted, with the margin of every gate) and the minimum-action barrier of a
double well with a curved instanton (recovered 5.00009 against the exact 5.00000, relative error about 2 × 10⁻⁵).  It runs in about 15 s on a
desktop computer.  `examples/example_02_discrimination.py` (five simulated regimes), `example_03_cross_domain.py`
(published non-neural series, Fig. 6; a few minutes) and `example_04_corpus.py` (a set of recordings) show further
uses.

## Instructions for use

```python
import numpy as np
from ndif import certify, recommend_estimator, Design

cert = certify(X, Design(dt=1.0, window_s=1.0, step_s=1.0, segments=bounds, labels=cond, name="sub-01"))
print(cert)                                   # the text report
cert.levels["L3"].status                      # 'admitted' | 'refused' | 'indeterminate'
cert.levels["L3"].binding_gate                # the first failed gate, e.g. 'G1'
cert.levels["L3"].detection_margin            # the fold change that would have been detected (80% power)
cert.to_json("sub-01.json")

certify(X, design, estimator="kalman")        # the observation-aware maximum-likelihood estimator
certify(X, design, configuration="g8_augmented")
recommend_estimator(X, design)                # the least costly estimator admitted at L3, if any
```

`X` has one row per state (window) and one column per dimension; `segments` gives `(start, stop)` pairs so that no
lagged pair straddles a segment boundary; `labels` (one per row) are needed for a two-condition L1 contrast.  The
command-line entry point `ndif-certify` reads a `.npy`/`.npz` file.  `docs/PACKAGE_USAGE.md` documents the API, the
nine gates and their costs; `docs/GATE_SPECIFICATION.md` gives each gate as pseudocode.  Per-gate settings are passed
as keyword dictionaries, e.g. `certify(X, design, G3=dict(n_rep=64))`.  The reported analyses are reproduced as
described under *Reproducing the reported results* below.

## Analysis configurations

The framework is one gate set run in three configurations (Methods, *Analysis configurations*;
`ndif/configurations.py`).  `certify(..., configuration=...)` selects which gates run by default, which gates govern
each claim class and in what order, and which form G1, G3 and G5 take.

| Configuration | Gates | Governing order for L3 | G1 / G3 / G5 | Used for |
|---|---|---|---|---|
| `primary` (default) | G0–G7 | G0, G1, G2, G3, G7, G4 | two-lag white share ≤ 0.20 / fitted generator / simulated zero-truth floor | 1,204 sleep cell assessments and 336 adjacent-stage pairs; 83 HCP MEG; 56 COG-BCI; Nile, sunspots, unemployment |
| `benchmark` | G0, G1, G2, G3, G7 | G0, G1, G2, G3, G7 (G4 not run) | as primary | 320 estimator outputs with known rates |
| `g8_augmented` | G0–G3, G5, G6, G8 (G4, G7 diagnostics) | G0, G2, G8, G1, G3 | profile-likelihood rate bound / indirect-inference-corrected generator, criteria beyond Monte Carlo error / permutation null | extended simulation evaluation (404 recordings, 808 outputs), misspecification and sleep-matched sets, calendar-preserving CO₂ and the non-neural series, re-evaluation of the 602 sleep cells |

L1 is governed by G6 then G5 and L2 by G0 then the variance channel of G3 in every configuration.

## Layout

| Path | Contents |
|---|---|
| `ndif/` | the package: `certify.py` (entry points), `configurations.py`, `gates.py` (G0–G8), `estimators.py`, `simulate.py` (calibration generators and injections), `baselines.py` (the five comparison methods), `action.py` (minimum-action solver), `report.py`, `cli.py` |
| `tests/` | closed-form recoveries (covariance identity below 10⁻¹²), gate behaviour on injected changes, the recovery check at the noise boundary, the G8 adequacy check, the governing rules |
| `examples/` | runnable examples: quick start, five simulated regimes, the non-neural series (Fig. 6; `--write` writes its stored inputs), the sleep corpus |
| `docs/` | `PACKAGE_USAGE.md` (API, gates, costs, limitations) and `GATE_SPECIFICATION.md` (each gate as pseudocode) |
| `analyses/` | index of the analyses (`analyses/README.md`): scripts, stored outputs and the results they support, run order, environment variables (`analyses/common/paths.py`) |
| `analyses/cohort_states/`, `landscape_readouts/`, `separation_floors/`, `recurrent_networks/`, `theory_checks/`, `rate_identifiability/`, `sleep_stage_contrasts/`, `sleep_telemetry/`, `task_location/` | analysis scripts and stored outputs behind Figs 1–5, Extended Data Figs 1, 4 and 7, Supplementary Figs 1 and 2 and Supplementary Tables 1, 2, 7–9, 11 and 14 (trained network weights in `recurrent_networks/weights/`) |
| `analyses/primary_configuration/` | drivers and stored outputs of the primary and benchmark configurations: the 320-output benchmark, the per-gate cost table, the 602 sleep cells × 2 estimators and 336 pairs, the 83 HCP MEG assessments |
| `analyses/workload_certificates/` | the 56 COG-BCI workload assessments after the duplicate exclusion, the load-cohort driver and the COG-BCI state construction |
| `analyses/sleep_feature_cache/` | the 153 Sleep-EDF feature caches read by the sleep drivers |
| `analyses/solver_validation/` | validation of the minimum-action solver against closed-form actions (Extended Data Fig. 5) |
| `analyses/g8_configuration/` | drivers, analyses and stored outputs of the G8-augmented configuration, the parametric-bootstrap comparators, the noise-aware bootstrap ranking, the matched comparison and the selection tables |
| `figure_reproduction/inputs/` | the stored summaries plotted in Figs 2–6, Extended Data Figs 1, 4 and 7 and Supplementary Fig. 2, with the analysis script that writes each (`figure_reproduction/README.md`) |
| `verification/recount_stored_outputs.py` | recounts the principal reported counts from the stored outputs and compares them with the manuscript |
| `package_file_manifest.json` | SHA-256 of every file in the repository |

Each `analyses/*/README.md` maps scripts to outputs and to the figures and tables they support.  Stored sleep outputs
use the stage codes `Wake`, `N1`–`N4`, `REM` for the R&K stages Wake, S1–S4, REM of the manuscript.

## Reproducing the reported results

```bash
python verification/recount_stored_outputs.py          # the principal reported counts from the stored outputs; no fitting
python analyses/g8_configuration/benchmark_09_analyse.py   # re-tabulates the extended evaluation from its jsonl outputs
python analyses/g8_configuration/benchmark_10_analyse.py   # re-tabulates the sleep re-evaluation and the CO2 segmentations
python analyses/solver_validation/solver_validation.py     # every value of Extended Data Fig. 5 (about three minutes)
python examples/example_03_cross_domain.py --write         # Fig. 6: the non-neural series and the CO2 sweep (about two minutes)
python analyses/primary_configuration/summarize_corpus_full.py          # sleep summaries and detection thresholds (Fig. 5c,d)
python analyses/g8_configuration/matched_comparison/make_matched_comparison.py   # matched comparison (Supplementary Fig. 3)
python analyses/g8_configuration/tabulate_selection.py     # selection tables and sensitivity (Supplementary Note 4)
```

These commands need no source data and reproduce the values of their stored outputs exactly.  Every other reported analysis
is re-run from the source data with the scripts listed in `analyses/README.md`, in the run order given there; the
summary steps of those analyses also re-run on the stored per-record tables.  The Sleep-EDF 1-s caches are supplied
(`analyses/sleep_feature_cache`); the other caches are rebuilt from the source data by the cache builders
(`analyses/cohort_states/`, `analyses/workload_certificates/features/cog_cache.py`,
`analyses/sleep_stage_contrasts/window_cache.py`), because data derived from the HCP and COG-BCI recordings are not
redistributed.  Locations are set with the environment variables of `analyses/common/paths.py`; with `NDIF_OUT` set,
outputs are written below another directory, so that a run can be compared with the stored files.  Stored outputs
record the settings that produced them; long drivers append to their output files and can be interrupted and resumed.

## Data

The analysed recordings are published datasets: OpenNeuro ds003838, the COG-BCI database (Zenodo), the Human
Connectome Project working-memory fMRI and MEG data, Sleep-EDF Expanded and the PhysioNet EEG Motor Movement/Imagery
Dataset; the non-neural series are the `nile`, `sunspots`, `co2` and `macrodata` datasets of statsmodels.  Accession
identifiers and access conditions are in the manuscript's Data availability statement.  Figure source data accompany
the submission as Source Data.

## Licence and citation

The code is released under the MIT License (`LICENSE`); `NOTICE.md` states its scope, the terms of the datasets and
the required acknowledgements.  Citation metadata are in `CITATION.cff`.
