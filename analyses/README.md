# Analyses: scripts, stored outputs and the results they support

Each directory has a README that maps every script (with the arguments of the reported run) to its outputs and to
the figures, tables and numbers of the manuscript it supports, and states its inputs, environment variables, run
order and runtime.  The values plotted in the figures are the stored outputs listed there (`figure_reproduction/inputs/`
and `analyses/*/out/`); the Source Data give them per figure.

| Directory | Contents | Reported in |
|---|---|---|
| `common/` | `paths.py`: locations of the source data, intermediate caches and outputs (environment variables, repository-relative defaults) | — |
| `cohort_states/` | state spaces of the five recording cohorts (ds003838, eegmmidb, HCP MEG, HCP fMRI, Sleep-EDF), the per-trial HCP MEG cache and the count cache of the 646 recordings | composition of the 646 recordings (64 + 104 + 83 + 198 + 197; 449 awake; 153 cassette and 44 telemetry nights; Methods, *Human recordings*; Supplementary Fig. 2a); inputs of Fig. 2b,c,f, Fig. 4d and the digit-span analyses |
| `landscape_readouts/` | drift replacement, separation threshold, known-barrier fit and the five landscape readouts | Fig. 2b–f |
| `separation_floors/` | single-basin separation floors and division thresholds of four cohorts | Fig. 1f |
| `recurrent_networks/` | trained flip-flop networks (weights supplied), joint and separate scaling, escape simulation, known-barrier cells, drive–gain plane | Fig. 1d,e; Fig. 2d,e; Fig. 3e; Supplementary Fig. 1; Supplementary Table 14 |
| `theory_checks/` | Lyapunov identity and covariance invariance grid | Fig. 1c |
| `solver_validation/` | minimum-action solver against closed-form actions | Extended Data Fig. 5; solver values of the Methods |
| `rate_identifiability/` | identifiability maps, lag profile, white-share control, window scaling, injection recovery, static reshaping control, autocorrelation tail excess and second-mode calibration | Fig. 3a–d; Extended Data Figs 1 and 7; Supplementary Tables 1 and 2 (control) |
| `sleep_stage_contrasts/` | Sleep-EDF window and noise caches, stage contrasts, random-direction rates, observation-aware rate corrections, Benjamini–Hochberg adjustment | Fig. 4c; Fig. 5b,e; Supplementary Tables 8 and 11; Methods (1-s ratios, adjusted P values) |
| `sleep_telemetry/` | bout spread rows, temazepam–placebo crossover, drug-contrast power | Fig. 4c; Supplementary Tables 8 (alternative processing) and 9 |
| `task_location/` | digit-span (ds003838) position slopes and COG-BCI workload axes | Fig. 4a,b; Extended Data Fig. 4; Supplementary Table 7; Supplementary Note 6 |
| `primary_configuration/` | NDIF drivers and stored outputs under the primary and benchmark configurations | Fig. 3f; Fig. 4d (HCP MEG); Fig. 5c,d; Fig. 6a; Extended Data Figs 3 and 6; Supplementary Tables 2, 3 and 10; gate runtimes |
| `workload_certificates/` | COG-BCI state construction and assessments, the load-cohort driver, the COG-BCI band-power cache | Fig. 4d (COG-BCI); Supplementary Table 2 |
| `g8_configuration/` | NDIF drivers and stored outputs under the G8-augmented configuration, comparators, noise-aware bootstrap, matched comparison and selection tables | Extended Data Fig. 2; Supplementary Note 4; Supplementary Tables 4–6; Supplementary Fig. 3; the sleep re-evaluation |
| `sleep_feature_cache/` | the 1-s Sleep-EDF cassette caches read by the sleep drivers (written by `sleep_stage_contrasts/window_cache.py`) | inputs of Fig. 5c,d |

The non-neural series of Fig. 6 are assessed by `examples/example_03_cross_domain.py --write`.

## Run order

1. Caches from the source data: `cohort_states/` (state spaces, per-trial HCP MEG cache, count cache),
   `workload_certificates/features/cog_cache.py` (COG-BCI), `sleep_stage_contrasts/window_cache.py` and
   `noise_cache.py` (Sleep-EDF).
2. Analyses of the recordings: `landscape_readouts/`, `separation_floors/`, `rate_identifiability/`,
   `sleep_stage_contrasts/`, `sleep_telemetry/`, `task_location/`.
3. Trained networks: `recurrent_networks/` (a CUDA GPU is used when available).
4. NDIF assessments: `primary_configuration/`, `workload_certificates/`, `g8_configuration/`.

The source datasets are obtained from their providers (top-level README, *Data*) and located through the
environment variables of `common/paths.py`.  Intermediate caches are written to `analyses/_work/` unless their
variables name other directories.  Setting `NDIF_OUT` writes every output below another directory, so that a run can
be compared with the stored files without replacing them.

## Reproducibility

Every script records the settings of its run.  Re-running a script with the stored settings reproduces its
deterministic outputs exactly; the summary steps reproduce the stored summaries exactly from the stored per-record
tables.  Random streams are drawn per recording, participant or job (fixed seeds, or a hash of the record
name).  The scripts set the thread counts of the numerical libraries.  The simulations of the trained networks draw
from seeded torch generators on a CUDA GPU, the device of the stored runs.
