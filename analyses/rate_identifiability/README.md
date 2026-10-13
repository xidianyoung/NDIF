# Rate identifiability on the recordings: scripts and stored outputs

These scripts measure how well a relaxation rate can be recovered at the geometry and noise level of each cohort
(the identifiability map), what short-time band power does to the autocorrelation of the state (phase-surrogate
share, calibrated white share, a single-timescale pipeline control) and how two rate estimators scale with the
feature window on the sleep recordings.

| Script | Output | Reported in |
|---|---|---|
| `sleep_lag_profile.py --workers 4 --calib-every 3 --windows 2.0 1.0 0.5` | `out/sleep_lag_profile.json`, `out/sleep_lag_profile_rows.csv` | per-stage white shares and per-recording rates of the 153 cassette and 44 telemetry recordings, read by the next two scripts |
| `lag_report.py` | `../../figure_reproduction/inputs/phase_surrogate_variance_by_stage.csv` | Fig. 3b: phase-surrogate to recorded variance ratios 0.36-0.57 across sleep stages at 1-s windows (bars); calibrated white share rising from 0.45 in wake to 0.83 in S4 (circles) |
| `window_scaling.py` | `../../figure_reproduction/inputs/window_scaling_exponent.json` | Fig. 3c and Supplementary Table 1: window-scaling exponent theta = 1.001 [0.978, 1.024] of the one-step rate and 0.493 [0.424, 0.566] of the six-lag calibrated rate between 2- and 0.5-s windows (78 participants); per-stage ratios lambda(1 s) / lambda(2 s) |
| `white_share_pipeline_control.py --reps 40 --n-blocks 12` | `../../figure_reproduction/inputs/white_share_control_simulation.json`, `out/white_share_control_rows.csv` | Fig. 3b shading: a synthetic single-timescale envelope processed like the recordings gives nu = 0.643 and exponentiality index 0.613 (arm `same_a08_step`, 1-s windows) |
| `identifiability_map.py --stage all --workers 13 --K 500` | `../../figure_reproduction/inputs/identifiability_map_grid.csv`, `identifiability_map_observed.csv`, `identifiability_map_contours.csv`, `identifiability_map_summary.json`; `out/identifiability_map_lever.csv`, `out/identifiability_map_obs.csv`, `out/identifiability_map_power.csv`, `out/identifiability_map_upstream.csv`, `out/identifiability_map_equivalence.json` (`.csv`) | Fig. 3a, Extended Data Fig. 1, Results and Methods: 2,304 configurations in the four displayed cohorts, 2,300 with usable fits; nominal coverage at 74 of 2,300, all with nu <= 0.2; COG-BCI interval width 0.53 -> 0.07 from 0.25 to 8 times the recording length; true-rate coverage zero at the observed points of COG-BCI, ds003838, Sleep-EDF and HCP fMRI; apparent-rate coverage 0.85 (COG-BCI) and 0.94 (Sleep-EDF) |

### Recovery calibration and tail excess

| Script | Output | Reported in |
|---|---|---|
| `../workload_certificates/features/cog_state.py --cboot 50` | `../workload_certificates/features/out/rescaling_cog_calibration.json` (and `rescaling_cog_subjects.csv`) | Fig. 3d (COG-BCI arm); COG-BCI observed point of the map |
| `injection_recovery_fmri.py --boot 150 --cboot 60` | `out/injection_recovery_fmri.json`, `_subjects.csv`, `_calibration.json` | Fig. 3d (HCP fMRI arm) |
| `injection_recovery_meg.py --boot 200 --cboot 80` | `out/injection_recovery_meg.json`, `_subjects.csv`, `_calibration.json` | Fig. 3d (HCP MEG arm) |
| `injection_recovery_ds003838.py --boot 200 --cboot 60` | `out/injection_recovery_ds003838.json`, `_subjects.csv`, `_calibration.json` | Fig. 3d (digit-span EEG arm) |
| `injection_recovery_report.py` | `../../figure_reproduction/inputs/injection_recovery.csv` | Fig. 3d: innovation-stream rescaling recovers 0.68-1.11 of the injected change in COG-BCI, HCP fMRI and HCP MEG and -0.007 to 0.58 in digit-span EEG; temporal decimation -0.157 to 0.663 |
| `rate_margin.py --cohort cog --draws 260 --boot 300` | `out/rate_margin_cog.json` | Supplementary Table 2 (static covariance reshaping control): log-trace error -0.0006 [-0.0022, 0.0010], apparent-rate shift 0.0009 [-0.0125, 0.0157] |
| `tail_excess_cog.py --nrep 200 --perm 2000`, `tail_excess_fmri.py --nrep 300 --perm 2000`, `tail_excess_meg.py --nrep 200`, `tail_excess_sleep.py --nrep 200 --prefix SC` | `out/tail_excess_<cohort>*` | per-cohort autocorrelation tail rows; observed points of the map |
| `instrument_white_share.py` | `out/band_white_share.json`, `out/band_white_share_rows.csv` | instrument-only white share (reference value of the map summary) |
| `tail_excess_ds003838.py --nrep 200 --perm 2000` | `../../figure_reproduction/inputs/tail_excess_ds003838.json` | Extended Data Fig. 7a (digit-span EEG, n = 50) |
| `tail_excess_contrast.py --perm 200` | `../../figure_reproduction/inputs/tail_excess_contrasts.json` | Extended Data Fig. 7a (MATB 27/15, n-back 15/10, HCP fMRI 95/90) |
| `tail_excess_calibration.py --cohort both` | `../../figure_reproduction/inputs/second_mode_calibration.json` | Extended Data Fig. 7b (thresholds 0.544 and 0.366) |

Recovery and tail-excess run order: the caches, then the four injection arms and `injection_recovery_report.py`;
`rate_margin.py`; `tail_excess_{cog,fmri,meg,sleep,ds003838}.py`, then `tail_excess_contrast.py` and
`tail_excess_calibration.py`; the identifiability map last.  Runtimes on one desktop: injection arms 1.5-10 min,
rate margin 10.5 min, tail excess 2-48 min per script.  These scripts run with the default multithreaded BLAS;
their permutation and surrogate draws are seeded per participant with Python's hash of the participant key.

## Run order

```bash
# sleep recordings (needs the sleep window cache, see Inputs)
python analyses/rate_identifiability/sleep_lag_profile.py        # about 3 min on 4 workers
python analyses/rate_identifiability/window_scaling.py           # seconds
python analyses/rate_identifiability/lag_report.py               # seconds
# synthetic, no data
python analyses/rate_identifiability/white_share_pipeline_control.py   # under 1 min
# identifiability map, last: the lever, obs and report stages read the per-recording rows of the
# tail-excess and injection-recovery scripts
python analyses/rate_identifiability/identifiability_map.py --stage equiv
python analyses/rate_identifiability/identifiability_map.py --stage grid --workers 13
python analyses/rate_identifiability/identifiability_map.py --stage lever
python analyses/rate_identifiability/identifiability_map.py --stage obs --workers 4
python analyses/rate_identifiability/identifiability_map.py --stage report
```

`--stage all` runs the five stages in this order.  `--cohorts` restricts the equiv, grid, lever and obs stages to
some of `COG-BCI ds003838 HCP_fMRI HCP_MEG SLEEP`; seeds are assigned over all five cohorts in this order, so a
restricted run draws the same random numbers as the full run for its cohorts.  The grid and lever stages append
every finished point to `out/identifiability_map_{grid,lever}_partial.jsonl`, resume from it and write every point
found there, so cohorts run in separate invocations end up in one grid; the obs stage writes the rows of the
cohorts it ran, so it is run for all five before the report stage, which always covers all five.
With `NDIF_OUT` set, every output (and every input written by another script) is taken below that directory.

## Inputs

- Sleep window cache, `analyses.common.paths.cache("sleep_states")` (`NDIF_SLEEP_STATES_CACHE`, default
  `analyses/_work/sleep_states`), written by `../sleep_stage_contrasts/window_cache.py` and
  `../sleep_stage_contrasts/noise_cache.py`; read by `sleep_lag_profile.py` through `lag_lib.py`:
  - `<stem>_win<w>.npz` for w = 2, 1, 0.5 and every cassette (`SC*`) and telemetry (`ST*`) recording:
    `X` (n_bout, n_win, 8) log10 band power of the used bouts (Fpz-Cz, Pz-Oz x delta, theta, alpha, beta;
    non-overlapping windows), `stage` (n_bout,) stage codes `Wake`, `N1`-`N4`, `REM`, `win_s`, `n_use`, `n_skip`;
  - `<stem>_noise.npz`: `"<stage>|<w>"` -> 8 x 8 covariance of the log band power of phase-randomised copies of
    the same bouts.
- Sleep-EDF directory (`NDIF_SLEEPEDF_RAW`): `ST-subjects.xls`, the temazepam/placebo assignment of the telemetry
  nights (drug contrast in `out/sleep_lag_profile.json`; read with xlrd).
- `identifiability_map.py` (stages lever, obs and report): the per-recording rows listed in `OBSERVED_ROWS` of the
  script, from which the observed point of each cohort is taken (medians with bootstrap intervals):
  calibrated single-timescale fits from `tail_excess_cog.py`, `tail_excess_ds003838.py` and `tail_excess_fmri.py`
  (`out/tail_excess_{cog,ds003838,fmri}_rows.csv`) and the corresponding rows of HCP MEG (trial clock, 0-back),
  Sleep-EDF (stage N2) and the COG-BCI instrument-only white share (`out/tail_excess_meg_rows.csv`,
  `out/tail_excess_sleep_rows.csv`, `out/band_white_share_rows.csv`); multivariate rates from
  `injection_recovery_{ds003838,fmri,meg}.py` (`out/injection_recovery_*_subjects.csv`) and from
  `../workload_certificates/features/cog_state.py --cboot 50` (COG-BCI).  The equiv, grid and report stages need
  no recordings.
- The equiv stage compares the vectorised bias correction with the loop implementation `fit_cell` / `fit_pairs` of
  `../workload_certificates/features/cog_fit_lib.py`.

## Runtimes and reproducibility

`sleep_lag_profile.py` takes about 3 min on 4 workers; `white_share_pipeline_control.py` under 1 min;
`window_scaling.py` and `lag_report.py` seconds.  The identifiability-map grid costs about one worker-minute per
configuration (2,880 configurations, about 48 worker-hours); the lever stage about one hour, obs about 16 min on 4
workers, equiv 10-15 min, report under one minute.

All random streams are seeded (20260922).  In `sleep_lag_profile.py` the injections use a per-job stream (CRC-32 of
the job key) and the group statistics one stream drawn in a fixed order, so the outputs do not depend on the
worker count.  In the map, a grid point whose simulation runs out of memory is retried with smaller simulation
chunks.  Grid rows are written in completion order.  The COG-BCI geometry has 28 participants (duplicate
exclusion).  The calibrated rate is `lam_calibrated` throughout (`window_scaling_exponent.json`:
`lam_calibrated`, `calibrated_ratio`, `calibrated_ci`); `p_second_timescale_rule` is the fraction of principal-
component profile slopes above the matched single-timescale 95th percentile.
