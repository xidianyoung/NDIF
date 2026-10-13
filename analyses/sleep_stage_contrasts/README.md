# Sleep-stage contrasts in the standardized four-dimensional sleep state

Directional spread and apparent relaxation rate of adjacent sleep stages in the 153 Sleep-EDF cassette nights
(78 participants), with 2-s and 1-s windows, and the observation-aware rate corrections.  Each night's eight
log band-power features (two EEG channels x four bands) are robust-standardized and reduced to four principal
components fitted on that night's pooled bouts (`sleep_common.state_space`).  Stage codes `Wake`, `N1`-`N4`,
`REM` stand for the R&K stages Wake, S1-S4 and REM.

| Script | Output | Reported in |
|---|---|---|
| `window_cache.py --library both --workers 4` | `<cache>/<stem>_win2.npz`, `_win1.npz`, `_win0.5.npz`, `<cache>/cache_meta.json` (197 recordings); the 153 cassette `<stem>_win1.npz` files are the files of `../sleep_feature_cache` | input of every sleep-stage analysis: Fig. 4c, Fig. 5b-e, Supplementary Tables 8, 10 and 11 (through `../sleep_feature_cache`), Fig. 3b,c (`../rate_identifiability`) |
| `noise_cache.py --workers 4 --draws 2` | `<cache>/<stem>_noise.npz` (instrument-noise covariance of phase-randomised copies of the same bouts) | the surrogate-subtraction rates of Fig. 5e and Supplementary Table 11; `../rate_identifiability` |
| `stage_contrasts.py --win 2.0 1.0 --workers 4` | `../../figure_reproduction/inputs/sleep_stage_contrasts.json`, `../../figure_reproduction/inputs/sleep_stage_cells.csv`, `out/stage_cell_covariances.npz`, `out/primary_contrasts_bh.json` | Fig. 4c and Supplementary Table 8 (first part): spread ratios Wake/S1 1.19 [1.11, 1.27] (n = 64), S1/S2 0.99 [0.93, 1.06] (76), S2/S3 1.20 [1.15, 1.28] (51), S3/S4 1.09 [1.03, 1.16] (20) with label-permutation P 0.0002, 0.60, 0.0002, 0.054 and the 1-s rows; Fig. 5b: apparent-rate ratios S1/Wake 1.25 [1.18, 1.33], S2/S1 1.24 [1.20, 1.28]; Methods: 1-s spread ratios 1.10, 0.96, 1.14, 1.09 and the Benjamini-Hochberg adjustment of the eight primary 2-s contrasts (four adjusted P <= 4 x 10^-4, S3/S4 spread 0.086); Supplementary Fig. 2 |
| `random_direction_rates.py --win 1.0 --workers 4` | `out/sleep_random_direction_two_lag.json`, `out/sleep_random_direction_two_lag_rows.csv` | random-direction two-lag rows of Fig. 5e and Supplementary Table 11: S1/Wake 1.446 [1.333, 1.663] (n = 57), S2/S1 1.445 [1.301, 1.654] (n = 76) |
| `rate_corrections.py` | `../../figure_reproduction/inputs/sleep_rate_corrections.csv` | Fig. 5e and Supplementary Table 11: two-lag IV 0.938 / 1.262, surrogate subtraction 1.182 / 1.611, random directions 1.446 / 1.445, six-lag grid 1.474 / 1.365 (S1/Wake / S2/S1) |

`sleep_common.py` holds the shared code (state space, VAR(1), bias-corrected, lag-2 instrument and
noise-corrected fits, participant-level summaries); it writes nothing.

Stored outputs in this repository: the three figure inputs, `out/primary_contrasts_bh.json` (computed from the
stored contrast P values) and `out/sleep_random_direction_two_lag.json`.  The per-direction rows of the random
directions are in the Source Data (`analysis_inputs/sleep/sleep_random_direction_two_lag_rows.csv`); the cell
covariances are not stored.

## Run order

```bash
python analyses/sleep_stage_contrasts/window_cache.py --library both --workers 4    # about 1 min
python analyses/sleep_stage_contrasts/noise_cache.py --workers 4 --draws 2          # about 1.2 min
python analyses/sleep_stage_contrasts/stage_contrasts.py --win 2.0 1.0 --workers 4  # about 5 min
python analyses/sleep_stage_contrasts/random_direction_rates.py --win 1.0 --workers 4   # about 1 min
python analyses/rate_identifiability/sleep_lag_profile.py                             # six-lag rows
python analyses/sleep_stage_contrasts/rate_corrections.py                            # under 1 s
```

## Inputs and environment variables

| Variable | Default | Content |
|---|---|---|
| `NDIF_SLEEPEDF_RAW` | `data/sleep-edf` | Sleep-EDF Expanded: `SC*-PSG.edf`, `ST*-PSG.edf` and their `*-Hypnogram.edf` |
| `NDIF_SLEEP_STATES_CACHE` | `analyses/_work/sleep_states` | the window and noise cache (`<cache>` above); also read by `../rate_identifiability` |
| `NDIF_OUT` | unset | if set, outputs are written below this directory instead of their stored locations |

Runtimes are for one desktop with four worker processes.

## Reproducibility

* The window cache is deterministic.
* `noise_cache.py`, the bootstrap-corrected rates of `stage_contrasts.py` (`_bc`, `_ncbc`) and the random
  directions of `random_direction_rates.py` draw from one generator per recording seeded with Python's hash of
  the recording name.  The group summaries run over the cells in the order in which
  the worker processes return them.
* The summaries are reproduced exactly from the stored tables, in their stored row order:
  `stage_contrasts.py --cells figure_reproduction/inputs/sleep_stage_cells.csv` rewrites
  `sleep_stage_contrasts.json` (and `out/primary_contrasts_bh.json`) byte for byte, and
  `random_direction_rates.py --rows <Source Data>/analysis_inputs/sleep/sleep_random_direction_two_lag_rows.csv`
  rewrites `out/sleep_random_direction_two_lag.json` byte for byte.  `rate_corrections.py` rewrites
  `sleep_rate_corrections.csv` byte for byte from the stored inputs.
* Benjamini-Hochberg: the family is the four 2-s spread contrasts (label-permutation P, at least
  1/(5,000 + 1)) and the four 2-s apparent-rate contrasts (Wilcoxon P, entered at the reporting bound 10^-4
  when smaller); the adjustment of the unbounded P values is recorded beside it.
