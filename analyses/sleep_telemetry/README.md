# Telemetry temazepam/placebo crossover and the alternative 2-s spread processing

Directional-spread contrasts of adjacent sleep stages from 2-s windows, with every bout truncated to the same
60 s (`spread_lib`, *THE 2-S BOUT CHAIN*): in each arm of the Sleep-EDF telemetry temazepam/placebo crossover
(42 nights, 21 participants), the paired temazepam/placebo contrast per stage with its resampling power, and
the alternative-processing contrasts of the cassette nights.  The state space is the robust z + four-component
PCA of `../sleep_stage_contrasts/sleep_common.py`.  Stage codes `Wake`, `N1`-`N4`, `REM` stand for the R&K
stages Wake, S1-S4 and REM.

| Script | Output | Reported in |
|---|---|---|
| `bout_spread_rows.py --library cassette --workers 2`; `bout_spread_rows.py --library telemetry --workers 1` | `<rows>/sleep_alternative_processing_rows.json` (153 cassette nights), `<rows>/telemetry_crossover_rows.json` (44 telemetry nights; ST7161 and ST7162 have fewer than two usable stages) | the per-direction rows behind Supplementary Tables 8 (second part) and 9 and Fig. 4c |
| `crossover.py --nperm 5000 --nboot 4000` | `../../figure_reproduction/inputs/sleep_telemetry_crossover.json`, `out/alternative_processing.csv` | Fig. 4c (arms) and Supplementary Table 9: placebo S2/S3 1.178 [1.120, 1.314] (n = 16), temazepam 1.196 [1.071, 1.284] (15), placebo S3/S4 1.282 [1.157, 1.328] (7), temazepam 1.185 [1.145, 1.239] (8); 42 nights from 21 participants; S2 temazepam/placebo Wilcoxon P 0.5854.  Supplementary Table 8 (second part, alternative processing): 1.186 [1.055, 1.245] (n = 52), 1.000 [0.933, 1.062] (67), 1.180 [1.104, 1.272] (40), 1.111 [1.070, 1.249] (14); label-permutation P 0.0004, 0.9942, 0.0004, 0.0500.  Supplementary Fig. 2 |
| `drug_power.py` | `../../figure_reproduction/inputs/sleep_drug_contrast_bounds.json` | Fig. 4c and Supplementary Table 9: S2 temazepam/placebo spread 1.002 [0.929, 1.123] (n = 21); smallest folds with 80% correct-sign power 1.125 upward and 1.20 downward; power at a 1.20-fold change 0.9295 upward and 0.8033 downward |

`spread_lib.py` holds the shared code (EEG loading, bout features, spread and lag-1 estimators, cells,
participant contrasts, size-matched within-night label permutation, resampling power, and a minimal reader of
`ST-subjects.xls`); it writes nothing.

The alternative-processing contrasts use 129 of the 153 cassette nights; the 24 nights left out are listed in
`crossover.py` (`NOT_IN_ALTERNATIVE_SET`).  Stored outputs in this repository: the two figure inputs and
`out/alternative_processing.csv`.  The per-night rows are in the Source Data
(`analysis_inputs/telemetry/stored_outputs/`), under the file names that `bout_spread_rows.py` writes.

## Run order

```bash
python analyses/sleep_telemetry/bout_spread_rows.py --library cassette --workers 2    # about 11 min
python analyses/sleep_telemetry/bout_spread_rows.py --library telemetry --workers 1   # about 1.5 min
python analyses/sleep_telemetry/crossover.py --nperm 5000 --nboot 4000                # about 20 s
python analyses/sleep_telemetry/drug_power.py                                         # about 10 s
```

`crossover.py` and `drug_power.py` run without the EEG on the stored rows of the Source Data:

```bash
NDIF_SLEEP_BOUT_ROWS_CACHE=<Source Data>/analysis_inputs/telemetry/stored_outputs python analyses/sleep_telemetry/crossover.py
```

(`ST-subjects.xls` is read from `NDIF_SLEEPEDF_RAW`; the Source Data hold a copy under
`analysis_inputs/telemetry/assignment_source/`.)

## Inputs and environment variables

| Variable | Default | Content |
|---|---|---|
| `NDIF_SLEEPEDF_RAW` | `data/sleep-edf` | Sleep-EDF Expanded EDF files and hypnograms, and `ST-subjects.xls` (placebo and temazepam nights) |
| `NDIF_SLEEP_BOUT_ROWS_CACHE` | `analyses/_work/sleep_bout_rows` | the per-night rows (`<rows>` above) |
| `NDIF_OUT` | unset | if set, outputs are written below this directory instead of their stored locations |

At most two worker processes: each holds a whole night of EEG in memory.

## Reproducibility

All random streams have fixed seeds (per night 20260921 + crc32(name) % 100000; contrasts, permutations,
bias curve and power from 20260921 + a fixed offset), so the scripts are deterministic.  From the stored rows,
`crossover.py` and `drug_power.py` rewrite `sleep_telemetry_crossover.json`, `sleep_drug_contrast_bounds.json`
and `out/alternative_processing.csv` byte for byte.
