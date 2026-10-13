# Landscape readouts of the recordings (Fig. 2b-f)

The equilibrium count of the mixture drift field on the 646 recordings (64 digit-span EEG, 104 motor EEG, 83 MEG,
198 fMRI runs of 99 participants, 197 sleep nights), recomputed with the dynamics replaced, against a single
separation threshold and against label-free readouts; and the density barrier of the known-barrier cells.

| Script | Output | Reported in |
|---|---|---|
| `count_recount.py --cohort all --workers 6` | one row per record in `paths.cache("count_rows")/<cohort>/<record>.json` (`NDIF_COUNT_ROWS_CACHE`, default `analyses/_work/count_rows`) | the recount of every record with the condition labels and with label-free assignments (Gaussian mixture with K fixed, segment- or sample-level; K by BIC; Gaussian HMM; permuted labels; one-dimensional Kramers-Moyal drift); input of `readout_table.py` |
| `ds003838_pooled_readouts.py` | `out/ds003838_pooled_readouts.json` | Fig. 2f, Gaussian mixture (BIC) 1 × 41, 2 × 17, 3 × 4, 5 × 2 and pairwise maximum entropy 2 × 7, 4 × 25, 6 × 20, 8 × 10, 10 × 2 on the 64 digit-span recordings |
| `readout_table.py --part all --workers 4 --cap 10000` | `out/sleep_ladder_readouts.json`; `../../figure_reproduction/inputs/landscape_readout_counts.csv` | Fig. 2f, five readouts on the 64 digit-span recordings: drift field with labels 1 × 64, labels hidden 1 × 62 and 2 × 2, model-free 1-D drift 1 × 63 and 2 × 1, and the two readouts above; separations and reference counts of all 646 records for the threshold of Fig. 2c |
| `drift_replacement.py --workers 4 --cohorts all` | `../../figure_reproduction/inputs/drift_replacement_records.csv`, `drift_replacement_summary.json`, `separation_threshold_summary.json`, `density_barrier_summary.json`; console log `out/drift_replacement.log` | Fig. 2b: with every drift matrix replaced by −I or by random stable matrices the count is unchanged in 445 of 449 awake recordings, a constant count of one agrees in 446, sleep 165 and 131 of 197; Fig. 2c: threshold 3.35 → 623 of 646, 2.31 → 557, constant one 607 (446 + 161, counted by `make_Fig2.py`), cohort median separations 0.49, 0.46, 0.48, 0.88 against 2.54 in sleep; Fig. 2d: density barrier = 0.1234 × separation² through the origin, R² 0.9994, n = 256; Fig. 2e (left): 256 of 256 one-well cells read exactly zero although their true barriers are 1.86-21.71 nats; recording composition 646 = 64 + 104 + 83 + 198 + 197 (Methods; Supplementary Fig. 2a) |

`landscape_lib.py` holds the functions shared by these scripts and by `analyses/separation_floors` (ridgeline mode
counting, the regime construction of the count, the Gaussian HMM, the Kramers-Moyal drift and the two pooled readouts);
the reference state space, the VAR(1) regime fit and the equilibrium search are imported from
`analyses/cohort_states/states_lib.py`. `elapy_functions.py` holds the seven functions of elapy
(https://github.com/okumakito/elapy; Apache License 2.0, `LICENSE_elapy.txt`) used by the pairwise maximum-entropy
readout, unchanged.

## Run order

```bash
# caches: analyses/cohort_states/ds003838_states.py, hcp_meg_states.py, count_cache.py (see that README)
python analyses/landscape_readouts/count_recount.py --cohort all --workers 6        # about 25 min per cohort group
python analyses/landscape_readouts/ds003838_pooled_readouts.py                      # about 6 min
python analyses/landscape_readouts/readout_table.py --part all --workers 4 --cap 10000   # sleep readout ~15 min
python analyses/landscape_readouts/drift_replacement.py --workers 4 --cohorts all   # about 11 min (C2-C4 < 1 s)
```

`drift_replacement.py` part C3/C4 reads `figure_reproduction/inputs/density_barrier_cells.csv` (the known-barrier cells
of `analyses/recurrent_networks`); `--skip-c1` runs C2-C4 alone on the stored tables.

## Inputs

| Input | Location |
|---|---|
| count cache (state series, segments, condition labels of the 646 records) | `paths.cache("count")`, `NDIF_COUNT_CACHE`; built by `analyses/cohort_states/count_cache.py` |
| digit-span feature cache | `paths.cache("ds003838")`, `NDIF_DS003838_CACHE`; built by `analyses/cohort_states/ds003838_states.py` |
| reference count per record | `analyses/cohort_states/reference_pipeline_counts.csv` (`readout_table.py --reference-counts`) |
| Gaussian-mixture and maximum-entropy readouts of the motor EEG, MEG and fMRI records and the maximum-entropy readout of the nights | optional JSON `{cohort: {record: {"ladder_basins", "ising_basins"}}}` (`readout_table.py --reference-readouts`); without it these cells of the table are empty. Fig. 2f uses the digit-span rows only |

Set `NDIF_OUT` to write every output below another directory instead of its stored location.

## Notes

- The random-stable arm of `drift_replacement.py` draws from one generator per record, seeded with Python's hash of
  (cohort, record).
- The scripts pin BLAS to one thread (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`), as in the runs
  behind the stored outputs.
- The fMRI separation of Fig. 2c (0.88) is the median over the 198 runs; Fig. 1f (`analyses/separation_floors`) reports
  the median over the 99 participants (0.86). The motor-EEG cohort has 104 records here (the eligibility rule of the
  count cache) and 106 participants in `analyses/separation_floors`.
- Dependencies beyond the package: scikit-learn, pandas, networkx (elapy functions).
