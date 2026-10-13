# Single-basin separation floors of the cohorts (Fig. 1f)

How large a separation between the two condition regimes the equilibrium count reports when the true separation is
zero, at each awake cohort's own geometry, and where a basin divides in that geometry.

| Script | Output | Reported in |
|---|---|---|
| `extract_generators.py --cohort hcp_fmri hcp_meg ds003838 eegmmidb --workers 8` | `paths.cache("separation_floors")/generators/<cohort>.pkl` and `<cohort>_summary.json` (`NDIF_SEPARATION_FLOORS_CACHE`, default `analyses/_work/separation_floors`) | the observed separations of Fig. 1f (per record, two-regime generator parameters of 198 fMRI runs of 99 participants, 83 MEG, 64 digit-span and 106 motor-EEG participants) |
| `floor_benchmark.py --reps 60 --workers 14` | `out/floor_benchmark.json`, `out/floor_benchmark.csv`, `out/floor_benchmark.log`; replicate rows `paths.cache("separation_floors")/floor_benchmark_rows.pkl` | the simulation benchmark: 4 cohorts × 7 generators × 12 separations × 60 participants |
| `floor_report.py --n-boot 4000` | `out/separation_floor_summary.json`, `out/separation_floor_ladder.csv`, `out/separation_floor_report.txt` | Fig. 1f: zero-separation floors 0.571 [0.515, 0.720] (HCP fMRI), 0.225 [0.210, 0.249] (HCP MEG), 0.178 [0.164, 0.197] (digit-span EEG), 0.173 [0.159, 0.193] (motor EEG), 95th percentiles 1.185, 0.298, 0.282, 0.267; observed participant medians 0.86 [0.42-1.98], 0.48 [0.18-1.07], 0.49 [0.21-1.01], 0.46 [0.14-1.13]; 8 of 99, 4 of 83, 0 of 64 and 1 of 106 participants at or below the floor; generators at d / dt / median segments per regime 7 / 0.72 s / (4, 4), 8 / 1 s / (150, 149), 8 / 1 s / (54, 108), 8 / 1 s / (180, 180) (`/cohorts/<cohort>/rungs/nonstationarity/{floor, floor_p95, bootstrap/floor_ci}`, `/cohorts/<cohort>/observed_separation`) |
| `division_threshold.py --workers 12` | `out/division_threshold.json`, `out/division_threshold.csv` | Fig. 1f, separation at which a basin divides in each cohort's geometry: participant medians 2.34 (HCP fMRI), 2.58 (HCP MEG), 2.76 (digit-span EEG), 2.33 (motor EEG) (`/cohorts/<cohort>/delta_div_real_subject_median`) |

The generators add one ingredient at a time to an isotropic reference (d = 8, Σ = I, relaxation rate 0.5 s⁻¹, one
contiguous segment per regime): the cohort's dimension and sampling interval, its anisotropic regime covariances, its
fitted one-step maps, its number of windows per regime, its segment structure and its measured between-segment
non-stationarity. The last generator (`nonstationarity`) is matched to the cohort; its floor is the floor of Fig. 1f.
The stored files in `out/` are the outputs of this chain for the run behind Fig. 1f (`floor_report.py` and
`division_threshold.py` applied to that run's replicate rows and generators reproduce them exactly).

## Run order

```bash
python analyses/separation_floors/extract_generators.py --workers 8      # a few minutes per cohort
python analyses/separation_floors/floor_benchmark.py --reps 60 --workers 14   # about 34 min on 14 workers
python analyses/separation_floors/floor_report.py --n-boot 4000          # about 1 min
python analyses/separation_floors/division_threshold.py --workers 12     # about 2 min
```

## Inputs

| Input | Location |
|---|---|
| HCP fMRI parcel series and the EV files of each run | `NDIF_HCP_FMRI_PARCELS`, layout of `analyses/cohort_states/hcp_fmri_states.py` (`<subject>__<run>__atlas_fpn7.nii.npy`, `<subject>__<run>__EVs/`) |
| HCP MEG per-trial state cache | `paths.cache("meg")`, `NDIF_MEG_CACHE`; built by `analyses/cohort_states/hcp_meg_states.py` |
| digit-span feature cache | `paths.cache("ds003838")`, `NDIF_DS003838_CACHE`; built by `analyses/cohort_states/ds003838_states.py` |
| PhysioNet EEG Motor Movement/Imagery recordings | `NDIF_EEGMMIDB_RAW` (read with `analyses/cohort_states/eegmmidb_states.py`) |

The state construction of every cohort is that of `analyses/cohort_states`; the regime fit, the equilibrium count and
the ridgeline mode count are those of `analyses/landscape_readouts/landscape_lib.py`. Set `NDIF_OUT` to write the
outputs below another directory instead of `out/`.

## Notes

- Seeds: participant draw `default_rng(20260921 + crc32(cohort) % 10000)` (60 distinct participants; one run per
  participant for fMRI); replicate `20260921 + 7919 (generator index + 1) + 104729 round(1000 δ) + 13 rep +
  crc32(cohort) % 1000`; equilibrium search `20260921 + rep`; participant bootstrap `424242 + generator index`.
- The observed fMRI separation here is the median over participants (0.86); Fig. 2c reports the median over the 198
  runs (0.88, `analyses/landscape_readouts`). The motor-EEG cohort has 106 participants here and 104 records in the
  count cache (different eligibility rules on the same recordings).
- BLAS is pinned to one thread, as in the runs behind the stored outputs.
- Dependencies beyond the package: scikit-learn, mne (EDF reading of the motor EEG).
