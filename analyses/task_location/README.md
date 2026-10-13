# Task location: digit-span position slopes and workload axes

Position slopes of the task variable in the two EEG task cohorts: the digit-span cohort (OpenNeuro ds003838, 64
participants; slope of the held-out projection on the expected number of items held, memory arm against the
listen-only arm) and the workload cohort (COG-BCI, 28 participants after the duplicate exclusion of sub-27 and
sub-28 session 1; slope of the projection on the graded n-back and MATB difficulty along other-participant and
within-person axes, with band-restricted variants).

| Script | Output | Reported in |
|---|---|---|
| `ds003838_position_slopes.py --workers 5 --perm 1000 --sims 200` | `out/digit_span_position_analysis.json` (settings, group summary, one row per participant), `out/digit_span_participants.csv`, `out/digit_span_rate_diffusion_cells.csv` | the per-participant position slopes and nulls behind Fig. 4a and Extended Data Fig. 4a; read by `ds003838_summary.py` |
| `ds003838_summary.py` | `../../figure_reproduction/inputs/digit_span_position_slopes.json`; `out/digit_span_supplementary_values.json` | memory 0.06399 [0.04164, 0.08640], listen-only -0.00022 [-0.01893, 0.01848], memory - listen 0.06421 [0.04565, 0.08285] (Fig. 4a; Extended Data Fig. 4a; Supplementary Table 7a); alternative inputs u1 0.01684 [0.01123, 0.02263] and u2 0.02267 [0.01547, 0.03014] (Extended Data Fig. 4a and Supplementary Table 7a, source-only rows); one-slot-shift control 0.03540 [0.02210, 0.04902] and slot-map permutation P 0.000999 with 1,000 draws (Supplementary Note 6) |
| `workload_axes.py --boot 10000 --perm 10000` | `../../figure_reproduction/inputs/workload_position_slopes.json`; `out/workload_axes_summary.json`, `out/workload_axes_cells.csv`, `out/workload_axes_dynamics_rows.csv`, `out/workload_axes_S248.npz`, `out/workload_axes_S8.npz` | other-participant (LOPO) axis 0.41453 [0.33730, 0.49233], n = 28 (Fig. 4a); n-back -> MATB 0.50183, MATB -> MATB 0.66709, MATB -> n-back 0.07192, n-back -> n-back 0.31080 (Fig. 4b; Extended Data Fig. 4b,c; Supplementary Table 7b,c) |
| `workload_within_axes.py --boot 10000 --perm 10000` | `out/workload_within_person.json` | own cross-task axis 0.27279 [0.13670, 0.40172] (key `S248/LP_cv`; Extended Data Fig. 4b; Supplementary Table 7b) and its 8-PC variant 0.24530 [0.11410, 0.37066] (key `S8/LP_cv`; source-only row) |
| `workload_band_variants.py` (with `workload_axes_lib.py`) | `out/workload_band_variants.json`, `out/workload_band_variant_rows.csv` | own cross-task axis on 4-12 Hz features 0.09558 [0.03846, 0.15231] (`S248_thetaAlpha/LP_cv`) and on 4-30 Hz features 0.14324 [0.03671, 0.24373] (`S248_noGamma/LP_cv`, source-only row); within-person n-back -> MATB 0.40108 [0.20417, 0.58979], MATB -> MATB in sample 0.87131, MATB -> n-back 0.14450 [0.05319, 0.23154], n-back -> n-back in sample 0.66904 (`S248_kss/LP_nb2mb`, `LP_in_mb`, `LP_mb2nb`, `LP_in_nb`; Extended Data Fig. 4b,d; Supplementary Table 7b,d) |

`ds003838_common.py` holds the cache reader, the input definitions u1/u2/u3 and the state-space and VAR(1)
helpers of the digit-span analysis; `workload_axes.py` and `workload_within_axes.py` use the COG-BCI state
construction of `../workload_certificates/features/cog_features.py` unchanged; `workload_axes_lib.py` is a second,
self-contained implementation of the within-person readouts (own cache reader and bootstrap, seed 777001) used by
`workload_band_variants.py`.

Stored outputs in `out/`: `digit_span_supplementary_values.json`, `workload_within_person.json`,
`workload_band_variants.json` and `workload_band_variant_rows.csv` (the values reported without a figure input);
the two figure inputs are in `figure_reproduction/inputs/`.  The other outputs listed above are rebuilt by the
scripts and are not stored.

## Run order

1. Feature caches: `../cohort_states/ds003838_states.py` (ds003838) and the COG-BCI band-power cache read by
   `../workload_certificates` (`NDIF_COGBCI_CACHE`).
2. `ds003838_position_slopes.py` -> `ds003838_summary.py`.
3. `workload_axes.py` -> `workload_within_axes.py` (imports `workload_axes.py`) -> `workload_band_variants.py`
   (independent of the other two).

`ds003838_summary.py --input <file>` reads another position-analysis output in place of the default.  Set
`NDIF_OUT` to write every output below another directory instead of its stored location.

## Inputs

| Variable | Default | Content |
|---|---|---|
| `NDIF_DS003838_CACHE` | `analyses/_work/ds003838` | ds003838 feature cache written by `../cohort_states/ds003838_states.py`: per participant `sub-XXX.npz` with, for every sequence `j`, `s{j:03d}_F_dist` (windows x 256: 64 channels x theta/alpha/beta/gamma log10 band power, 1-s windows every 0.5 s), `_F_dyn` (non-overlapping 1-s windows), `_flag_d`/`_flag_y` (window QC flags), `_t_start`/`_t_dyn` (s), `_slot_true`/`_slot_true_dyn` (item slot from the true item onsets), `_slot_pub`/`_slot_pub_dyn` (item slot from the nominal 2.0-s inter-item interval), `_nuis`, `_resid`, `_periph` (nuisance, residual-artifact and peripheral columns), `_item_ok` (per-item recall outcome) and `_onsets`; and `sub-XXX_meta.json` (channels, settings, one entry per sequence: `cond` 0 = listen-only, 1 = memory, `block`, `role`, `n_digits`, `t0`, ...) |
| `NDIF_COGBCI_CACHE` | `analyses/workload_certificates/features/cog_feature_cache` | COG-BCI feature cache, one `sub-XX.npz` per participant (248 log band-power features of 62 channels x 4 bands per window, keys `<session>\|<recording>\|<dist or dyn>\|F`, `\|keep`, `\|seg`, and `meta`) |
| `NDIF_COGBCI_RAW` | `COG-BCI` | the COG-BCI release; `KSS.txt` and `RSME.txt` are read |

Neither cache is redistributed; both are derived from the source datasets (top-level README, *Data*).

## Runtimes and settings

On one desktop: `ds003838_position_slopes.py` about 4 min with 2 worker processes (about 7 s per participant;
2 BLAS threads per worker, as in the stored run), `ds003838_summary.py` under 1 s, `workload_axes.py` 20 s,
`workload_within_axes.py` 15 s, `workload_band_variants.py` 25 s.  The three workload scripts pin BLAS to one thread
(`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` = 1), the setting of the stored outputs.

The permutation draws of the slot -> u map and the matched simulation of the rate slope are seeded per participant
with `20260921 + hash(participant) mod 100000` (and `+ 7` with the arm name for the simulation).
