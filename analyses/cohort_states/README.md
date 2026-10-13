# Cohort state spaces and the count cache

These scripts build the state series of the five recording cohorts from the source data and assemble the count cache:
the pooled state series, segment structure and condition labels of the 646 recordings whose equilibrium counts are
analysed in `analyses/landscape_readouts` (Fig. 2b,c,f). They write intermediate caches only; no stored output of the
repository is rewritten.

| Script | Output | Reported in |
|---|---|---|
| `ds003838_states.py --workers 6` | digit-span feature cache, `paths.cache("ds003838")` (`NDIF_DS003838_CACHE`, default `analyses/_work/ds003838`): `sub-XXX.npz`, `sub-XXX_meta.json` per participant, `_cache_report.json` | the 64 digit-span recordings (Methods, *Human recordings*); input of `count_cache.py` and of the digit-span analyses in `analyses/task_location` and `analyses/rate_identifiability` |
| `hcp_meg_states.py --workers 6` | per-trial HCP MEG state cache, `paths.cache("meg")` (`NDIF_MEG_CACHE`, default `analyses/primary_configuration/meg_feature_cache`): `<subject>.npz` per participant | the 83 HCP MEG participants (Methods); input of `count_cache.py`, `../primary_configuration/benchmark_04_meg_clock.py` and `../workload_certificates/benchmark_06_load_cohorts.py --cohorts meg` (Fig. 4d; Supplementary Table 2) |
| `count_cache.py --cohort all --workers 6` | count cache, `paths.cache("count")` (`NDIF_COUNT_CACHE`, default `analyses/_work/count`): `{ds003838,eegmmidb,meg,fmri,sleep}/<record>.npz` and `_meta.json` per cohort | the 646 recordings: 64 digit-span EEG, 104 motor EEG, 83 MEG, 198 fMRI runs (99 participants) and 197 nights (153 cassette, 44 telemetry; 100 participants); 449 awake recordings (Methods, *Human recordings*; Supplementary Fig. 2a); input of the drift-replacement, separation-threshold and readout analyses (Fig. 2b,c,f) |

Modules without a command line: `states_lib.py` (band power, log features, robust z with the reference PCA of
`random_state` 20260728, segment splitting, the VAR(1) regime fit and the equilibrium search), `eegmmidb_states.py`,
`hcp_fmri_states.py` and `sleep_states.py` (the states of these three cohorts are built by `count_cache.py`).
`reference_pipeline_counts.csv` lists, per record, the equilibrium count of the per-participant reference landscape
pipeline (one VAR(1) regime per condition on the condition labels; for the nights also the ridgeline mode count of the
three stage Gaussians). `count_cache.py` copies it into the cache metadata (`pub_n_stable`, `pub_n_eq`, `pub_modes_K`,
`labels`), takes the motor-EEG and sleep cohorts from it, and compares the fMRI counts recomputed from the cached labels
with it (`label_check_ok`, `label_reconstruction_check`). Its `n_stable` column is the `reference_pipeline_field`
column of `figure_reproduction/inputs/landscape_readout_counts.csv`.

## Run order

```bash
python analyses/cohort_states/ds003838_states.py --workers 6      # 0.1-0.4 min per participant, about 4 min
python analyses/cohort_states/hcp_meg_states.py --workers 6       # about 7 min for 83 participants
python analyses/cohort_states/count_cache.py --cohort all --workers 6   # about 25 min (sleep 9, eegmmidb 2, fMRI 3)
```

`count_cache.py --cohort <name> --records <record ...>` rebuilds selected records; the cohort metadata then lists only
those records. Existing motor-EEG and sleep records are skipped; delete them to rebuild.

## Inputs

| Variable | Default | Content |
|---|---|---|
| `NDIF_DS003838_DERIV` | `data/ds003838_preproc` | preprocessed digit-span recordings, one `sub-XXX/` per participant: `sub-XXX_task-memory_preproc_250Hz.npy`, `_sequences.json`, `_qc.json`, `_nuisance_1s.npz`, `_periph_1s.npz`, `_pupil_1s.npz` (where recorded), `_done.json`. The derivative was produced by standard preprocessing as described in the Methods (*Human recordings*); that step is not part of this repository. |
| `NDIF_DS003838_RAW` | `data/ds003838` | OpenNeuro ds003838 (BIDS); only `sub-XXX/eeg/sub-XXX_task-memory_events.tsv` is read (item onsets and recall outcomes) |
| `NDIF_EEGMMIDB_RAW` | `data/eegmmidb` | PhysioNet EEG Motor Movement/Imagery, `S001/S001R03.edf` ... |
| `NDIF_HCP_MEG_RAW` | `data/hcp_meg` | HCP MEG, `<subject>/MEG/Wrkmem/tmegpreproc/<subject>_MEG_{6,7}-Wrkmem_tmegpreproc_TIM.mat` |
| `NDIF_HCP_FMRI_PARCELS` | `data/hcp_wm_parcels` | per working-memory run: `<subject>__<run>__atlas_fpn7.nii.npy` (405 x 7 parcel series) and `<subject>__<run>__EVs/` holding the run's HCP task EV files (`{0bk,2bk}_{body,faces,places,tools}.txt` from `MNINonLinear/Results/tfMRI_<run>/EVs/`). The parcel series are detrended parcel means of the MNI-space volumes; the parcellation of the volumes is not part of this repository. |
| `NDIF_SLEEPEDF_RAW` | `data/sleep-edf` | Sleep-EDF Expanded cassette and telemetry recordings, `*-PSG.edf` and `*-Hypnogram.edf` in one directory |
| `NDIF_DS003838_CACHE`, `NDIF_MEG_CACHE`, `NDIF_COUNT_CACHE` | see the table above | the caches written here; `count_cache.py` reads the first two |

The HCP and Sleep-EDF derived caches are not redistributed (access terms of the source data).

## Recordings per cohort

| Cohort | Records | Rule |
|---|---|---|
| ds003838 | 64 participants | every participant of the derivative; the feature cache would drop a participant marked excluded by the preprocessing QC, with fewer than 8 ICA components labelled brain or more than 30 removed, or whose events file and sequence list disagree (none of the 64) |
| eegmmidb | 104 of 109 participants | at least 20 rest/task blocks and 10 of each class in runs R03-R14; S088, S092 and S100 are 128-Hz recordings whose annotated blocks cannot be read at the 160-Hz rate of the cohort; for S035 and S063 the per-participant reference pipeline stopped at the singular covariance of their artifact proxies, so they have no reference count. The separation floors of Fig. 1f use 106 participants under the loader rule of `analyses/separation_floors`. |
| meg | 83 participants | every participant with a Wrkmem run (all 83 have runs 6 and 7) and at least 40 task trials |
| fmri | 198 runs | WM_LR and WM_RL of 99 participants; the run median of the separations (0.88, Fig. 2c) and the participant median (0.86, Fig. 1f) summarize the same per-run values at the two levels |
| sleep | 197 nights | every night with a hypnogram and at least two of the stage groups Wake, NREM, REM (153 cassette nights of 78 participants, 44 telemetry nights of 22 participants) |

## Notes

- Thread counts follow the stored runs: two threads per process for `ds003838_states.py` and `count_cache.py`, the
  libraries' default for `hcp_meg_states.py`.
- No random draw enters the caches except through the equilibrium search of the fMRI count check (seed 20260819) and
  the PCA `random_state` 20260728; nothing is seeded from Python's `hash()`.
- Feature-cache keys `slot_pub` and `slot_pub_dyn` hold the item slot from the nominal 2.0-s inter-item interval
  (`soa_published` = 2.0 in the metadata); `slot_true` and `slot_true_dyn` use the onsets of the events file.
- A rebuild of records of every cohort (one participant each for ds003838, eegmmidb and MEG, all 198 fMRI runs, one
  cassette and one telemetry night) reproduces, array by array, the caches from which the reported analyses were run.
