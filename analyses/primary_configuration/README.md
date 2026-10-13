# Primary and benchmark configurations: drivers and stored outputs

The primary configuration runs gates G0–G7 with the governing order of the Methods (L1: G6, G5; L2: G0, G3 variance
channel; L3: G0, G1, G2, G3, G7, G4); the benchmark configuration runs G0, G1, G2, G3 and G7 on simulated recordings
with known rates (`ndif.configurations`).

| Script | Output | Reported in |
|---|---|---|
| `benchmark_01_estimators.py --reps 20 --workers 3` | `out/benchmark_01.json` (console output in `out/run_01.log`) | the 320-output benchmark: 91 accurate and 23 inaccurate admissions, 24 accurate and 182 inaccurate non-admissions (Fig. 3f; Extended Data Figs 3 and 6; Supplementary Table 3) |
| `benchmark_02_gate_cost.py --reps 5` | `out/gate_cost.json` (console output in `out/gate_cost.log`) | per-gate runtimes at 24,000 samples × 3 dimensions × 60 segments (Methods; Extended Data Fig. 3a) |
| `benchmark_05_corpus_full.py --workers 6` | `out/corpus_full/` (602 cells × 2 estimators, 336 adjacent-stage pairs; console output in `out/corpus_full_run.log`) | the sleep cell assessments (Fig. 5c,d; Supplementary Table 10) |
| `summarize_corpus_full.py` | `out/corpus_full_summary.json`, `out/corpus_full_L1.json`, `out/corpus_full.log`; `figure_reproduction/inputs/sleep_detection_thresholds.json` (detection thresholds per stage, Fig. 5c,d) and `figure_reproduction/inputs/adjacent_stage_pairs_index.json` (the 336 pair assessments, Supplementary Fig. 2) | least squares 0/590/12, maximum likelihood 87/511/4 by stage; G6 per cell and G5 on the 336 pairs (Methods) |
| `benchmark_04_meg_clock.py` | `out/meg_clock.json` | HCP MEG two-lag solutions on the nominal 2.5-s trial clock (Methods, *Human recordings*) |
| `../workload_certificates/benchmark_06_load_cohorts.py --cohorts meg` | `out/corpus_meg/` (83), `out/corpus_meg.log`, `out/corpus_load_summary_meg.json` | HCP MEG assessments: L1 72/11, L2 83, L3 refused 83 (Fig. 4d; Supplementary Table 2); the G3 rate channel is evaluated from the generator fit also where it places the white share at its boundary (`boundary_rule=False`, `MEG_GATE_KW` of the driver) |
| `../../examples/example_03_cross_domain.py --write` | `out/cross_domain.log` (gate statistics of the three regularly indexed series); also `figure_reproduction/inputs/nonneural_series_assessments.csv` and `co2_block_sweep.{json,csv}` | the three regularly indexed non-neural series under the primary configuration with G2 `rate="propagator"` and the G3 rate channel decided on the slope (`REGULAR_SERIES_KW`) (Fig. 6a); the CO₂ calendar sweep under the G8-augmented configuration (Fig. 6b) |

Stored sleep outputs use the stage codes `Wake`, `N1`–`N4`, `REM` for the R&K stages Wake, S1–S4 and REM of the
manuscript.  The sleep drivers read `../sleep_feature_cache`; the MEG driver reads a per-trial state cache derived from
the HCP data (not redistributed; set `NDIF_MEG_CACHE`).  Logs record the wall time of each run; paths inside them are
relative to the repository root.
