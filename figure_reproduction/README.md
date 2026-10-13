# Stored figure inputs

`inputs/` holds the stored summaries that the figures plot: every value drawn in Figs 2–6, Extended Data Figs 1, 4 and
7 and Supplementary Fig. 2 that is not read directly from an `analyses/*/out/` file.  Each file is written by the
analysis script named below (`analyses/README.md` gives run order, inputs and runtimes); the scripts write here by
default, and `NDIF_OUT` redirects them for comparison.  The same values are given per figure in the Source Data.

| File | Figure | Written by |
|---|---|---|
| `drift_replacement_records.csv`, `drift_replacement_summary.json` | Fig. 2b; Supplementary Fig. 2 | `analyses/landscape_readouts/drift_replacement.py` |
| `separation_threshold_summary.json` | Fig. 2c | `analyses/landscape_readouts/drift_replacement.py` |
| `density_barrier_summary.json` | Fig. 2d,e (fits) | `analyses/landscape_readouts/drift_replacement.py` |
| `density_barrier_cells.csv` | Fig. 2d,e (cells) | `analyses/recurrent_networks/known_barrier_cells.py` |
| `arrhenius_slopes.json` | Fig. 2e (right) | `analyses/recurrent_networks/arrhenius_summary.py` |
| `landscape_readout_counts.csv` | Fig. 2f | `analyses/landscape_readouts/readout_table.py` |
| `identifiability_map_grid.csv`, `_observed.csv`, `_contours.csv`, `_summary.json` | Fig. 3a; Extended Data Fig. 1 | `analyses/rate_identifiability/identifiability_map.py` |
| `phase_surrogate_variance_by_stage.csv` | Fig. 3b (bars, circles) | `analyses/rate_identifiability/lag_report.py` |
| `white_share_control_simulation.json` | Fig. 3b (shading) | `analyses/rate_identifiability/white_share_pipeline_control.py` |
| `window_scaling_exponent.json` | Fig. 3c; Supplementary Table 1 | `analyses/rate_identifiability/window_scaling.py` |
| `injection_recovery.csv` | Fig. 3d | `analyses/rate_identifiability/injection_recovery_report.py` |
| `drive_gain_separation.csv` | Fig. 3e | `analyses/recurrent_networks/drive_gain_report.py` |
| `digit_span_position_slopes.json` | Fig. 4a; Extended Data Fig. 4a | `analyses/task_location/ds003838_summary.py` |
| `workload_position_slopes.json` | Fig. 4a,b; Extended Data Fig. 4b–d | `analyses/task_location/workload_axes.py` |
| `sleep_stage_contrasts.json`, `sleep_stage_cells.csv` | Fig. 4c; Fig. 5b; Supplementary Table 8; Supplementary Fig. 2 | `analyses/sleep_stage_contrasts/stage_contrasts.py` |
| `sleep_telemetry_crossover.json` | Fig. 4c; Supplementary Table 9 | `analyses/sleep_telemetry/crossover.py` |
| `sleep_drug_contrast_bounds.json` | Fig. 4c | `analyses/sleep_telemetry/drug_power.py` |
| `sleep_detection_thresholds.json` | Fig. 5c,d | `analyses/primary_configuration/summarize_corpus_full.py` |
| `sleep_rate_corrections.csv` | Fig. 5e; Supplementary Table 11 | `analyses/sleep_stage_contrasts/rate_corrections.py` |
| `adjacent_stage_pairs_index.json` | Supplementary Fig. 2 | `analyses/primary_configuration/summarize_corpus_full.py` |
| `nonneural_series_assessments.csv` | Fig. 6a | `examples/example_03_cross_domain.py --write` |
| `co2_block_sweep.json`, `co2_block_sweep.csv` | Fig. 6b | `examples/example_03_cross_domain.py --write` |
| `tail_excess_contrasts.json`, `tail_excess_ds003838.json` | Extended Data Fig. 7a | `analyses/rate_identifiability/tail_excess_contrast.py`, `tail_excess_ds003838.py` |
| `second_mode_calibration.json` | Extended Data Fig. 7b | `analyses/rate_identifiability/tail_excess_calibration.py` |

Figures also plot stored outputs kept under `analyses/`: the HCP MEG and COG-BCI assessments
(`analyses/primary_configuration/out/corpus_meg/`, `analyses/workload_certificates/corpus_cog/`; Fig. 4d), the sleep
corpus summary (`analyses/primary_configuration/out/corpus_full_summary.json`; Fig. 5c,d), the known-rate benchmark
(`analyses/primary_configuration/out/benchmark_01.json`; Fig. 3f) and the gate statistics of the non-neural series
(`analyses/primary_configuration/out/cross_domain.log`; Fig. 6a).
