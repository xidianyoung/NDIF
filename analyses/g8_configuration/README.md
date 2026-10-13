# G8-augmented configuration: drivers and stored outputs

The G8-augmented configuration (Methods, *Analysis configurations*; `ndif.configurations.CONFIGURATIONS["g8_augmented"]`)
governs L3 by G0, G2, G8, G1 and G3, with G1 as the profile-likelihood bound on the log rate over the white-share set,
G3 with an indirect-inference-corrected generator and criteria decided beyond Monte Carlo error, and G4 and G7 as
diagnostics.

| Script | Output | Reported in |
|---|---|---|
| `benchmark_09_extended_evaluation.py --part single/variants/missp/neg/twins/stab/abc/mdc` | `out/benchmark_09_<part>.jsonl` | extended simulation evaluation: 404 recordings and 808 outputs (`single` + `variants`), the misspecification set (`missp`, 100 recordings), the negative controls (`neg`), the sleep-matched simulation (`twins`, 320 recordings from 160 source cells) |
| `benchmark_09_analyse.py` | `out/benchmark_09_results.json` | admission risk, accurate-estimate retention, leave-one-check-out exclusions, binding gates (Extended Data Fig. 2; Supplementary Note 4; Supplementary Table 6) |
| `benchmark_09b_comparator.py` | `out/benchmark_09b_comparator.jsonl`, `out/benchmark_09b_comparator_results.json` | estimator-matched maximum-likelihood bootstrap on the 394 paired recordings (Extended Data Fig. 2b; Supplementary Table 5) |
| `noise_aware_bootstrap/noise_aware_bootstrap.py plan/test/pilot/run` | `noise_aware_bootstrap/results.json`, `paired_records.csv`, `protocol.json` | noise-aware least-squares bootstrap rankings at 72 admissions (Extended Data Fig. 2a; Supplementary Table 4; Supplementary Fig. 3) |
| `matched_comparison/make_matched_comparison.py --draws 5000 --seed 20261009` | `matched_comparison/matched_coverage_results.json`, `matched_comparison/paired_records.csv`, `matched_comparison/risk_coverage_curves.csv` | matched-admission comparison on the 394 paired recordings: maximum likelihood 179 admissions, 21 (certificate) vs 16 (estimator-matched bootstrap), risk difference -2.793 [-7.345, 1.156] percentage points (Extended Data Fig. 2b; Supplementary Table 5; Supplementary Note 4); least-squares bootstrap ranking unmatched 277/365 and coverage-matched 63/72 vs 5/72, risk-coverage curves and G3-margin sweeps (Supplementary Fig. 3) |
| `tabulate_selection.py` | `out/g8_outputs_truth_and_gates.csv`, `out/selection_performance.csv`, `out/check_removal_ablation.csv`, `out/selection_sensitivity.json` | Supplementary Table 6 and Supplementary Note 4: 261/808 admitted, 28 inaccurate, 233/339 accurate admitted; misspecification 10 (all inaccurate)/187/3; two-timescale 40/40 refused; sleep-matched 16/210 (633 scoreable); single-gate removals; offset subtraction 28 -> 4 of 261; accuracy-tolerance sweep 0.10-0.30 (109/78/48/28/19 inaccurate of 261); matched rows 5 vs 4 at 72 and 21 vs 16 at 179 (Extended Data Fig. 2; Supplementary Tables 4-5); sleep re-evaluation 11/583/8 and 6/590/6 |
| `benchmark_10_real_recordings.py --part sleep/pairs/series/co2` | `out/benchmark_10_<part>.jsonl` | re-evaluation of the 602 sleep cells and the 336 adjacent-stage pairs (Supplementary Note 4), the calendar-preserving CO₂ analysis (Fig. 6b) and the non-neural series |
| `benchmark_10_analyse.py` | `out/benchmark_10_results.json` | the sleep re-evaluation counts (11/583/8 least squares; 6/590/6 maximum likelihood) and the CO₂ segmentations |

The real-recording driver reads the sleep feature cache (`../sleep_feature_cache`) and the statsmodels non-neural
series.  Stored records carry the settings and gate details that produced them.  The drivers append to their output
files and skip records already written, so a run can be interrupted and resumed; `--budget` stops submitting new jobs
after the given number of seconds.  `NDIF_SMOKE=1` runs the extended evaluation on other seeds into separate `_smoke`
files, as a quick test.  `python verification/recount_stored_outputs.py` recounts the principal reported numbers from
these files.

Run `matched_comparison/make_matched_comparison.py` and then `tabulate_selection.py` after `benchmark_09_analyse.py`,
`benchmark_09b_comparator.py` and `noise_aware_bootstrap/noise_aware_bootstrap.py`; both are tabulations of stored outputs
(seconds) and reproduce the stored files byte for byte.
