# Sleep-EDF feature cache

One `.npz` per primary cassette night of Sleep-EDF Expanded (153 nights, 78 participants), the input of the sleep
assessments and the sleep re-evaluation.  Each file holds `X` (scored 60-s bouts, one row per bout, each an array of
1-s windows × 8 unstandardized log band-power features: 4 bands × 2 channels), `stage` (the Rechtschaffen–Kales stage
of each bout) and `win_s` (the window length, 1 s).

Stage labels in the cache and in the stored outputs use the codes `Wake`, `N1`, `N2`, `N3`, `N4`, `REM`, which stand
for the R&K stages Wake, S1, S2, S3, S4 and REM of the manuscript (S3 and S4 are kept separate).

Readers: `analyses/primary_configuration/benchmark_05_corpus_full.py` (602 night–stage cells with at least six scored
bouts, both estimators, and the 336 adjacent-stage pairs), `examples/example_04_corpus.py`
and `analyses/g8_configuration/benchmark_10_real_recordings.py --part sleep`.

The features derive from the Sleep-EDF Expanded database (PhysioNet, https://doi.org/10.13026/C2X676), whose terms
permit redistribution of derived data.
