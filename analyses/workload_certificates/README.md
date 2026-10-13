# COG-BCI workload assessments (56 inputs, primary configuration)

The stored outputs cited for the COG-BCI workload cohort (Fig. 4d; Results; Supplementary Table 2) after the duplicate
exclusion of the public release (all sub-27 recordings duplicate sub-17; sub-28 session 1 duplicates sub-25 session 1):
56 participant–task inputs from 28 independent participants.

* `corpus_cog/*.json` — the 56 per-input assessments (gates G0–G7, lag-1 least squares, non-overlapping 1-s windows,
  the three difficulty levels pooled).  In one input the fitted white share reaches the model boundary, where the G3
  rate channel is unavailable while the variance channel is still evaluated (Methods, G3).
* `corpus_cog.log` — the driver summary (L1 indeterminate 56; L2 admitted 56; L3 refused 55 and indeterminate 1;
  first failed gates G1 × 32, G2 × 10, G3 × 13; median G3 rate recovery 0.107; median detection threshold × 10.50).
* `corpus_load_summary_cog.json` — the COG-BCI summary written by the driver.
* `benchmark_06_load_cohorts.py` — the driver for both load cohorts.  Its COG-BCI branch builds the state series with
  the modules in `features/` (248 log band-power features, participant-specific Flanker-referenced median/IQR
  standardization, an eight-component pooled Flanker PCA) and writes the files above; its HCP MEG branch reads the
  per-trial MEG state cache and writes the 83 assessments, `corpus_meg.log` and `corpus_load_summary_meg.json` to
  `../primary_configuration/out/`.  For HCP MEG the rate channel of G3 is evaluated from the generator fit also where
  the fit places the white share at its boundary (`MEG_GATE_KW = dict(G3=dict(boundary_rule=False))`).
* `features/` — `cog_features.py` (loading, duplicate exclusion, readouts), `cog_state.py` (state construction and the
  time-rescaling calibrations), `cog_fit_lib.py` (block-wise propagator fit) and `cog_rescaling_lib.py` (time-rescaling
  estimators and injections), and `cog_cache.py` (the band-power cache from the raw COG-BCI archives).

Both branches read feature caches derived from the source datasets (COG-BCI: Zenodo, https://doi.org/10.5281/zenodo.6874128;
HCP: Connectome Coordination Facility, data-use terms).  The caches are not redistributed; they are rebuilt from the
raw recordings by `features/cog_cache.py` (COG-BCI archives in `NDIF_COGBCI_RAW`, about 2 min per participant) and
`../cohort_states/hcp_meg_states.py` (HCP MEG `tmegpreproc` data in `NDIF_HCP_MEG_RAW`).  Set `NDIF_COGBCI_CACHE` to the directory of the COG-BCI
band-power cache (one `sub-XX.npz` per participant) and `NDIF_MEG_CACHE` to the directory of the per-trial MEG state
cache (one `<subject>.npz` per participant).
