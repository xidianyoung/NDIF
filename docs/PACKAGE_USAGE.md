# NDIF — package usage

NDIF assesses whether a specified neural recording, its preprocessing and an estimator can support a dynamical claim.
Claims are of three classes: state geometry (L1), state variability (L2) and intrinsic dynamics (L3; here relaxation
rates under linear calibration generators).  A stationary density is a joint function of drift and diffusion, so an
observation constrains the pair only up to an equivalence class: functionals of the distribution (L1, L2) can be
assessed with finite-sample controls, whereas intrinsic rates (L3) require temporal information sufficient to recover
latent rates.  The package returns that assessment per recording and estimator.

```python
import numpy as np
from ndif import certify, Design

cert = certify(X, Design(dt=0.5, window_s=0.5, step_s=0.5, segments=bounds,
                         labels=cond, name="demo-2cond", modality="synthetic"))
print(cert)
print(cert.levels["L3"].status)        # 'admitted' | 'refused' | 'indeterminate'
print(cert.levels["L3"].binding_gate)  # the first failed gate, e.g. 'G1'
print(cert.levels["L3"].detection_margin)
cert.to_json("demo-2cond.json")
```

For the simulated two-condition recording of `examples/example_01_quickstart.py` (40 segments × 150 samples, three
dimensions; the conditions differ in where the state sits, not in how fast it relaxes), run with the example's
reduced replicate counts, the printed certificate is:

```
==========================================================================
NDIF identifiability certificate   demo-2cond   [synthetic]
==========================================================================
  6000 samples x 3 dims in 40 segments; dt = 0.5 s; cost 4.4 s

  level status         binding gate      margin
  ----------------------------------------------
  L1    ADMITTED       G6                  0.05
  L2    ADMITTED       G0                     0
  L3    ADMITTED       G0                     0

  L1 (state geometry - where the system is)
      grounds: all governing gates cleared; tightest is G6 at margin 0.05
      detection margin: separations above 0.217 clear the zero-truth floor at this geometry
  L2 (state variability - how it explores)
      grounds: all governing gates cleared; tightest is G0 at margin 0
      detection margin: a change in state variance of at least x1.10 would have been detected at 80% power
  L3 (intrinsic dynamics - how it evolves)
      grounds: all governing gates cleared; tightest is G0 at margin 0
      detection margin: a change in relaxation rate of at least x1.26 would have been detected at 80% power

  gate  status        statistic                                         value
  --------------------------------------------------------------------------
  G0    cleared       step/window                                           1
  G1    cleared       nu (white variance share)                       0.01575
  G2    cleared       theta in lambda ~ dt^(-theta)                   0.02633
  G3    cleared       recovered fraction (rate channel)                0.8944
  G4    cleared       agreement with dynamics-deleted surrogate             0
  G5    cleared       separation (Mahalanobis)                         0.9603
  G6    cleared       TPR at delta=3.0 / FPR                                1
  G7    cleared       share of rate summary from identified modes      0.9549
==========================================================================
```

A refusal is reported with the size of effect the recording could have shown (the detection threshold), never as
"no effect".  Admission means that every governing gate was evaluated and passed under the configured calibration
generators and observation models; it is not an accuracy certificate and not a controlled-risk selection guarantee.

## An assessment is about a *(recording, estimator)* pair

Identifiability is not a property of a recording alone.  A 40 × 150 recording of a latent AR(1) process
(persistence 0.85) with an additive white share of 0.4 gives, under the primary configuration:

```python
import numpy as np
from ndif import Design, certify
from ndif.simulate import simulate_var1

a, nu, seed = 0.85, 0.4, 1
rng = np.random.default_rng(seed)
segs = simulate_var1(np.array([[a]]), np.array([[1 - a * a]]), n_segments=40, n_block=150, rng=seed)
segs = [s + rng.normal(scale=np.sqrt(nu / (1 - nu)), size=s.shape) for s in segs]
edges = np.cumsum([0] + [len(s) for s in segs])
design = Design(dt=1.0, window_s=1.0, step_s=1.0, segments=list(zip(edges[:-1], edges[1:])))
for est in ("ols", "kalman"):
    print(est, certify(np.concatenate(segs), design, estimator=est, rng=1).levels["L3"].status)
```

| estimator | G1 (ν) | G2 (θ) | G3 recovered | L3 | detection threshold |
|---|---|---|---|---|---|
| `ols` — least squares on lag-1 pairs | **fails** 0.408 | **fails** 0.548 | **0.237** | **refused** | ×1.88 |
| `kalman` — maximum likelihood, AR(1) plus additive white noise | not binding | passes −0.131 | **0.950** | **admitted** | ×1.28 |

Same data: the rate is recoverable, the estimator most people run cannot recover it, and under that estimator the
rate tracks the sampling interval (θ = 0.55, where an identified rate gives θ = 0).  The package therefore does not
stop at a refusal:

```python
from ndif import recommend_estimator

r = recommend_estimator(X, design)
r["recommended"]   # 'kalman'
r["reason"]        # "kalman (maximum likelihood, AR(1) plus additive white) is admitted at level 3, ..."
```

A cost-based recommendation depends on the machine and does not rank scientific accuracy; the report includes every
candidate.  When no candidate is admitted, the reasons for refusal or indeterminacy and the detection thresholds are
returned instead.

## Analysis configurations

`certify(..., configuration=...)` selects one of three configurations of the same gate set
(`ndif/configurations.py`; Methods, *Analysis configurations*):

| configuration | gates run | governing order | G1 | G3 | G5 | gate simulations |
|---|---|---|---|---|---|---|
| `primary` (default) | G0–G7 | L1: G6, G5; L2: G0, G3 variance channel; L3: G0, G1, G2, G3, G7, G4 | two-lag white share ν ≤ 0.20 for an estimator without an observation model; not binding otherwise | recovery from the fitted generator: slope within 0.20 of 1 and offset within 0.20 log units | 95th percentile of 200 simulations of the fitted generator without a condition effect | the recording's number of segments at the rounded mean length |
| `benchmark` | G0, G1, G2, G3, G7 | as primary; G4 not run | as primary | as primary | — | as primary |
| `g8_augmented` | G0–G3, G5, G6, G8; G4 and G7 as diagnostics | L3: G0, G2, G8, G1, G3 | log rate varies by at most 0.20 over the 95% profile-likelihood set of the white share | indirect-inference-corrected generator; slope, offset and spread decided beyond Monte Carlo error | permutation of the condition labels across segments | the recording's own segment lengths |

Every gate keyword can still be overridden per call (`G3=dict(n_rep=64)`, `G1=dict(rule="profile")`).

## System requirements and installation

Pure Python; Windows, Linux and macOS (tested on Windows 11, Python 3.13.9, numpy 2.4.4, scipy 1.17.1).  No compiled
extension, no GPU; one core suffices (the benchmarks accept `--workers`).

```bash
pip install -e .               # numpy + scipy only
pip install -e ".[test]"       # + pytest
pip install -e ".[examples]"   # + statsmodels, for the non-neural series example
pip install -r requirements.txt   # the versions used for the reported analyses and figures
```

## Demo

```bash
python examples/example_01_quickstart.py
```

The demo simulates its own recording: part 1 assesses a three-dimensional VAR(1) series of 40 segments × 150 samples
in two conditions that differ in where the state sits, not in how fast it relaxes, and prints the full report; part 2
runs `minimum_action` on the curved-instanton double well with `D = 0.05` and prints the exact barrier (5.000000), the
recovered barrier (about 5.0001) and the peak `|y|` of the converged path (about 1.0).  About a minute on one core.

## Instructions for use

Build a state series `X` of shape `(n_samples, n_dims)` (one row per window) and describe how it was made with a
`Design`: `dt` is the sampling interval *of the state series* (the window step, not the raw sampling period);
`window_s` and `step_s` are the feature window and step (G0 needs them); `segments` gives the segment boundaries as
`(start, stop)` pairs so that no lag-1 pair straddles a boundary; `labels` gives a condition label per sample when a
two-condition L1 contrast is wanted; `axis` names a direction of interest when the rate is read along a task axis.

```python
from ndif import certify, Design, recommend_estimator

design = Design(dt=0.72, window_s=0.72, step_s=0.72, segments=runs, labels=cond, name="sub-01")   # one state per 0.72-s volume
cert = certify(X, design, estimator="ols")   # or "kalman", or a registered name
cert.levels["L3"].status                     # 'admitted' | 'refused' | 'indeterminate'
cert.levels["L3"].binding_gate               # the first failed gate
cert.gate("G3").value                        # any gate's statistic
cert.to_json("sub-01.json")
```

`certify` accepts a subset of gates (`which=["G1", "G2"]`; a level governed by a gate that was not run is
indeterminate), per-gate keyword overrides and a global `rng`.  A list of per-segment arrays can be passed in place
of `X`.  A state series stored in a `.npy` or `.npz` file can be assessed from the command line with `ndif-certify`
(`ndif-certify --help`).

**Registering an estimator.** An estimator other than the built-in `ols`, `kalman` and `abc` must be registered before
it can be assessed:

```python
from ndif import register_estimator

def my_rate(segments_1d, dt):
    ...                                      # -> relaxation rate per unit time, or NaN
    return rate

register_estimator("mine", my_rate, observation_aware=False)
cert = certify(X, design, estimator="mine")
```

`fn(segments_1d, dt)` receives the one-dimensional segments the gates project onto and is called exactly as the
built-in estimators are, including inside G2's refits at each decimation level and G3's injection ladder.
`observation_aware` declares whether the estimator separates an additive white component from the latent process; it
decides whether the two-lag G1 binds and cannot be inferred.

## Reproducing the paper's numbers

| what | script | needs |
|---|---|---|
| report and barrier (above) | `examples/example_01_quickstart.py` | nothing |
| five simulated regimes | `examples/example_02_discrimination.py` | nothing |
| the non-neural series: the three regularly indexed series (Fig. 6a) and the CO₂ calendar sweep (Fig. 6b) | `examples/example_03_cross_domain.py --write` | the series bundled with `statsmodels` |
| benchmark with known rates: 320 outputs, 85.3% agreement (Fig. 3f) | `analyses/primary_configuration/benchmark_01_estimators.py --reps 20 --workers 3` | nothing |
| per-gate cost table (Extended Data Fig. 3a) | `analyses/primary_configuration/benchmark_02_gate_cost.py --reps 5` | nothing |
| Sleep-EDF cells, both estimators (Fig. 5c,d) | `analyses/primary_configuration/benchmark_05_corpus_full.py`, then `summarize_corpus_full.py` | `analyses/sleep_feature_cache` (supplied) |
| HCP MEG and COG-BCI load cohorts (Fig. 4d; HCP MEG with the G3 setting `boundary_rule=False`) | `analyses/workload_certificates/benchmark_06_load_cohorts.py` | MEG and COG-BCI state caches (source data) |
| extended simulation evaluation, CO₂, sleep re-evaluation (G8-augmented) | `analyses/g8_configuration/` | see its README |
| the principal reported counts, from the stored outputs | `verification/recount_stored_outputs.py` | nothing |

## The three claim classes

| level | question | quantities | status |
|---|---|---|---|
| **L1** state geometry | *Where is the system?* | displacement, separation, distribution shape | assessable from the distribution with finite-sample controls |
| **L2** state variability | *How does it explore?* | covariance, variance magnitude, dimensionality | assessable, and **not** a stability measure |
| **L3** intrinsic dynamics | *How does it evolve?* | relaxation rates (restoring forces, basins, escape are outside the current calibration generators) | **requires temporal information sufficient to recover latent rates** |

## The nine gates

Each isolates one failure mode and returns a **margin**, not a verdict alone.  A gate that cannot be evaluated
returns *indeterminate*, never a silent pass.

| gate | asks | statistic | governs | cost† |
|---|---|---|---|---|
| **G0** | Are windows overlapping? | step / window | L2, L3 | < 1 ms |
| **G1** | Is the rate protected from the white share of the variance? | ν, two-lag white share (primary) or the profile-likelihood rate bound (G8-augmented) | L3 | 0.01 s |
| **G2** | Is the rate an estimate or the sampling rate? | θ in λ ∝ Δt^(−θ) | L3 | 0.01 s |
| **G3** | What fraction of a known injected change comes back? | recovery slope, two channels | L2, L3 | 5.8 s |
| **G4** | Does the readout change when the fitted dynamics are replaced? | agreement with density-matched surrogates | L3 (primary); diagnostic (G8-augmented) | 1.4 s |
| **G5** | Does the separation clear the finite-sample floor? | observed vs zero-truth 95th percentile (primary) or permutation P (G8-augmented) | L1 | 1.9 s |
| **G6** | What are the counter's error rates here? | TPR / FPR under known truth | L1 | 15.7 s |
| **G7** | Is the rate summary carried by distinguishable modes? | share from distinguishable modes | L3 (primary); diagnostic (G8-augmented) | 1.3 s |
| **G8** | Does the estimator's single-timescale model describe the recording? | largest standardised autocorrelation deviation over lags 1–8 and a two-timescale likelihood ratio, both simulation-calibrated | L3 (G8-augmented) | not timed in `gate_cost.json` |

† median of five runs on 24,000 samples × 3 dimensions in 60 segments, one core, default replicate counts, `ols`;
stored in `analyses/primary_configuration/out/gate_cost.json`.  The gate medians sum to **26.1 s** for least squares
and **441.1 s** for maximum likelihood, because G2 and G3 refit the likelihood at every decimation level and every rung
of the injection ladder; G3 alone goes from 5.8 s to 412 s.

**G1, G2 and G3 are estimator-conditional.**  They ask their question of the rate that is actually going to be
reported.  **Every threshold is a keyword argument.**  Thresholds are operational settings, not a bound on the
proportion of inaccurate estimates among admissions; in the count comparison of the manuscript the best-agreeing
separation threshold is 3.35, against 2.31 under the reference definition.

## What the gates do on known truth

Five simulated regimes, same generator, one property changed at a time (`examples/example_02_discrimination.py`,
primary configuration, `ols`):

| regime | L2 | L3 | first failed gate | ν measured | rate recovery | variance recovery |
|---|---|---|---|---|---|---|
| clean, long blocks | admitted | **admitted** | — | 0.011 | 0.943 | 1.008 |
| additive white ν = 0.25 | admitted | refused | G1 | 0.256 | 0.228 | 1.004 |
| additive white ν = 0.50 | admitted | refused | G1 | 0.503 | 0.096 | 1.001 |
| 50 % window overlap | **refused** | refused | G0 | 0.011 | 0.943 | 1.008 |
| 8-sample blocks | admitted | refused | G3 | n/a | 0.743 | 1.010 |

The variance channel recovers an injected change at 1.00 ± 0.01 in every regime while the rate channel falls to 0.10:
L2 and L3 are not the same kind of quantity.

## Against the comparison methods

Ground truth: AR(1), a = 0.85, plus an additive white share ν, 20 replicates per cell (`benchmark_01_estimators.py`;
Extended Data Fig. 6).  Point estimate as a ratio to the truth, typical geometry (40 × 150):

| method | ν=0 | ν=0.2 | ν=0.4 | ν=0.6 | coverage at ν=0.4 | s per call |
|---|---|---|---|---|---|---|
| least squares AR(1) | 1.09 | 2.61 | 4.54 | **7.18** | **0.00** | 0.04 |
| AIC/BIC adequacy check | 1.09 | 2.61 | 4.54 | 7.18 | no interval | 0.01 |
| ABC, single timescale | 1.00 | 1.44 | 2.20 | **4.21** | **0.00** | 0.98 |
| profile likelihood | 1.09 | 1.14 | **1.15** | 1.21 | 0.65 | 5.8 |
| ABC, observation model | 0.88 | **1.02** | **1.09** | **1.09** | **1.00** | 1.06 |

NDIF wraps the same maximum-likelihood estimator that profile likelihood uses; what it adds is the *decision*: over 320 (recording, estimator) outputs of the
benchmark configuration, 91 accurate and 23 inaccurate estimates were admitted and 24 accurate and 182 inaccurate
estimates were not (273/320 agree with known accuracy, 85.3%; 23/114 admissions inaccurate).  Thirteen of the 23
inaccurate admissions lie in one cell (20 segments × 60 samples, ν = 0) where the median relative error is 0.285, just
outside the 25% tolerance.

## On the human recordings

Sleep-EDF, 602 night–stage cells (153 nights, 78 participants), 1-s windows, primary configuration
(`analyses/primary_configuration/out/corpus_full/`): least squares admits no L3 claim (590 refused, 557 of them first
at G1; 12 indeterminate); observation-aware maximum likelihood admits 87 (511 refused, 263 at G2 and 248 at G3; 4
indeterminate).  L2 is admitted in all 602 cells under both estimators.  HCP MEG (83 participants, trial clock): L1
72 admitted and 11 refused at G5, L2 83 admitted, L3 83 refused (82 at G3, 1 at G2).  COG-BCI workload (56 inputs):
L1 indeterminate (three labels), L2 56 admitted, L3 55 refused and 1 indeterminate.

## The minimum-action solver

`ndif.minimum_action` computes Freidlin–Wentzell quasipotential barriers: it minimises the discretised geometric action
with the exact analytic gradient with respect to every interior node, redistributes the nodes in arclength between
outer iterations, and refines on a coarse-to-fine node schedule, so a curved instanton is found without seeding the
solver with its shape.  `analyses/solver_validation/solver_validation.py` reproduces every value of the solver
validation reported in the paper (Extended Data Fig. 5).  A translation-only descent is kept as a regression case
(`tests/test_action.py::test_translation_only_descent_fails_where_ours_succeeds`): its gradient assigns one scalar per
coordinate to every interior node, so the curve can only translate, never bend.

| problem | exact | `ndif.minimum_action` | translation-only descent |
|---|---|---|---|
| 1-D double well, well → saddle | 5.0 | 5.000 | 5.000 (cannot discriminate) |
| 2-D curved instanton, min → saddle | 5.0 | 5.000021 (401 nodes) | 12.9 (regression test: 101 nodes, 300 iterations; 25.6 on the straight line) |
| 2-D curved instanton, min → min through the saddle | 5.0 | 5.000569 (641 nodes) | 69.178 (control of `analyses/solver_validation`: 81 nodes, 600 iterations; 108.18 on the straight line) |
| linear non-normal OU, d = 2–4 | ½xᵀΣ⁻¹x | rel. err 5 × 10⁻¹² – 9 × 10⁻⁷ | rel. err 1 × 10⁻² – 1.9 × 10⁻¹ |

Second-order convergence in node count; the analytic gradient agrees with finite differences to 2 × 10⁻⁹.

## Limitations

- **The white-share model is two-lag.** `white_share` assumes one relaxation timescale plus additive white noise.
  When that is wrong the two-lag solution leaves the admissible range and G1 returns **indeterminate** rather than a
  number — check `detail.two_lag.ok`.
- **ν is direction dependent.** Estimated on a task axis it is typically *lower* than on the orthogonal directions of
  the same state, so a task-axis ν is the most favourable value available, not a representative one.  G1 reports both.
- **Block-mean centring is not neutral.** It attenuates the propagator by roughly `1 − c/n_block` and leaves a bias
  the bootstrap does not remove, because the bootstrap regenerates a clean process whose long-run variance is already
  too small.  The report records the block length for this reason.
- **`bootstrap_bias_correct` removes finite-sample bias only.** Where white noise or block centring dominates it can
  make the estimate worse.  G1 and G2 are what detect that; do not read the bias-corrected number as clean.
- **G3 is an upper bound on what a recording can deliver**, not an estimate of it.  The simulated generator is linear
  with a single relaxation timescale; a real recording that violates either is worse off, not better.
- **Gate costs scale with replicate counts.** G6 at `n_rep=200` costs about 52 s.  Budget accordingly.
- **Assessments are per recording.** Pooling them across a cohort is a decision about the cohort, not a property the
  gates establish.

## Layout

```
ndif/
  certify.py        certify() and recommend_estimator() -> Certificate
  configurations.py the primary, benchmark and G8-augmented configurations
  gates.py          G0-G8, each gate(rec, **kw) -> GateResult
  estimators.py     propagator, rates, white share, bias correction, Lyapunov, register_estimator()
  action.py         geometric action and the minimum-action solver
  simulate.py       known-truth generators and the injections G3 pushes through
  baselines.py      the five comparison methods, for the benchmark
  types.py          Recording, Design, GateResult, Certificate
  report.py         text and matrix rendering
  cli.py            the ndif-certify command
tests/              pytest; closed-form recoveries, gate behaviour, governing rules
examples/           runnable examples
docs/               this file and GATE_SPECIFICATION.md
analyses/           drivers, settings and stored outputs of every reported analysis
figure_reproduction/  the stored summaries plotted in the figures (inputs/)
verification/       the recount of every reported number from the stored outputs
```

## Citation and licence

`CITATION.cff` carries the manuscript and software references in machine-readable form; see `LICENSE`.
