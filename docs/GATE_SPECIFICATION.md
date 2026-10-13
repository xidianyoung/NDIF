# The gate sequence — formal specification

Notation. A recording is a state series `Z = {z_t}` of dimension `d`, partitioned
into `S` segments of length `B` (trials, runs, epochs) between which lag-1 pairs
are not formed, sampled at interval `Δt`. `N = S·B`. Where a scalar rate is
required, `y_t = eᵀz_t` for a unit axis `e` (the design's axis, else the leading
principal component).

Observation model. Throughout,

&nbsp;&nbsp;&nbsp;&nbsp;`y_t = x_t + ε_t`, &nbsp;&nbsp; `x_t = a·x_{t−1} + w_t`, &nbsp;&nbsp; `ε ⊥ w` white,

with `λ = −log a / Δt` the relaxation rate and `ν = Var(ε)/Var(y)` the additive
white variance share. Two identities are used repeatedly and are exact under
this model:

&nbsp;&nbsp;&nbsp;&nbsp;`a = r(2)/r(1)`, &nbsp;&nbsp; `1 − ν = r(1)²/r(2)`, &nbsp;&nbsp; `λ_eff = λ − log(1−ν)/Δt`.

The third says a one-lag rate reports the instrument as well as the system. It
is the reason G1 exists, and the reason G1's verdict can depend on the estimator.

**An assessment is issued for a pair (recording, estimator), never for a
recording alone.** `ESTIMATORS` records, per estimator, whether it models the
observation process; a gate that fails for one may not fail for the other.

Three analysis configurations of this one gate set are defined in
`ndif/configurations.py` and in the section *Configurations* below: they fix
which gates run, which gates govern each claim class and in what order, and
which form G1, G3 and G5 take.

---

## Interface

```
certify(X, design, which=None, configuration="primary", estimator="ols", **gate_kwargs) -> Certificate
```

**Inputs.** `X`: `(N, d)` array, or a list of segments. `design`: `Δt`, feature
window and step, segment boundaries, condition labels, axis. `estimator`: the
estimator the assessment is about. `configuration`: `"primary"`, `"benchmark"`
or `"g8_augmented"`.

**Output.** For each level `L1, L2, L3`: `admitted | refused | indeterminate`,
the first failed gate, the margin at that gate, and — for a refusal — the
detection threshold. Plus the full gate trace and provenance.

**Dependencies.** `numpy ≥ 1.21`, `scipy ≥ 1.7`. Nothing else; no compiled
extension, no GPU, no configuration file.

---

## G0 — window overlap

*Are successive states built from shared samples?*

```
ratio ← step_s / window_s
pass  ← ratio ≥ 1
```

Governs **L2, L3**. Overlap injects a moving-average component that biases the
propagator toward spuriously slow relaxation. Indeterminate when the window and
step are not supplied — the gate cannot be inferred from the series.

Cost `O(1)`.

## G1 — additive white share

*Is the reported rate protected from the uncorrelated (white) share of the variance?*

Two-lag form (`rule="two_lag"`; primary and benchmark configurations):

```
r1, r2 ← pooled autocorrelations at lags 1, 2
a ← r2/r1 ;  ν ← 1 − r1²/r2
if ν ∉ [−tol, 1) or a ∉ (0,1):  return INDETERMINATE      # model violated
pass ← observation_aware(estimator)  or  ν ≤ ν_max        # default ν_max = 0.20
```

A small negative `ν` is a boundary fluctuation and is clipped; a large excursion
is a violation of the single-timescale-plus-white model and is reported as
indeterminate, not as a number. `ν` is also computed on `n_dir` random
directions, because a task axis typically carries a *lower* white share than the
orthogonal directions of the same state. Cost `O(N·d + n_dir·N)`.

Profile-likelihood form (`rule="profile"`; G8-augmented configuration):

```
ℓ(ν) ← profile log-likelihood of the observation model over a grid of ν, with a(ν) at the maximum
set ← {ν : 2(max ℓ − ℓ(ν)) ≤ χ²₁(level)}                 # default level = 0.95
dev ← max over the set of |log λ(ν) − log λ(ν_max)|
pass ← dev ≤ tol                                          # default tol = 0.20 log units
```

The same check for every estimator: does the recording determine the rate
despite the unknown white share? Indeterminate when the profile cannot be
computed. The two-lag estimate of `ν` is reported alongside.

Governs **L3**.

## G2 — sampling-interval dependence

*Is the rate an estimate, or the sampling rate?*

```
for m in factors:  λ̂(m) ← rate refitted on Z decimated by m, at Δt·m
θ ← −slope of  log λ̂  on  log(Δt·m)
CI(θ) ← residual bootstrap over decimation levels
pass ← |θ| ≤ θ_max  and  CI excludes 1
```

Governs **L3**. An identified rate is invariant to the sampling interval, so
`θ = 0`. A statistic that is merely the reciprocal of the sampling interval gives
`θ = 1`. Being far from 1 is not the same as being at 0, so both conditions are
required. Indeterminate with fewer than three usable decimation levels.

`rate` selects how `λ̂(m)` is obtained:

* `rate="estimator"` (the default; every configuration): the declared estimator is refitted on the
  projected series decimated by `m`; a factor whose fit returns no positive rate (for least squares, a lag-1
  factor outside (0, 1)) is not used, and a factor needs at least 12 decimated samples.
* `rate="propagator"` (optional): the geometric mean of the modal rates `−ln|μ_k| / (m·Δt)` of the
  least-squares propagator refitted on the decimated state, each segment centred on its own mean; each
  eigenvalue is read by its modulus, and a factor needs at least `4d + 4` decimated samples.  The rate does not
  depend on the declared estimator.  The three regularly indexed series of Fig. 6a use this form
  (`examples/example_03_cross_domain.py`).

Cost `O(|factors|·N·d² + n_boot)`.

## G3 — injection recovery

*What fraction of an injected change comes back through this chain?*

Fitted-generator form (`generator="fitted"`; primary and benchmark configurations):

```
g ← AR(1)+white maximum-likelihood fit to the projected series      # a₀, ν
if ν ≥ 1 − 10⁻⁶ and boundary_rule:  rate channel INDETERMINATE (no autocorrelated signal to simulate); variance channel still evaluated
for f in ladder, for r in 1..n_rep:
    simulate  a ← a₀^f  at the recording's own (S, B), re-inject ν
    λ̂ ← ESTIMATOR(simulated)                       # the declared estimator
    record (log f, log λ̂)
slope ← slope of log λ̂ on log f                    # the injected change is exactly log f
offset ← mean log λ̂ at the un-injected rung − log λ₀
pass ← |slope − 1| ≤ tol  and  |offset| ≤ bias_tol   # defaults 0.20, 0.20
variance channel: Q ← g·Q, log tr Σ̂ on log g; pass ← |slope − 1| ≤ tol
```

`boundary_rule=True` is the default.  With `boundary_rule=False` the rate channel is evaluated from the fit also at
the white-share boundary; the HCP MEG assessments (Fig. 4d; `analyses/workload_certificates/benchmark_06_load_cohorts.py`)
use this setting.  With `bias_tol=inf` the rate channel is decided on the slope alone, as for the three regularly
indexed series of Fig. 6a.

Corrected-generator form (`generator="corrected"`; G8-augmented configuration):

```
(a*, ν*) ← fit corrected by indirect inference at the recording's own segment lengths
simulate the ladder from (a*, ν*) with the recording's segment lengths; refit with the declared estimator
criteria on log λ̂: slope within tol of 1; offset within bias_tol; spread (s.d. at the un-injected rung) ≤ spread_tol
each criterion decided only beyond mc_z standard errors; repetitions doubled (up to max_rep_factor) while ambiguous,
then INDETERMINATE
variance channel: slope and spread of log tr Σ̂, decided in the same way
```

Governs **L3** (rate channel) and **L2** (variance channel). `a → a^f`
multiplies every relaxation rate by exactly `f`, so the injected change is exact
and the recovered fraction is a slope, not a ratio of noisy quantities.

The detection threshold follows from the same run: the spread at the un-injected
rung, divided by the recovered fraction, gives the smallest fold change
detectable at 80 % power at a two-sided 5 % level (equation (22) of the Methods).

**This gate is an upper bound on what the recording can deliver, not an estimate
of it.** The simulated generator is linear with a single relaxation timescale; a
recording that violates either is worse off, never better.

Cost `O(|ladder|·n_rep·(S·B·d² + C_est))`, where `C_est` is the declared
estimator's own cost — the dominant term for a maximum-likelihood estimator.

## G4 — drift replacement

*Does the readout change when the fitted dynamics are replaced while the density is kept?*

```
Σ ← stationary covariance of the fit
A_iso ← exp(−λ_ref·Δt)·I ,   Q_iso ← Σ − A_iso Σ A_isoᵀ      # exactly the same Σ; λ_ref = 1 per second
A_rnd ← random stable, Q_rnd ← Σ − A_rnd Σ A_rndᵀ      # largest modulus log-uniform in [0.05, 0.95]
if both surrogates match the fitted spectrum:  return INDETERMINATE
agreement ← max over surrogates of  P(|readout_surrogate − readout_fitted| ≤ scale)
pass ← agreement ≤ agree_max                            # default 0.95
```

**The reference rate is fixed and is deliberately not read from the fit.** A
surrogate built from the fit's own geometric-mean eigenvalue would preserve the
rate instead of replacing it, and in one dimension it *is* the fitted propagator
— so the gate would compare the process against itself. Where the surrogate
cannot be separated from the fit, the gate returns indeterminate.

Governs **L3** in the primary configuration; a diagnostic in the G8-augmented
configuration. Both surrogates preserve the stationary density *exactly*. A
readout that cannot tell either apart is a functional of the density — a
level-1 quantity, however it is interpreted. `readout` is user-supplied; the
default is the mean log rate relative to the reference rate.

Cost `O(3·n_rep·S·B·d²)`.

## G5 — finite-sample separation floor

*Does the observed separation exceed what this geometry produces without a condition effect?*

```
obs ← unbiased two-sample squared Mahalanobis separation between conditions (equation (18) of the Methods)
```

Simulated form (`null="simulated"`; primary and benchmark configurations):

```
null ← separations from n_rep draws of the FITTED process at the recording's
        own (S, B, d), same label vector — no condition structure by construction
floor ← q-th percentile of null                      # defaults n_rep = 200, q = 95
pass ← obs > floor
```

Permutation form (`null="permutation"`; G8-augmented configuration):

```
null ← separations after permuting the condition labels across segments (all arrangements when ≤ n_perm)
P ← share of the null at or above obs
pass ← P ≤ alpha beyond Monte Carlo error             # default alpha = 0.05; permutations doubled while ambiguous
```

Governs **L1**. The comparison is never against zero. Indeterminate without a
two-condition contrast, when a segment mixes conditions, or (permutation form)
when too few distinct arrangements exist.

Cost `O(n_rep·(S·B·d² + d³))`.

## G6 — basin-counter calibration

*What are the basin counter's error rates at this geometry?*

```
for n_rep draws:
    negative ← the FITTED linear process at this (S, B, d) — unimodal by
               construction, and carrying the recording's own autocorrelation
    positive ← the same, with half the samples displaced by δ·sd along e     # δ = 3.0
    count ← BIC prefers two components AND the fitted mixture has two modes
FPR ← P(count ≥ 2 | negative) ;  TPR ← P(count ≥ 2 | positive)
pass ← TPR ≥ tpr_min and FPR ≤ fpr_max               # defaults 0.80, 0.05
```

Governs **L1**. Mode counting is exact for a two-component normal mixture: any
additional mode must lie between the component means, so a dense scan of that
interval settles it. The calibration concerns the first state coordinate and
this counter; it does not establish a multivariate basin count.

**Error rates are a property of the recording, not of the counter.** This is why
the gate is per recording and why its threshold is refitted rather than imported.

Cost `O(n_rep·(S·B·d² + EM))`; the dominant gate in practice.

## G7 — contribution of distinguishable modes

*Is the reported rate summary carried by modes that are distinguishable from the surrogate?*

```
if the surrogate matches the fitted spectrum:  return INDETERMINATE
null ← relaxation spectra from n_rep draws of the uniform-decay surrogate of G4
identified_k ← λ̂_k outside the null's central 95 % for rank k
share ← Σ_{k identified} |log(λ̂_k/λ_ref)|  /  Σ_k |log(λ̂_k/λ_ref)|
pass ← share ≥ frac_min                               # default 0.5
```

Governs **L3** in the primary configuration; a diagnostic in the G8-augmented
configuration. A spectrum can look structured while most of its modes are
indistinguishable from the surrogate.

Cost `O(n_rep·S·B·d²)`.

## G8 — model adequacy

*Does the estimator's single-timescale model describe the recording?*

```
model ← AR(1) (least squares) or AR(1) + white (maximum likelihood), fitted to the projected series
(a*, ν*) ← indirect-inference correction of the fit at the recording's segment lengths (n_bias_iter steps of n_bias simulations)
for n_rep simulations of (a*, ν*) at the recording's segment lengths:
    T1 ← largest standardised deviation of the pooled autocorrelation from the simulated mean at lags 1..short_lag   # 8
    T2 ← likelihood ratio of a two-timescale-plus-white model against the single-timescale model,
          both refitted by the Whittle likelihood of the pooled periodograms
P ← the smaller of the two simulation-calibrated P values, itself calibrated on the same simulations
pass ← P > alpha beyond Monte Carlo error             # default alpha = 0.05; simulations doubled (up to max_rep_factor) while ambiguous
```

Indeterminate when a decision cannot be reached within Monte Carlo error after
the doublings, and when the segments are shorter than `min_cycles` (4)
relaxation times of the corrected model, because the within-segment
autocorrelation is then dominated by the centring of each segment.

Governs **L3** in the G8-augmented configuration only.

Cost `O(n_rep·(S·B + FFT))` plus the correction fits.

---

## Scoring

Gates are consulted per level in a fixed order; the **first gate that fails** is
the first failed (binding) gate.

| level | primary and benchmark configurations | G8-augmented configuration |
|---|---|---|
| L1 | G6, G5 | G6, G5 |
| L2 | G0, G3 (variance channel) | G0, G3 (variance channel) |
| L3 | G0, G1, G2, G3, G7, G4 (benchmark: G4 not run) | G0, G2, G8, G1, G3 |

- any governing gate fails → **refused**, with that gate and its margin;
- else any governing gate is indeterminate or was not run → **indeterminate**, naming which and why;
- else → **admitted**, with the tightest margin among the passed gates.

A failed gate takes precedence over an unavailable gate. An unevaluable gate is
never silently a pass. A refusal always carries its detection threshold, so it
is reported as a bound and never as "no effect".

## Configurations

| configuration | gates run | G1 | G3 | G5 | gate simulations |
|---|---|---|---|---|---|
| `primary` | G0–G7 | two-lag | fitted generator | simulated floor | the recording's number of segments at the rounded mean length |
| `benchmark` | G0, G1, G2, G3, G7 | two-lag | fitted generator | — | as primary |
| `g8_augmented` | G0–G3, G5, G6, G8; G4 and G7 as diagnostics | profile likelihood | corrected generator, Monte Carlo decision | permutation | the recording's own segment lengths |

The primary configuration governs the human applications (1,204 sleep cell
assessments and 336 adjacent-stage pairs, 83 HCP MEG and 56 COG-BCI assessments,
the three regularly indexed non-neural series); the benchmark configuration
assesses the rate gates on 320 estimator outputs with known rates; the
G8-augmented configuration is used for the extended simulation evaluation, the
calendar-preserving CO₂ analysis and the re-evaluation of the 602 sleep cells.

## Decision rule

```
recommend_estimator(X, design, candidates=None, configuration="primary") -> dict   # candidates default: ["ols", "kalman"]
```

Assesses once per candidate and returns the least costly estimator whose L3
verdict is *admitted*, with the full comparison. Where none is admitted it
returns `None` and the instruction to report the quantity as a bound with its
detection threshold. A cost-based ranking depends on the machine and does not
rank scientific accuracy.

## Complexity summary

Let `C = S·B·d²` be the cost of one propagator fit.

| gate | cost | default replicates |
|---|---|---|
| G0 | `O(1)` | — |
| G1 | `O(N·d)` | `n_dir = 8` (two-lag); profile grid (profile) |
| G2 | `O(4C + n_boot)` | `n_boot = 200` |
| G3 | `O(5·n_rep·(C + C_est))` | `n_rep = 24` |
| G4 | `O(3·n_rep·C)` | `n_rep = 40` |
| G5 | `O(n_rep·(C + d³))` | `n_rep = 200` (simulated); `n_perm = 1000` (permutation) |
| G6 | `O(n_rep·(C + EM))` | `n_rep = 60` |
| G7 | `O(n_rep·C)` | `n_rep = 120` |
| G8 | `O(n_rep·(S·B + FFT))` | `n_rep = 100`, `n_bias = 24` |

Every threshold and every replicate count is a keyword argument. Thresholds are
operational settings, not a bound on the proportion of inaccurate estimates
among admissions; in the count comparison of the manuscript the best-agreeing
separation threshold is 3.35, against 2.31 under the reference definition.
