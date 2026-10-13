"""The identifiability gate sequence, G0-G8.

Each gate isolates one way in which a recording can fail to determine a
dynamical quantity, and each returns a **margin** rather than a verdict alone.
A gate that cannot be evaluated returns ``passed=None`` and is reported as
*indeterminate*; it is never silently treated as a pass.

Every gate has the same signature ``gate(rec, **kw) -> GateResult`` and every
threshold is a keyword argument with a documented default.  Defaults are
starting points, not constants of nature: the framework's own rule is that a
threshold is refitted on the user's own records and never imported.  In the
count comparison of the manuscript the best-agreeing separation threshold is
3.35, against 2.31 under the reference definition, which is why nothing here
is hard-coded out of reach.
"""

from __future__ import annotations

import time

import numpy as np

from .estimators import (as_segments, fit_propagator, relaxation_rates,
                         white_share)
from .simulate import (add_white, is_degenerate_surrogate, matched_isotropic,
                       random_hurwitz_propagator, scale_rate, simulate_var1,
                       stationary_cov_discrete)
from .types import GateResult

__all__ = ["g0_overlap", "g1_instrument", "G1_RULES", "profile_noise_fraction", "g2_aliasing", "G2_RATES", "g3_recovery", "G3_GENERATORS", "G5_NULLS",
           "g4_deletion", "g5_floor", "g6_counter", "g7_modes", "g8_adequacy",
           "GATES", "DEFAULT_GATES", "DIAGNOSTICS", "run_gates"]


# ------------------------------------------------------------------ helpers
def _geometry(rec):
    segs = rec.segments()
    n_seg = len(segs)
    n_blk = int(round(float(np.mean([len(s) for s in segs]))))
    return segs, n_seg, max(n_blk, 3)


def _sim_lengths(rec, segment_lengths="actual"):
    """Segment lengths of the simulations.

    ``segment_lengths="actual"`` (G8-augmented configuration): those of the recording (at least three samples
    each).  ``"mean"`` (primary and benchmark configurations): the recording's number of segments, each of the
    rounded mean length.  Returns the list of lengths and the longest one; with segments of equal length the
    two are the same computation.
    """
    if segment_lengths == "mean":
        _, n_seg, n_blk = _geometry(rec)
        return [n_blk] * n_seg, n_blk
    if segment_lengths != "actual":
        raise ValueError(f"segment_lengths must be 'mean' or 'actual', not {segment_lengths!r}")
    lens = [max(int(len(s)), 3) for s in rec.segments()]
    return lens, max(lens)


def _simulate(A, Q, sl, rng=None):
    """Independent VAR(1) segments with the segment lengths of the recording.

    All segments are generated at the longest length with one random stream and each is cut to its own length,
    so that a recording with equal segments gives exactly the draws of :func:`simulate_var1`.
    """
    lens, n_sim = sl
    return [s[:L] for s, L in zip(simulate_var1(A, Q, len(lens), n_sim, rng=rng), lens)]


def _project(rec):
    """The one-dimensional series a scalar rate is actually read from."""
    segs = rec.segments()
    if rec.d == 1:
        return [s[:, 0] - s[:, 0].mean() for s in segs]
    axis = rec.design.axis
    if axis is None:
        Z = np.concatenate(segs, 0)
        Z = Z - Z.mean(0)
        _, _, Vt = np.linalg.svd(Z, full_matrices=False)
        axis = Vt[0]
    e = np.asarray(axis, float).ravel()
    e = e / np.linalg.norm(e)
    return [(s @ e) - (s @ e).mean() for s in segs]


def _log_rate_summary(A, dt, ref=1.0):
    """Mean of ln(lambda_k / ref) over the modes; ``ref`` is in the unit of the rates."""
    lam = relaxation_rates(A, dt)
    lam = lam[np.isfinite(lam) & (lam > 0)]
    if lam.size == 0:
        return np.nan
    return float(np.mean(np.log(lam / float(ref))))


def _reference_rate(rec, ref_rate, ref_unit):
    """Rate of the uniform-decay surrogate of G4 and G7, per unit of ``rec.design.dt``.

    The default is 1 per second for state series whose sampling interval is given in seconds.  It is converted with the unit of the design, so writing the same recording
    in milliseconds changes neither the surrogate nor the certificate.
    """
    from .types import rate_in_unit
    d = rec.design
    return rate_in_unit(ref_rate, ref_unit, getattr(d, "time_unit", "s"), d.dt)


def _fit(rec, demean="block"):
    return fit_propagator(rec.X, rec.design.segments, dt=rec.design.dt,
                          demean=demean)


def _timed(fn):
    def wrapped(*a, **kw):
        t0 = time.perf_counter()
        res = fn(*a, **kw)
        res.cost_s = float(time.perf_counter() - t0)
        return res
    wrapped.__name__ = fn.__name__
    wrapped.__doc__ = fn.__doc__
    return wrapped


# ---------------------------------------------------------------------- G0
@_timed
def g0_overlap(rec, **kw):
    """Are the feature windows overlapping?

    Overlapping windows inject a moving-average correlation that biases the
    propagator toward spuriously slow relaxation.  The statistic is
    ``step / window``; anything below 1 means successive states share samples.
    """
    d = rec.design
    if d.window_s is None or d.step_s is None or not d.window_s:
        return GateResult("G0", "Are the feature windows overlapping?",
                          "step/window", float("nan"),
                          "ratio >= 1 (no shared samples)", None, float("nan"),
                          ("L2", "L3"),
                          note="window_s/step_s not supplied; cannot be evaluated")
    ratio = float(d.step_s) / float(d.window_s)
    return GateResult("G0", "Are the feature windows overlapping?",
                      "step/window", ratio, "ratio >= 1 (no shared samples)",
                      bool(ratio >= 1.0), float(ratio - 1.0), ("L2", "L3"),
                      detail=dict(window_s=d.window_s, step_s=d.step_s,
                                  overlap_fraction=max(0.0, 1.0 - ratio)))


# ---------------------------------------------------------------------- G1
#: Grid of noise fractions of the profile likelihood of G1.
NU_PROFILE_GRID = np.round(np.concatenate([np.arange(0.0, 0.5, 0.025), np.arange(0.5, 0.95, 0.05)]), 4)


def profile_noise_fraction(proj, grid=None):
    """Profile log-likelihood of equation (6) over the noise fraction.

    At every noise fraction of ``grid`` the exact likelihood of the projected, segment-centred series is maximised
    over the persistence and the latent variance (Nelder-Mead from four starting persistences, the best kept).
    Returns the grid, the profile log-likelihoods and the persistences at their maxima.
    """
    from .estimators import _nelder_mead_batch, kalman_ll_batch, padded_centred
    grid = NU_PROFILE_GRID if grid is None else np.asarray(grid, float)
    Y, lens = padded_centred([proj])
    mask = None if len(set(lens)) == 1 else (np.arange(Y.shape[0])[:, None] < np.asarray(lens)[None, :]).astype(float)
    g = len(grid)
    Yb = np.repeat(Y, g, axis=1)
    v = float(np.sum(Y * Y) / sum(lens))
    if not v > 0:
        return grid, np.full(g, np.nan), np.full(g, np.nan)

    def nll(idx, Z):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            a = 1 / (1 + np.exp(-Z[:, 0]))
            s2 = np.exp(Z[:, 1])
        nu = grid[np.asarray(idx)]
        sw2 = s2 * (1 - a * a)
        se2 = np.where(nu > 0, nu / np.maximum(1 - nu, 1e-9) * s2, 1e-300)
        ok = (a > 0) & (a < 0.99999)
        out = np.full(len(idx), 1e12)
        if ok.any():
            iv = np.asarray(idx)[ok]
            val = -kalman_ll_batch(Yb[:, iv, :], a[ok], sw2[ok], se2[ok], mask=mask)
            out[ok] = np.where(np.isfinite(val), val, 1e12)
        return out

    best = None
    for a0 in (0.2, 0.5, 0.8, 0.95):
        Z0 = np.column_stack([np.full(g, np.log(a0 / (1 - a0))), np.log(np.maximum((1 - grid) * v, 1e-9 * v))])
        X, F, _ = _nelder_mead_batch(nll, Z0, maxiter=400, xatol=1e-6, fatol=1e-8)
        if best is None:
            best = (X, F)
        else:
            m = F < best[1]
            best = (np.where(m[:, None], X, best[0]), np.where(m, F, best[1]))
    X, F = best
    with np.errstate(over="ignore"):
        a = 1 / (1 + np.exp(-X[:, 0]))
    ll = np.where(F < 1e11, -F, np.nan)
    return grid, ll, a


def _g1_profile(rec, tol=0.20, level=0.95, grid=None, estimator="ols", rng=0, **kw):
    """G1 under the G8-augmented configuration: does the noise fraction leave the rate determined?

    Uncorrelated noise and fast relaxation have similar effects on the autocorrelation of a short or fast series, so
    the noise fraction can be poorly determined, and with it the rate: an estimator without a noise model then reads
    noise as fast relaxation, and one with a noise model can settle on the wrong noise fraction.  The check computes
    the profile likelihood of equation (6) over the noise fraction (:func:`profile_noise_fraction`), forms the
    ``level`` profile-likelihood confidence set of the noise fraction (likelihood-ratio statistic at most the
    chi-square(1) quantile) and reads the relaxation rate implied at each noise fraction of the set.  It passes when
    none of these rates differs by more than ``tol`` logarithmic units from the rate at the profile maximum, i.e. when
    the half-width of the profile-likelihood interval of the log rate along the noise fraction is at most ``tol``.  The
    check is the same for every estimator: it asks whether the recording determines the rate despite the unknown
    noise fraction, while the bias of a particular estimator is the subject of the recovery check (G3).  It is
    indeterminate when the profile cannot be computed.  The two-lag moment
    estimate of the white share is also reported.
    """
    from scipy.stats import chi2
    q = "Does the noise fraction leave the rate determined?"
    st = "half-width of the profile-likelihood interval of the log rate along the noise fraction"
    crit_txt = f"<= {tol} log units ({100 * level:g}% profile set)"
    dt = rec.design.dt
    proj = _project(rec)
    w = white_share(rec.X, rec.design.segments, axis=rec.design.axis)
    info = dict(two_lag=w, estimator=estimator)
    grid_, ll, a = profile_noise_fraction(proj, grid)
    if not np.any(np.isfinite(ll)):
        return GateResult("G1", q, st, float("nan"), crit_txt, None, float("nan"), ("L3",),
                          note="the profile likelihood could not be computed", detail=info)
    lmax = float(np.nanmax(ll))
    thr = float(chi2.ppf(level, 1))
    ok = np.isfinite(ll) & (2.0 * (lmax - ll) <= thr) & (a > 0) & (a < 1)
    if not ok.any():
        return GateResult("G1", q, st, float("nan"), crit_txt, None, float("nan"), ("L3",),
                          note="no noise fraction of the profile set has a usable persistence", detail=info)
    i_max = int(np.nanargmax(ll))
    if not (0 < a[i_max] < 1):
        return GateResult("G1", q, st, float("nan"), crit_txt, None, float("nan"), ("L3",),
                          note="the profile maximum has no usable persistence", detail=info)
    lam_max = -np.log(a[i_max]) / dt
    lam_p = -np.log(a[ok]) / dt
    dev = np.log(lam_p) - np.log(lam_max)
    d = float(np.max(np.abs(dev)))
    passed = bool(d <= tol)
    note = "" if passed else (f"noise fractions from {grid_[ok].min():.3f} to {grid_[ok].max():.3f} are compatible with "
                              f"the recording and change the rate by up to {d:.2f} log units")
    info.update(nu_profile_max=float(grid_[i_max]), rate_profile_max=float(lam_max),
                nu_set=[float(grid_[ok].min()), float(grid_[ok].max())],
                rate_set=[float(lam_p.min()), float(lam_p.max())],
                log_rate_deviation=[float(dev.min()), float(dev.max())], chi2_threshold=thr, level=float(level),
                grid=grid_.tolist(), profile_loglik=[float(v) for v in ll], profile_a=[float(v) for v in a])
    return GateResult("G1", q, st, d, crit_txt, passed, float(tol - d), ("L3",), note=note, detail=info)


def _g1_two_lag(rec, nu_max=0.20, n_dir=8, estimator="ols", rng=0, **kw):
    """G1 under the primary and benchmark configurations: how much of the uncorrelated share is the instrument?

    Estimates the additive white variance share ``nu`` from two lags.  A large
    ``nu`` inflates a one-lag rate by ``-ln(1-nu)/dt`` regardless of the
    system.

    **Whether that binds depends on the estimator being certified.**  An
    estimator that has no observation model -- least squares on lag-1 pairs,
    which is what is most often run -- takes the inflation straight into the
    rate, so the gate binds.  An estimator that fits the white component
    jointly does not, and the gate reports ``nu`` without binding.  This is
    why a certificate is issued *for an estimator on a recording*, not for a
    recording alone: identifiability is a property of the pair.

    ``nu`` is also direction dependent.  The gate reports the value on the
    design's own axis when one is given *and* the median over random
    directions, because a task axis typically carries a lower white share than
    the orthogonal directions of the same state -- i.e. it is the most
    favourable value available, not a representative one.
    """
    from .estimators import ESTIMATORS
    aware = bool(ESTIMATORS.get(estimator, {}).get("observation_aware", False))
    rng = np.random.default_rng(rng)
    w = white_share(rec.X, rec.design.segments, axis=rec.design.axis)
    nus = []
    if rec.d > 1:
        for _ in range(int(n_dir)):
            e = rng.normal(size=rec.d)
            wi = white_share(rec.X, rec.design.segments, axis=e)
            if wi["ok"]:
                nus.append(wi["nu"])
    nu_med = float(np.median(nus)) if nus else float("nan")
    nu = w["nu"]
    if not w["ok"]:
        return GateResult("G1", "How much of the uncorrelated share is the instrument?",
                          "nu (white variance share)", float(nu) if np.isfinite(nu) else float("nan"),
                          f"nu <= {nu_max}", None, float("nan"), ("L3",),
                          detail=dict(two_lag=w, nu_random_directions=nu_med),
                          note="two-lag solution outside the admissible range; the "
                               "single-timescale-plus-white model does not hold here")
    crit = (f"nu <= {nu_max} for an observation-naive estimator; "
            f"not binding for an observation-aware one")
    passed = True if aware else bool(nu <= nu_max)
    return GateResult("G1", "How much of the uncorrelated share is the instrument?",
                      "nu (white variance share)", float(nu), crit,
                      passed, float(np.inf if aware else nu_max - nu), ("L3",),
                      detail=dict(two_lag=w, nu_random_directions=nu_med,
                                  estimator=estimator, observation_aware=aware,
                                  rate_inflation_per_dt=float(-np.log(max(1 - nu, 1e-12)))),
                      note=("" if not aware else
                            f"nu = {nu:.3f} is carried by the estimator's own "
                            f"observation model, so it does not bind here; it would "
                            f"inflate an OLS lag-1 rate by "
                            f"{-np.log(max(1 - nu, 1e-12)):.3f} per dt"))


G1_RULES = ("two_lag", "profile")


@_timed
def g1_instrument(rec, rule="two_lag", **kw):
    """G1, additive white share: is the rate protected from the uncorrelated (white) share of the variance?

    ``rule`` selects the form of the check that the analysis configuration prescribes
    (:mod:`ndif.configurations`):

    * ``"two_lag"`` (primary and benchmark configurations): the two-lag moment estimate of the white share
      ``nu`` must not exceed ``nu_max`` (0.20) for an estimator without an observation model; the check does
      not bind for an observation-aware estimator (:func:`_g1_two_lag`).
    * ``"profile"`` (G8-augmented configuration): the log rate may vary by at most ``tol`` (0.20) over the
      ``level`` profile-likelihood set of the white share, for every estimator (:func:`_g1_profile`).
    """
    if rule == "two_lag":
        return _g1_two_lag(rec, **kw)
    if rule == "profile":
        return _g1_profile(rec, **kw)
    raise ValueError(f"unknown G1 rule {rule!r}; choose one of {G1_RULES}")


# ---------------------------------------------------------------------- G2
def _g2_theta(xs, ys, per, theta_max, n_boot, rng, **detail):
    """The decision of G2 from the log rates ``ys`` read at the log sampling intervals ``xs``.

    ``theta`` is minus the least-squares slope of ``ys`` on ``xs``.  Its interval is the 2.5-97.5 percentile range of
    ``n_boot`` refits after resampling the residuals over the decimation levels (residual bootstrap).  The check
    passes when ``|theta| <= theta_max`` and the interval excludes 1, and is indeterminate with fewer than three usable
    decimation levels.  ``detail`` is added to the result's detail.
    """
    if len(xs) < 3:
        return GateResult("G2", "Is the rate an estimate or the sampling rate?",
                          "theta in lambda ~ dt^(-theta)", float("nan"),
                          f"|theta| <= {theta_max} and CI excludes 1", None,
                          float("nan"), ("L3",), detail=dict(per_factor=per),
                          note="too few usable decimation levels")
    xs, ys = np.asarray(xs), np.asarray(ys)
    theta = float(-np.polyfit(xs, ys, 1)[0])
    # residual bootstrap over decimation levels
    res = ys - np.polyval(np.polyfit(xs, ys, 1), xs)
    bs = []
    for _ in range(int(n_boot)):
        yb = np.polyval(np.polyfit(xs, ys, 1), xs) + rng.choice(res, size=res.size, replace=True)
        bs.append(-np.polyfit(xs, yb, 1)[0])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    excludes_one = bool(hi < 1.0 or lo > 1.0)
    ok = bool(abs(theta) <= theta_max and excludes_one)
    return GateResult("G2", "Is the rate an estimate or the sampling rate?",
                      "theta in lambda ~ dt^(-theta)", theta,
                      f"|theta| <= {theta_max} and CI excludes 1", ok,
                      float(theta_max - abs(theta)), ("L3",),
                      detail=dict(ci=[float(lo), float(hi)], per_factor=per,
                                  excludes_one=excludes_one, **detail))


def _g2_estimator(rec, factors=(1, 2, 3, 4), theta_max=0.5, n_boot=200,
                  estimator="ols", rng=0, **kw):
    """G2 with the rate refitted by the declared estimator (the form used by every configuration).

    The projected series (:func:`_project`) is decimated by each factor ``m`` and the rate is refitted on it with
    the declared estimator (:func:`fit_rate`) at the sampling interval ``m dt``; a factor whose fit returns no
    positive rate (for least squares, a lag-1 factor outside (0, 1)) is not used.  A factor is used when the segments
    of at least ``3 m`` samples hold at least 12 decimated samples together.

    **The rate is refitted with the declared estimator**, not with a fixed one.
    Aliasing is a property of the estimator as much as of the sampling: a rate
    that is the reciprocal of the sampling interval under least squares need
    not be under an estimator that models the observation process.  Asking the
    question of a rate the user is not going to report would bind the gate for
    a reason that does not apply to them.
    """
    from .estimators import fit_rate
    rng = np.random.default_rng(rng)
    dt = rec.design.dt
    proj = _project(rec)
    xs, ys = [], []
    per = {}
    for m in factors:
        sub = [s[::int(m)] for s in proj if s.shape[0] >= 3 * int(m)]
        if len(sub) == 0 or sum(len(s) for s in sub) < 12:
            continue
        try:
            lam = fit_rate(sub, dt=dt * int(m), method=estimator,
                           seed=int(rng.integers(1 << 30)))
        except Exception:
            continue
        if not np.isfinite(lam) or lam <= 0:
            continue
        lr = float(np.log(lam))
        xs.append(np.log(dt * int(m)))
        ys.append(lr)
        per[int(m)] = float(lam)
    return _g2_theta(xs, ys, per, theta_max, n_boot, rng, estimator=estimator,
                     factors=[int(m) for m in factors])


def _g2_propagator(rec, factors=(1, 2, 3, 4), theta_max=0.5, n_boot=200, rng=0, **kw):
    """G2 with the rate read from the least-squares propagator of the decimated state.

    Every segment of the state is decimated by each factor ``m``, and the least-squares propagator is refitted on
    the decimated segments, each centred on its own mean (:func:`fit_propagator` with ``demean="block"``).  The rate
    at that factor is the geometric mean of the relaxation rates of the propagator's modes,
    ``lambda_k = -ln|mu_k| / (m dt)``: each eigenvalue ``mu_k`` is read by its modulus, so a negative lag-1 factor of
    a decimated series gives a finite rate rather than none.  A factor is used when the segments of at least ``3 m``
    samples hold at least ``4 d + 4`` decimated samples together.  The rate does not depend on the declared
    estimator.

    No configuration uses this form by default; it is selected with ``G2=dict(rate="propagator")``, as for the three
    regularly indexed series of ``examples/example_03_cross_domain.py``.
    """
    rng = np.random.default_rng(rng)
    segs = rec.segments()
    dt = rec.design.dt
    xs, ys = [], []
    per = {}
    for m in factors:
        sub = [s[::int(m)] for s in segs if s.shape[0] >= 3 * int(m)]
        if len(sub) == 0 or sum(len(s) for s in sub) < 4 * rec.d + 4:
            continue
        try:
            f = fit_propagator(sub, dt=dt * int(m), demean="block")
        except Exception:
            continue
        lr = _log_rate_summary(f.A, dt * int(m))
        if not np.isfinite(lr):
            continue
        xs.append(np.log(dt * int(m)))
        ys.append(lr)
        per[int(m)] = float(np.exp(lr))
    return _g2_theta(xs, ys, per, theta_max, n_boot, rng, rate="propagator",
                     factors=[int(m) for m in factors])


#: The ways G2 reads the rate at each decimation factor; the first is the default.
G2_RATES = ("estimator", "propagator")


@_timed
def g2_aliasing(rec, rate="estimator", **kw):
    """G2, sampling-interval dependence: is the rate an estimate, or the sampling rate?

    Refits the rate after decimating the series by each factor and fits
    ``lambda ~ dt^(-theta)``.  An identified rate is invariant to the sampling
    interval, so ``theta = 0``.  A statistic that is simply the reciprocal of
    the sampling interval gives ``theta = 1``.

    The gate binds unless ``|theta|`` is small *and* the bootstrap interval
    excludes 1 -- being far from 1 is not the same as being at 0.

    ``rate`` selects how the rate at each decimation factor is obtained:

    * ``"estimator"`` (default; every configuration): the rate refitted with the declared estimator on the decimated
      projected series (:func:`_g2_estimator`).
    * ``"propagator"``: the geometric mean of the modal relaxation rates of the least-squares propagator refitted on
      the decimated state, each eigenvalue read by its modulus, the same for every estimator
      (:func:`_g2_propagator`).
    """
    if rate == "estimator":
        return _g2_estimator(rec, **kw)
    if rate == "propagator":
        return _g2_propagator(rec, **kw)
    raise ValueError(f"unknown G2 rate {rate!r}; choose one of {G2_RATES}")


# ---------------------------------------------------------------------- Monte Carlo decisions
def _mc_criterion(stat, target, tol, se, z):
    """Pass / fail / ambiguous for one criterion ``|stat - target| <= tol`` (``target=None``: ``stat <= tol``).

    The criterion is decided only beyond Monte Carlo error: it passes when the margin exceeds ``z`` standard errors,
    fails when the margin is below ``-z`` standard errors, and is ambiguous otherwise.
    """
    if stat is None or not np.isfinite(stat):
        return "nan", float("nan")
    margin = float(tol - (abs(stat - target) if target is not None else stat))
    se = float(se) if se is not None and np.isfinite(se) else 0.0
    if margin > z * se:
        return "pass", margin
    if margin < -z * se:
        return "fail", margin
    return "ambiguous", margin


def _mc_combine(states):
    """Fail if any criterion fails; pass if every criterion passes; otherwise ambiguous (or not evaluable)."""
    if any(s == "fail" for s in states):
        return "fail"
    if any(s == "nan" for s in states):
        return "nan"
    if all(s == "pass" for s in states):
        return "pass"
    return "ambiguous"


def _decision_to_passed(dec):
    return {"pass": True, "fail": False}.get(dec, None)


# ---------------------------------------------------------------------- G3
#: A maximum-likelihood noise fraction at or above this value means that the fit placed the variance of the
#: autocorrelated component at zero. A recovery simulation from that fit contains no autocorrelated signal,
#: so it cannot test an estimator of the rate: the rate channel of G3 is then indeterminate, not binding.
NU_BOUNDARY = 1.0 - 1e-6


def _corrected_generator(a_hat, nu_hat, lens, n_ii, n_ii_iter, include_white, rng):
    """Indirect inference for the generator of equation (6).

    The maximum-likelihood fit of equation (6) to a short recording is biased (towards faster relaxation, mainly
    through the centring of each segment).  The generator is moved, in ``n_ii_iter`` additive steps of ``n_ii``
    simulations at the recording's segment lengths, so that the same fit applied to simulated data returns the
    observed fit on average: on the scale of the log-rate for the persistence and on the raw scale for the noise
    fraction.  Returns the corrected persistence and noise fraction, the Monte Carlo standard error of the last
    step's mean log-rate and the steps.
    """
    from .estimators import _kalman_fit_padded, simulate_eq6_padded
    rho_hat = float(np.log(-np.log(a_hat)))
    rho, nu_k = rho_hat, (float(nu_hat) if include_white else 0.0)
    se_gen, steps = 0.0, []
    for _ in range(int(n_ii_iter)):
        a_k = float(np.clip(np.exp(-np.exp(rho)), 1e-3, 0.999))
        Y = simulate_eq6_padded(a_k, nu_k, lens, int(n_ii), rng)
        res = _kalman_fit_padded(Y, lens)
        aj = np.array([r["a"] for r in res])
        nj = np.array([r["nu"] for r in res])
        ok = np.array([r["converged"] for r in res]) & (aj > 0) & (aj < 1)
        if int(ok.sum()) < 5:
            break
        rj = np.log(-np.log(aj[ok]))
        rho = rho + (rho_hat - float(np.mean(rj)))
        if include_white:
            nu_k = float(np.clip(nu_k + (float(nu_hat) - float(np.mean(nj[ok]))), 0.0, 0.95))
        se_gen = float(np.std(rj, ddof=1) / np.sqrt(ok.sum()))
        steps.append(dict(a=float(np.exp(-np.exp(rho))), nu=nu_k, mean_fit_log_rate=float(np.mean(rj)),
                          mean_fit_nu=float(np.mean(nj[ok])), n_fits=int(ok.sum())))
    a_star = float(np.clip(np.exp(-np.exp(rho)), 1e-6, 0.999999))
    return a_star, nu_k, se_gen, steps


def _mdc_fold(spread, slope):
    """Smallest fold change between two recordings of this size detected at 80% power, two-sided 5%."""
    if not (np.isfinite(spread) and np.isfinite(slope)) or abs(slope) < 1e-6:
        return float("nan")
    log_fold = (1.96 + 0.84) * np.sqrt(2.0) * spread / abs(slope)
    if not np.isfinite(log_fold) or log_fold > 20.0:
        return float("inf")
    return float(np.exp(log_fold))


def _regress(x, y):
    """Slope, its standard error, and the fitted line, for the recovery regression."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 4 or np.ptp(x) <= 0:
        return float("nan"), float("nan")
    b, c = np.polyfit(x, y, 1)
    r = y - (b * x + c)
    sxx = float(np.sum((x - x.mean()) ** 2))
    se = float(np.sqrt(np.sum(r * r) / max(x.size - 2, 1) / sxx)) if sxx > 0 else float("nan")
    return float(b), se


def _g3_corrected(rec, ladder=(0.6, 0.8, 1.0, 1.25, 1.7), n_rep=24, tol=0.20, bias_tol=0.20, spread_tol=0.20,
                  include_white=True, n_ii=32, n_ii_iter=2, mc_z=2.0, max_rep_factor=4, rng=0, **kw):
    """G3 under the G8-augmented configuration: does a known change come back, without bias, and with useful
    precision?

    The end-to-end check.  A generator of equation (6) is fitted to the recording by maximum likelihood and
    corrected by indirect inference (:func:`_corrected_generator`), because the fit is itself biased in short
    recordings and a check calibrated at a biased generator underestimates the bias it is meant to find.  Known changes are imposed on the corrected generator, simulated recordings with the recording's
    segment lengths and noise fraction are passed through the estimator being certified, and three criteria are
    evaluated on the logarithm of the rate:

    * **slope** of the recovered on the imposed change, within ``tol`` of 1 (sensitivity);
    * **offset**, the mean recovered value without a change minus the generator's own value, within ``bias_tol``
      (bias at the corrected generator);
    * **spread**, the standard deviation of the recovered value without a change, at most ``spread_tol``
      (precision at the size of the recording; this bounds the minimum detectable change).

    Each criterion is decided only beyond Monte Carlo error (``mc_z`` standard errors).  When a criterion is within
    that error of its threshold, the check is repeated with twice the repetitions, up to ``max_rep_factor`` times the
    default; a criterion that is still within Monte Carlo error is reported, and the check is indeterminate rather
    than decided by the random stream.

    The variance channel (level 2) imposes changes on the innovation covariance of a first-order autoregressive model
    of all coordinates and evaluates the slope and spread of the logarithm of the total variance in the same way.
    """
    from .estimators import fit_ar1_kalman, fit_ar1_kalman_batch, fit_rates_padded, simulate_eq6_padded
    rng = np.random.default_rng(rng)
    dt = rec.design.dt
    estimator = kw.get("estimator", "ols")
    fit = _fit(rec)
    segs, n_seg, n_blk = _geometry(rec)
    sl = _sim_lengths(rec)
    proj = _project(rec)
    lens = [max(int(len(s)), 3) for s in proj]

    g = fit_ar1_kalman_batch([proj])[0]
    if not g.get("converged", True):
        g = fit_ar1_kalman(proj, seed=int(rng.integers(1 << 30)))
    a_hat = float(np.clip(g["a"], 1e-6, 0.999999))
    nu_hat = float(g["nu"]) if include_white else 0.0
    nu_boundary = bool(include_white and nu_hat >= NU_BOUNDARY)

    # ---- rate channel (level 3)
    rate = dict(decision="nan", rounds=[])
    factor = 1
    while not nu_boundary:
        n_r, n_i = int(n_rep) * factor, int(n_ii) * factor
        a_star, nu_star, se_gen, steps = _corrected_generator(a_hat, nu_hat, lens, n_i, n_ii_iter,
                                                              include_white, rng)
        lam_star = -np.log(a_star) / dt
        A = np.concatenate([np.full(n_r, a_star ** float(f)) for f in ladder])
        X = np.concatenate([np.full(n_r, np.log(float(f))) for f in ladder])
        Y = simulate_eq6_padded(A, nu_star, lens, A.size, rng)
        lam = fit_rates_padded(Y, lens, dt=dt, method=estimator)
        ok = np.isfinite(lam) & (lam > 0)
        x, y = X[ok], np.log(lam[ok])
        slope, se_slope = _regress(x, y)
        at0 = y[np.isclose(x, 0.0)]
        if at0.size >= 3:
            offset = float(np.mean(at0) - np.log(lam_star))
            spread = float(np.std(at0, ddof=1))
            se_off = float(np.sqrt(spread ** 2 / at0.size + se_gen ** 2))
            se_spr = float(spread / np.sqrt(2.0 * (at0.size - 1)))
        else:
            offset = spread = se_off = se_spr = float("nan")
        crit = dict(slope=_mc_criterion(slope, 1.0, tol, se_slope, mc_z),
                    offset=_mc_criterion(offset, 0.0, bias_tol, se_off, mc_z),
                    spread=_mc_criterion(spread, None, spread_tol, se_spr, mc_z))
        dec = _mc_combine([c[0] for c in crit.values()])
        rate = dict(decision=dec, slope=slope, se_slope=se_slope, offset=offset, se_offset=se_off,
                    spread=spread, se_spread=se_spr, criteria={k: c[0] for k, c in crit.items()},
                    margins={k: c[1] for k, c in crit.items()}, a_generator=a_star, nu_generator=nu_star,
                    rate_generator=float(lam_star), se_generator_log_rate=se_gen, correction_steps=steps,
                    n_rep=n_r, n_ii=n_i, n_usable=int(ok.sum()), n_total=int(ok.size),
                    rounds=rate.get("rounds", []) + [dict(n_rep=n_r, decision=dec)])
        if dec != "ambiguous" or factor >= int(max_rep_factor):
            break
        factor *= 2

    # ---- variance channel (level 2)
    var = dict(decision="nan", rounds=[])
    nu_v = (rate.get("nu_generator", nu_hat) if include_white else 0.0)
    nu_v = 0.0 if nu_boundary else nu_v
    factor = 1
    while True:
        n_r = int(n_rep) * factor
        inj, got = [], []
        for f in ladder:
            for _ in range(n_r):
                sd = int(rng.integers(1 << 30))
                sim = _simulate(fit.A, fit.Q * float(f), sl, rng=sd)
                if nu_v > 0:
                    sim = add_white(sim, nu_v, rng=sd + 1)
                f2 = fit_propagator(sim, dt=dt, demean=fit.demean)
                Sig = stationary_cov_discrete(f2.A, f2.Q)
                v = float(np.log(max(np.trace(Sig), 1e-300)))
                if np.isfinite(v):
                    inj.append(np.log(float(f)))
                    got.append(v)
        inj, got = np.asarray(inj), np.asarray(got)
        s_v, se_sv = _regress(inj, got)
        at0 = got[np.isclose(inj, 0.0)]
        spr_v = float(np.std(at0, ddof=1)) if at0.size >= 3 else float("nan")
        se_spr_v = float(spr_v / np.sqrt(2.0 * (at0.size - 1))) if at0.size >= 3 else float("nan")
        crit_v = dict(slope=_mc_criterion(s_v, 1.0, tol, se_sv, mc_z),
                      spread=_mc_criterion(spr_v, None, spread_tol, se_spr_v, mc_z))
        dec_v = _mc_combine([c[0] for c in crit_v.values()])
        var = dict(decision=dec_v, slope=s_v, se_slope=se_sv, spread=spr_v, se_spread=se_spr_v,
                   criteria={k: c[0] for k, c in crit_v.items()}, margins={k: c[1] for k, c in crit_v.items()},
                   n_rep=n_r, rounds=var.get("rounds", []) + [dict(n_rep=n_r, decision=dec_v)])
        if dec_v != "ambiguous" or factor >= int(max_rep_factor):
            break
        factor *= 2

    passed = None if nu_boundary else _decision_to_passed(rate["decision"])
    margins = [m for m in rate.get("margins", {}).values() if np.isfinite(m)]
    margin = float(min(margins)) if margins else float("nan")
    if nu_boundary:
        note = (f"the maximum-likelihood fit places the variance of the autocorrelated component at zero "
                f"(nu = {nu_hat:.6f}); a recovery simulation from this fit contains no autocorrelated signal, so the "
                f"rate channel is indeterminate")
    elif rate["decision"] == "ambiguous":
        amb = [k for k, v in rate["criteria"].items() if v == "ambiguous"]
        note = (f"{', '.join(amb)} within Monte Carlo error of the criterion after {rate['n_rep']} repetitions per "
                f"imposed change; the check is indeterminate")
    elif rate["decision"] == "fail":
        failed = [k for k, v in rate["criteria"].items() if v == "fail"]
        note = f"fails on {', '.join(failed)}"
    elif rate["decision"] == "nan":
        note = "too few usable simulated estimates"
    else:
        note = ""
    var_passed = _decision_to_passed(var["decision"])
    vmargins = [m for m in var.get("margins", {}).values() if np.isfinite(m)]
    return GateResult("G3", "Does a known change come back, without bias and with useful precision?",
                      "recovered fraction (rate channel)", float(rate.get("slope", float("nan"))),
                      f"|slope - 1| <= {tol}, |offset| <= {bias_tol} and spread <= {spread_tol} log units, "
                      f"each beyond Monte Carlo error ({mc_z:g} s.e.)",
                      passed, margin, ("L3",), note=note,
                      detail=dict(rate_recovery=float(rate.get("slope", float("nan"))),
                                  bias_log=float(rate.get("offset", float("nan"))),
                                  spread_log=float(rate.get("spread", float("nan"))),
                                  rate=rate, variance=var,
                                  variance_recovery=float(var.get("slope", float("nan"))),
                                  variance_ok=var_passed,
                                  variance_margin=float(min(vmargins)) if vmargins else float("nan"),
                                  detection_margin_fold=_mdc_fold(rate.get("spread", float("nan")),
                                                                  rate.get("slope", float("nan"))),
                                  detection_margin_fold_variance=_mdc_fold(var.get("spread", float("nan")),
                                                                           var.get("slope", float("nan"))),
                                  a_fit=a_hat, nu_fit=nu_hat, nu_at_boundary=nu_boundary,
                                  estimator=estimator, ladder=list(ladder), tol=float(tol),
                                  bias_tol=float(bias_tol), spread_tol=float(spread_tol), mc_z=float(mc_z),
                                  n_segments=n_seg, n_block=n_blk,
                                  simulated_lengths="segment lengths of the recording",
                                  n_block_min=int(min(sl[0])), n_block_max=int(sl[1])))



def _g3_fitted(rec, ladder=(0.6, 0.8, 1.0, 1.25, 1.7), n_rep=24, tol=0.20,
                bias_tol=0.20, include_white=True, boundary_rule=True, rng=0, **kw):
    """G3 under the primary and benchmark configurations: what fraction of a known truth comes back?

    The end-to-end gate.  A generator is fitted to the recording, a *known*
    change is injected into it, the simulated data are pushed through the same
    estimator the user ran, and the recovered change is regressed on the
    injected one.  The slope is the recovered fraction; 1 is perfect recovery,
    0 is none.

    Two channels are reported separately because they behave differently:

    * **rate channel** -- every relaxation rate multiplied by ``f`` (a matrix
      power of the propagator), so the injected change in mean log-rate is
      exactly ``log f``;
    * **variance channel** -- innovation covariance multiplied by ``g``, so the
      injected change in ``log tr(Sigma)`` is exactly ``log g``.

    ``include_white`` re-injects the recording's own measured white share into
    the simulation, which is what makes the gate end-to-end rather than a test
    of the estimator in isolation.

    **Two conditions, not one.**  The slope answers "does an injected change
    come back", which is sensitivity.  It does not answer "is the returned
    value right": an estimator that is biased but linear scores a slope of 1
    while being systematically wrong, and precision and accuracy routinely move
    in opposite directions in exactly this setting.  The gate therefore also
    measures the **offset** at the un-injected rung, where the simulated truth
    is the fitted generator's own rate, and binds if either condition fails.
    Slope alone does not bound the bias of the returned rate, so the gate
    requires both conditions; this lowers false admissions at the cost of more
    refusals.  For a certification tool a false admit is the dangerous error
    and a miss is merely conservative.

    **White share at its boundary.**  When the maximum-likelihood fit places
    the white share at or above :data:`NU_BOUNDARY`, a simulation from the fit
    contains no autocorrelated signal, and with ``boundary_rule=True`` (the
    default) the rate channel is indeterminate while the variance channel is
    still evaluated.  With ``boundary_rule=False`` the rate channel is
    evaluated from that fit like any other; the HCP MEG assessments
    (``analyses/workload_certificates/benchmark_06_load_cohorts.py``) use this
    setting.
    """
    from .estimators import fit_ar1_kalman, fit_rate
    rng = np.random.default_rng(rng)
    dt = rec.design.dt
    estimator = kw.get("estimator", "ols")
    fit = _fit(rec)
    segs, n_seg, n_blk = _geometry(rec)

    # --- rate channel, on the scalar series the rate is actually read from.
    # Scaling a -> a**f multiplies the relaxation rate by exactly f, so the
    # injected truth in log-rate is exactly log f with no approximation.
    proj = _project(rec)
    g = fit_ar1_kalman(proj, seed=int(rng.integers(1 << 30)), equal_length="truncate")
    nu = g["nu"] if include_white else 0.0
    a0 = float(np.clip(g["a"], 1e-6, 0.999999))
    nu_boundary = bool(boundary_rule and include_white and nu >= NU_BOUNDARY)

    def measure_rate(f, seed):
        a = a0 ** float(f)
        r2 = np.random.default_rng(seed)
        Y = np.zeros((n_blk, n_seg))
        E = r2.normal(scale=np.sqrt(max(1 - a * a, 1e-12)), size=(n_blk, n_seg))
        Y[0] = r2.normal(size=n_seg)
        for t in range(1, n_blk):
            Y[t] = a * Y[t - 1] + E[t]
        if nu > 0:
            Y = Y + r2.normal(scale=np.sqrt(nu / max(1 - nu, 1e-9)), size=Y.shape)
        lam = fit_rate([Y[:, j] for j in range(n_seg)], dt=dt, method=estimator,
                       seed=seed)
        return np.log(lam) if np.isfinite(lam) and lam > 0 else np.nan

    def measure_var(gfac, seed):
        sim = simulate_var1(fit.A, fit.Q * gfac, n_seg, n_blk, rng=seed)
        if nu > 0:
            sim = add_white(sim, nu, rng=seed + 1)
        f2 = fit_propagator(sim, dt=dt, demean=fit.demean)
        Sig = stationary_cov_discrete(f2.A, f2.Q)
        return float(np.log(max(np.trace(Sig), 1e-300)))

    inj_r, got_r, inj_v, got_v = [], [], [], []
    for f in ladder:
        for _ in range(int(n_rep)):
            seed_r = int(rng.integers(1 << 30))     # drawn in every case: the variance
            if nu_boundary:                          # channel keeps its random stream
                continue
            lr = measure_rate(f, seed=seed_r)
            if np.isfinite(lr):
                inj_r.append(np.log(f))
                got_r.append(lr)
        for _ in range(int(n_rep)):
            lv = measure_var(f, seed=int(rng.integers(1 << 30)))
            if np.isfinite(lv):
                inj_v.append(np.log(f))
                got_v.append(lv)

    def slope(x, y):
        x, y = np.asarray(x), np.asarray(y)
        if x.size < 4 or np.ptp(x) <= 0:
            return float("nan")
        return float(np.polyfit(x, y, 1)[0])

    s_rate = slope(inj_r, got_r)
    s_var = slope(inj_v, got_v)

    def mde_fold(inj, got, s):
        """Smallest injected fold-change detectable at 80% power, two-sided 5%.

        The spread is taken at the un-injected rung and divided by the recovered
        fraction, because an effect that only partly survives the chain needs to
        be correspondingly larger to be seen.
        """
        inj, got = np.asarray(inj), np.asarray(got)
        at0 = got[np.isclose(inj, 0.0)]
        if at0.size < 3 or not np.isfinite(s) or abs(s) < 1e-6:
            return float("nan")
        sd = float(np.std(at0, ddof=1))
        log_fold = (1.96 + 0.84) * np.sqrt(2.0) * sd / abs(s)
        # A near-zero recovered fraction sends this to infinity; report that as
        # "no fold change would have been detected" rather than overflowing.
        if not np.isfinite(log_fold) or log_fold > 20.0:
            return float("inf")
        return float(np.exp(log_fold))

    # Offset at the un-injected rung: the simulated truth there is exactly the
    # fitted generator's own rate, so this is the chain's bias for this
    # estimator at this geometry, in log units.
    lam0 = -np.log(a0) / dt
    at0 = np.asarray([g for i, g in zip(inj_r, got_r) if np.isclose(i, 0.0)], float)
    bias_log = float(np.mean(at0) - np.log(lam0)) if at0.size >= 3 else float("nan")

    ok_slope = bool(np.isfinite(s_rate) and abs(s_rate - 1.0) <= tol)
    ok_bias = bool(np.isfinite(bias_log) and abs(bias_log) <= bias_tol)
    ok = bool(ok_slope and ok_bias)
    if nu_boundary:
        ok = None
    margin = min(tol - abs(s_rate - 1.0) if np.isfinite(s_rate) else np.nan,
                 bias_tol - abs(bias_log) if np.isfinite(bias_log) else np.nan)
    return GateResult("G3", "What fraction of a known truth comes back?",
                      "recovered fraction (rate channel)",
                      s_rate,
                      f"|slope - 1| <= {tol} AND |bias| <= {bias_tol} log units",
                      ok, float(margin), ("L3",),
                      note=((f"the maximum-likelihood fit places the variance of the autocorrelated "
                             f"component at zero (nu = {nu:.6f}); a recovery simulation from this fit "
                             f"contains no autocorrelated signal, so the rate channel is indeterminate")
                            if nu_boundary else "" if ok or not ok_slope else
                            f"slope is acceptable ({s_rate:.3f}) but the chain "
                            f"returns a biased rate at this geometry "
                            f"({bias_log:+.3f} log units): the estimate is "
                            f"precise about the wrong value"),
                      detail=dict(rate_recovery=s_rate, variance_recovery=s_var,
                                  bias_log=bias_log, slope_ok=ok_slope,
                                  bias_ok=ok_bias, bias_tol=float(bias_tol),
                                  variance_ok=bool(np.isfinite(s_var) and abs(s_var - 1) <= tol),
                                  detection_margin_fold=mde_fold(inj_r, got_r, s_rate),
                                  detection_margin_fold_variance=mde_fold(inj_v, got_v, s_var),
                                  nu_injected=float(nu), nu_at_boundary=nu_boundary,
                                  estimator=estimator,
                                  ladder=list(ladder),
                                  n_rep=int(n_rep), n_segments=n_seg, n_block=n_blk))


G3_GENERATORS = ("fitted", "corrected")


@_timed
def g3_recovery(rec, generator="fitted", **kw):
    """G3, injection recovery: does a known change imposed on a generator fitted to the recording come back
    through the estimator being assessed?

    ``generator`` selects the form of the check that the analysis configuration prescribes
    (:mod:`ndif.configurations`):

    * ``"fitted"`` (primary and benchmark configurations): the generator is the maximum-likelihood fit itself.
      The rate channel requires a recovery slope within ``tol`` of 1 and an offset within ``bias_tol``; the
      variance channel requires a slope within ``tol`` of 1.  When the fit places the white share at its
      boundary the rate channel is indeterminate and the variance channel is still evaluated, unless
      ``boundary_rule=False`` (:func:`_g3_fitted`).
    * ``"corrected"`` (G8-augmented configuration): the generator is corrected by indirect inference and the
      slope, offset and spread criteria are decided beyond Monte Carlo error (:func:`_g3_corrected`).
    """
    if generator == "fitted":
        return _g3_fitted(rec, **kw)
    if generator == "corrected":
        return _g3_corrected(rec, **kw)
    raise ValueError(f"unknown G3 generator {generator!r}; choose one of {G3_GENERATORS}")


# ---------------------------------------------------------------------- G4
Q_G4 = "Does the read-out change when the fitted dynamics are replaced?"
S_G4 = "agreement with dynamics-deleted surrogate"


@_timed
def g4_deletion(rec, readout=None, n_rep=40, agree_max=0.95, rtol=0.05, rng=0,
                ref_rate=1.0, ref_unit="s", segment_lengths="mean", **kw):
    """Does the read-out change when the fitted dynamics are replaced?

    Two surrogate dynamics are built that keep the stationary covariance
    *exactly* and are unrelated to how the recorded state moves: a uniform
    decay at the reference rate (``ref_rate`` per ``ref_unit``, default 1 per
    second, converted to the unit of ``dt``), and a random stable propagator.
    A read-out that cannot tell either apart from the fitted process is a
    functional of the density, and therefore a level-1 quantity however it is
    interpreted.

    For a level-3 claim the gate is cleared when the readout **does** move.

    ``readout`` is ``f(segments, dt) -> float``; the default is the mean of
    ln(lambda_k / lambda_ref) over the modes, the mean log-relaxation-rate
    measured from the reference rate, which does not depend on the time unit.
    ``segment_lengths`` sets the geometry of the surrogate recordings: the
    rounded mean segment length (``"mean"``, primary and benchmark
    configurations) or the recording's own segment lengths (``"actual"``,
    G8-augmented configuration).
    """
    rng = np.random.default_rng(rng)
    dt = rec.design.dt
    fit = _fit(rec)
    segs, n_seg, n_blk = _geometry(rec)
    sl = _sim_lengths(rec, segment_lengths)
    lam_ref = _reference_rate(rec, ref_rate, ref_unit)
    if readout is None:
        def readout(sim, dt):
            f = fit_propagator(sim, dt=dt, demean="block")
            return _log_rate_summary(f.A, dt, ref=lam_ref)

    A_iso, Q_iso = matched_isotropic(fit.A, fit.Q, dt=dt, rate=lam_ref)
    A_rnd, Q_rnd = random_hurwitz_propagator(fit.A, fit.Q, rng=rng)
    if is_degenerate_surrogate(fit.A, A_iso) and (
            A_rnd is None or is_degenerate_surrogate(fit.A, A_rnd)):
        return GateResult("G4", Q_G4, S_G4, float("nan"),
                          f"agreement <= {agree_max}", None, float("nan"), (),
                          note="the surrogate dynamics are not distinguishable "
                               "from the fitted propagator at this geometry, so the "
                               "gate would compare the process against itself")

    def sample(A, Q, n):
        out = []
        for _ in range(int(n)):
            sim = _simulate(A, Q, sl, rng=int(rng.integers(1 << 30)))
            v = readout(sim, dt)
            if np.isfinite(v):
                out.append(float(v))
        return np.asarray(out)

    base = sample(fit.A, fit.Q, n_rep)
    iso = sample(A_iso, Q_iso, n_rep)
    rnd = sample(A_rnd, Q_rnd, n_rep) if A_rnd is not None else np.array([])
    if base.size < 3 or iso.size < 3:
        return GateResult("G4", Q_G4, S_G4, float("nan"),
                          f"agreement <= {agree_max}", None, float("nan"), (),
                          note="too few usable replicates")

    scale = max(float(np.std(base)), abs(float(np.mean(base))) * rtol, 1e-12)
    agree_iso = float(np.mean(np.abs(iso[:, None] - base[None, :]) <= scale))
    agree_rnd = float(np.mean(np.abs(rnd[:, None] - base[None, :]) <= scale)) if rnd.size else float("nan")
    agree = float(np.nanmax([agree_iso, agree_rnd]))
    ok = bool(agree <= agree_max)
    return GateResult("G4", Q_G4, S_G4, agree,
                      f"agreement <= {agree_max}", ok, float(agree_max - agree),
                      (),
                      detail=dict(agreement_isotropic=agree_iso,
                                  agreement_random_hurwitz=agree_rnd,
                                  readout_mean_fitted=float(np.mean(base)),
                                  readout_mean_isotropic=float(np.mean(iso)),
                                  readout_sd_fitted=float(np.std(base)),
                                  random_surrogate_available=A_rnd is not None,
                                  reference_rate=float(lam_ref),
                                  time_unit=getattr(rec.design, "time_unit", "s")))


# ---------------------------------------------------------------------- G5
def _segment_labels(rec):
    """One label per segment, or None when a segment mixes conditions."""
    lab = np.asarray(rec.design.labels)
    segs = rec.design.segments
    if segs is None:
        bounds = [(0, rec.n_t)]
    else:
        segs = np.asarray(segs)
        if segs.ndim == 1 and segs.size == rec.n_t:
            bounds = [tuple(np.flatnonzero(segs == u)[[0, -1]] + [0, 1]) for u in np.unique(segs)]
        else:
            bounds = [(int(a), int(b)) for a, b in segs]
    out = []
    for a, b in bounds:
        u = np.unique(lab[a:b])
        if u.size != 1:
            return None, bounds
        out.append(u[0])
    return np.asarray(out), bounds


def _g5_permutation(rec, n_perm=1000, alpha=0.05, min_arrangements=20, mc_z=2.0, max_perm_factor=4, rng=0, **kw):
    """G5 under the G8-augmented configuration: does the observed separation exceed what relabelling the segments
    produces?

    The statistic is the distance between the condition means in standard deviations of the state (the
    bias-corrected Mahalanobis distance).  Its null distribution is obtained by permuting the condition labels across
    segments, keeping the number of segments of each condition: under no effect of condition, segments are
    exchangeable, and differences between segments of the same condition (drifts, blocks, nights) are part of the
    null.  The check passes when the permutation P value is at most ``alpha`` beyond Monte
    Carlo error; all arrangements are enumerated when there are at most ``n_perm`` of them.  It is indeterminate when
    the labels do not define two conditions, when a segment mixes conditions, or when fewer than
    ``min_arrangements`` distinct arrangements exist (the smallest attainable P value would exceed ``alpha``).
    """
    import itertools
    from math import comb
    q = "Does the observed separation exceed what relabelling the segments produces?"
    st = "separation (Mahalanobis)"
    crit_txt = f"permutation P <= {alpha} (labels permuted across segments)"
    rng = np.random.default_rng(rng)
    lab = rec.design.labels
    if lab is None:
        return GateResult("G5", q, st, float("nan"), crit_txt, None, float("nan"), ("L1",),
                          note="no condition labels supplied")
    lab = np.asarray(lab)
    if lab.shape[0] != rec.n_t or np.unique(lab).size != 2:
        return GateResult("G5", q, st, float("nan"), crit_txt, None, float("nan"), ("L1",),
                          note="labels must be one per sample with exactly two levels")
    seg_lab, bounds = _segment_labels(rec)
    if seg_lab is None:
        return GateResult("G5", q, st, float("nan"), crit_txt, None, float("nan"), ("L1",),
                          note="a segment contains both conditions, so labels cannot be permuted across segments")
    u = np.unique(lab)
    X = rec.X
    p = X.shape[1]
    S1 = np.array([X[a:b].sum(0) for a, b in bounds])                       # (n_seg, p)
    S2 = np.array([X[a:b].T @ X[a:b] for a, b in bounds])                  # (n_seg, p, p)
    N = np.array([b - a for a, b in bounds], float)

    def sep(is1):
        n1, n2 = N[is1].sum(), N[~is1].sum()
        if n1 < 2 or n2 < 2:
            return np.nan
        m1, m2 = S1[is1].sum(0) / n1, S1[~is1].sum(0) / n2
        W = (S2[is1].sum(0) - n1 * np.outer(m1, m1)) + (S2[~is1].sum(0) - n2 * np.outer(m2, m2))
        S = W / (n1 + n2 - 2) + 1e-10 * np.eye(p)
        dmu = m1 - m2
        d2 = float(dmu @ np.linalg.solve(S, dmu))
        nu_ = n1 + n2 - 2
        if nu_ > p + 2:
            d2 = (nu_ - p - 1) / nu_ * d2 - p * (1.0 / n1 + 1.0 / n2)
        return float(np.sqrt(max(d2, 0.0)))

    is1 = seg_lab == u[0]
    obs = sep(is1)
    n_seg, k1 = len(seg_lab), int(is1.sum())
    n_arr = comb(n_seg, k1)
    if k1 < 2 or n_seg - k1 < 2 or n_arr < int(min_arrangements):
        return GateResult("G5", q, st, float(obs), crit_txt, None, float("nan"), ("L1",),
                          note=f"{n_arr} distinct arrangements of the segment labels; the smallest attainable P value "
                               f"exceeds {alpha}",
                          detail=dict(n_segments=n_seg, n_segments_condition1=k1, n_arrangements=n_arr))
    if n_arr <= int(n_perm):
        null = []
        for idx in itertools.combinations(range(n_seg), k1):
            m = np.zeros(n_seg, bool)
            m[list(idx)] = True
            null.append(sep(m))
        null = np.asarray(null)
        pval = float(np.mean(null >= obs - 1e-12))
        exact, n_used, dec = True, int(n_arr), ("pass" if pval <= alpha else "fail")
    else:
        null = []
        factor = 1
        while True:
            target = int(n_perm) * factor
            while len(null) < target:
                null.append(sep(rng.permutation(is1)))
            nn = np.asarray(null)
            pval = float((1 + np.sum(nn >= obs - 1e-12)) / (1 + nn.size))
            se = float(np.sqrt(pval * (1 - pval) / nn.size))
            dec = "pass" if pval + mc_z * se <= alpha else ("fail" if pval - mc_z * se > alpha else "ambiguous")
            if dec != "ambiguous" or factor >= int(max_perm_factor):
                break
            factor *= 2
        null = np.asarray(null)
        exact, n_used = False, int(null.size)
    floor = float(np.percentile(null, 100 * (1 - alpha)))
    passed = _decision_to_passed(dec)
    note = ("" if dec != "ambiguous" else
            f"permutation P value {pval:.3f} within Monte Carlo error of {alpha} after {n_used} permutations")
    return GateResult("G5", q, st, float(obs), crit_txt, passed, float(obs - floor), ("L1",), note=note,
                      detail=dict(floor=floor, p_value=pval, null_median=float(np.median(null)),
                                  n_permutations=n_used, exact=exact, n_segments=n_seg,
                                  n_segments_condition1=k1, alpha=float(alpha)))



def _g5_simulated(rec, n_rep=200, q=95.0, rng=0, **kw):
    """G5 under the primary and benchmark configurations: does the observed separation clear the realistic floor?

    Builds a zero-truth generator at the recording's *own* geometry -- same
    dimension, same segment count, same block length, same stationary
    covariance, one basin -- and reads the ``q``-th percentile of the
    separation it produces by chance.  The observed separation is compared
    against that, never against zero.
    """
    rng = np.random.default_rng(rng)
    lab = rec.design.labels
    if lab is None:
        return GateResult("G5", "Does observed separation clear the realistic floor?",
                          "separation (Mahalanobis)", float("nan"),
                          f"observed > {q}th pct of zero-truth", None, float("nan"),
                          ("L1",), note="no condition labels supplied")
    lab = np.asarray(lab)
    if lab.shape[0] != rec.n_t or np.unique(lab).size != 2:
        return GateResult("G5", "Does observed separation clear the realistic floor?",
                          "separation (Mahalanobis)", float("nan"),
                          f"observed > {q}th pct of zero-truth", None, float("nan"),
                          ("L1",), note="labels must be one per sample with exactly two levels")
    u = np.unique(lab)

    def sep(X, l):
        a, b = X[l == u[0]], X[l == u[1]]
        if a.shape[0] < 2 or b.shape[0] < 2:
            return np.nan
        S = ((a.shape[0] - 1) * np.cov(a, rowvar=False) +
             (b.shape[0] - 1) * np.cov(b, rowvar=False)) / (a.shape[0] + b.shape[0] - 2)
        S = np.atleast_2d(S) + 1e-10 * np.eye(X.shape[1])
        dmu = a.mean(0) - b.mean(0)
        d2 = float(dmu @ np.linalg.solve(S, dmu))
        # unbiased two-sample correction for the squared Mahalanobis distance
        n1, n2, p = a.shape[0], b.shape[0], X.shape[1]
        nu_ = n1 + n2 - 2
        if nu_ > p + 2:
            d2 = (nu_ - p - 1) / nu_ * d2 - p * (1.0 / n1 + 1.0 / n2)
        return float(np.sqrt(max(d2, 0.0)))

    obs = sep(rec.X, lab)
    fit = _fit(rec)
    segs, n_seg, n_blk = _geometry(rec)
    # The zero-truth generator is the fitted process itself: same geometry,
    # same temporal correlation, and no condition structure by construction.
    # Substituting a rate-deleted surrogate here would change the effective
    # sample size and therefore the floor.
    A_iso, Q_iso = fit.A, fit.Q
    null = []
    for _ in range(int(n_rep)):
        sim = np.concatenate(simulate_var1(A_iso, Q_iso, n_seg, n_blk,
                                           rng=int(rng.integers(1 << 30))), axis=0)
        m = min(sim.shape[0], rec.n_t)
        v = sep(sim[:m], lab[:m])
        if np.isfinite(v):
            null.append(v)
    if len(null) < 20:
        return GateResult("G5", "Does observed separation clear the realistic floor?",
                          "separation (Mahalanobis)", float(obs),
                          f"observed > {q}th pct of zero-truth", None, float("nan"),
                          ("L1",), note="null did not produce enough usable replicates")
    floor = float(np.percentile(null, q))
    ok = bool(np.isfinite(obs) and obs > floor)
    return GateResult("G5", "Does observed separation clear the realistic floor?",
                      "separation (Mahalanobis)", float(obs),
                      f"observed > {q}th pct of zero-truth ({floor:.3f})", ok,
                      float(obs - floor), ("L1",),
                      detail=dict(floor=floor, null_median=float(np.median(null)),
                                  n_null=len(null), q=q))


G5_NULLS = ("simulated", "permutation")


@_timed
def g5_floor(rec, null="simulated", **kw):
    """G5, finite-sample separation floor: does the observed separation between two conditions exceed what the
    recording produces without a condition effect?

    ``null`` selects the reference distribution that the analysis configuration prescribes
    (:mod:`ndif.configurations`):

    * ``"simulated"`` (primary and benchmark configurations): the ``q``-th percentile (95) of the separation over
      ``n_rep`` (200) simulations of the fitted generator with the recording's dimension, segment count, block
      length and label vector and no condition effect (:func:`_g5_simulated`).
    * ``"permutation"`` (G8-augmented configuration): the condition labels are permuted across segments and the
      check passes when the permutation P value is at most ``alpha`` beyond Monte Carlo error
      (:func:`_g5_permutation`).
    """
    if null == "simulated":
        return _g5_simulated(rec, **kw)
    if null == "permutation":
        return _g5_permutation(rec, **kw)
    raise ValueError(f"unknown G5 null {null!r}; choose one of {G5_NULLS}")


# ---------------------------------------------------------------------- G6
def _bimodal_1d(x, n_restart=2, rng=0, n_max=3000, tol=1e-7, max_iter=200):
    """Two-component 1-D Gaussian mixture by EM, with an exact mode count.

    Returns ``(n_modes, bic_prefers_two)``.  The mode count is exact for a
    two-component normal mixture -- any extra mode must lie between the
    component means, so a dense scan of that interval settles it -- rather than
    a grid search over the ambient space.
    """
    rng = np.random.default_rng(rng)
    x = np.asarray(x, float).ravel()
    if x.size > n_max:
        x = rng.choice(x, size=n_max, replace=False)
    n = x.size
    if n < 20:
        return 1, False
    v = float(np.var(x))
    if v <= 0:
        return 1, False
    ll1 = -0.5 * n * (np.log(2 * np.pi * v) + 1.0)
    bic1 = -2 * ll1 + 2 * np.log(n)

    def em(m0):
        m = np.asarray(m0, float).copy()
        s = np.full(2, np.sqrt(v))
        p = np.full(2, 0.5)
        ll_prev = -np.inf
        for _ in range(int(max_iter)):
            lp = np.stack([np.log(p[k]) - 0.5 * np.log(2 * np.pi * s[k] ** 2)
                           - 0.5 * ((x - m[k]) / s[k]) ** 2 for k in range(2)], 1)
            mx = lp.max(1, keepdims=True)
            ex = np.exp(lp - mx)
            tot = ex.sum(1, keepdims=True)
            ll = float(np.sum(mx.ravel() + np.log(tot.ravel())))
            w = ex / tot
            nk = w.sum(0) + 1e-12
            m = (w * x[:, None]).sum(0) / nk
            s = np.sqrt(np.maximum((w * (x[:, None] - m) ** 2).sum(0) / nk, 1e-10 * v))
            p = nk / n
            if ll - ll_prev < tol * max(1.0, abs(ll)):
                break
            ll_prev = ll
        return ll, m, s, p

    # one deterministic start at the quartiles, then random starts
    inits = [np.percentile(x, [25, 75])]
    inits += [rng.choice(x, size=2, replace=False) for _ in range(max(0, int(n_restart) - 1))]
    best = max((em(m0) for m0 in inits), key=lambda t: t[0])
    ll2, m, s, p = best
    bic2 = -2 * ll2 + 5 * np.log(n)
    # exact mode count of the fitted two-component mixture on a fine grid of
    # the interval between the component means (where any extra mode must lie)
    lo, hi = float(min(m)), float(max(m))
    if hi - lo <= 0:
        return 1, bool(bic2 < bic1)
    t = np.linspace(lo - 1e-9, hi + 1e-9, 2001)
    dens = sum(p[k] * np.exp(-0.5 * ((t - m[k]) / s[k]) ** 2) / (s[k] * np.sqrt(2 * np.pi))
               for k in range(2))
    interior = dens[1:-1]
    n_modes = 1 + int(np.sum((interior > dens[:-2]) & (interior > dens[2:])))
    return int(min(n_modes, 2)), bool(bic2 < bic1)


@_timed
def g6_counter(rec, n_rep=60, tpr_min=0.80, fpr_max=0.05, delta=3.0, rng=0, segment_lengths="mean", **kw):
    """What are the basin counter's error rates at *this* geometry?

    Runs the counter on a known single-basin process and on a known bistable
    mixture with separation ``delta``, both built at the recording's own
    dimension, segment count and block length.  Reports the false-positive rate
    (a one-basin truth called multistable) and the true-positive rate.

    A counter's error rates are a property of the recording, not of the
    counter, which is why this is a per-recording gate.
    """
    rng = np.random.default_rng(rng)
    fit = _fit(rec)
    segs, n_seg, n_blk = _geometry(rec)
    sl = _sim_lengths(rec, segment_lengths)
    # Known single-basin truth at this recording's own geometry and temporal
    # correlation: the fitted linear process is unimodal by construction.
    A_iso, Q_iso = fit.A, fit.Q
    Sig = stationary_cov_discrete(A_iso, Q_iso)
    e = np.zeros(rec.d)
    e[0] = 1.0
    sd = float(np.sqrt(max(e @ Sig @ e, 1e-300)))

    fp = tp = 0
    n_fp = n_tp = 0
    for _ in range(int(n_rep)):
        s1 = np.concatenate(_simulate(A_iso, Q_iso, sl, rng=int(rng.integers(1 << 30))), axis=0)
        nm, bic2 = _bimodal_1d(s1 @ e, rng=int(rng.integers(1 << 30)))
        fp += int(nm >= 2 and bic2)
        n_fp += 1
        s2 = np.concatenate(_simulate(A_iso, Q_iso, sl, rng=int(rng.integers(1 << 30))), axis=0)
        shift = (np.arange(s2.shape[0]) < s2.shape[0] // 2).astype(float)
        s2 = s2 + np.outer(shift * delta * sd, e)
        nm2, bic2b = _bimodal_1d(s2 @ e, rng=int(rng.integers(1 << 30)))
        tp += int(nm2 >= 2 and bic2b)
        n_tp += 1
    fpr = fp / max(n_fp, 1)
    tpr = tp / max(n_tp, 1)
    ok = bool(tpr >= tpr_min and fpr <= fpr_max)
    return GateResult("G6", "What are the counter's error rates under known truth?",
                      "TPR at delta=%.1f / FPR" % delta, float(tpr),
                      f"TPR >= {tpr_min} and FPR <= {fpr_max}", ok,
                      float(min(tpr - tpr_min, fpr_max - fpr)), ("L1",),
                      detail=dict(tpr=float(tpr), fpr=float(fpr), delta=float(delta),
                                  n_rep=int(n_rep)))


# ---------------------------------------------------------------------- G7
@_timed
def g7_modes(rec, n_rep=120, frac_min=0.5, rng=0, ref_rate=1.0, ref_unit="s", segment_lengths="mean", **kw):
    """Is the rate summary carried by identified modes?

    Each fitted mode is compared against the spread of the fitted rates under
    the uniform-decay surrogate of G4: the same stationary covariance, every
    mode relaxing at the reference rate lambda_ref (``ref_rate`` per
    ``ref_unit``, default 1 per second, converted to the unit of ``dt``).  A
    mode whose fitted rate lies inside the central 95% of that spread is not
    distinguishable from the surrogate.  The statistic is the share of
    sum_k |ln(lambda_k / lambda_ref)| carried by the modes that *are*
    distinguishable: each mode is weighted by its log-distance from the
    reference rate, which does not depend on the unit of time.
    """
    rng = np.random.default_rng(rng)
    dt = rec.design.dt
    fit = _fit(rec)
    segs, n_seg, n_blk = _geometry(rec)
    sl = _sim_lengths(rec, segment_lengths)
    lam_ref = _reference_rate(rec, ref_rate, ref_unit)
    A_iso, Q_iso = matched_isotropic(fit.A, fit.Q, dt=dt, rate=lam_ref)
    if is_degenerate_surrogate(fit.A, A_iso):
        return GateResult("G7", "Is the rate summary carried by identified modes?",
                          "share of rate summary from identified modes", float("nan"),
                          f"share >= {frac_min}", None, float("nan"), (),
                          note="the uniform-decay surrogate coincides with the fitted "
                               "propagator at this geometry, so no mode could be "
                               "distinguished from it even in principle")

    obs = relaxation_rates(fit.A, dt)
    null = []
    for _ in range(int(n_rep)):
        sim = _simulate(A_iso, Q_iso, sl, rng=int(rng.integers(1 << 30)))
        try:
            f2 = fit_propagator(sim, dt=dt, demean=fit.demean)
        except Exception:
            continue
        null.append(relaxation_rates(f2.A, dt))
    if len(null) < 20:
        return GateResult("G7", "Is the rate summary carried by identified modes?",
                          "share of rate summary from identified modes", float("nan"),
                          f"share >= {frac_min}", None, float("nan"), (),
                          note="null did not produce enough usable replicates")
    null = np.asarray(null)
    lo = np.percentile(null, 2.5, axis=0)
    hi = np.percentile(null, 97.5, axis=0)
    ident = (obs < lo) | (obs > hi)
    ok_obs = np.isfinite(obs) & (obs > 0)
    # weight: log-distance of each mode from the reference rate (unit-free)
    w = np.where(ok_obs, np.abs(np.log(np.clip(obs, 1e-300, None) / lam_ref)), 0.0)
    share = float(w[ident & ok_obs].sum() / w.sum()) if w.sum() > 0 else float("nan")
    ok = bool(np.isfinite(share) and share >= frac_min)
    return GateResult("G7", "Is the rate summary carried by identified modes?",
                      "share of rate summary from identified modes", share,
                      f"share >= {frac_min}", ok,
                      float(share - frac_min) if np.isfinite(share) else float("nan"),
                      (),
                      detail=dict(n_modes=int(obs.size),
                                  n_identified=int(ident.sum()),
                                  rates=obs.tolist(),
                                  null_lo=lo.tolist(), null_hi=hi.tolist(),
                                  weights=w.tolist(),
                                  reference_rate=float(lam_ref),
                                  time_unit=getattr(rec.design, "time_unit", "s")))


# ---------------------------------------------------------------------- G8
@_timed
def g8_adequacy(rec, short_lag=8, min_cycles=4.0, n_rep=100, n_bias=24, n_bias_iter=2, alpha=0.05, mc_z=2.0,
                max_rep_factor=4, rng=0, **kw):
    """Does the estimator's single-timescale model describe the recording?

    The certified estimator assumes a first-order autoregressive process (``model="ar1"``, least squares) or that
    process plus additive uncorrelated noise (``"ar1_white"``, equation (6)).  The model is fitted to the projected
    series (maximum likelihood of equation (6), or the least-squares lag-1 factor) and corrected by indirect inference
    (``n_bias_iter`` additive steps of ``n_bias`` simulations), because centring each segment biases the fit of short
    segments.  The corrected model is simulated ``n_rep`` times with the recording's segment lengths, and two
    statistics are computed on the recording and on every simulation:

    * the largest standardised deviation of the pooled autocorrelation from the simulated mean at lags 1 to
      ``short_lag`` (localised misfit such as autocorrelated noise or oscillation);
    * the likelihood ratio of a model with two latent timescales plus uncorrelated noise against equation (6), both
      refitted on every simulated recording by maximising the spectral (Whittle) likelihood of the pooled
      periodograms (several timescales, long memory, autocorrelated noise).

    The smaller of their two simulation-calibrated P values is itself calibrated on the same simulations, so a
    recording that the model describes is refused with a nominal probability of ``alpha``.  The combined P value is
    decided only beyond Monte Carlo error (``mc_z`` standard errors); otherwise the simulations are doubled, up to
    ``max_rep_factor`` times, and the check is then indeterminate.

    The check is indeterminate when the segments are shorter than ``min_cycles`` relaxation times of the corrected
    model: the autocorrelation within such segments is dominated by the centring of each segment, so it does not
    determine the model (without this condition the check refuses most recordings at persistence 0.98 in segments of
    40 samples although the model holds).  The recovery check (G3) still evaluates such recordings.
    """
    from .estimators import (ESTIMATORS, _kalman_fit_padded, fit_model, ols_persistence_padded, padded_centred,
                             pooled_acf_padded, simulate_eq6_padded, whittle_lr_two_timescale)
    rng = np.random.default_rng(rng)
    estimator = kw.get("estimator", "ols")
    model = ESTIMATORS.get(estimator, {}).get("model", "ar1")
    question = "Does the estimator's single-timescale model describe the recording?"
    K = int(short_lag)
    stat = (f"combined P value of the largest standardised autocorrelation deviation at lags 1-{K} and of the "
            f"likelihood ratio against two timescales")
    crit_txt = f"combined P value > {alpha} beyond Monte Carlo error"
    proj = _project(rec)
    lens = [int(len(s)) for s in proj]
    L_med = float(np.median(lens))

    def indet(note, **extra):
        return GateResult("G8", question, stat, float("nan"), crit_txt, None, float("nan"), ("L3",), note=note,
                          detail=dict(estimator=estimator, model=model, alpha=float(alpha), **extra))

    a, nu = fit_model(proj, model)
    if not (np.isfinite(a) and 0.0 < a < 1.0):
        return indet("the estimator's model returned no usable lag-1 factor, so there is no model to test")
    if nu >= NU_BOUNDARY:
        return indet("the maximum-likelihood fit places the variance of the autocorrelated component at zero "
                     f"(nu = {nu:.6f}); the fitted model has no autocorrelation to compare", a=a, nu=nu)
    Yobs, _ = padded_centred([proj])
    r_obs = pooled_acf_padded(Yobs, K)[0]
    cnt = np.array([sum(max(L - k, 0) for L in lens) for k in range(1, K + 1)], float)
    use = np.isfinite(r_obs) & (cnt >= 20)
    if int(use.sum()) < 2:
        return indet("fewer than two lags have 20 pooled pairs", a=a, nu=nu)
    lr_obs = float(whittle_lr_two_timescale(Yobs, lens)[0])

    def fit_many(Y):
        if model == "ar1_white":
            res = _kalman_fit_padded(Y, lens)
            return (np.array([r["a"] for r in res]), np.array([r["nu"] for r in res]),
                    np.array([r["converged"] for r in res]))
        aa = ols_persistence_padded(Y, lens)
        return aa, np.zeros_like(aa), np.isfinite(aa)

    factor, rounds = 1, []
    while True:
        nb, nr = int(n_bias) * factor, int(n_rep) * factor
        a1, nu1, steps = a, (nu if model == "ar1_white" else 0.0), []
        for _ in range(int(n_bias_iter)):
            Y = simulate_eq6_padded(a1, nu1, lens, nb, rng)
            fa, fn, okf = fit_many(Y)
            ok = okf & np.isfinite(fa) & (fa > 0) & (fa < 1)
            if int(ok.sum()) < 5:
                break
            a1 = float(np.clip(a1 + (a - float(np.mean(fa[ok]))), 1e-3, 0.999))
            if model == "ar1_white":
                nu1 = float(np.clip(nu1 + (nu - float(np.mean(fn[ok]))), 0.0, 0.95))
            steps.append(dict(a=a1, nu=nu1, mean_fit_a=float(np.mean(fa[ok])), mean_fit_nu=float(np.mean(fn[ok])),
                              n_fits=int(ok.sum())))
        tau = float(-1.0 / np.log(a1))
        if tau > L_med / float(min_cycles):
            return indet(f"the segments (median {L_med:g} samples) are shorter than {min_cycles:g} relaxation times of "
                         f"the corrected model ({tau:.1f} samples), so their autocorrelation does not determine the "
                         f"model", a=a, nu=nu, a_generator=a1, nu_generator=nu1, relaxation_time=tau, L_median=L_med)
        Y = simulate_eq6_padded(a1, nu1, lens, nr, rng)
        R = pooled_acf_padded(Y, K)
        mu = np.nanmean(R, axis=0)
        sd = np.nanstd(R, axis=0, ddof=1)
        ok_lag = use & np.isfinite(mu) & np.isfinite(sd) & (sd > 1e-9)
        if int(ok_lag.sum()) < 2:
            return indet("the simulated autocorrelation has no spread at the usable lags", a=a, nu=nu)
        with np.errstate(invalid="ignore", divide="ignore"):
            z_obs = (r_obs - mu) / sd
            Z = (R - mu) / sd
        T1o, T1s = float(np.max(np.abs(z_obs[ok_lag]))), np.nanmax(np.abs(Z[:, ok_lag]), axis=1)
        T2s = whittle_lr_two_timescale(Y, lens)
        n = nr
        p1o = (1.0 + np.sum(T1s >= T1o)) / (n + 1.0)
        p2o = (1.0 + np.sum(T2s >= lr_obs)) / (n + 1.0)
        r1 = (n - np.argsort(np.argsort(T1s, kind="stable"), kind="stable")) / float(n)
        r2 = (n - np.argsort(np.argsort(T2s, kind="stable"), kind="stable")) / float(n)
        minp_obs = float(min(p1o, p2o))
        P = float((1.0 + np.sum(np.minimum(r1, r2) <= minp_obs)) / (n + 1.0))
        se = float(np.sqrt(P * (1.0 - P) / n))
        dec = "pass" if P - mc_z * se > alpha else ("fail" if P + mc_z * se <= alpha else "ambiguous")
        rounds.append(dict(n_rep=nr, n_bias=nb, P=P, decision=dec))
        if dec != "ambiguous" or factor >= int(max_rep_factor):
            break
        factor *= 2
    passed = _decision_to_passed(dec)
    lags = np.arange(1, K + 1)
    worst = int(lags[ok_lag][int(np.argmax(np.abs(z_obs[ok_lag])))])
    if dec == "fail":
        which = ("two timescales describe the recording better (likelihood ratio "
                 f"{lr_obs:.1f})") if p2o < p1o else f"the autocorrelation deviates most at lag {worst}"
        note = f"the single-timescale model of the estimator is refused (combined P = {P:.4f}): {which}"
    elif dec == "ambiguous":
        note = f"combined P value {P:.3f} within Monte Carlo error of {alpha} after {nr} simulations"
    else:
        note = ""
    return GateResult("G8", question, stat, P, crit_txt, passed, float(P - alpha), ("L3",), note=note,
                      detail=dict(estimator=estimator, model=model, a_fit=a, nu_fit=nu, a_generator=a1,
                                  nu_generator=nu1, bias_steps=steps, relaxation_time=tau, L_median=L_med,
                                  short_lag=K, n_rep=nr, n_bias=nb, n_bias_iter=int(n_bias_iter), alpha=float(alpha),
                                  lags_used=lags[ok_lag].tolist(), p_autocorrelation=float(p1o),
                                  p_likelihood_ratio=float(p2o), likelihood_ratio=lr_obs,
                                  statistic_autocorrelation=T1o, combined_p=P, rounds=rounds,
                                  r_observed=[float(v) for v in r_obs], r_model_mean=[float(v) for v in mu],
                                  r_model_sd=[float(v) for v in sd], z_by_lag=[float(v) for v in z_obs],
                                  worst_lag=worst, n_segments=len(lens), n_samples=int(sum(lens))))


GATES = [g0_overlap, g1_instrument, g2_aliasing, g3_recovery,
         g4_deletion, g5_floor, g6_counter, g7_modes, g8_adequacy]

#: Checks run by default: the gates of the primary configuration (G0-G7).  The G8-augmented configuration
#: (:mod:`ndif.configurations`) runs G0-G3, G5, G6 and G8; there G4 and G7, which concern summaries over several
#: modes of a multivariate least-squares fit, are diagnostics run only when requested and govern no level.
DEFAULT_GATES = ("G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7")
DIAGNOSTICS = ("G4", "G7")


#: Keywords passed to every gate unless that gate's own dict overrides them.
GLOBAL_KW = ("rng", "estimator")


def run_gates(rec, which=None, verbose=False, **kw):
    """Run the gate sequence and return the list of :class:`GateResult`.

    Keys in :data:`GLOBAL_KW` are forwarded to every gate; anything else must
    be addressed to a gate by name, e.g. ``G3=dict(n_rep=32)``.  ``estimator``
    is global because a certificate is issued for one estimator and a gate that
    silently used a different one would be answering a question the user did
    not ask.
    """
    out = []
    if which is None:
        which = DEFAULT_GATES
    for g in GATES:
        name = g.__name__.split("_")[0].upper()
        if name not in which:
            continue
        gkw = dict(kw.get(name, {})) if isinstance(kw.get(name), dict) else {}
        for k in GLOBAL_KW:
            if k in kw and k not in gkw:
                gkw[k] = kw[k]
        try:
            r = g(rec, **gkw)
        except Exception as exc:                                 # pragma: no cover
            r = GateResult(name, g.__doc__.splitlines()[0] if g.__doc__ else "",
                           "n/a", float("nan"), "n/a", None, float("nan"), (),
                           note=f"gate raised {type(exc).__name__}: {exc}")
        if verbose:
            print(f"  {r.gate}  {r.status:<13} {r.statistic} = {r.value:.4g}")
        out.append(r)
    return out
