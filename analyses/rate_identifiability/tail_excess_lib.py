# -*- coding: utf-8 -*-
r"""Tail excess of the state autocorrelation: one relaxation timescale plus white noise against several timescales.

THE QUESTION
------------
The state autocorrelation of these cohorts is NOT exponential: ln r(2) / (2 ln r(1)) is well below the value 1
that an OU / VAR(1) process is obliged to return.  Two very different causes produce exactly that:

  single timescale   ONE timescale plus an ADDITIVE WHITE component.  If y_t = x_t + w_t with x an AR(1) of
                     one-step factor `a` and w white carrying a fraction `nu` of the variance,
                         r(0) = 1 ,   r(k) = (1 - nu) a^k   for every k >= 1 ,
                     so the successive ratio r(k+1)/r(k) is EXACTLY `a`, constant in k, for every k >= 1.
  several timescales a mixture of exponentials, a stretched exponential or a power law: the fast components
                     die first, so the successive ratio RISES with k.

The successive ratio discriminates with no fitting and no free parameter, and it is immune to a multiplicative
attenuation, because a constant factor kappa in r(k) -> kappa r(k) cancels in r(k+1)/r(k).

WHAT THE ESTIMATOR DOES TO IT
-----------------------------
Every block is centred on its OWN mean (a block's location is a condition effect and must not enter a dynamical
statistic).  Inside a block of length n the mean-removed autocovariances are forced to sum to nearly zero, so r(k)
is biased DOWNWARD, increasingly with k, and the estimated successive-ratio profile FALLS with k even when the truth
is a single exponential.  For the estimator actually used,

    num(k) = sum_{t=0}^{m-1} (x_t - xbar)(x_{t+k} - xbar) ,  den = num(0) ,  m = n - K

(the base sample is held fixed across lags so r(k) is comparable in k), the expectation of num(k) under ANY
stationary autocorrelation rho is available in closed form:

    E num(k) = m gamma(k) - (1/n) sum_{t<m} [S(t) + S(t+k)] + m V
    S(t)     = sum_{s=0}^{n-1} gamma(|t-s|) = G(t) + G(n-1-t) - gamma(0) ,  G(m) = sum_{j<=m} gamma(j)
    V        = (1/n^2) sum_{|h|<n} (n - |h|) gamma(h)

summed over blocks (`expected_num`).  E[num] is EXACTLY LINEAR in the mixture weights, because independent
components' autocovariances add; `components` returns one E[num] curve per basis component and every model fit
below is a (non-negative) linear solve on those curves.

THE DECISION
------------
  1.  Fit the single-timescale model EXACTLY THROUGH THE ESTIMATOR: (a, nu) such that the PREDICTED r(1), r(2)
      equal the OBSERVED r(1), r(2).
  2.  Compare predicted and observed r(3..K): zero in expectation under the single-timescale model; above it under
      several timescales.
  3.  PRIMARY STATISTIC, per participant:   TAIL EXCESS
          E_tail = mean_{k=3..K} [ r_obs(k) - r_pred(k) ] / r_obs(1)
      scaled by r_obs(1) so a multiplicative attenuation cancels.
  4.  NULL: a parametric bootstrap under the single-timescale model at each participant's own fitted (a, nu),
      block lengths and block count, pushed through the identical fit.
  5.  DISPLAY: the successive-ratio profile rho(k) = r(k+1)/r(k) beside the predicted profile, and its
      parity-immune two-step twin r(k+2)/r(k).

A PERIODIC component at the task's own stimulus rate adds c cos(2 pi k / p) to r(k); at p = 2 it makes
r(2) > r(1) with no white component and no second timescale.  `parity_amplitude` measures it, and the
single-timescale basis carries an optional parity component.

Imported by tail_excess_{cog,fmri,ds003838,contrast,calibration}.py.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


# ==================================================================== the estimator
def acf_blocks(blocks, w=None, maxlag=10, center="block"):
    """Pooled autocorrelation of the projected state over a list of blocks.

    blocks : list of (n_t, d) arrays of NON-OVERLAPPING, CONTIGUOUS state windows.  A block never
             spans a gap in the window index; the cohort scripts split at gaps (`split_contiguous`).
    w      : projection direction (d,), or None to pool the coordinates.
    center : 'block'  each block centred on its own mean -- the standard rule, and the
                      CONSERVATIVE choice here, since it biases the profile toward a single timescale;
             'global' all blocks centred on the common mean -- a sensitivity only, because a
                      block-to-block offset then appears as a slow component and manufactures a second timescale;
             'linear' each block additionally DETRENDED with its own straight line.  This is the
                      control for the one alternative that would fake several timescales from a single one:
                      a slow within-block drift (fatigue, electrode or scanner drift, a block-level
                      task effect) gives a fat autocorrelation tail with no second relaxation mode.
                      Linear detrending removes any such drift, and also removes part of a genuine
                      slow mode, so a tail excess that SURVIVES it is not a drift.

    Returns (r, lens), r[0] = 1, lens the block lengths used; (None, None) if nothing is usable.
    """
    num = np.zeros(maxlag + 1)
    den = 0.0
    lens = []
    gm = None
    if center == "global":
        xs = [np.asarray(b, float) @ w if w is not None else np.asarray(b, float) for b in blocks]
        xs = [x for x in xs if len(x) >= maxlag + 2]
        if not xs:
            return None, None
        gm = np.concatenate([np.atleast_2d(x.T).T for x in xs], 0).mean(axis=0)
    for b in blocks:
        b = np.asarray(b, float)
        if len(b) < maxlag + 2:
            continue
        x = b @ w if w is not None else b
        if x.ndim == 1:
            x = x[:, None]
        if center == "linear":
            t = np.arange(len(x), dtype=float)
            t = (t - t.mean()) / (np.sqrt(np.sum((t - t.mean()) ** 2)) + EPS)
            x = x - x.mean(axis=0) - np.outer(t, t @ x)
        else:
            x = x - (gm if gm is not None else x.mean(axis=0))
        m = len(x) - maxlag
        den += float(np.sum(x[:m] ** 2))
        for k in range(maxlag + 1):
            num[k] += float(np.sum(x[:m] * x[k:m + k]))
        lens.append(len(x))
    if den <= 0 or not lens:
        return None, None
    return num / den, lens


def split_contiguous(Z, idx, min_len):
    """Split a window series into maximal runs of CONSECUTIVE window indices.

    The feature caches drop artefact windows from the middle of a record; concatenating what is left
    would silently turn a lag-1 pair into a lag-2 pair.  Every cohort's blocks go through this, so
    lag k always means exactly k windows and exactly k * dt seconds.
    """
    idx = np.asarray(idx)
    if len(idx) == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) != 1) + 1
    return [p for p in np.split(np.asarray(Z, float), cuts) if len(p) >= min_len]


# ==================================================================== estimator expectations
def expected_num(gam, lens, maxlag):
    """E[num(0..maxlag)] and E[den] for the block-mean-centred estimator, given autocovariance gam.

    gam must cover lags 0 .. max(lens)-1.  O(n) per DISTINCT block length: the expectation depends
    on a block only through its length, so repeated lengths -- which is every cohort whose blocks
    are a fixed trial, bout or task block -- are computed once and multiplied by their count.
    Returns (NUM, DEN)."""
    gam = np.asarray(gam, float)
    NUM = np.zeros(maxlag + 1)
    DEN = 0.0
    uniq = {}
    for n in lens:
        uniq[n] = uniq.get(n, 0) + 1
    for n, mult in uniq.items():
        if n < maxlag + 2:
            continue
        g = gam[:n]
        G = np.cumsum(g)                                   # G[m] = sum_{j<=m} gamma(j)
        t = np.arange(n)
        S = G[t] + G[n - 1 - t] - g[0]                     # S(t) = sum_s gamma(|t-s|)
        cS = np.concatenate([[0.0], np.cumsum(S)])
        h = np.arange(n)
        V = float(np.sum((n - h) * g * np.where(h == 0, 1.0, 2.0))) / (n * n)
        m = n - maxlag
        base_sum = cS[m] - cS[0]
        for k in range(maxlag + 1):
            NUM[k] += mult * (m * g[k] - (base_sum + (cS[m + k] - cS[k])) / n + m * V)
        DEN += mult * (m * g[0] - 2.0 * base_sum / n + m * V)
    return NUM, DEN


def _gam_exp(a, K):
    return a ** np.arange(K, dtype=float)


def _gam_white(K):
    g = np.zeros(K)
    g[0] = 1.0
    return g


def _gam_parity(K):
    """Unit-variance period-2 component: gamma(h) = (-1)^h."""
    return (-1.0) ** np.arange(K)


def rho_stretch(amp, tau, beta, K):
    """Stretched exponential, gamma(h) = amp exp(-(h/tau)^beta), gamma(0) = 1.

    beta < 1 is the classic signature of a CONTINUUM of relaxation times: it is exactly a positive
    mixture of exponentials, so it belongs to the several-timescale class.
    """
    g = np.empty(K)
    g[0] = 1.0
    h = np.arange(1, K, dtype=float)
    g[1:] = amp * np.exp(-((h / max(tau, 1e-6)) ** beta))
    return g


def rho_power(amp, alpha, K):
    """Power law, gamma(h) = amp h^-alpha, gamma(0) = 1: the scale-free limit of several timescales."""
    g = np.empty(K)
    g[0] = 1.0
    h = np.arange(1, K, dtype=float)
    g[1:] = amp * h ** (-alpha)
    return g


def components(lens, maxlag, a_list, white=True, parity=False):
    """E[num] curves (and E[den]) for each unit-variance basis component, at these block lengths.

    Because independent components' autocovariances add, E[num] is EXACTLY LINEAR in the component
    variances, so every model below is a linear solve on these curves -- including the timescale
    spectrum of step 2B, which is a non-negative least squares over an `a` grid.
    Returns (NUM, DEN) with NUM of shape (n_comp, maxlag+1) and DEN of shape (n_comp,).
    """
    K = max(maxlag + 1, max(lens))
    NUM, DEN = [], []
    for a in a_list:
        nu_, de_ = expected_num(_gam_exp(a, K), lens, maxlag)
        NUM.append(nu_); DEN.append(de_)
    if white:
        nu_, de_ = expected_num(_gam_white(K), lens, maxlag)
        NUM.append(nu_); DEN.append(de_)
    if parity:
        nu_, de_ = expected_num(_gam_parity(K), lens, maxlag)
        NUM.append(nu_); DEN.append(de_)
    return np.array(NUM), np.array(DEN)


# ==================================================================== single-timescale fit through the estimator
A_GRID = np.concatenate([np.linspace(0.02, 0.90, 89), np.linspace(0.905, 0.995, 19)])


def fit_single_timescale(robs, lens, maxlag, a_grid=A_GRID, parity=False):
    """(a, nu) chosen so the PREDICTED r(1), r(2) equal the OBSERVED r(1), r(2), exactly.

    For fixed `a` the predicted r(k) = [(1-nu) A(k) + nu W(k)] / [(1-nu) A0 + nu W0] is a ratio of
    two affine functions of nu, so matching r(1) is one linear solve; `a` is then scanned to match
    r(2).  With `parity` the period-2 amplitude is solved jointly and r(3) is used as the third
    moment, so the parity term costs a lag rather than being fitted to the tail it is meant to
    leave alone.
    """
    if robs is None or len(robs) <= maxlag or robs[1] <= 0:
        return None
    K = max(maxlag + 1, max(lens))
    Wn, Wd = expected_num(_gam_white(K), lens, maxlag)
    Pn, Pd = (expected_num(_gam_parity(K), lens, maxlag) if parity else (None, None))
    best, bd = None, np.inf
    for a in a_grid:
        An, Ad = expected_num(_gam_exp(a, K), lens, maxlag)
        if parity:
            # unknowns s = (var of exponential part, var of white, var of parity), scale-free:
            # solve  [A(k) W(k) P(k)] s = robs[k] * [A0 W0 P0] s   for k = 1, 2, with sum s = 1.
            M = np.array([[An[k] - robs[k] * Ad, Wn[k] - robs[k] * Wd, Pn[k] - robs[k] * Pd]
                          for k in (1, 2)] + [[1.0, 1.0, 1.0]])
            try:
                s = np.linalg.solve(M, np.array([0.0, 0.0, 1.0]))
            except np.linalg.LinAlgError:
                continue
            if np.any(s < -1e-6):
                continue
            pred = (s[0] * An + s[1] * Wn + s[2] * Pn) / (s[0] * Ad + s[1] * Wd + s[2] * Pd)
            d = abs(pred[3] - robs[3]) if maxlag >= 3 else 0.0
            cand = (a, float(1.0 - s[0] / max(s.sum(), EPS)), pred, float(s[2] / max(s.sum(), EPS)))
        else:
            # match r(1): (1-nu) An1 + nu Wn1 = robs1 * [(1-nu) Ad + nu Wd]
            c1 = (An[1] - robs[1] * Ad)
            c2 = (Wn[1] - robs[1] * Wd)
            if abs(c1 - c2) < EPS:
                continue
            nu = c1 / (c1 - c2)
            if not (-1e-9 <= nu < 0.9999):
                continue
            nu = min(max(nu, 0.0), 0.9999)
            pred = ((1 - nu) * An + nu * Wn) / ((1 - nu) * Ad + nu * Wd)
            d = abs(pred[2] - robs[2])
            cand = (a, float(nu), pred, 0.0)
        if d < bd:
            bd, best = d, cand
    if best is None:
        return None
    a, nu, pred, cpar = best
    return dict(a=float(a), nu=float(nu), c_parity=float(cpar), pred=pred, resid=float(bd),
                ok=bool(bd < 0.01))


def tail_excess(robs, pred, k0=3, k1=None, scale=True):
    """(observed - predicted by the single-timescale model) averaged over the tail lags, in units of r_obs(1).

    `scale=False` returns the raw difference in autocorrelation units.  The scaled version is the
    primary, because dividing by r(1) is what makes the statistic immune to the multiplicative
    attenuation of the whole autocorrelation; but it needs r(1) to be safely away from zero, which
    is why every caller admits a unit only above R1_FLOOR and reports the admitted fraction.
    """
    if robs is None or pred is None:
        return np.nan
    k1 = len(robs) - 1 if k1 is None else k1
    if k1 < k0 or robs[1] <= 0:
        return np.nan
    d = float(np.mean(np.asarray(robs)[k0:k1 + 1] - np.asarray(pred)[k0:k1 + 1]))
    return d / robs[1] if scale else d


R1_FLOOR = 0.08          # a direction whose lag-1 autocorrelation is below this cannot carry a
                         # ratio profile: every ratio divides by a number indistinguishable from
                         # zero.  Admissibility is declared, counted and reported, never silent.


# ==================================================================== simulation under the single-timescale model
def sim_single_timescale(a, nu, lens, maxlag, nrep, rng, c_parity=0.0):
    """nrep replicates of AR(1)+white (+parity) at these block lengths, through the SAME estimator.

    Returns (nrep, maxlag+1) of simulated r_hat.  This is the parametric bootstrap the decision uses:
    it carries every finite-sample distortion, including the ratio-of-expectations bias the closed
    form does not.
    """
    out = np.zeros((nrep, maxlag + 1))
    num = np.zeros((nrep, maxlag + 1))
    den = np.zeros(nrep)
    burn = 200
    sa = np.sqrt(max(1.0 - nu - abs(c_parity), 0.0))
    for n in lens:
        if n < maxlag + 2:
            continue
        e = rng.standard_normal((nrep, n + burn))
        x = np.zeros((nrep, n + burn))
        for t in range(1, n + burn):
            x[:, t] = a * x[:, t - 1] + e[:, t]
        x = x[:, burn:] * np.sqrt(max(1.0 - a * a, EPS))
        y = sa * x + np.sqrt(nu) * rng.standard_normal((nrep, n))
        if c_parity > 0:
            ph = rng.standard_normal((nrep, 1)) * np.sqrt(c_parity)
            y = y + ph * ((-1.0) ** np.arange(n))[None, :]
        y = y - y.mean(axis=1, keepdims=True)
        m = n - maxlag
        den += np.sum(y[:, :m] ** 2, axis=1)
        for k in range(maxlag + 1):
            num[:, k] += np.sum(y[:, :m] * y[:, k:m + k], axis=1)
    ok = den > 0
    out[ok] = num[ok] / den[ok][:, None]
    return out[ok]


# ==================================================================== profile statistics
def ratio_profile(r, step=1, kmin=1, kmax=None, floor=1e-3):
    """rho(k) = r(k+step)/r(k) with its lag index.  NaN where the denominator is not safely
    positive -- a ratio through zero carries no information and must not be averaged."""
    r = np.asarray(r, float)
    kmax = len(r) - 1 - step if kmax is None else kmax
    ks = np.arange(kmin, kmax + 1)
    vs = np.array([r[k + step] / r[k] if r[k] > floor else np.nan for k in ks], float)
    return ks.astype(float), vs


def profile_slope(ks, vs):
    """OLS slope of the successive ratio on the lag index.  0 = constant (single timescale), > 0 = rising."""
    m = np.isfinite(vs)
    if m.sum() < 3:
        return np.nan
    x = ks[m] - ks[m].mean()
    den = float(np.sum(x ** 2))
    return float(np.sum(x * (vs[m] - vs[m].mean())) / den) if den > 0 else np.nan


def parity_amplitude(r, kmin=1, kmax=None):
    """Period-2 amplitude: mean_k (-1)^k r(k) / r(1).  Positive = EVEN lags elevated."""
    r = np.asarray(r, float)
    kmax = len(r) - 1 if kmax is None else kmax
    k = np.arange(kmin, kmax + 1)
    return float(np.mean(((-1.0) ** k) * r[k]) / r[1]) if r[1] > 0 else np.nan


def exp_index(r):
    """ln r(2) / (2 ln r(1)); 1.0 for any OU / VAR(1) in any fixed direction."""
    if r is None or not (0 < r[1] < 1) or r[2] <= 0:
        return np.nan
    return float(np.log(r[2]) / (2.0 * np.log(r[1])))


# ==================================================================== injections (end-to-end)
def inject_white(F, nu, rng):
    """Add an iid white component carrying a fraction `nu` of the variance of the RESULT.

    Handed the RAW features, before standardisation and before any projection, which is where an end-to-end
    injection must enter.
    """
    F = np.asarray(F, float)
    s = np.sqrt(np.maximum(F.var(axis=0, ddof=1), 0) * nu / max(1.0 - nu, 1e-9))
    return F + rng.standard_normal(F.shape) * s


def inject_slow(F, seg, weight, a_slow, rng):
    """Add a SECOND TIMESCALE: an independent AR(1) of one-step factor `a_slow` carrying a fraction
    `weight` of the variance of the result, generated within each contiguous segment.

    The injected component is a scalar process times a fixed random loading over features, so the
    truth is a genuine extra relaxation mode of stated weight, not per-feature noise.
    """
    F = np.asarray(F, float)
    load = rng.standard_normal(F.shape[1])
    load /= np.linalg.norm(load)
    amp = np.sqrt(np.maximum(F.var(axis=0, ddof=1), 0).sum() * weight / max(1.0 - weight, 1e-9))
    out = F.copy()
    for g in np.unique(seg):
        m = seg == g
        n = int(m.sum())
        e = rng.standard_normal(n + 200)
        s = np.zeros(n + 200)
        for t in range(1, n + 200):
            s[t] = a_slow * s[t - 1] + e[t]
        s = s[200:] * np.sqrt(max(1.0 - a_slow ** 2, EPS))
        out[m] += amp * np.outer(s, load)
    return out


# ==================================================================== inference
def boot_ci(x, n=10000, seed=20260922, stat=np.mean):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=int(len(x)),
                    n_pos=0, sd=float("nan"))
    rng = np.random.default_rng(seed)
    b = np.array([stat(rng.choice(x, len(x), replace=True)) for _ in range(n)])
    return dict(mean=float(stat(x)), lo=float(np.percentile(b, 2.5)),
                hi=float(np.percentile(b, 97.5)), n=int(len(x)), n_pos=int((x > 0).sum()),
                sd=float(x.std(ddof=1)))


def signflip_p(x, n=10000, seed=20260922):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    rng = np.random.default_rng(seed + 7)
    obs = abs(float(np.mean(x)))
    s = rng.choice([-1.0, 1.0], size=(n, len(x)))
    return float((1 + np.sum(np.abs((s * x).mean(axis=1)) >= obs)) / (n + 1))


def lagperm_null(profiles, n=2000, seed=20260922):
    """Size-matched permutation null for 'the profile RISES with k': within each participant the
    lag labels of the successive-ratio profile are permuted, which destroys any ordering in k while
    keeping every value, every participant and the sample size.  Returns the null distribution of
    the group-mean slope."""
    rng = np.random.default_rng(seed + 13)
    out = []
    for _ in range(n):
        sl = []
        for ks, vs in profiles:
            m = np.isfinite(vs)
            if m.sum() < 3:
                continue
            v = vs.copy()
            v[m] = rng.permutation(v[m])
            s = profile_slope(ks, v)
            if np.isfinite(s):
                sl.append(s)
        if sl:
            out.append(float(np.mean(sl)))
    return np.array(out)


def perm_p(null, obs, tail="greater"):
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    if len(null) == 0 or not np.isfinite(obs):
        return float("nan")
    if tail == "greater":
        return float((1 + np.sum(null >= obs)) / (len(null) + 1))
    return float((1 + np.sum(np.abs(null) >= abs(obs))) / (len(null) + 1))


def orth_directions(w, n, rng):
    """n unit directions orthogonal to w, in w's own space -- the matched control set."""
    d = len(w)
    w = np.asarray(w, float) / (np.linalg.norm(w) + EPS)
    out = []
    for _ in range(n):
        v = rng.standard_normal(d)
        v -= (v @ w) * w
        nv = np.linalg.norm(v)
        if nv > EPS:
            out.append(v / nv)
    return out



# ==================================================================== the per-unit test
def analyse_unit(B, w, rng, nrep, maxlag=10, tail=(3, 10), parity=False, r1_floor=R1_FLOOR,
                 center="block"):
    """Everything the decision needs for ONE unit (participant x condition x direction).

    Returns None when the direction is INADMISSIBLE, i.e. r(1) < r1_floor, because every successive
    ratio then divides by a number indistinguishable from zero.  Callers count and report the
    admitted fraction; a direction is never dropped silently.
    """
    r, lens = acf_blocks(B, w, maxlag, center)
    if r is None or not lens or r[1] < r1_floor:
        return None
    f = fit_single_timescale(r, lens, maxlag, parity=parity)
    if f is None:
        return None
    ks, rho = ratio_profile(r, 1)
    ks2, rho2 = ratio_profile(r, 2)
    kp, rhop = ratio_profile(f["pred"], 1)
    out = dict(n_blocks=len(lens), n_win=int(sum(lens)), lens=lens, r=r.tolist(),
               pred=f["pred"].tolist(), a=f["a"], nu=f["nu"], c_parity=f["c_parity"],
               fit_ok=f["ok"], exp_index=exp_index(r), parity_amp=parity_amplitude(r, 1, maxlag),
               E_tail=tail_excess(r, f["pred"], tail[0], tail[1]),
               E_tail_raw=tail_excess(r, f["pred"], tail[0], tail[1], scale=False),
               slope=profile_slope(ks, rho), slope2=profile_slope(ks2, rho2),
               rho=rho.tolist(), rho_pred=rhop.tolist(), rho2=rho2.tolist(), ks=ks.tolist(),
               null_E=[], null_E_raw=[], null_slope=[], n_null_drop=0)
    if nrep:
        sim = sim_single_timescale(f["a"], f["nu"], lens, maxlag, nrep, rng, f["c_parity"])
        for rr in sim:
            if rr[1] < r1_floor:                       # the same admissibility as the real unit
                out["n_null_drop"] += 1
                continue
            g = fit_single_timescale(rr, lens, maxlag, parity=parity)
            if g is None:
                out["n_null_drop"] += 1
                continue
            out["null_E"].append(float(tail_excess(rr, g["pred"], tail[0], tail[1])))
            out["null_E_raw"].append(float(tail_excess(rr, g["pred"], tail[0], tail[1], False)))
            kk, vv = ratio_profile(rr, 1)
            s = profile_slope(kk, vv)
            if np.isfinite(s):
                out["null_slope"].append(float(s))
        out["null_E"] = [x for x in out["null_E"] if np.isfinite(x)]
        out["null_E_raw"] = [x for x in out["null_E_raw"] if np.isfinite(x)]
        out["E_tail_corr"] = (out["E_tail"] - float(np.mean(out["null_E"]))
                              if out["null_E"] else np.nan)
        out["E_tail_raw_corr"] = (out["E_tail_raw"] - float(np.mean(out["null_E_raw"]))
                                  if out["null_E_raw"] else np.nan)
    else:
        out["E_tail_corr"] = np.nan
        out["E_tail_raw_corr"] = np.nan
    return out


def group_stat(vals, label, nullmat=None, seed=20260922):
    """Bootstrap CI over participants, plus the single-timescale parametric-bootstrap GROUP null if supplied.

    The group null is formed by taking replicate b's value from every participant, so it is
    size-matched to the observed sample by construction and carries the same block structure,
    the same block counts and the same per-participant (a, nu).
    """
    g = boot_ci(vals, seed=seed)
    g["label"] = label
    g["p_signflip"] = signflip_p(vals, seed=seed)
    if nullmat:
        nullmat = [v for v in nullmat if len(v)]
        if nullmat:
            nb = min(len(v) for v in nullmat)
            gm = np.array([v[:nb] for v in nullmat]).mean(axis=0)
            g["null_mean"] = float(np.mean(gm))
            g["null_lo"] = float(np.percentile(gm, 2.5))
            g["null_hi"] = float(np.percentile(gm, 97.5))
            g["p_single_timescale_model"] = perm_p(gm, g["mean"], "greater")
            g["excess_over_single_timescale_model"] = g["mean"] - g["null_mean"]
            g["n_null_rep"] = int(nb)
    return g


def half_ceiling(B, w, rng, maxlag=10, tail=(3, 10), parity=False, r1_floor=R1_FLOOR):
    """The matched ceiling: the same statistic between two DISJOINT halves of ONE condition.

    Returns (E_half1 - E_half2) / 2, which estimates the sampling error of the FULL-data estimate,
    since two independent halves each carry twice the variance.  Nothing changes between the two
    halves, so the group mean of this quantity is zero and the width of its interval is exactly how
    much profile difference the design produces when there is nothing to find.
    """
    if len(B) >= 4:
        idx = rng.permutation(len(B))
        H = [[B[i] for i in idx[0::2]], [B[i] for i in idx[1::2]]]
    else:
        H = [[b[: len(b) // 2] for b in B], [b[len(b) // 2:] for b in B]]
    v = []
    for h in H:
        u = analyse_unit(h, w, None, 0, maxlag, tail, parity, r1_floor)
        v.append(u["E_tail"] if u else np.nan)
    return float((v[0] - v[1]) / 2.0) if all(np.isfinite(x) for x in v) else np.nan


# ==================================================================== step 2B: the spectrum
def fit_mixture(robs, lens, maxlag, a_list, kmax=None, white=True, nonneg=True):
    """Non-negative least squares for the TIMESCALE SPECTRUM, fitted THROUGH the estimator.

    Because E[num] is exactly linear in the component variances (independent components'
    autocovariances add), the observed r_hat(1..kmax) is, up to a common denominator, a
    non-negative linear combination of the basis curves `components` returns.  Writing s for the
    component variances the model is

        r_pred(k) = (NUM' s)_k / (DEN' s) ,

    a ratio of two linear forms, so the weights come from one non-negative solve of
    NUM_k' s - r_obs(k) DEN' s = 0 subject to sum(s) = 1.  This fits the autocorrelation FUNCTION,
    which is smooth and well determined here, instead of a VAR -- the object that has failed in
    the propagator fits of these recordings.

    Returns dict(weights, a_list, nu, pred, sse, mean_a, n_effective) or None.
    """
    if robs is None:
        return None
    kmax = maxlag if kmax is None else kmax
    NUM, DEN = components(lens, maxlag, a_list, white=white, parity=False)
    K = np.arange(1, kmax + 1)
    Mrow = NUM[:, K].T - np.outer(np.asarray(robs)[K], DEN)
    Mrow = np.vstack([Mrow, np.ones(len(DEN))])
    y = np.zeros(len(K) + 1)
    y[-1] = 1.0
    try:
        if nonneg:
            from scipy.optimize import nnls
            s, _ = nnls(Mrow, y)
        else:
            s = np.linalg.lstsq(Mrow, y, rcond=None)[0]
    except Exception:
        return None
    tot = float(s.sum())
    if tot <= 0:
        return None
    s = s / tot
    pred = (s @ NUM) / max(float(s @ DEN), EPS)
    w = s[: len(a_list)]
    sw = max(float(w.sum()), EPS)
    return dict(weights=w.tolist(), a_list=[float(x) for x in a_list],
                nu=float(s[len(a_list)]) if white else 0.0, pred=pred,
                sse=float(np.sum((pred[K] - np.asarray(robs)[K]) ** 2)),
                mean_a=float(np.sum(w * np.asarray(a_list)) / sw),
                n_effective=float((w.sum() ** 2) / max(float(np.sum(w ** 2)), EPS)))


def _fit_shape(robs, lens, maxlag, base, kmax):
    """One white component plus one fixed correlated SHAPE; solves the white fraction exactly."""
    K = np.arange(1, kmax + 1)
    ro = np.asarray(robs)
    Kg = max(maxlag + 1, max(lens))
    n1, d1 = expected_num(base, lens, maxlag)
    nw, dw = expected_num(_gam_white(Kg), lens, maxlag)
    c1 = n1[K] - ro[K] * d1
    c2 = nw[K] - ro[K] * dw
    den = float(np.sum((c1 - c2) ** 2))
    if den <= 0:
        return None
    nu = float(np.clip(np.sum(c1 * (c1 - c2)) / den, 0.0, 0.999))
    pred = ((1 - nu) * n1 + nu * nw) / max((1 - nu) * d1 + nu * dw, EPS)
    return nu, pred, float(np.sum((pred[K] - ro[K]) ** 2))


def fit_family(robs, lens, maxlag, family, kmax=None, grid=None):
    """One named model class fitted through the estimator by grid search on its own parameters.

    family in {'exp1', 'exp2', 'exp3', 'stretch', 'power'}.  'exp1' IS the single-timescale model.  Every class carries a
    white component, because every recording has one, so the classes differ ONLY in the shape of the
    correlated part: the comparison is about timescales, not about noise.
    """
    kmax = maxlag if kmax is None else kmax
    ro = np.asarray(robs)
    Kg = max(maxlag + 1, max(lens))
    if grid is None:
        grid = np.concatenate([np.linspace(0.05, 0.88, 18), np.linspace(0.90, 0.99, 8)])
    best = None
    if family in ("exp1", "exp2", "exp3"):
        from itertools import combinations
        n = int(family[-1])
        cands = [(g,) for g in grid] if n == 1 else list(combinations(grid, n))
        for al in cands:
            f = fit_mixture(ro, lens, maxlag, list(al), kmax)
            if f and (best is None or f["sse"] < best["sse"]):
                best = dict(f, family=family)
    elif family == "stretch":
        for tau in np.linspace(0.4, 25.0, 26):
            for beta in np.linspace(0.25, 1.6, 14):
                q = _fit_shape(ro, lens, maxlag, rho_stretch(1.0, tau, beta, Kg), kmax)
                if q and (best is None or q[2] < best["sse"]):
                    best = dict(family=family, tau=float(tau), beta=float(beta), nu=q[0],
                                pred=q[1], sse=q[2])
    elif family == "power":
        for al in np.linspace(0.15, 4.0, 40):
            q = _fit_shape(ro, lens, maxlag, rho_power(1.0, al, Kg), kmax)
            if q and (best is None or q[2] < best["sse"]):
                best = dict(family=family, alpha=float(al), nu=q[0], pred=q[1], sse=q[2])
    return best


def cv_families(B, w, maxlag, families, k_fit, k_test, rng=None, r1_floor=R1_FLOOR):
    """OUT-OF-SAMPLE comparison across HELD-OUT LAGS and HELD-OUT BLOCKS, per unit.

    Each class is fitted on lags 1..k_fit of one half of the blocks and scored on lags
    k_fit+1..k_test of the OTHER half, so neither the lags nor the data it is scored on entered the
    fit.  Returns {family: rms held-out prediction error}, averaged over the two folds.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    if len(B) < 4:
        return None
    idx = rng.permutation(len(B))
    halves = [[B[i] for i in idx[0::2]], [B[i] for i in idx[1::2]]]
    K = np.arange(k_fit + 1, k_test + 1)
    out = {f: [] for f in families}
    for f0, f1 in ((0, 1), (1, 0)):
        rf, lf = acf_blocks(halves[f0], w, maxlag)
        rt, lt = acf_blocks(halves[f1], w, maxlag)
        if rf is None or rt is None or rf[1] < r1_floor or rt[1] < r1_floor:
            continue
        for fam in families:
            m = fit_family(rf, lf, maxlag, fam, kmax=k_fit)
            if m is None:
                continue
            if fam.startswith("exp"):
                p = fit_mixture(rf, lt, maxlag, m["a_list"], k_fit)
                pred = p["pred"] if p else m["pred"]
            else:
                pred = m["pred"]
            out[fam].append(float(np.sqrt(np.mean((pred[K] - rt[K]) ** 2))))
    return {f: (float(np.mean(v)) if v else np.nan) for f, v in out.items()}
