"""Competing methods, implemented so the comparison is fair.

Each function takes the same input a gate does and returns the same shape of
answer: a point estimate of the relaxation rate, an interval, a runtime, and --
where the method offers one -- its own verdict on whether the rate is
identifiable.  The last column is the one that matters: a method can be
accurate and still say nothing about whether *this* recording could have
supported the claim.

The estimand throughout is ``lambda = -log(a)/dt`` for the latent process, on a
one-dimensional projection of the state (the leading principal component
unless an axis is given).  The observation model is

    y_t = x_t + e_t ,   x_t = a x_{t-1} + w_t ,

so a method that ignores ``e_t`` is fitting a model the data does not obey --
which is the situation every one of these methods is normally used in.

Methods
-------
``ols_ar1``
    What is most often done: least squares on lag-1 pairs, bootstrap interval.
``profile_likelihood``
    Raue et al. 2009.  Exact Gaussian likelihood by Kalman filter, profiled
    over the rate, interval from the likelihood-ratio threshold.  The standard
    test for *practical* identifiability.
``sloppiness``
    Gutenkunst et al. 2007.  Hessian eigenspectrum at the optimum; reports how
    much of the rate direction lies in the sloppy subspace.
``aic_bic_adequacy``
    The conventional adequacy check: compare candidate models by information
    criterion and test the residuals for whiteness.
``abc_timescale``
    Approximate Bayesian computation matching the autocorrelation of data
    simulated at the recording's own geometry -- the approach of Zeraati et
    al. 2022, the closest prior work.  Run with a single-timescale generative
    model (``with_white=False``) and with a correctly specified one
    (``with_white=True``).
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

from .estimators import as_segments, fit_ar1_kalman, kalman_ll

__all__ = ["project", "ols_ar1", "profile_likelihood", "sloppiness",
           "aic_bic_adequacy", "abc_timescale", "BASELINES"]


# --------------------------------------------------------------------- setup
def project(X, segments=None, axis=None):
    """Reduce to one dimension: a supplied axis, or the leading PC."""
    segs = as_segments(X, segments)
    d = segs[0].shape[1]
    if d == 1:
        return [s[:, 0] - s[:, 0].mean() for s in segs]
    if axis is None:
        Z = np.concatenate(segs, 0)
        Z = Z - Z.mean(0)
        _, _, Vt = np.linalg.svd(Z, full_matrices=False)
        axis = Vt[0]
    e = np.asarray(axis, float).ravel()
    e = e / np.linalg.norm(e)
    return [(s @ e) - (s @ e).mean() for s in segs]


def _stack(segs):
    """Equal-length segments as one (n_t, n_seg) array, for vectorised passes."""
    n = min(len(s) for s in segs)
    return np.column_stack([s[:n] for s in segs])


def _acf(segs, max_lag):
    """Segment-pooled autocorrelation, lags 1..max_lag."""
    num = np.zeros(max_lag + 1)
    for s in segs:
        s = s - s.mean()
        for k in range(max_lag + 1):
            if len(s) > k:
                num[k] += float(np.dot(s[k:], s[:len(s) - k]))
    return num[1:] / num[0] if num[0] > 0 else np.full(max_lag, np.nan)


# ----------------------------------------------------------- Kalman likelihood

def _fit_kalman(Y, a_fixed=None, n_restart=3, seed=0):
    """Kalman ML fit on a stacked (n_t, n_seg) array."""
    return fit_ar1_kalman([Y[:, j] for j in range(Y.shape[1])],
                          a_fixed=a_fixed, n_restart=n_restart, seed=seed)


# ------------------------------------------------------------------ baselines
def ols_ar1(X, segments=None, dt=1.0, axis=None, n_boot=400, seed=0, **kw):
    """Least squares on lag-1 pairs with a segment bootstrap interval.

    The commonest estimator in practice.  It has no notion of an observation
    process, so an additive white component enters the estimate directly.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    segs = project(X, segments, axis)

    def lam_of(ss):
        num = den = 0.0
        for s in ss:
            s = s - s.mean()
            num += float(np.dot(s[1:], s[:-1]))
            den += float(np.dot(s[:-1], s[:-1]))
        if den <= 0:
            return np.nan
        a = num / den
        return -np.log(a) / dt if 0 < a < 1 else np.nan

    lam = lam_of(segs)
    bs = []
    idx = np.arange(len(segs))
    for _ in range(int(n_boot)):
        pick = rng.choice(idx, size=len(segs), replace=True)
        v = lam_of([segs[i] for i in pick])
        if np.isfinite(v):
            bs.append(v)
    lo, hi = (np.percentile(bs, [2.5, 97.5]) if len(bs) > 20 else (np.nan, np.nan))
    return dict(method="ols_ar1", lam=float(lam), lo=float(lo), hi=float(hi),
                flags_unidentifiable=None, runtime_s=time.perf_counter() - t0,
                detail=dict(n_boot=len(bs)))


def profile_likelihood(X, segments=None, dt=1.0, axis=None, n_grid=25,
                       seed=0, **kw):
    """Raue et al. 2009: profile the exact likelihood over the rate.

    The interval is ``{a : 2 (ll_max - ll_profile(a)) <= chi2_1(0.95)}``.  A
    profile that is flat to the edge of the admissible range is the signature
    of practical non-identifiability, and is reported as such.
    """
    t0 = time.perf_counter()
    segs = project(X, segments, axis)
    Y = _stack(segs)
    full = _fit_kalman(Y, None, seed=seed)
    thr = chi2.ppf(0.95, 1) / 2.0
    a_hat = float(full["a"])
    lo_b, hi_b = 0.005, 0.9995

    def excess(a):
        """Log-likelihood drop below the threshold; > 0 inside the interval."""
        return _fit_kalman(Y, float(a), n_restart=1, seed=seed)["ll"] - (full["ll"] - thr)

    def walk(direction):
        """March outward from the optimum, then bisect the crossing."""
        edge = hi_b if direction > 0 else lo_b
        prev, step = a_hat, abs(edge - a_hat) / n_grid
        for _ in range(int(n_grid)):
            nxt = a_hat + direction * step * (_ + 1)
            nxt = min(nxt, hi_b) if direction > 0 else max(nxt, lo_b)
            if excess(nxt) < 0:
                lo_x, hi_x = (prev, nxt) if direction > 0 else (nxt, prev)
                for _ in range(30):
                    mid = 0.5 * (lo_x + hi_x)
                    if excess(mid) >= 0:
                        lo_x, hi_x = (mid, hi_x) if direction > 0 else (lo_x, mid)
                    else:
                        lo_x, hi_x = (lo_x, mid) if direction > 0 else (mid, hi_x)
                    if hi_x - lo_x < 1e-6:
                        break
                return 0.5 * (lo_x + hi_x), False
            prev = nxt
            if nxt in (lo_b, hi_b):
                break
        return edge, True          # never crossed: flat to the edge

    a_hi, flat_hi = walk(+1)
    a_lo, flat_lo = walk(-1)
    # flat to an edge of the admissible range -> practically non-identifiable
    flat = bool(flat_hi or flat_lo)
    lam = -np.log(full["a"]) / dt
    return dict(method="profile_likelihood", lam=float(lam),
                lo=float(-np.log(a_hi) / dt), hi=float(-np.log(a_lo) / dt),
                flags_unidentifiable=flat, runtime_s=time.perf_counter() - t0,
                detail=dict(a_hat=full["a"], sw2=full["sw2"], se2=full["se2"],
                            nu_implied=full["se2"] / (full["se2"] + full["sw2"] /
                                                      max(1 - full["a"] ** 2, 1e-9)),
                            profile_flat_to_edge=flat))


def sloppiness(X, segments=None, dt=1.0, axis=None, seed=0, h=1e-3, **kw):
    """Gutenkunst et al. 2007: Hessian eigenspectrum at the optimum.

    Returns the condition number of the Hessian of the negative log-likelihood
    in log-parameter space and the share of the rate direction that lies in the
    sloppiest half of the spectrum.  A model can fit perfectly and still leave
    the rate unconstrained; that is what a large spread reports.
    """
    t0 = time.perf_counter()
    segs = project(X, segments, axis)
    Y = _stack(segs)
    f = _fit_kalman(Y, None, seed=seed)
    theta = np.array([np.log(f["a"] / (1 - f["a"])), np.log(f["sw2"]), np.log(f["se2"])])

    def nll(z):
        a = 1.0 / (1.0 + np.exp(-z[0]))
        return -kalman_ll(Y, a, np.exp(z[1]), np.exp(z[2]))

    n = len(theta)
    H = np.zeros((n, n))
    f0 = nll(theta)
    for i in range(n):
        for j in range(i, n):
            ei, ej = np.zeros(n), np.zeros(n)
            ei[i] = h
            ej[j] = h
            H[i, j] = H[j, i] = (nll(theta + ei + ej) - nll(theta + ei - ej)
                                 - nll(theta - ei + ej) + nll(theta - ei - ej)) / (4 * h * h)
    w, V = np.linalg.eigh(0.5 * (H + H.T))
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    cond = float(np.max(np.abs(w)) / np.min(np.abs(w)))
    order = np.argsort(np.abs(w))
    sloppy = order[:max(1, n // 2)]
    share = float(np.sum(V[0, sloppy] ** 2))
    lam = -np.log(f["a"]) / dt
    return dict(method="sloppiness", lam=float(lam), lo=np.nan, hi=np.nan,
                flags_unidentifiable=bool(cond > 1e3),
                runtime_s=time.perf_counter() - t0,
                detail=dict(condition_number=cond, eigenvalues=w.tolist(),
                            rate_share_in_sloppy_subspace=share))


def aic_bic_adequacy(X, segments=None, dt=1.0, axis=None, seed=0, max_lag=12, **kw):
    """The conventional adequacy check: information criterion plus residual test.

    Fits AR(1) and AR(2) by least squares and reports which an information
    criterion prefers, together with a Ljung-Box test of residual whiteness.
    This is what "the model fits" normally means.  It is included because a
    degeneracy is invisible to it: a model that fits perfectly may be one of
    many that fit identically.
    """
    t0 = time.perf_counter()
    segs = project(X, segments, axis)
    n = sum(len(s) - 2 for s in segs)

    def ar_fit(p):
        Xs, ys = [], []
        for s in segs:
            if len(s) <= p:
                continue
            for t in range(p, len(s)):
                Xs.append(s[t - p:t][::-1])
                ys.append(s[t])
        Xs, ys = np.asarray(Xs), np.asarray(ys)
        beta, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
        res = ys - Xs @ beta
        s2 = float(np.mean(res ** 2))
        ll = -0.5 * len(ys) * (np.log(2 * np.pi * s2) + 1.0)
        return beta, res, ll, len(ys)

    b1, r1, ll1, n1 = ar_fit(1)
    b2, r2, ll2, n2 = ar_fit(2)
    aic1, aic2 = -2 * ll1 + 2 * 2, -2 * ll2 + 2 * 3
    bic1, bic2 = -2 * ll1 + 2 * np.log(n1), -2 * ll2 + 3 * np.log(n2)

    # Ljung-Box on the AR(1) residuals
    rr = r1 - r1.mean()
    denom = float(np.dot(rr, rr))
    q = 0.0
    for k in range(1, max_lag + 1):
        rk = float(np.dot(rr[k:], rr[:-k])) / denom
        q += rk * rk / (len(rr) - k)
    Q = len(rr) * (len(rr) + 2) * q
    p_lb = float(1 - chi2.cdf(Q, max_lag - 1))

    a = float(b1[0])
    lam = -np.log(a) / dt if 0 < a < 1 else np.nan
    adequate = bool(p_lb > 0.05 and aic1 <= aic2)
    return dict(method="aic_bic_adequacy", lam=float(lam), lo=np.nan, hi=np.nan,
                flags_unidentifiable=False,          # it has no such concept
                runtime_s=time.perf_counter() - t0,
                detail=dict(aic_ar1=aic1, aic_ar2=aic2, bic_ar1=bic1, bic_ar2=bic2,
                            prefers="AR(1)" if aic1 <= aic2 else "AR(2)",
                            ljung_box_p=p_lb, declares_model_adequate=adequate))


def abc_timescale(X, segments=None, dt=1.0, axis=None, with_white=False,
                  n_sim=4000, n_keep=200, max_lag=8, rounds=2, seed=0, **kw):
    """Zeraati et al. 2022, in essence: ABC on the autocorrelation.

    Simulates from a generative model **at the recording's own geometry** --
    same segment count, same block length -- and keeps the parameters whose
    simulated autocorrelation is closest to the observed one.  Simulating at
    the data's geometry is what removes the finite-sample bias that an analytic
    estimator carries, which is the method's contribution.

    ``with_white=False`` uses a single-timescale generative model, which is how
    the method is normally applied.  ``with_white=True`` adds the white share
    as a second free parameter, i.e. a correctly specified model.  Running both
    is the point: the difference between them is not visible in either one's
    own output.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    segs = project(X, segments, axis)
    n_seg = len(segs)
    n_blk = int(np.mean([len(s) for s in segs]))
    obs = _acf(segs, max_lag)

    lo_a, hi_a = 0.02, 0.995
    lo_nu, hi_nu = 0.0, 0.9
    post = None
    for rd in range(int(rounds)):
        if post is None:
            A = rng.uniform(lo_a, hi_a, n_sim)
            NU = rng.uniform(lo_nu, hi_nu, n_sim) if with_white else np.zeros(n_sim)
        else:  # adaptive: resample around the retained set
            pa, pn = post
            A = np.clip(rng.choice(pa, n_sim) + rng.normal(0, max(pa.std(), 1e-3), n_sim),
                        lo_a, hi_a)
            NU = (np.clip(rng.choice(pn, n_sim) + rng.normal(0, max(pn.std(), 1e-3), n_sim),
                          lo_nu, hi_nu) if with_white else np.zeros(n_sim))
        dist = np.empty(n_sim)
        for i in range(n_sim):
            a, nu = A[i], NU[i]
            Y = np.zeros((n_blk, n_seg))
            E = rng.normal(scale=np.sqrt(1 - a * a), size=(n_blk, n_seg))
            Y[0] = rng.normal(size=n_seg)
            for t in range(1, n_blk):
                Y[t] = a * Y[t - 1] + E[t]
            if nu > 0:
                Y = Y + rng.normal(scale=np.sqrt(nu / max(1 - nu, 1e-9)),
                                   size=Y.shape)
            sim = _acf([Y[:, j] for j in range(n_seg)], max_lag)
            dist[i] = float(np.nanmean((sim - obs) ** 2))
        keep = np.argsort(dist)[:n_keep]
        post = (A[keep], NU[keep])
    pa, pn = post
    lam_draws = -np.log(np.clip(pa, 1e-9, 0.999999)) / dt
    lam = float(np.median(lam_draws))
    lo, hi = np.percentile(lam_draws, [2.5, 97.5])
    return dict(method=f"abc_timescale({'with_white' if with_white else 'single'})",
                lam=lam, lo=float(lo), hi=float(hi),
                flags_unidentifiable=None,
                runtime_s=time.perf_counter() - t0,
                detail=dict(posterior_nu_median=float(np.median(pn)),
                            n_sim=int(n_sim), n_keep=int(n_keep), rounds=int(rounds)))


BASELINES = {
    "ols_ar1": ols_ar1,
    "profile_likelihood": profile_likelihood,
    "sloppiness": sloppiness,
    "aic_bic_adequacy": aic_bic_adequacy,
    "abc_single": lambda *a, **k: abc_timescale(*a, with_white=False, **k),
    "abc_with_white": lambda *a, **k: abc_timescale(*a, with_white=True, **k),
}
