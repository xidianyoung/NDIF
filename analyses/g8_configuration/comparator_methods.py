"""Parametric-bootstrap comparators used by benchmark_09b_comparator.py.

``zeraati_precheck``
    The parametric-bootstrap pre-check of Zeraati, Engel & Levina (Nat. Comput. Sci. 2022;
    ``abcTau.preprocessing.check_expEstimates``), one timescale.
``estimator_bootstrap``
    The same parametric-bootstrap logic applied to an assessed estimator's own output.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # the repository root

from ndif.estimators import fit_ar1_kalman, fit_rate                 # noqa: E402


# ============================================ Zeraati, Engel & Levina pre-check
# abcTau (github.com/roxana-zeraati/abcTau, abcTau/summary_stats.py and
# abcTau/basic_functions.py), transcribed.
def comp_ac_fft(data):
    """abcTau.summary_stats.comp_ac_fft: per-trial mean removed, right zero
    padding, unnormalised autocovariance divided by arange(1, n)[::-1], averaged
    over trials."""
    n = np.shape(data)[1]
    xp = data - data.mean(1)[:, None]
    xp = np.concatenate((xp, np.zeros_like(xp)), axis=1)
    f = np.fft.fft(xp)
    p = np.absolute(f) ** 2
    pi = np.fft.ifft(p)
    ac_all = np.real(pi)[:, :n - 1] / np.arange(1, n)[::-1]
    return np.mean(ac_all, axis=0)


def single_exp(time_, a, tau):
    """abcTau.basic_functions.single_exp."""
    return a * np.exp(-time_ / tau)


def fit_one_tau(ac, max_lag):
    """abcTau.preprocessing.fit_oneTauExponential with binSize = 1 sample:
    curve_fit of single_exp to ac[0:max_lag] at lags 0 .. max_lag-1, default
    start values, maxfev = 2000.  Returns tau in samples or NaN."""
    from scipy.optimize import curve_fit
    xdata = np.arange(0, max_lag, 1.0)
    ydata = ac[0:max_lag]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(single_exp, xdata, ydata, maxfev=2000)
        tau = float(popt[1])
        return tau if np.isfinite(tau) and tau > 0 else float("nan")
    except Exception:
        return float("nan")


def _ar1_trials(a, n_trials, n_t, rng, var=1.0):
    """Stationary AR(1) = OU sampled exactly at the sampling interval."""
    Y = np.zeros((n_trials, n_t))
    Y[:, 0] = rng.normal(size=n_trials)
    E = rng.normal(scale=np.sqrt(max(1 - a * a, 1e-12)), size=(n_trials, n_t))
    for t in range(1, n_t):
        Y[:, t] = a * Y[:, t - 1] + E[:, t]
    return Y * np.sqrt(var)


def zeraati_precheck(segs, seed, max_lag=10, n_iter=500):
    """check_expEstimates, one timescale.

    Direct fit of single_exp to the trial-averaged sample autocorrelation of the
    recording (trials = segments); ``n_iter`` synthetic datasets from the
    one-timescale OU model with the fitted timescale, the recording's number of
    trials, trial duration and variance; each refitted the same way.  The
    bootstrap error is ``int((tau_hat - mean(tau_bs)) / mean(tau_bs) * 100)``
    per cent, as printed by abcTau.
    """
    t0 = time.perf_counter()
    data = np.stack([np.asarray(s, float) for s in segs])
    n_trials, n_t = data.shape
    tau_hat = fit_one_tau(comp_ac_fft(data), max_lag)
    out = dict(max_lag=int(max_lag), n_iter=int(n_iter), tau_hat=tau_hat,
               lam_direct=(1.0 / tau_hat if np.isfinite(tau_hat) else None))
    if not np.isfinite(tau_hat):
        out.update(err_pct=None, n_ok=0, mean_tau_bs=None, cost_s=time.perf_counter() - t0)
        return out
    a = float(np.exp(-1.0 / tau_hat))
    rng = np.random.default_rng(seed)
    var = float(np.var(data))
    taus = []
    for _ in range(int(n_iter)):
        syn = _ar1_trials(a, n_trials, n_t, rng, var=var)
        tb = fit_one_tau(comp_ac_fft(syn), max_lag)
        if np.isfinite(tb):
            taus.append(tb)
    taus = np.asarray(taus)
    if taus.size == 0:
        out.update(err_pct=None, n_ok=0, mean_tau_bs=None, cost_s=time.perf_counter() - t0)
        return out
    x1 = float(np.mean(taus))
    out.update(err_pct=int(((tau_hat - x1) / x1) * 100), err_exact=float((tau_hat - x1) / x1),
               mean_tau_bs=x1, sd_tau_bs=float(np.std(taus)), n_ok=int(taus.size),
               cost_s=float(time.perf_counter() - t0))
    return out


def estimator_bootstrap(segs, est, seed, n_iter=200):
    """The same parametric-bootstrap logic applied to an assessed estimator.

    The estimator is fitted to the recording; ``n_iter`` synthetic recordings
    of the same size are drawn from the model that estimator assumes (LS: AR(1)
    at the fitted persistence; ML: AR(1) plus white noise at the fitted
    persistence and noise fraction); the estimator is refitted to each; the
    error is ``(tau_hat - mean(tau_bs)) / mean(tau_bs)`` in timescale units, as
    in check_expEstimates.
    """
    t0 = time.perf_counter()
    data = np.stack([np.asarray(s, float) for s in segs])
    n_trials, n_t = data.shape
    var = float(np.var(data))
    if est == "kalman":
        g = fit_ar1_kalman(list(data), seed=seed, n_restart=1, maxiter=300)
        a_hat, nu_hat = float(g["a"]), float(g["nu"])
    else:
        lam = fit_rate(list(data), dt=1.0, method="ols")
        a_hat, nu_hat = (float(np.exp(-lam)) if np.isfinite(lam) else float("nan")), 0.0
    out = dict(estimator=est, n_iter=int(n_iter), a_hat=a_hat, nu_hat=nu_hat)
    if not (np.isfinite(a_hat) and 0 < a_hat < 1):
        out.update(err_pct=None, n_ok=0, cost_s=time.perf_counter() - t0)
        return out
    tau_hat = -1.0 / np.log(a_hat)
    rng = np.random.default_rng(seed)
    nu_sim = min(nu_hat, 1 - 1e-6)
    taus = []
    for i in range(int(n_iter)):
        syn = _ar1_trials(a_hat, n_trials, n_t, rng, var=1.0)
        if nu_sim > 0:
            syn = syn + rng.normal(scale=np.sqrt(nu_sim / (1 - nu_sim)), size=syn.shape)
        lam_b = fit_rate(list(syn), dt=1.0, method=est, seed=int(rng.integers(1 << 30)))
        if np.isfinite(lam_b) and lam_b > 0:
            taus.append(1.0 / lam_b)
    taus = np.asarray(taus)
    if taus.size == 0:
        out.update(err_pct=None, n_ok=0, cost_s=time.perf_counter() - t0)
        return out
    x1 = float(np.mean(taus))
    out.update(tau_hat=float(tau_hat), mean_tau_bs=x1, sd_tau_bs=float(np.std(taus)),
               err_pct=int(((tau_hat - x1) / x1) * 100), err_exact=float((tau_hat - x1) / x1),
               n_ok=int(taus.size), cost_s=float(time.perf_counter() - t0))
    return out
