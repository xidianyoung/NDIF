# -*- coding: utf-8 -*-
"""Shared helpers of the cohort state-space builders in this directory.

Contents
  short_time_band_power, log_features   Hann-window spectrogram band power per channel and its log10 features
  artifact_proxies                      per-window 30-45 Hz power, log peak-to-peak amplitude, spike fraction
  RefSpace, RANDOM_STATE                robust z (median / IQR) + PCA fitted on a reference set of windows
  median_iqr_z                          robust z of each column (median / IQR, IQR floored at 1e-9), no PCA
  split_runs                            runs of True in a boolean mask
  fit_linear_regime, find_equilibria    VAR(1) regime fit within segments and the equilibria of the
                                        mixture-weighted drift field (used by hcp_fmri_states.reference_count)

Every builder calls short_time_band_power with float64 signals.  RefSpace is the state space of the digit-span
EEG, motor EEG and HCP MEG cohorts (d = 8, reference = the windows of one condition) and of the Sleep-EDF nights
(d = 4, reference = all scored windows of the night).  No computation is run on import.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy import linalg as sla
from scipy import signal as sps
from scipy import stats as sst
from sklearn.decomposition import PCA

#: random_state of every reference PCA
RANDOM_STATE = 20260728


# ----------------------------------------------------------------------------- features
def short_time_band_power(x, sfreq, win_s, step_s, bands):
    """Band power per window and channel: (n_windows, n_channels, n_bands).

    Hann window of win_s seconds moved in steps of step_s seconds, constant detrend, power spectral density summed
    over [lo, hi) of each band times the frequency resolution.
    """
    nperseg = int(round(win_s * sfreq)); nstep = int(round(step_s * sfreq))
    if x.shape[1] < nperseg:
        return np.zeros((0, x.shape[0], len(bands)))
    f, t, S = sps.spectrogram(x, fs=sfreq, window="hann", nperseg=nperseg, noverlap=nperseg - nstep,
                              detrend="constant", scaling="density", mode="psd", axis=-1)
    df = f[1] - f[0]
    out = np.empty((S.shape[-1], x.shape[0], len(bands)))
    for bi, (lo, hi) in enumerate(bands.values()):
        m = (f >= lo) & (f < hi)
        out[:, :, bi] = (S[:, m, :].sum(axis=1) * df).T
    return out


def log_features(P):
    """log10 band power, channels x bands flattened per window."""
    return np.log10(np.maximum(P, 1e-30)).reshape(P.shape[0], -1)


def artifact_proxies(x, sfreq, win_s, step_s):
    """Per window: log10 mean 30-45 Hz power, log10 median peak-to-peak amplitude, fraction of samples beyond
    5 robust standard deviations."""
    nperseg = int(round(win_s * sfreq)); nstep = int(round(step_s * sfreq))
    n_win = 1 + max(0, (x.shape[1] - nperseg) // nstep)
    sd = 1.4826 * np.median(np.abs(x - np.median(x, axis=1, keepdims=True)), axis=1, keepdims=True)
    sd[sd == 0] = np.inf
    ptp, nspike = np.empty(n_win), np.empty(n_win)
    for i in range(n_win):
        seg = x[:, i * nstep: i * nstep + nperseg]
        ptp[i] = np.median(seg.max(axis=1) - seg.min(axis=1))
        nspike[i] = float(np.mean(np.abs(seg) > 5 * sd))
    P = short_time_band_power(x, sfreq, win_s, step_s, {"emg": (30.0, 45.0)})
    emg = np.log10(np.maximum(P[:, :, 0].mean(axis=1), 1e-30))
    n = min(n_win, len(emg))
    return np.column_stack([emg[:n], np.log10(np.maximum(ptp[:n], 1e-12)), nspike[:n]])


# ----------------------------------------------------------------------------- state spaces
class RefSpace:
    """Robust z (median and interquartile range of the reference windows; a non-finite or non-positive IQR is
    replaced by 1) followed by a PCA to d components fitted on the z-scored reference windows."""

    def __init__(self, d, seed=RANDOM_STATE):
        self.d, self.seed = d, seed

    def fit(self, F_ref):
        self.med = np.median(F_ref, axis=0)
        iqr = sst.iqr(F_ref, axis=0)
        iqr[~np.isfinite(iqr) | (iqr <= 0)] = 1.0
        self.iqr = iqr
        self.pca = PCA(n_components=self.d, random_state=self.seed).fit(self.z(F_ref))
        return self

    def z(self, F):
        return (F - self.med) / self.iqr

    def transform(self, F):
        return self.pca.transform(self.z(F))


def median_iqr_z(F):
    """Robust z of each column over all rows: (F - median) / max(IQR, 1e-9)."""
    med = np.median(F, axis=0)
    iqr = np.subtract(*np.percentile(F, [75, 25], axis=0))
    return (F - med) / np.maximum(iqr, 1e-9)


def split_runs(mask):
    """Index runs of True in a boolean mask -> list of index arrays."""
    runs, cur = [], []
    for i, m in enumerate(mask):
        if m:
            cur.append(i)
        elif cur:
            runs.append(np.array(cur)); cur = []
    if cur:
        runs.append(np.array(cur))
    return runs


# ----------------------------------------------------------------------------- regime fit and equilibria
def ridgeline_points(mus, Sigmas, alphas):
    """Ridgeline of a Gaussian mixture (Ray and Lindsay, 2005): for weights alpha on the simplex,
    x*(alpha) = [sum_k alpha_k S_k^-1]^-1 [sum_k alpha_k S_k^-1 mu_k]."""
    Ps = [np.linalg.inv(np.asarray(S, float)) for S in Sigmas]
    Pm = [P @ np.asarray(m, float) for P, m in zip(Ps, mus)]
    out = np.empty((len(alphas), len(np.asarray(mus[0]))))
    for i, a in enumerate(alphas):
        M = sum(ai * P for ai, P in zip(a, Ps))
        b = sum(ai * v for ai, v in zip(a, Pm))
        out[i] = np.linalg.solve(M, b)
    return out


@dataclass
class LinearRegime:
    label: str
    mu: np.ndarray
    Sigma: np.ndarray
    A: np.ndarray             # continuous-time drift, A = log(M)/dt
    M: np.ndarray             # discrete one-step map
    Q: np.ndarray             # innovation covariance
    n: int
    logm_ok: bool


def fit_linear_regime(segments, label, dt=1.0, ridge=1e-6):
    """VAR(1) fit on a list of contiguous state segments, increments taken within segments only.

    A = log(M)/dt with the principal matrix logarithm; where expm(A dt) does not reproduce M the Euler form
    (M - I)/dt is used and logm_ok is False.
    """
    X, Y = [], []
    for S in segments:
        S = np.asarray(S, float)
        if len(S) >= 2:
            X.append(S[:-1])
            Y.append(S[1:])
    X, Y = np.vstack(X), np.vstack(Y)
    Z = np.vstack([np.asarray(s, float) for s in segments])
    mu = Z.mean(axis=0)
    Xc, Yc = X - mu, Y - mu
    G = Xc.T @ Xc + ridge * np.eye(Xc.shape[1]) * np.trace(Xc.T @ Xc) / Xc.shape[1]
    M = np.linalg.solve(G, Xc.T @ Yc).T
    R = Yc - Xc @ M.T
    Q = np.cov(R, rowvar=False, ddof=1)
    Sigma = np.cov(Z, rowvar=False, ddof=1)
    ok = True
    try:
        A = np.real(sla.logm(M)) / dt
        if not np.allclose(sla.expm(A * dt), M, atol=1e-6, rtol=1e-4):
            A, ok = (M - np.eye(len(M))) / dt, False
    except Exception:
        A, ok = (M - np.eye(len(M))) / dt, False
    return LinearRegime(label, mu, Sigma, A, M, Q, int(len(X)), ok)


def multilinear_drift(x, regimes, softness=1.0):
    """Drift field b(x) = sum_k w_k(x) A_k (x - mu_k), with w_k(x) the Gaussian responsibilities of the regime
    means and covariances tempered by softness."""
    x = np.atleast_2d(np.asarray(x, float))
    L = np.empty((len(x), len(regimes)))
    for k, r in enumerate(regimes):
        dv = x - r.mu
        P = np.linalg.inv(r.Sigma)
        L[:, k] = -0.5 * (np.linalg.slogdet(r.Sigma)[1]
                          + np.einsum("nk,kl,nl->n", dv, P, dv))
    L /= max(softness, 1e-6)
    L -= L.max(axis=1, keepdims=True)
    Wt = np.exp(L)
    Wt /= Wt.sum(axis=1, keepdims=True)
    b = np.zeros_like(x)
    for k, r in enumerate(regimes):
        b += Wt[:, [k]] * ((x - r.mu) @ r.A.T)
    return b


def find_equilibria(regimes, softness=1.0, n_starts=300, seed=0):
    """Zeros of the drift field from Newton (fsolve) starts at the regime means, nine ridgeline points between
    every pair of regimes and n_starts // K draws from each regime's Gaussian; distinct zeros with their Jacobian
    eigenvalues (central differences) and stability."""
    from scipy.optimize import fsolve
    rng = np.random.default_rng(seed)
    d = len(regimes[0].mu)
    starts = [np.asarray(r.mu, float) for r in regimes]
    for i, j in itertools.combinations(range(len(regimes)), 2):
        a = np.linspace(0, 1, 9)
        starts += list(ridgeline_points([regimes[i].mu, regimes[j].mu],
                                        [regimes[i].Sigma, regimes[j].Sigma],
                                        np.column_stack([1 - a, a])))
    for r in regimes:
        starts += list(rng.multivariate_normal(r.mu, r.Sigma,
                                               size=max(1, n_starts // len(regimes))))

    def f(z):
        return multilinear_drift(z, regimes, softness)[0]

    eqs = []
    for s in starts:
        try:
            z, _, ier, _ = fsolve(f, s, full_output=True, xtol=1e-12)
        except Exception:
            continue
        if ier != 1 or np.linalg.norm(f(z)) > 1e-6:
            continue
        if not any(np.linalg.norm(z - e) < 1e-4 * (1 + np.linalg.norm(e)) for e in eqs):
            eqs.append(z)
    out = []
    for z in eqs:
        J = np.empty((d, d))
        h = 1e-5 * (1 + np.abs(z))
        for k in range(d):
            e = np.zeros(d)
            e[k] = h[k]
            J[:, k] = (f(z + e) - f(z - e)) / (2 * h[k])
        ev = np.linalg.eigvals(J)
        out.append(dict(x=z, eig=ev, stable=bool(np.all(ev.real < 0)),
                        n_unstable=int(np.sum(ev.real > 0))))
    return out
