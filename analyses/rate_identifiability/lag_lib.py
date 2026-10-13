# -*- coding: utf-8 -*-
r"""Shared code of the lag-ratio analyses: the calibrated single-timescale fit, the sleep state and
the band-power pipeline.

Imported by sleep_lag_profile.py, white_share_pipeline_control.py, window_scaling.py and
identifiability_map.py.

THE INSTRUMENT
--------------
    g(k) = r(k+1) / r(k)

  single-timescale model   one timescale + an additive white component of variance share nu:
                           r(k) = (1 - nu) a^k for k >= 1  ->  g(k) = a, CONSTANT in k.
  second timescale         several timescales (mixture / stretched / power law):
                           the fast parts die first  ->  g(k) RISES with k.

A constant factor C in r(k) = C a^k cancels in the ratio, which is why the instrument survives a
multiplicative attenuation of the autocorrelation.

A non-overlapping window average of a continuous OU process has, exactly, r(k) = C a^k with C > 1,
so windowing pushes the exponential index ln r(2) / (2 ln r(1)) ABOVE 1 and can never produce a
value below 1; an additive white component gives C = 1 - nu < 1, which does.

THE BLOCK-CENTRING BIAS, MODELLED EXACTLY
-----------------------------------------
Every block (bout, trial, task block) is centred on its OWN mean and no pair is taken across a
block boundary.  That biases the sample autocorrelation downward, and NOT by a constant factor: r(k)
at large k is biased proportionally more, so the measured g(k) FALLS with k even for an exact OU
process.

`expected_r` computes E[r_hat(1..K)] exactly (to first order in the ratio of expectations) from the
true ACF, the block length and the block count, by summing the exact expectation of the
mean-centred sample autocovariance

    E[sum_t c_t c_{t+k}] = sum_t [ gam(k) - m(t) - m(t+k) + M ]
    m(t) = (1/n) sum_j gam(t-j),   M = (1/n^2) sum_{i,j} gam(i-j)

and the two matching sums of squares.  Every estimator below uses it, so the reported (a, nu) are
the values that REPRODUCE the observed r(1..K) at the observed geometry, not the naive ratio.

THE SLEEP STATE
---------------
Sleep-EDF cassette and telemetry recordings, log10 short-time band power of two EEG channels
(Fpz-Cz, Pz-Oz) in four bands on NON-OVERLAPPING windows of 2.0, 1.0 and 0.5 s, per scored bout,
read from the sleep window cache (`load_cell`, `load_noise_R`), then the robust z + PCA (d = 4)
state space fitted on the recording's pooled bouts (`build_state_space`, `state_space`).

Cache directory: analyses.common.paths.cache("sleep_states") (environment variable
NDIF_SLEEP_STATES_CACHE, default analyses/_work/sleep_states).  Files read:

  <stem>_win<w>.npz   one per recording and window (w = 2, 1, 0.5): X (n_bout, n_win, 8) log10 band
                      power of the used bouts (channels x bands delta/theta/alpha/beta), stage
                      (n_bout,) stage code Wake/N1/N2/N3/N4/REM, win_s, n_use, n_skip
  <stem>_noise.npz    one per recording: "<stage>|<w>" -> 8 x 8 covariance of the log band power of
                      phase-randomised copies of the same bouts (the instrument-noise covariance)

with <stem> the six-character Sleep-EDF recording name (SC4001, ST7011, ...).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats as sst
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from analyses.common import paths                                                # noqa: E402

CACHE = Path(paths.cache("sleep_states"))

STAGES = ("Wake", "N1", "N2", "N3", "N4", "REM")
RUNGS = (("Wake", "N1"), ("N1", "N2"), ("N2", "N3"), ("N3", "N4"))
MIN_BOUTS = 5
D_STATE = 4
RANDOM_STATE = 20260728          # PCA random state of the sleep state space

# band-power pipeline: two channels x four bands, 100 Hz
BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0),
         "alpha": (8.0, 12.0), "beta": (12.0, 30.0)}
SFREQ = 100.0
N_CHAN = 2


# --------------------------------------------------------------- ACF shapes
def acf_ou(a, K):
    return np.asarray(a, float) ** np.arange(K + 1)


def acf_ou_white(a, nu, K):
    r = (1.0 - nu) * acf_ou(a, K)
    r[0] = 1.0
    return r


# ----------------------------------------------------- exact centring bias
def expected_r(gam, n_win, kmax):
    """E[r_hat(1..kmax)] for mean-centred blocks of length n_win.

    `gam` is the true autocorrelation at lags 0..n_win-1 (gam[0] = 1).
    The estimator modelled is exactly the one in `pooled_acf_many`:
        r(k) = sum_t c_t c_{t+k} / sqrt(sum_t c_t^2 * sum_t c_{t+k}^2)
    with the two sums of squares taken over the SAME index ranges as the numerator
    and c the block-centred series.  Block count cancels to first order (every
    block contributes the same expectation), so only n_win enters.
    """
    g = np.asarray(gam, float)
    n = int(n_win)
    if len(g) < n:
        g = np.concatenate([g, np.zeros(n - len(g))])
    idx = np.arange(n)
    G = g[np.abs(idx[:, None] - idx[None, :])]          # (n, n) true covariance
    m = G.mean(axis=1)                                   # m(t) = (1/n) sum_j gam(t-j)
    M = float(G.mean())                                  # grand mean
    # E[c_t c_s] = gam(t-s) - m(t) - m(s) + M
    Ecc = G - m[:, None] - m[None, :] + M
    out = np.full(kmax, np.nan)
    for k in range(1, kmax + 1):
        if n <= k + 1:
            continue
        ta = idx[: n - k]
        tb = idx[k:]
        num = float(Ecc[ta, tb].sum())
        da = float(Ecc[ta, ta].sum())
        db = float(Ecc[tb, tb].sum())
        out[k - 1] = num / np.sqrt(da * db) if da > 0 and db > 0 else np.nan
    return out


def expected_r_single_timescale(a, nu, n_win, kmax):
    return expected_r(acf_ou_white(a, nu, n_win), n_win, kmax)


# --------------------------------------------------------------- estimators
def pooled_acf_many(X, kmax):
    """(..., n_blocks, n_win) -> (..., kmax).  Block-centred, no cross-block pair."""
    X = np.asarray(X, float)
    c = X - X.mean(axis=-1, keepdims=True)
    n_win = c.shape[-1]
    r = np.full(c.shape[:-2] + (kmax,), np.nan)
    for k in range(1, kmax + 1):
        if n_win <= k + 1:
            continue
        a, b = c[..., :-k], c[..., k:]
        num = (a * b).sum(axis=(-1, -2))
        den = np.sqrt((a * a).sum(axis=(-1, -2)) * (b * b).sum(axis=(-1, -2)))
        with np.errstate(invalid="ignore", divide="ignore"):
            r[..., k - 1] = np.where(den > 0, num / den, np.nan)
    return r


def pooled_acf_ragged(blocks, kmax):
    """Blocks of unequal length (real bouts).  Same rule."""
    num = np.zeros(kmax); da = np.zeros(kmax); db = np.zeros(kmax)
    for b in blocks:
        c = np.asarray(b, float)
        c = c - c.mean()
        for k in range(1, kmax + 1):
            if len(c) <= k + 1:
                continue
            x, y = c[:-k], c[k:]
            num[k - 1] += float(x @ y)
            da[k - 1] += float(x @ x)
            db[k - 1] += float(y @ y)
    den = np.sqrt(da * db)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.maximum(den, 1e-300), np.nan)


def profile_many(R):
    """(..., K) of r(1..K) -> the profile statistics, vectorised."""
    R = np.asarray(R, float)
    K = R.shape[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        G = np.where(R[..., :-1] > 0.02, R[..., 1:] / R[..., :-1], np.nan)
    k = np.arange(1, K, dtype=float)
    ok = np.isfinite(G)
    n_ok = ok.sum(axis=-1)
    Gz = np.where(ok, G, 0.0)
    kk = np.where(ok, k, 0.0)
    n = np.maximum(n_ok, 1)
    mk = kk.sum(axis=-1) / n
    mg = Gz.sum(axis=-1) / n
    sxy = np.where(ok, (kk - mk[..., None]) * (Gz - mg[..., None]), 0.0).sum(axis=-1)
    sxx = np.where(ok, (kk - mk[..., None]) ** 2, 0.0).sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = np.where((n_ok >= 2) & (sxx > 0), sxy / np.where(sxx > 0, sxx, 1.0), np.nan)
        ghat = np.nanmedian(np.where(ok, G, np.nan), axis=-1)
        ei = np.where((R[..., 0] > 0) & (R[..., 1] > 0),
                      np.log(np.where(R[..., 1] > 0, R[..., 1], np.nan))
                      / (2.0 * np.log(np.where(R[..., 0] > 0, R[..., 0], np.nan))), np.nan)
        nu_naive = np.where(ghat > 0.02, 1.0 - R[..., 0] / ghat, np.nan)
    even = (k % 2 == 0)
    with np.errstate(invalid="ignore"):
        par = (np.nanmean(np.where(even & ok, G, np.nan), axis=-1)
               - np.nanmean(np.where((~even) & ok, G, np.nan), axis=-1))
    return dict(slope=slope, exp_index=ei, ghat=ghat, nu_naive=nu_naive,
                parity=par, g1=G[..., 0], G=G, n_g=n_ok)


# ------------------------------------------ calibrated single-timescale two-parameter fit
_A_GRID = np.concatenate([np.linspace(0.02, 0.90, 45), np.linspace(0.905, 0.985, 17)])
_NU_GRID = np.linspace(0.0, 0.92, 47)


def single_timescale_grid(n_win, kmax, a_grid=None, nu_grid=None):
    """Pre-compute E[r_hat(1..kmax)] on the (a, nu) grid at this block length."""
    a_grid = _A_GRID if a_grid is None else np.asarray(a_grid, float)
    nu_grid = _NU_GRID if nu_grid is None else np.asarray(nu_grid, float)
    # E[r] is linear in gam up to the normalisation, so it is NOT separable in nu;
    # compute the whole grid once.
    T = np.empty((len(a_grid), len(nu_grid), kmax))
    for i, a in enumerate(a_grid):
        for j, nu in enumerate(nu_grid):
            T[i, j] = expected_r_single_timescale(a, nu, n_win, kmax)
    return dict(a=a_grid, nu=nu_grid, T=T, n_win=int(n_win), kmax=int(kmax))


def fit_single_timescale(r_obs, grid, w=None):
    """Least-squares (a, nu) whose EXPECTED r_hat matches the observed r_hat.

    This is the calibrated version of "nu = 1 - r(1)/a": it inverts the block
    centring bias instead of ignoring it.  Weights default to 1 on every usable
    lag.  Returns (a, nu, residual rms, fitted r).
    """
    r = np.asarray(r_obs, float)
    ok = np.isfinite(r)
    if ok.sum() < 2:
        return np.nan, np.nan, np.nan, None
    W = np.ones(len(r)) if w is None else np.asarray(w, float)
    T = grid["T"]
    d = (T[:, :, ok] - r[ok]) * np.sqrt(W[ok])
    sse = (d ** 2).sum(axis=-1)
    i, j = np.unravel_index(int(np.argmin(sse)), sse.shape)
    return (float(grid["a"][i]), float(grid["nu"][j]),
            float(np.sqrt(sse[i, j] / ok.sum())), T[i, j].copy())


def fit_single_timescale_many(R, grid, chunk=400):
    """Batched `fit_single_timescale`.  R is (N, kmax); returns (a, nu, resid) each (N,).

    Lags that are non-finite in a row are dropped for that row only, so rows with
    different usable lag counts are handled without a Python loop over the grid.
    """
    R = np.asarray(R, float)
    N, K = R.shape
    T = grid["T"]                                   # (na, nnu, K)
    na, nnu = T.shape[0], T.shape[1]
    A = np.full(N, np.nan); NU = np.full(N, np.nan); RES = np.full(N, np.nan)
    Tf = T.reshape(na * nnu, K)
    for s0 in range(0, N, chunk):
        r = R[s0:s0 + chunk]
        ok = np.isfinite(r)
        rf = np.where(ok, r, 0.0)
        d = Tf[None, :, :] - rf[:, None, :]
        d = np.where(ok[:, None, :], d, 0.0)
        sse = (d ** 2).sum(axis=-1)                 # (n, na*nnu)
        n_ok = ok.sum(axis=1)
        best = np.argmin(sse, axis=1)
        i, j = np.unravel_index(best, (na, nnu))
        good = n_ok >= 2
        A[s0:s0 + chunk] = np.where(good, grid["a"][i], np.nan)
        NU[s0:s0 + chunk] = np.where(good, grid["nu"][j], np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            RES[s0:s0 + chunk] = np.where(
                good, np.sqrt(sse[np.arange(len(r)), best] / np.maximum(n_ok, 1)), np.nan)
    return A, NU, RES


# --------------------------------------------------------------- generators
def chol_from_acf(rho, n_win):
    """Cholesky of the Toeplitz correlation, with the PSD repair reported."""
    from scipy.linalg import toeplitz
    r = np.asarray(rho, float)[:n_win].copy()
    if len(r) < n_win:
        r = np.concatenate([r, np.zeros(n_win - len(r))])
    C = toeplitz(r)
    w, V = np.linalg.eigh(C)
    clipped = float(-w.min()) if w.min() < 0 else 0.0
    C = V @ np.diag(np.clip(w, 1e-10, None)) @ V.T
    dg = np.sqrt(np.diag(C))
    C = C / np.outer(dg, dg)
    return np.linalg.cholesky(C + 1e-12 * np.eye(n_win)), clipped


def gen_blocks(L, n_series, n_blocks, n_win, rng):
    return rng.standard_normal((n_series, n_blocks, n_win)) @ L.T


# --------------------------------------------------------------- group stats
def boot_mean(v, n_boot=10000, rng=None):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return dict(mean=np.nan, lo=np.nan, hi=np.nan, n=0, n_pos=0, p_sign=np.nan)
    rng = np.random.default_rng(0 if rng is None else rng)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    bm = v[idx].mean(axis=1)
    npos = int((v > 0).sum())
    # two-sided sign-flip P on the mean
    fl = rng.choice([-1.0, 1.0], size=(n_boot, len(v)))
    null = (v * fl).mean(axis=1)
    p = float((np.abs(null) >= abs(np.mean(v))).mean())
    return dict(mean=float(np.mean(v)), lo=float(np.percentile(bm, 2.5)),
                hi=float(np.percentile(bm, 97.5)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                n=int(len(v)), n_pos=npos, p_sign=max(p, 1.0 / n_boot))


def paired(v_a, v_b, n_boot=10000, rng=None):
    a = np.asarray(v_a, float); b = np.asarray(v_b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return boot_mean(a[m] - b[m], n_boot=n_boot, rng=rng)


# --------------------------------------------------------------- sleep state
def build_state_space(features):
    """Robust z -> PCA (d = D_STATE).  Fitted on pooled features."""
    Fpool = np.vstack(list(features.values()))
    center = np.median(Fpool, axis=0)
    scale = sst.iqr(Fpool, axis=0)
    scale[~np.isfinite(scale) | (scale <= 0)] = 1.0
    Z = (Fpool - center) / scale
    pca = PCA(n_components=D_STATE, random_state=RANDOM_STATE).fit(Z)

    def transform(F):
        Zt = (np.asarray(F, dtype=float) - center) / scale
        return pca.transform(Zt)

    return transform, pca.explained_variance_ratio_


def load_cell(stem: str, win_s: float, tag: str = ""):
    """{stage: (n_bout, n_win, 8) float64} for one recording at one window."""
    p = CACHE / f"{stem}{tag}_win{win_s:g}.npz"
    if not p.exists():
        return None, None
    z = np.load(p, allow_pickle=False)
    X, stage = z["X"], z["stage"]
    if X.size == 0:
        return None, None
    out = {}
    for g in STAGES:
        m = stage == g
        if m.sum() >= MIN_BOUTS:
            out[g] = X[m].astype(np.float64)
    meta = dict(win_s=float(z["win_s"]), n_use=int(z["n_use"]),
                n_skip=int(z["n_skip"]))
    return (out if len(out) >= 2 else None), meta


def cache_stems(prefix="SC", win_s=1.0, tag=""):
    return sorted({p.name[:6] for p in CACHE.glob(f"{prefix}*{tag}_win{win_s:g}.npz")
                   if p.name[6:].startswith(tag + "_win")})


def state_space(feat, with_jacobian=False):
    """Robust z + PCA(d = 4), fitted on this recording's pooled bouts.

    `with_jacobian` additionally returns the exact 4x8 linear map J of the affine
    transform, obtained from the transform itself (state = (F - c)/s @ V' is affine,
    so J[:, j] = transform(e_j) - transform(0) exactly).  J carries the measured
    instrument-noise covariance from feature space into the state space.
    """
    pooled = {g: v.reshape(-1, v.shape[-1]) for g, v in feat.items()}
    transform, evr = build_state_space(pooled)
    Z = {g: np.stack([transform(b) for b in v]) for g, v in feat.items()}
    if not with_jacobian:
        return Z, np.asarray(evr, float)
    n_f = pooled[next(iter(pooled))].shape[1]
    base = transform(np.zeros((1, n_f)))[0]
    J = np.stack([transform(np.eye(n_f)[j:j + 1])[0] - base
                  for j in range(n_f)], axis=1)          # (d, n_f)
    return Z, np.asarray(evr, float), J


def load_noise_R(stem, stage, win_s, J):
    """Instrument-noise covariance in the state space: J R_F J'.

    R_F is measured on phase-randomised copies of this recording's own bouts
    (the <stem>_noise.npz file of the sleep window cache), so it contains no
    information about the dynamics.
    """
    p = CACHE / f"{stem}_noise.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    key = f"{stage}|{win_s:g}"
    if key not in z.files:
        return None
    RF = np.asarray(z[key], float)
    R = J @ RF @ J.T
    return 0.5 * (R + R.T)


# --------------------------------------------------------------- band-power pipeline
def band_power(x, sfreq, win_s):
    """Non-overlapping short-time band power, (n_win, n_chan, n_band).

    scipy spectrogram, Hann window, nperseg = win_s * sfreq, noverlap = 0, constant
    detrend, power spectral density; band power = sum over the band's bins x bin width.
    """
    from scipy import signal as sps
    nperseg = int(round(win_s * sfreq))
    if x.shape[1] < nperseg:
        return np.zeros((0, x.shape[0], len(BANDS)))
    f, t, S = sps.spectrogram(x, fs=sfreq, window="hann", nperseg=nperseg,
                              noverlap=0, detrend="constant",
                              scaling="density", mode="psd", axis=-1)
    df = f[1] - f[0]
    out = np.empty((S.shape[-1], x.shape[0], len(BANDS)))
    for bi, (lo, hi) in enumerate(BANDS.values()):
        m = (f >= lo) & (f < hi)
        out[:, :, bi] = (S[:, m, :].sum(axis=1) * df).T
    return out


def n_bins_per_band(sfreq, win_s):
    """How many FFT bins each band gets at this window - the cost of shortening."""
    nperseg = int(round(win_s * sfreq))
    f = np.fft.rfftfreq(nperseg, 1.0 / sfreq)
    return {b: int(((f >= lo) & (f < hi)).sum()) for b, (lo, hi) in BANDS.items()}


def band_limited(n, lo, hi, rng, sfreq=SFREQ):
    """Unit-variance noise whose power sits strictly inside [lo, hi)."""
    w = rng.standard_normal(n)
    F = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, 1.0 / sfreq)
    F[(f < lo) | (f >= hi)] = 0.0
    x = np.fft.irfft(F, n)
    s = x.std()
    return x / s if s > 0 else x


def synth_recording(a_per_band, sd_per_band, n_win, win_s, rng, envelope="step",
                    sfreq=SFREQ):
    """A two-channel signal whose per-band log10 power is a known AR(1).

    envelope="step"       the log-power is constant inside each window, so the
                          window's TRUE log10 band power is exactly the injected
                          process and only the spectral-estimation noise is added
    envelope="continuous" the log-power is an AR(1) at the sample rate with the
                          matching per-window coefficient, so the window ALSO
                          averages a varying envelope

    Returns (signal (n_chan, n_samp), U (n_win, n_chan, n_band) injected log10
    power per window) - U is the ground truth the pipeline is measured against.
    """
    nper = int(round(win_s * sfreq))
    n_samp = nper * n_win
    bands = list(BANDS.items())
    sig = np.zeros((N_CHAN, n_samp))
    U = np.zeros((n_win, N_CHAN, len(bands)))
    for ci in range(N_CHAN):
        for bi, (bname, (lo, hi)) in enumerate(bands):
            a = float(a_per_band[bname])
            sd = float(sd_per_band[bname])
            u = np.empty(n_win)
            u[0] = rng.standard_normal() * sd
            for t in range(1, n_win):
                u[t] = a * u[t - 1] + sd * np.sqrt(1 - a ** 2) * rng.standard_normal()
            U[:, ci, bi] = u
            if envelope == "step":
                env = np.repeat(10.0 ** (0.5 * u), nper)
            else:
                from scipy.signal import lfilter
                a_s = a ** (1.0 / nper)
                sd_s = sd * np.sqrt(1 - a_s ** 2)
                nz = rng.standard_normal(n_samp) * sd_s
                nz[0] = u[0]
                v = lfilter([1.0], [1.0, -a_s], nz)
                env = 10.0 ** (0.5 * v)
                U[:, ci, bi] = v.reshape(n_win, nper).mean(axis=1)
            sig[ci] += env * band_limited(n_samp, lo, hi, rng)
    return sig, U


def features_from_signal(sig, win_s, overlap=0.0, sfreq=SFREQ):
    """The pipeline's band power -> log10, (n_win, 8).

    overlap = 0 calls `band_power` unchanged; a non-zero overlap re-runs the same
    spectrogram with noverlap set.
    """
    if overlap == 0.0:
        P = band_power(sig, sfreq, win_s)
    else:
        from scipy import signal as sps
        nper = int(round(win_s * sfreq))
        nov = int(round(nper * overlap))
        f, t, S = sps.spectrogram(sig, fs=sfreq, window="hann", nperseg=nper,
                                  noverlap=nov, detrend="constant",
                                  scaling="density", mode="psd", axis=-1)
        df = f[1] - f[0]
        P = np.empty((S.shape[-1], sig.shape[0], len(BANDS)))
        for bi, (lo, hi) in enumerate(BANDS.values()):
            m = (f >= lo) & (f < hi)
            P[:, :, bi] = (S[:, m, :].sum(axis=1) * df).T
    return np.log10(np.maximum(P, 1e-30)).reshape(P.shape[0], -1), P.shape[0]
