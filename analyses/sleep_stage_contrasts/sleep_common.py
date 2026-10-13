# -*- coding: utf-8 -*-
r"""Shared code of the sleep-stage contrasts: the standardized four-dimensional sleep state, the
VAR(1) rate estimators fitted in it and the participant-level contrast statistics.

Imported by stage_contrasts.py and random_direction_rates.py (and, for `build_state_space`, by
../sleep_telemetry), so that the scripts share one implementation.

THE STATE
---------
The sleep window cache (window_cache.py, noise_cache.py) holds, per Sleep-EDF recording and window
length, the log10 band power of two EEG channels in four bands for every used 60-s span of a scored
bout.  `state_space` fits the robust z (median / IQR) and a PCA with d = 4 components on the pooled
bouts of one recording (`build_state_space`, PCA random state 20260728) and projects every bout.

Cache directory: analyses.common.paths.cache("sleep_states") (environment variable
NDIF_SLEEP_STATES_CACHE, default analyses/_work/sleep_states).

THREE ESTIMATORS OF THE RELAXATION RATE
---------------------------------------
  (1) one-step, `lam_1`        -(1/d) sum ln|eig M| / dt for the least-squares VAR(1) propagator M.
                               If the observed series is x_t + e_t with e white, then
                               r_obs(k) = (1 - s) rho^k for every k >= 1, so r_obs(1) under-reads
                               rho by the signal share (1 - s) and the rate reads high.
  (2) bias-corrected, `lam_bc` the same functional on the bootstrap bias-corrected propagator
                               (`bias_correct`), which removes the order-1/n shrinkage of M-hat but
                               not the attenuation by the white share.
  (3) lag-2 instrument, `lam_iv`  M = C2 C1^{-1}, Sigma_x = M^{-1} C1, Q = Sigma_x - M Sigma_x M',
                               white share s = 1 - tr(Sigma_x)/tr(C0); free of an additive white
                               component, but it needs r(2) measurably above zero.
  and the noise-corrected fits `lam_nc` / `lam_ncbc`: the propagator from lag 0 and lag 1 after
  subtracting the instrument-noise covariance R measured on phase-randomised copies of the same
  bouts (noise_cache.py), M = C1 (C0 - R)^{-1}, without and with a bootstrap bias correction
  matched to this estimator.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import linalg as sla
from scipy import stats as sst
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from analyses.common import paths  # noqa: E402

CACHE = Path(paths.cache("sleep_states"))

STAGES = ("Wake", "N1", "N2", "N3", "N4", "REM")
CONTRASTS = (("Wake", "N1"), ("N1", "N2"), ("N2", "N3"), ("N3", "N4"))
MIN_BOUTS = 5
D_STATE = 4
RANDOM_STATE = 20260728          # PCA random state of the sleep state space
SEED = 20260922


# ----------------------------------------------------------------- state space
def build_state_space(features):
    """Robust z -> PCA (d=D_STATE). Fitted on pooled features."""
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


# ----------------------------------------------------------------- VAR(1) fit
@dataclass
class OUFit:
    M: np.ndarray            # discrete one-step propagator, exp(A dt)
    Sigma_eps: np.ndarray    # one-step residual covariance
    Sigma0: np.ndarray       # lag-0 (stationary) covariance
    Sigma1: np.ndarray       # lag-1 cross-covariance E[(x_t+1-mu)(x_t-mu)']
    dt: float
    n_pairs: int


def fit_ou(segments, dt, lag=1, ridge=1e-6):
    """Fit a VAR(1) propagator from one or more continuous state segments.

    `segments` is a list of (n_t, d) arrays.  Each is centred on its own mean
    before pooling, so a pooled fit describes the shared relaxation dynamics
    rather than the displacement between segments.  Lagged pairs are never
    formed across a segment boundary.
    """
    S0 = S1 = None
    npair = 0
    d = segments[0].shape[1]
    S0 = np.zeros((d, d))
    S1 = np.zeros((d, d))
    ntot = 0
    for X in segments:
        X = np.asarray(X, dtype=float)
        if len(X) < lag + 3:
            continue
        Xc = X - X.mean(axis=0)
        a, b = Xc[:-lag], Xc[lag:]
        S0 += Xc.T @ Xc
        S1 += b.T @ a
        ntot += len(Xc)
        npair += len(a)
    if npair < d + 2:
        return None
    S0 /= max(ntot - 1, 1)
    S1 /= max(npair - 1, 1)
    reg = ridge * np.trace(S0) / d * np.eye(d)
    M = S1 @ np.linalg.inv(S0 + reg)
    Seps = S0 - M @ S0 @ M.T
    return OUFit(M=M, Sigma_eps=0.5 * (Seps + Seps.T), Sigma0=S0, Sigma1=S1,
                 dt=dt * lag, n_pairs=npair)


def _simulate_var1_batch(M, L, n, B, rng, burn=200):
    """Vectorised simulation of B independent VAR(1) paths. Returns (B, n, d)."""
    d = M.shape[0]
    x = np.zeros((B, d))
    for _ in range(burn):
        x = x @ M.T + rng.standard_normal((B, d)) @ L.T
    X = np.empty((B, n, d))
    for t in range(n):
        x = x @ M.T + rng.standard_normal((B, d)) @ L.T
        X[:, t] = x
    return X


def _refit_M_batch(X, ridge=1e-6):
    """Least-squares VAR(1) propagator for each path in a (B, n, d) batch."""
    Xc = X - X.mean(axis=1, keepdims=True)
    a, b = Xc[:, :-1], Xc[:, 1:]
    S0 = np.einsum("bti,btj->bij", Xc, Xc) / (X.shape[1] - 1)
    S1 = np.einsum("bti,btj->bij", b, a) / (a.shape[1] - 1)
    d = X.shape[2]
    reg = ridge * (np.trace(S0, axis1=1, axis2=2) / d)[:, None, None] * np.eye(d)
    return S1 @ np.linalg.inv(S0 + reg)


def bias_correct(fit: OUFit, n_boot=200, rng=None):
    """Bootstrap bias correction of the VAR(1) propagator, and lambda from it.

    Least-squares estimates of autoregressive coefficients are biased toward
    zero (the Hurwicz/Kendall bias, of order 1/n), which shrinks |eig(M)| and
    therefore inflates any relaxation rate computed as -log|eig|.  Simulating
    from M-hat and re-estimating gives E[M*|M-hat]; the bootstrap bias estimate
    is E[M*] - M-hat, and the corrected estimate is M_bc = 2 M-hat - E[M*].
    """
    rng = np.random.default_rng(rng)
    d = fit.M.shape[0]
    Se = 0.5 * (fit.Sigma_eps + fit.Sigma_eps.T)
    w, Vv = np.linalg.eigh(Se)
    L = Vv @ np.diag(np.sqrt(np.maximum(w, 0))) @ Vv.T
    n = max(fit.n_pairs + 1, 4 * d)

    X = _simulate_var1_batch(fit.M, L, n, n_boot, rng)
    M_star = _refit_M_batch(X).mean(axis=0)
    M_bc = 2 * fit.M - M_star

    def rates(M):
        ev = np.abs(np.linalg.eigvals(M))
        ev_ok = ev[(ev > 0) & (ev < 1)]
        lam_mean = float(-np.mean(np.log(ev_ok)) / fit.dt) if len(ev_ok) == d else np.nan
        rho = float(np.max(ev))
        lam_max = float(-np.log(rho) / fit.dt) if 0 < rho < 1 else np.nan
        return lam_mean, lam_max

    lm_bc, lx_bc = rates(M_bc)
    lm_raw, lx_raw = rates(fit.M)
    return dict(M_bc=M_bc, lambda_mean_bc=lm_bc, lambda_max_bc=lx_bc,
                lambda_mean_raw=lm_raw, lambda_max_raw=lx_raw,
                bias_fro=float(np.linalg.norm(M_star - fit.M, "fro")))


# ----------------------------------------------------------------- loading
def load_cell(stem: str, win_s: float):
    """{stage: (n_bout, n_win, 8) float64} for one recording at one window."""
    p = CACHE / f"{stem}_win{win_s:g}.npz"
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


def cache_stems(prefix="SC", win_s=1.0):
    """Recordings with a cache file at this window (six-character Sleep-EDF names)."""
    return sorted({p.name[:6] for p in CACHE.glob(f"{prefix}*_win{win_s:g}.npz")
                   if p.name[6:].startswith("_win")})


def state_space(feat, with_jacobian=False):
    """Robust-z + PCA(d=4), fitted on this recording's pooled bouts.

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

    R_F is measured by noise_cache.py on phase-randomised copies of this
    recording's own bouts, so it contains no information about the dynamics.
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


# ----------------------------------------------------------------- 1-D ACF
def pooled_acf(blocks, kmax=6):
    """Autocorrelation at lags 1..kmax, pooled over blocks, no pair across a
    block boundary; each block centred on its own mean."""
    num = np.zeros(kmax)
    da = np.zeros(kmax)
    db = np.zeros(kmax)
    for b in blocks:
        c = np.asarray(b, float)
        c = c - c.mean()
        for k in range(1, kmax + 1):
            if len(c) <= k + 1:
                continue
            a, z = c[:-k], c[k:]
            num[k - 1] += float(a @ z)
            da[k - 1] += float(a @ a)
            db[k - 1] += float(z @ z)
    den = np.sqrt(da * db)
    r = np.where(den > 0, num / np.maximum(den, 1e-300), np.nan)
    return r


# ----------------------------------------------------------------- d-D fits
def lagged_cov(blocks, kmax=2):
    """C0, C1, C2 pooled over blocks, each block centred on its own mean."""
    d = blocks[0].shape[1]
    C = [np.zeros((d, d)) for _ in range(kmax + 1)]
    n = [0] * (kmax + 1)
    for b in blocks:
        c = np.asarray(b, float)
        c = c - c.mean(axis=0)
        C[0] += c.T @ c
        n[0] += len(c)
        for k in range(1, kmax + 1):
            if len(c) <= k + 1:
                continue
            C[k] += c[k:].T @ c[:-k]
            n[k] += len(c) - k
    for k in range(kmax + 1):
        if n[k] > 1:
            C[k] /= n[k] - 1
    return C, n


def rates_from_M(M, dt, d):
    ev = np.abs(np.linalg.eigvals(M))
    ok = ev[(ev > 0) & (ev < 1)]
    lam_mean = float(-np.mean(np.log(ok)) / dt) if len(ok) == d else np.nan
    rho = float(np.max(ev))
    lam_slow = float(-np.log(rho) / dt) if 0 < rho < 1 else np.nan
    per_mode = np.sort(-np.log(np.clip(ev, 1e-12, 0.999999)) / dt)[::-1]
    return lam_mean, lam_slow, per_mode, int(len(ok))


def logD_from_Q(Q, dt, d):
    """(1/d) log det (Q / 2dt): the mean log diffusion per dimension."""
    Qs = 0.5 * (Q + Q.T)
    w = np.linalg.eigvalsh(Qs)
    if np.any(w <= 0):
        return np.nan, np.nan
    return (float(np.mean(np.log(w / (2 * dt)))),
            float(np.log(np.trace(Qs) / (d * 2 * dt))))


def _shrink_to_pd(C0, R, floor=0.05):
    """Largest alpha <= 1 with C0 - alpha R positive definite with a margin.

    Returns (Sigma_x, alpha).  alpha < 1 means the measured instrument noise is
    larger than the observed covariance can carry in some direction, which is
    itself reported: it is the direction in which the state is pure instrument.
    """
    R = 0.5 * (R + R.T)
    lo, hi = 0.0, 1.0
    thr = floor * np.trace(C0) / C0.shape[0]
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        S = C0 - mid * R
        if np.min(np.linalg.eigvalsh(0.5 * (S + S.T))) > thr:
            lo = mid
        else:
            hi = mid
    S = 0.5 * ((C0 - lo * R) + (C0 - lo * R).T)
    return S, float(lo)


def nc_propagator(C0, C1, R, dt, d, ridge=1e-6):
    """Noise-corrected propagator from lag 0 and lag 1 with R measured separately."""
    Sx, alpha = _shrink_to_pd(C0, R)
    reg = ridge * np.trace(Sx) / d * np.eye(d)
    M = C1 @ np.linalg.inv(Sx + reg)
    Q = Sx - M @ Sx @ M.T
    return M, 0.5 * (Q + Q.T), Sx, alpha


def _simulate_nc(M, Q, R, n_blocks, n_win, B, rng):
    """B draws of (n_blocks, n_win, d) from x' = Mx + w, observed as x + v."""
    d = M.shape[0]
    Lw = np.linalg.cholesky(Q + 1e-12 * np.eye(d) * max(np.trace(Q), 1e-12))
    Lv = np.linalg.cholesky(R + 1e-12 * np.eye(d) * max(np.trace(R), 1e-12))
    x = np.zeros((B, n_blocks, d))
    for _ in range(100):
        x = x @ M.T + rng.standard_normal(x.shape) @ Lw.T
    Y = np.empty((B, n_blocks, n_win, d))
    for t in range(n_win):
        x = x @ M.T + rng.standard_normal(x.shape) @ Lw.T
        Y[:, :, t] = x + rng.standard_normal(x.shape) @ Lv.T
    return Y


def bias_correct_nc(M, Q, R, n_blocks, n_win, dt, d, n_boot=120, rng=None,
                    ridge=1e-6):
    """Bootstrap bias correction matched to THIS estimator and THIS geometry.

    The least-squares propagator is biased toward zero at order 1/n (Hurwicz), and
    centring every bout on its own mean adds a further downward bias of the same
    order.  Both are reproduced by simulating from the fitted (M, Q) with the
    measured observation noise R added, re-estimating with the identical
    noise-corrected procedure, and taking M_bc = 2 M - E[M*].
    """
    rng = np.random.default_rng(rng)
    Y = _simulate_nc(M, Q, R, n_blocks, n_win, n_boot, rng)
    Yc = Y - Y.mean(axis=2, keepdims=True)
    n0 = n_blocks * n_win
    n1 = n_blocks * (n_win - 1)
    C0s = np.einsum("bkti,bktj->bij", Yc, Yc) / max(n0 - 1, 1)
    C1s = np.einsum("bkti,bktj->bij", Yc[:, :, 1:], Yc[:, :, :-1]) / max(n1 - 1, 1)
    Ms = []
    for b in range(n_boot):
        try:
            Mb, _, _, _ = nc_propagator(C0s[b], C1s[b], R, dt, d, ridge)
            Ms.append(Mb)
        except Exception:  # noqa: BLE001
            continue
    if len(Ms) < max(10, n_boot // 4):
        return M, np.nan
    Mstar = np.mean(Ms, axis=0)
    return 2 * M - Mstar, float(np.linalg.norm(Mstar - M, "fro"))


def fit_cell(blocks, dt, d=D_STATE, n_boot=200, rng=None, ridge=1e-6, R=None):
    """One stage of one recording: the three propagators and their diffusions.

    Returns the one-step fit, the bootstrap bias-corrected fit and the lag-2
    instrument fit (and, with R, the noise-corrected fits), plus the exact
    discrete Lyapunov solution Sigma = M Sigma M' + Q of the FITTED (M, Q)
    compared with the OBSERVED lag-0 covariance.
    """
    blocks = [np.asarray(b, float) for b in blocks]
    C, npair = lagged_cov(blocks, kmax=2)
    C0, C1, C2 = C
    out = dict(n_blocks=len(blocks), n_win=int(np.mean([len(b) for b in blocks])),
               n_pairs=int(npair[1]), tr_C0=float(np.trace(C0)))

    fit = fit_ou(blocks, dt=dt, lag=1, ridge=ridge)
    if fit is None:
        return None
    M1 = fit.M
    Q1 = 0.5 * (fit.Sigma_eps + fit.Sigma_eps.T)
    lm, ls, pm, nok = rates_from_M(M1, dt, d)
    ld, ltr = logD_from_Q(Q1, dt, d)
    out.update(lam_mean_1=lm, lam_slow_1=ls, logD_1=ld, logDtr_1=ltr,
               modes_1=pm.tolist(), n_eig_ok_1=nok)

    # exact discrete Lyapunov on the fitted pair, against the observed covariance
    try:
        Sig = sla.solve_discrete_lyapunov(M1, Q1)
        out["lyap_ratio_1"] = float(np.trace(Sig) / max(np.trace(C0), 1e-300))
        out["lyap_logdet_ratio_1"] = float(
            (np.linalg.slogdet(Sig)[1] - np.linalg.slogdet(C0)[1]) / d)
    except Exception:  # noqa: BLE001
        out["lyap_ratio_1"] = np.nan
        out["lyap_logdet_ratio_1"] = np.nan

    bc = bias_correct(fit, n_boot=n_boot, rng=rng)
    Mb = bc["M_bc"]
    lmb, lsb, pmb, nokb = rates_from_M(Mb, dt, d)
    Qb = C0 - Mb @ C0 @ Mb.T
    ldb, ltrb = logD_from_Q(Qb, dt, d)
    out.update(lam_mean_bc=lmb, lam_slow_bc=lsb, logD_bc=ldb, logDtr_bc=ltrb,
               modes_bc=pmb.tolist(), n_eig_ok_bc=nokb,
               bias_fro=bc["bias_fro"])
    try:
        Sigb = sla.solve_discrete_lyapunov(Mb, 0.5 * (Qb + Qb.T))
        out["lyap_ratio_bc"] = float(np.trace(Sigb) / max(np.trace(C0), 1e-300))
    except Exception:  # noqa: BLE001
        out["lyap_ratio_bc"] = np.nan

    # ---- lag-2 instrument
    reg = ridge * np.trace(C0) / d * np.eye(d)
    try:
        Miv = C2 @ np.linalg.inv(C1 + reg)
        Sx = np.linalg.solve(Miv + reg, C1)
        Sx = 0.5 * (Sx + Sx.T)
        Qiv = Sx - Miv @ Sx @ Miv.T
        lmi, lsi, pmi, noki = rates_from_M(Miv, dt, d)
        ldi, ltri = logD_from_Q(Qiv, dt, d)
        share = float(np.clip(1.0 - np.trace(Sx) / max(np.trace(C0), 1e-300), -1, 1))
        out.update(lam_mean_iv=lmi, lam_slow_iv=lsi, logD_iv=ldi, logDtr_iv=ltri,
                   modes_iv=pmi.tolist(), n_eig_ok_iv=noki, white_share_iv=share,
                   sx_pd=bool(np.all(np.linalg.eigvalsh(Sx) > 0)))
    except Exception:  # noqa: BLE001
        out.update(lam_mean_iv=np.nan, lam_slow_iv=np.nan, logD_iv=np.nan,
                   logDtr_iv=np.nan, modes_iv=[np.nan] * d, n_eig_ok_iv=0,
                   white_share_iv=np.nan, sx_pd=False)

    # ---- noise-corrected: R measured on phase-randomised copies of this signal
    if R is not None:
        try:
            Mn, Qn, Sx, alpha = nc_propagator(C0, C1, R, dt, d, ridge)
            lmn, lsn, pmn, nokn = rates_from_M(Mn, dt, d)
            ldn, ltrn = logD_from_Q(Qn, dt, d)
            Mnb, bias = bias_correct_nc(Mn, Qn, alpha * R, out["n_blocks"],
                                        out["n_win"], dt, d, n_boot=120, rng=rng,
                                        ridge=ridge)
            lmnb, lsnb, pmnb, noknb = rates_from_M(Mnb, dt, d)
            Qnb = Sx - Mnb @ Sx @ Mnb.T
            ldnb, ltrnb = logD_from_Q(0.5 * (Qnb + Qnb.T), dt, d)
            out.update(lam_mean_nc=lmn, lam_slow_nc=lsn, logD_nc=ldn, logDtr_nc=ltrn,
                       modes_nc=pmn.tolist(), n_eig_ok_nc=nokn,
                       lam_mean_ncbc=lmnb, lam_slow_ncbc=lsnb, logD_ncbc=ldnb,
                       logDtr_ncbc=ltrnb, modes_ncbc=pmnb.tolist(),
                       n_eig_ok_ncbc=noknb, nc_alpha=alpha, nc_bias_fro=bias,
                       noise_share_measured=float(alpha * np.trace(R)
                                                  / max(np.trace(C0), 1e-300)),
                       logSx=float(np.linalg.slogdet(Sx)[1] / d),
                       trSx=float(np.trace(Sx)))
            out["Sx"] = Sx.tolist()
            out["M_ncbc"] = Mnb.tolist()
        except Exception:  # noqa: BLE001
            out.update(lam_mean_nc=np.nan, lam_slow_nc=np.nan, logD_nc=np.nan,
                       lam_mean_ncbc=np.nan, lam_slow_ncbc=np.nan, logD_ncbc=np.nan,
                       modes_nc=[np.nan] * d, modes_ncbc=[np.nan] * d,
                       nc_alpha=np.nan, nc_bias_fro=np.nan,
                       noise_share_measured=np.nan, logSx=np.nan, trSx=np.nan)
    out["C0"] = C0.tolist()
    return out


# ------------------------------------------------------------- 1-D spread
def spread(blocks):
    """Stationary spread: each block centred on its own mean."""
    n = sum(len(b) for b in blocks)
    if n == 0:
        return float("nan")
    ss = sum(float(((b - b.mean()) ** 2).sum()) for b in blocks)
    return float(np.sqrt(ss / n))


def escape_widths(Z, stage):
    """Per-cell directional spread: median over the directions towards every other
    stage, with the per-direction values kept."""
    means = {g: v.reshape(-1, v.shape[-1]).mean(axis=0) for g, v in Z.items()}
    vals, dirs = [], []
    for b in Z:
        if b == stage:
            continue
        u = means[b] - means[stage]
        nu = float(np.linalg.norm(u))
        if not np.isfinite(nu) or nu <= 0:
            continue
        u = u / nu
        p = [z @ u for z in Z[stage]]
        vals.append(spread(p))
        dirs.append(dict(to=b, sep=nu, width=spread(p), r=pooled_acf(p, 4).tolist()))
    if not vals:
        return np.nan, dirs
    return float(np.median(vals)), dirs


# ------------------------------------------------------------- statistics
def boot_ci(vals, n_boot=4000, rng=None, stat=np.median):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if len(v) < 2:
        return dict(n=len(v), est=float(stat(v)) if len(v) else np.nan,
                    lo=np.nan, hi=np.nan)
    rng = np.random.default_rng(rng)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    bs = stat(v[idx], axis=1)
    return dict(n=int(len(v)), est=float(stat(v)),
                lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                mean=float(np.mean(v)))


def paired_summary(log_ratios, n_boot=4000, rng=None):
    """Participant-level paired contrast on the log scale: fold, CI, sign and Wilcoxon."""
    v = np.asarray([x for x in log_ratios if np.isfinite(x)], float)
    n = len(v)
    if n < 2:
        return dict(n=n)
    ci = boot_ci(v, n_boot=n_boot, rng=rng)
    k = int((v > 0).sum())
    o = dict(n=n, k_pos=k, fold=float(np.exp(np.median(v))),
             log_median=float(np.median(v)),
             fold_lo=float(np.exp(ci["lo"])), fold_hi=float(np.exp(ci["hi"])),
             mean_log=float(np.mean(v)),
             sign_p=float(sst.binomtest(k, n, 0.5).pvalue))
    try:
        o["wilcoxon_p"] = float(sst.wilcoxon(v).pvalue)
    except Exception:  # noqa: BLE001
        o["wilcoxon_p"] = np.nan
    return o


def read_table(path):
    """Rows of a stored per-cell or per-direction table (csv.DictWriter output), in file order.

    Every value is restored to the type the scripts gave it: numbers (written as their shortest
    round-trip representation, so the floats are recovered exactly), True/False, or text; an
    empty field is a key the row did not have and is left out.
    """
    import csv
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            row = {}
            for k, v in raw.items():
                if v == "":
                    continue
                if v in ("True", "False"):
                    row[k] = v == "True"
                    continue
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
            rows.append(row)
    return rows
