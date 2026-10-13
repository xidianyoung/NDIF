# -*- coding: utf-8 -*-
"""Shared estimators for the time-rescaling hypothesis.

Time rescaling.  Load raises the diffusion AND the relaxation rate, in the form  A -> cA, D -> cD.  Every
timescale divides by c and the stationary distribution is EXACTLY unchanged, because the Lyapunov
solution of (cA, cD) equals that of (A, D).  Predictions:

  P1  the stationary distribution is invariant in SHAPE - the state's correlation matrix does not
      move with load, while its timescales shorten;
  P2  the RATIOS of the relaxation rates across modes are preserved and only their common
      magnitude scales - a ratio survives a multiplicative attenuation of the rate channel;
  P3  the contrast lies on the UNIT-SLOPE DIAGONAL of the (dlog lambda, dlog D) plane;
  P4  the autocorrelation functions at different loads collapse onto one curve when time is
      rescaled by the fitted common factor c.

WHAT THE DISCRETE ESTIMANDS DO UNDER TIME RESCALING
---------------------------------------
Sample a stable linear system at interval tau.  Write M = exp(A tau), a_k = |eig_k(M)|, P the
stationary covariance, Q = P - M P M' the one-step innovation covariance.  Under A -> cA with the
stationary distribution held fixed (which is what D -> cD does):

    M -> M^c          so  |eig_k| -> a_k^c        and    lambda_k = -ln a_k / tau -> c lambda_k
    P -> P            exactly
    Q -> P - M^c P M^c'

Therefore, reading the per-step estimands with dt held at the sampling interval:

    dlog lambda = ln c                                                   EXACT, any regime
    dlog D      = ln [ tr(P - M^c P M^c') / tr(P - M P M') ]             NOT ln c in general

In one dimension dlog D = ln[(1 - a^{2c})/(1 - a^2)], which tends to ln c only as a -> 1 (sampling
fast relative to the relaxation time) and tends to ZERO as a -> 0 (sampling at or beyond the
correlation time).  **The unit-slope diagonal of P3 is a property of the well-sampled regime, not
of the hypothesis.**  In a coarsely sampled cohort a true pure time rescaling lands on the
horizontal (pure-gain) axis of the plane, and the plane cannot tell the two apart.  Every cohort's
own a is measured and reported with its plane, and the decimation calibration measures the landing
place directly rather than assuming it.

CALIBRATORS (all end to end, all on the real recordings)
--------------------------------------------------------
  INJ-S  temporal decimation, stride k vs stride k+1 of the SAME real window series.  Decimation
         changes no data value and multiplies the per-step rate estimand by exactly (k+1)/k for any
         VAR(1); the truth for dlog lambda is ln((k+1)/k), which sweeps 0.049 (k=20) to 0.693 (k=1)
         and therefore covers the range of the effect sizes at issue.
  INJ-T  an exact time rescaling of the real innovation stream: refit M and P on the real blocks,
         rebuild the series as x'_t = M^c x'_{t-1} + e'_t where e'_t are the REAL residuals
         recoloured to the covariance Q_c = P - M^c P M^c' that holds the stationary distribution
         fixed.  Truth: dlog lambda = ln c exactly, dlog D exactly computable, dCorr = 0 exactly.
  INJ-D  a pure drive injection: the within-block residuals of the high condition are multiplied by
         g along a fixed direction.  This is a similarity transform of the state, so the truth for
         lambda is exactly zero, and the truth for tr(Q) and for the correlation matrix is exactly
         computable.
  INJ-R  a pure reshaping: M -> V diag(|mu_k|^{gamma_k}) V^{-1} with the gamma chosen so that the
         COMMON factor is exactly zero and only the spectrum's shape moves, with Q recoloured as in
         INJ-T so the stationary distribution is again held fixed.

Imported by cog_state.py.  The propagator fit itself is cog_fit_lib.fit_cell.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
from scipy import linalg as sla
from scipy import stats as sst

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cog_fit_lib as R                                               # noqa: E402

SEED = 20260922
EPS = 1e-12


# =============================================================================== spectra (P2)
def log_rates(M, dt=1.0, floor=1e-8):
    """ell_k = ln lambda_k = ln(-ln|eig_k(M)| / dt), sorted ascending, plus the raw pieces."""
    ev = np.linalg.eigvals(M)
    a = np.abs(ev)
    ok = np.all((a > floor) & (a < 1.0))
    lam = -np.log(np.clip(a, floor, 1.0 - 1e-15)) / dt
    o = np.argsort(lam)
    return dict(ok=bool(ok), ev=ev[o], absev=a[o], lam=lam[o], ell=np.log(np.maximum(lam[o], EPS)),
                loga=np.log(np.maximum(a[o], floor)))


def match_modes(M_lo, M_hi):
    """Pair the eigenmodes of two propagators by eigenvector overlap (Hungarian on |v^H u|).

    Returns (idx_lo, idx_hi) index arrays of equal length d, and the mean matched |cos|.
    """
    from scipy.optimize import linear_sum_assignment
    wl, Vl = np.linalg.eig(M_lo)
    wh, Vh = np.linalg.eig(M_hi)
    Vl = Vl / (np.linalg.norm(Vl, axis=0, keepdims=True) + EPS)
    Vh = Vh / (np.linalg.norm(Vh, axis=0, keepdims=True) + EPS)
    Cst = np.abs(Vl.conj().T @ Vh)
    r, c = linear_sum_assignment(-Cst)
    return r, c, float(np.mean(Cst[r, c]))


def p2_decompose(M_lo, M_hi, dt=1.0):
    """Split the change of the relaxation spectrum into a COMMON factor and a RESHAPING.

    In ell = ln lambda the hypothesis is an exact translation: ell_k -> ell_k + ln c for every k.
    So the decomposition of the change vector d_k = ell_k(hi) - ell_k(lo) is

        d_k = dbar + rho_k ,   dbar = mean_k d_k ,   sum_k rho_k = 0

    with sum_k d_k^2 = d * dbar^2 + ||rho||^2, which apportions the total squared change between the
    two components.  The sorted spectrum (order statistics) is the primary pairing because it needs
    no mode correspondence; the eigenvector-matched pairing is reported beside it, since only the
    matched version can see a change of RANK ORDER.
    """
    A, B = log_rates(M_lo, dt), log_rates(M_hi, dt)
    if not (A["ok"] and B["ok"]):
        return None
    d = len(A["ell"])
    dv = B["ell"] - A["ell"]                       # sorted / order-statistic pairing
    dbar = float(np.mean(dv))
    rho = dv - dbar
    tot = float(np.sum(dv ** 2))
    out = dict(
        d=d, c_common=float(np.exp(dbar)), dbar=dbar,
        reshape_rms=float(np.sqrt(np.mean(rho ** 2))),
        reshape_norm=float(np.linalg.norm(rho)),
        ss_total=tot,
        frac_common=float(d * dbar ** 2 / tot) if tot > 0 else np.nan,
        frac_reshape=float(np.sum(rho ** 2) / tot) if tot > 0 else np.nan,
        sd_ell_lo=float(np.std(A["ell"], ddof=1)), sd_ell_hi=float(np.std(B["ell"], ddof=1)),
        dlog_sd_ell=float(np.log(np.std(B["ell"], ddof=1) / max(np.std(A["ell"], ddof=1), EPS))),
        # the same contrast parametrised on log|eig| rather than on ln lambda
        mean_loga_lo=float(np.mean(A["loga"])), mean_loga_hi=float(np.mean(B["loga"])),
        dlog_mean_loga=float(np.log(abs(np.mean(B["loga"])) / max(abs(np.mean(A["loga"])), EPS))),
        lam_lo=A["lam"].tolist(), lam_hi=B["lam"].tolist(),
        ell_lo=A["ell"].tolist(), ell_hi=B["ell"].tolist(),
    )
    # --- eigenvector-matched pairing: rank order and its own common/reshape split
    try:
        rl, ch, cosm = match_modes(M_lo, M_hi)
        el = log_rates(M_lo, dt); eh = log_rates(M_hi, dt)
        # log_rates returns sorted arrays; recompute unsorted for the matched pairing
        al = np.abs(np.linalg.eigvals(M_lo)); ah = np.abs(np.linalg.eigvals(M_hi))
        ll = np.log(np.maximum(-np.log(np.clip(al, 1e-8, 1 - 1e-15)) / dt, EPS))
        lh = np.log(np.maximum(-np.log(np.clip(ah, 1e-8, 1 - 1e-15)) / dt, EPS))
        dm = lh[ch] - ll[rl]
        dbm = float(np.mean(dm)); rm = dm - dbm
        tm = float(np.sum(dm ** 2))
        out.update(match_cos=cosm, match_dbar=dbm,
                   match_reshape_rms=float(np.sqrt(np.mean(rm ** 2))),
                   match_frac_common=float(len(dm) * dbm ** 2 / tm) if tm > 0 else np.nan,
                   rank_spearman=float(sst.spearmanr(ll[rl], lh[ch]).statistic)
                   if len(dm) > 2 else np.nan)
        _ = el, eh
    except Exception:                                                     # noqa: BLE001
        out.update(match_cos=np.nan, match_dbar=np.nan, match_reshape_rms=np.nan,
                   match_frac_common=np.nan, rank_spearman=np.nan)
    return out


# =============================================================================== P1, stationary shape
def _offdiag(C):
    d = C.shape[0]
    iu = np.triu_indices(d, 1)
    return C[iu]


def corr_of(blocks, centre=True):
    """Correlation matrix of the state.  With centre=True every block is centred on its own mean
    first, which removes the displacement and any slow drift and leaves the fluctuation shape."""
    X = []
    for b in blocks:
        b = np.asarray(b, float)
        if len(b) < 3:
            continue
        X.append(b - b.mean(axis=0, keepdims=True) if centre else b)
    if not X:
        return None
    X = np.vstack(X)
    s = X.std(axis=0)
    if np.any(s <= 0):
        return None
    return np.corrcoef(X.T)


def corr_dist(C1, C2):
    """RMS difference of the off-diagonal correlations (a scale-free shape distance)."""
    if C1 is None or C2 is None:
        return np.nan
    return float(np.sqrt(np.mean((_offdiag(C1) - _offdiag(C2)) ** 2)))


def corr_agree(C1, C2):
    """Pearson agreement of the off-diagonal correlations."""
    if C1 is None or C2 is None:
        return np.nan
    a, b = _offdiag(C1), _offdiag(C2)
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def halves(blocks, rng=None, interleave=True, nchunk=6, nmin=12):
    """Split a block list in two, balanced in time so that a drift is shared by both halves.

    With four or more blocks the split is block-alternating.  With fewer - a COG-BCI MATB record is
    ONE 597-window segment - each block is cut into `nchunk` contiguous chunks that are dealt
    alternately to the two halves, which keeps the split interleaved in time rather than
    first-half / second-half.  Returns None when neither route leaves usable pieces.
    """
    n = len(blocks)
    if n >= 4:
        if interleave:
            return [blocks[0::2], blocks[1::2]]
        idx = np.arange(n)
        if rng is not None:
            rng.shuffle(idx)
        return [[blocks[i] for i in idx[: n // 2]], [blocks[i] for i in idx[n // 2:]]]
    A, B = [], []
    for b in blocks:
        b = np.asarray(b, float)
        m = len(b) // nchunk
        if m < nmin:
            continue
        for j in range(nchunk):
            (A if j % 2 == 0 else B).append(b[j * m: (j + 1) * m])
    if not A or not B:
        return None
    return [A, B]


def p1_invariance(blocks_lo, blocks_hi, centre=True, rng=None):
    """Between-load correlation distance against the matched WITHIN-condition split-half ceiling.

    Every distance is computed from half a condition's blocks, so sample size is matched between
    the between-load comparison and its ceiling.  Returns None if either condition is too thin.

    Only the CROSS half pairs (half 1 of one condition against half 2 of the other) enter the
    between-condition distance.  In a real load contrast all four pairs are disjoint and the choice
    is cosmetic, but in a calibration - where the injected condition is built from the SAME windows
    as the baseline - the same-index pairs share their windows and would drive the ratio to 1/2 at
    zero injected truth.  One statistic is used everywhere so the calibration is on the real one.
    """
    hl, hh = halves(blocks_lo, rng), halves(blocks_hi, rng)
    if hl is None or hh is None:
        return None
    Cl = [corr_of(h, centre) for h in hl]
    Ch = [corr_of(h, centre) for h in hh]
    if any(c is None for c in Cl + Ch):
        return None
    d_within = np.mean([corr_dist(Cl[0], Cl[1]), corr_dist(Ch[0], Ch[1])])
    a_within = np.mean([corr_agree(Cl[0], Cl[1]), corr_agree(Ch[0], Ch[1])])
    db, ab = [], []
    for i, j in ((0, 1), (1, 0)):
        db.append(corr_dist(Cl[i], Ch[j])); ab.append(corr_agree(Cl[i], Ch[j]))
    d_between, a_between = float(np.mean(db)), float(np.mean(ab))
    Cfl, Cfh = corr_of(blocks_lo, centre), corr_of(blocks_hi, centre)
    return dict(d_within=float(d_within), d_between=d_between,
                d_ratio=float(d_between / d_within) if d_within > 0 else np.nan,
                d_excess=float(d_between - d_within),
                a_within=float(a_within), a_between=a_between,
                a_frac_ceiling=float(a_between / a_within) if a_within > 0 else np.nan,
                d_full=corr_dist(Cfl, Cfh), a_full=corr_agree(Cfl, Cfh),
                max_abs_dcorr=float(np.max(np.abs(_offdiag(Cfl) - _offdiag(Cfh))))
                if (Cfl is not None and Cfh is not None) else np.nan)


# =============================================================================== P4, the collapse
def acf_blocks(blocks, w=None, maxlag=8):
    """Autocorrelation of the state (projected on w, or the mean over coordinates if w is None),
    pooled over blocks with each block centred on its own mean."""
    num = np.zeros(maxlag + 1)
    den = 0.0
    for b in blocks:
        b = np.asarray(b, float)
        if len(b) < maxlag + 2:
            continue
        x = b @ w if w is not None else b
        x = x - (x.mean(axis=0) if x.ndim > 1 else x.mean())
        if x.ndim == 1:
            x = x[:, None]
        den += float(np.sum(x[: len(x) - maxlag] ** 2))
        for k in range(maxlag + 1):
            num[k] += float(np.sum(x[: len(x) - maxlag] * x[k: len(x) - maxlag + k]))
    if den <= 0:
        return None
    return num / den


def collapse_residual(r_lo, r_hi, c, lags=None):
    """RMS residual of r_hi(k) against r_lo evaluated at the rescaled lag c*k.

    Under time rescaling the high-load ACF is the low-load ACF with time compressed by c, so r_hi(k) =
    r_lo(c k).  c = 1 gives the residual BEFORE the collapse.  Lags beyond the low-load curve's
    support are dropped and the number used is returned.
    """
    K = len(r_lo) - 1
    g = np.arange(K + 1, dtype=float)
    ks = np.arange(1, len(r_hi)) if lags is None else np.asarray(lags)
    tgt = c * ks
    m = tgt <= K
    if m.sum() < 2:
        return np.nan, 0
    pred = np.interp(tgt[m], g, r_lo)
    res = r_hi[ks[m]] - pred
    return float(np.sqrt(np.mean(res ** 2))), int(m.sum())


def p4_collapse(blocks_lo, blocks_hi, c, w=None, maxlag=8):
    """Residual after the parameter-free collapse, before it, and the within-condition ceiling.

    SAMPLE SIZE IS MATCHED between the test and its ceiling.  The ceiling is the residual between
    two HALVES of one condition; if the between-condition residual were computed from the full
    data it would use twice as many windows and would beat that ceiling for no reason but n.  So
    the between-condition residual reported against the ceiling (`res_raw_matched`,
    `res_collapsed_matched`) is averaged over the two CROSS half pairs - half 1 of one condition
    against half 2 of the other - exactly as the P1 statistic is.  The full-sample versions are
    kept beside them, and the reported ACF curves are the full-sample ones.
    """
    r_lo = acf_blocks(blocks_lo, w, maxlag)
    r_hi = acf_blocks(blocks_hi, w, maxlag)
    if r_lo is None or r_hi is None:
        return None
    res_c, n_c = collapse_residual(r_lo, r_hi, c)
    res_1, n_1 = collapse_residual(r_lo, r_hi, 1.0)
    hl, hh = halves(blocks_lo), halves(blocks_hi)
    ceil, rm1, rmc = np.nan, np.nan, np.nan
    if hl is not None and hh is not None:
        A = [acf_blocks(h, w, maxlag) for h in hl]
        B = [acf_blocks(h, w, maxlag) for h in hh]
        if all(x is not None for x in A + B):
            cs = [collapse_residual(A[0], A[1], 1.0)[0], collapse_residual(B[0], B[1], 1.0)[0]]
            ceil = float(np.mean(cs))
            x1 = [collapse_residual(A[i], B[j], 1.0)[0] for i, j in ((0, 1), (1, 0))]
            xc = [collapse_residual(A[i], B[j], c)[0] for i, j in ((0, 1), (1, 0))]
            rm1, rmc = float(np.mean(x1)), float(np.mean(xc))
    return dict(c=float(c), res_collapsed=res_c, res_raw=res_1, n_lags=n_c,
                res_ceiling=float(ceil), res_raw_matched=rm1, res_collapsed_matched=rmc,
                gain=float(res_1 - res_c) if np.isfinite(res_c) and np.isfinite(res_1) else np.nan,
                r_lo=np.asarray(r_lo).tolist(), r_hi=np.asarray(r_hi).tolist())


def fit_c_from_acf(r_lo, r_hi, grid=None):
    """The c that minimises the collapse residual - reported only as a cross-check on P2's c."""
    grid = np.arange(0.5, 2.51, 0.01) if grid is None else grid
    best, bc = np.inf, np.nan
    for c in grid:
        v, n = collapse_residual(r_lo, r_hi, c)
        if np.isfinite(v) and n >= 2 and v < best:
            best, bc = v, c
    return float(bc), float(best)


# =============================================================================== injections
def decimate(blocks, s, phase=0, nmin=12):
    """Stride-s subsample of every block.  Changes no data value."""
    out = []
    for b in blocks:
        bb = np.asarray(b, float)[phase::s]
        if len(bb) >= nmin:
            out.append(bb)
    return out


def match_n(A, B, nmin=12):
    """Truncate two block lists to the same block count and the same per-block length."""
    nb = min(len(A), len(B))
    if nb == 0:
        return None, None
    A = sorted(A, key=lambda x: -len(x))[:nb]
    B = sorted(B, key=lambda x: -len(x))[:nb]
    oa, ob = [], []
    for a, b in zip(A, B):
        n = min(len(a), len(b))
        if n >= nmin:
            oa.append(np.asarray(a, float)[:n]); ob.append(np.asarray(b, float)[:n])
    if not oa:
        return None, None
    return oa, ob


def _sqrtm_psd(S):
    w, V = np.linalg.eigh(0.5 * (S + S.T))
    return V @ np.diag(np.sqrt(np.maximum(w, 1e-18))) @ V.T


def _isqrtm_psd(S):
    w, V = np.linalg.eigh(0.5 * (S + S.T))
    return V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-18))) @ V.T


def eig_power(M, gam):
    """M with every eigenvalue modulus raised to its own exponent, arguments untouched.

    gam is a scalar (pure time rescaling, M -> M^c in modulus) or a per-eigenvalue vector.
    Real by construction because conjugate pairs get conjugate treatment when gam is constant on
    each pair; the caller supplies gam sorted with the eigenvalues, and the imaginary residue is
    dropped after checking that it is negligible.
    """
    ev, V = np.linalg.eig(M)
    g = np.full(len(ev), float(gam)) if np.ndim(gam) == 0 else np.asarray(gam, float)
    a = np.abs(ev)
    new = (np.maximum(a, 1e-12) ** g) * np.exp(1j * np.angle(ev))
    Mn = V @ np.diag(new) @ np.linalg.inv(V)
    return np.real(Mn), float(np.max(np.abs(np.imag(Mn))))


def inject_rescale(blocks, c, dt=1.0, gam_vec=None, rng=None, n_boot=0):
    """INJ-T / INJ-R: rebuild the REAL series under an exactly known change of the propagator,
    holding the stationary distribution fixed.

    Fit (M, Q, P) on the real blocks, form M' = eig_power(M, c) (or with a per-mode exponent
    vector), set Q' = P - M' P M'^T so the Lyapunov solution is unchanged, and rebuild
        x'_t = M' x'_{t-1} + e'_t ,   e'_t = (Q'^{1/2} Q^{-1/2}) e_t
    from the REAL one-step residuals e_t of the real fit, with x'_0 = x_0.  No value is simulated:
    the innovation stream is the recording's own, only recoloured.

    Returns (new_blocks, truth) with truth carrying the exact dlog lambda, dlog D and the exact
    statement that the stationary correlation matrix is unchanged.
    """
    rng = np.random.default_rng(rng)
    f = R.fit_cell([np.asarray(b, float) for b in blocks], dt, n_boot=n_boot, rng=rng)
    if f is None or not f["stable"]:
        return None, None
    M, P = f["M"], f["P_emp"]
    Mn, imres = eig_power(M, c if gam_vec is None else gam_vec)
    if np.max(np.abs(np.linalg.eigvals(Mn))) >= 0.999:
        return None, None
    Qn = P - Mn @ P @ Mn.T
    Qn = 0.5 * (Qn + Qn.T)
    if np.min(np.linalg.eigvalsh(Qn)) <= 0:
        return None, None
    Q = f["Q"]
    T = _sqrtm_psd(Qn) @ _isqrtm_psd(Q)
    out = []
    for b in blocks:
        b = np.asarray(b, float)
        if len(b) < 4:
            continue
        bc = b - b.mean(axis=0, keepdims=True)
        e = bc[1:] - bc[:-1] @ M.T
        e = e @ T.T
        x = np.empty_like(bc)
        x[0] = bc[0]
        for t in range(1, len(bc)):
            x[t] = Mn @ x[t - 1] + e[t - 1]
        out.append(x)
    lam_lo = log_rates(M, dt)["ell"]
    lam_hi = log_rates(Mn, dt)["ell"]
    truth = dict(dlog_lam=float(np.mean(lam_hi - lam_lo)),
                 dlog_D=float(np.log(np.trace(Qn) / np.trace(Q))),
                 dbar=float(np.mean(lam_hi - lam_lo)),
                 reshape_rms=float(np.sqrt(np.mean(
                     ((lam_hi - lam_lo) - np.mean(lam_hi - lam_lo)) ** 2))),
                 d_corr=0.0, imag_residue=imres)
    return out, truth


def inject_drive(blocks, g, u=None, rng=None):
    """INJ-D: multiply the within-block residuals by g along the unit direction u.

    This is a similarity transform T = I + (g-1) u u' of the state, so the propagator's spectrum is
    EXACTLY unchanged (truth dlog lambda = 0) while tr(Q) and the correlation matrix move by an
    exactly computable amount.  It is the pure-drive injection: D -> D with an extra g^2-fold
    variance along u and A untouched in the transformed coordinates.
    """
    rng = np.random.default_rng(rng)
    d = np.asarray(blocks[0]).shape[1]
    if u is None:
        u = rng.standard_normal(d)
    u = np.asarray(u, float); u = u / np.linalg.norm(u)
    T = np.eye(d) + (g - 1.0) * np.outer(u, u)
    out = [np.asarray(b, float) @ T.T for b in blocks]
    return out, T, u


# =============================================================================== inference helpers
def boot_ci(x, n=10000, seed=SEED):
    return R.boot_ci(x, n=n, seed=seed)


def fmt(c, unit=""):
    return R.fmt(c, unit)


def paired_ci(x, y, n=10000, seed=SEED):
    """CI on the paired difference x - y with the sign-flip test."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    return R.boot_ci(x[m] - y[m], n=n, seed=seed)


def plane_angle(dl, dd, n=10000, seed=SEED):
    """Where a cohort's contrast sits in the (dlog lambda, dlog D) plane.

    Reports the group-mean vector's angle - 0 deg is the pure-gain axis (rate only), 90 deg the
    pure-drive axis (diffusion only), 45 deg the unit-slope diagonal that time rescaling predicts - with a
    bootstrap CI over participants, and the OLS slope of dlog D on dlog lambda tested against 1.
    """
    dl = np.asarray(dl, float); dd = np.asarray(dd, float)
    m = np.isfinite(dl) & np.isfinite(dd)
    dl, dd = dl[m], dd[m]
    if len(dl) < 4:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(dl), size=(n, len(dl)))
    ang = np.degrees(np.arctan2(dd[idx].mean(1), dl[idx].mean(1)))
    sl = []
    for b in range(n):
        x, y = dl[idx[b]], dd[idx[b]]
        v = np.var(x)
        sl.append(np.cov(x, y, ddof=1)[0, 1] / v if v > 0 else np.nan)
    sl = np.array(sl, float)
    sl = sl[np.isfinite(sl)]
    obs_ang = float(np.degrees(np.arctan2(dd.mean(), dl.mean())))
    v = np.var(dl, ddof=1)
    obs_sl = float(np.cov(dl, dd, ddof=1)[0, 1] / v) if v > 0 else np.nan
    # paired diagonal test: time rescaling predicts dlog D - dlog lambda = 0 participant by participant
    diag = R.boot_ci(dd - dl, seed=seed)
    return dict(n=int(len(dl)), angle_deg=obs_ang,
                angle_lo=float(np.percentile(ang, 2.5)), angle_hi=float(np.percentile(ang, 97.5)),
                slope=obs_sl,
                slope_lo=float(np.percentile(sl, 2.5)) if len(sl) else np.nan,
                slope_hi=float(np.percentile(sl, 97.5)) if len(sl) else np.nan,
                p_slope_vs_1=float(2 * min(np.mean(sl >= 1.0), np.mean(sl <= 1.0)))
                if len(sl) else np.nan,
                p_slope_vs_0=float(2 * min(np.mean(sl >= 0.0), np.mean(sl <= 0.0)))
                if len(sl) else np.nan,
                r=float(np.corrcoef(dl, dd)[0, 1]) if len(dl) > 2 else np.nan,
                diagonal_gap=diag, dlog_lam=R.boot_ci(dl, seed=seed), dlog_D=R.boot_ci(dd, seed=seed))


def csv_write(path, rows):
    keys = sorted({k for r in rows for k in r})
    lead = [k for k in ("subject", "cohort", "arm", "contrast") if k in keys]
    keys = lead + [k for k in keys if k not in lead]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            vals = []
            for k in keys:
                v = r.get(k)
                if v is None or (isinstance(v, float) and not np.isfinite(v)):
                    vals.append("")
                elif isinstance(v, float):
                    vals.append(f"{v:.6g}")
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")


def reshape_gamma(M, size):
    """Per-mode exponents gamma_k with mean(ln gamma) = 0 and rms(ln gamma) = `size`.

    A pure RESHAPING of the relaxation spectrum: the common factor is exactly zero by construction
    and only the spread and ordering of ln lambda move.  Tied eigenvalue moduli (conjugate pairs)
    get identical exponents, so the reshaped propagator stays real.
    """
    a = np.abs(np.linalg.eigvals(M))
    key = np.round(np.log(np.maximum(a, 1e-12)), 10)
    uk = np.unique(key)
    if len(uk) < 2:
        return np.ones(len(a)), 0.0
    pos = {v: i for i, v in enumerate(np.sort(uk))}
    r = np.array([pos[v] for v in key], float)
    r = r - r.mean()
    s = np.sqrt(np.mean(r ** 2))
    if s <= 0:
        return np.ones(len(a)), 0.0
    rho = size * r / s
    rho = rho - rho.mean()
    return np.exp(rho), float(np.sqrt(np.mean(rho ** 2)))


def exponentiality(blocks, w=None, maxlag=4):
    """How far the state's autocorrelation is from the exponential an OU/VAR(1) process requires.

    For a VAR(1), r(2) = r(1)^2 exactly in any fixed direction, so ln r(2) / (2 ln r(1)) = 1.  The
    index is > 1 when the ACF falls FASTER than exponential and < 1 when it has a heavy tail.  This
    is the quantity that decides whether temporal decimation can multiply the fitted rate: it can
    only do so to the extent the process really is first-order Markov at that sampling interval.
    """
    r = acf_blocks(blocks, w, maxlag)
    if r is None or len(r) < 3:
        return None
    r1, r2 = float(r[1]), float(r[2])
    out = dict(r1=r1, r2=r2, r2_pred=r1 ** 2, r2_excess=r2 - r1 ** 2)
    out["exp_index"] = (float(np.log(max(r2, 1e-9)) / (2 * np.log(max(r1, 1e-9))))
                        if 0 < r1 < 1 and r2 > 0 else np.nan)
    if len(r) > 3 and r[3] > 0 and 0 < r1 < 1:
        out["exp_index_3"] = float(np.log(r[3]) / (3 * np.log(r1)))
    return out


def simulate_var1(M, Q, lens, rng, burn=200):
    """A genuine VAR(1) surrogate: Gaussian white innovations of covariance Q driven through M,
    one block per entry of `lens`.

    This is NOT a reconstruction of the recording.  inject_rescale(..., c = 1) returns the
    recording bit for bit (T = I, M' = M), so it cannot serve as a first-order-Markov control;
    this does, because its innovations are white by construction while its (M, Q, block lengths,
    dimension and n) are the participant's own.
    """
    rng = np.random.default_rng(rng)
    L = _sqrtm_psd(Q)
    d = M.shape[0]
    out = []
    for n in lens:
        e = rng.standard_normal((n + burn, d)) @ L.T
        x = np.empty((n + burn, d))
        x[0] = e[0]
        for t in range(1, n + burn):
            x[t] = M @ x[t - 1] + e[t]
        out.append(x[burn:])
    return out


# =============================================================================== the per-unit test
def analyse_pair(B0, B1, dt, boot, rng, w=None, maxlag=6, D0=None, D1=None, split_fit=True):
    """P1 - P4 for ONE unit (participant, or participant x cell) from its two matched block lists.

    B0 / B1 are the low- and high-condition blocks used for everything dynamical.  D0 / D1, if
    given, are the corresponding DISTRIBUTIONAL block lists (the 50 %-overlap `dist`
    series) and are used for P1 only; overlap inflates both conditions identically and cancels in
    the between-minus-within comparison, while the extra windows make the correlation matrix far
    better determined.  Every cohort uses this one function.
    """
    f0 = R.fit_cell(B0, dt, n_boot=boot, rng=rng)
    f1 = R.fit_cell(B1, dt, n_boot=boot, rng=rng)
    if f0 is None or f1 is None or not (f0["stable"] and f1["stable"]):
        return None
    out = dict(n_blocks=f0["n_blocks"], n_pairs=f0["n_pairs"], d=f0["d"],
               absev_lo=float(np.mean(np.abs(np.linalg.eigvals(f0["M"])))),
               absev_hi=float(np.mean(np.abs(np.linalg.eigvals(f1["M"])))),
               lam_lo=f0["lam_mean"], lam_hi=f1["lam_mean"],
               lam_slow_lo=f0["lam_max"], lam_slow_hi=f1["lam_max"],
               D_lo=f0["D_tr"], D_hi=f1["D_tr"], sp_lo=f0["sp_emp_gm"], sp_hi=f1["sp_emp_gm"],
               dlog_lam=float(np.log(f1["lam_mean"] / f0["lam_mean"])),
               dlog_D=float(np.log(f1["D_tr"] / f0["D_tr"])),
               dlog_D_gm=float(np.log(f1["D_gm"] / f0["D_gm"])),
               dlog_sp=float(np.log(f1["sp_emp_gm"] / f0["sp_emp_gm"])),
               dlog_lam_slowest=float(np.log(f1["lam_max"] / f0["lam_max"])))
    # ---- P2
    p2 = p2_decompose(f0["M"], f1["M"], dt)
    if p2 is not None:
        for k in ("dbar", "c_common", "reshape_rms", "frac_common", "frac_reshape", "dlog_sd_ell",
                  "sd_ell_lo", "sd_ell_hi", "match_cos", "match_dbar", "match_reshape_rms",
                  "match_frac_common", "rank_spearman"):
            out["p2_" + k] = p2.get(k, np.nan)
    if split_fit:
        hs = []
        for B in (B0, B1):
            hh = halves(B)
            if hh is None:
                continue
            fa = R.fit_cell(hh[0], dt, n_boot=boot, rng=rng)
            fb = R.fit_cell(hh[1], dt, n_boot=boot, rng=rng)
            if fa is None or fb is None or not (fa["stable"] and fb["stable"]):
                continue
            q = p2_decompose(fa["M"], fb["M"], dt)
            if q is not None:
                hs.append(q)
        if hs:
            out["p2_ceiling_reshape_rms"] = float(np.mean([q["reshape_rms"] for q in hs]))
            out["p2_ceiling_dbar_abs"] = float(np.mean([abs(q["dbar"]) for q in hs]))
            out["p2_ceiling_rank_spearman"] = float(np.mean([q.get("rank_spearman", np.nan)
                                                             for q in hs]))
            if out["p2_ceiling_reshape_rms"] > 0 and np.isfinite(out.get("p2_reshape_rms", np.nan)):
                out["p2_reshape_over_ceiling"] = (out["p2_reshape_rms"]
                                                  / out["p2_ceiling_reshape_rms"])
    # ---- P1
    A0, A1 = (D0 if D0 else B0), (D1 if D1 else B1)
    p1 = p1_invariance(A0, A1, True, rng)
    if p1 is not None:
        for k, v in p1.items():
            out["p1_" + k] = v
    p1u = p1_invariance(A0, A1, False, rng)
    if p1u is not None:
        out["p1u_d_ratio"] = p1u["d_ratio"]
    # ---- Markov check and the one-dimensional rate along w
    e0, e1 = exponentiality(B0, w, 4), exponentiality(B1, w, 4)
    if e0 and e1:
        for tag, e in (("lo", e0), ("hi", e1)):
            out[f"exp_index_{tag}"] = e["exp_index"]
            out[f"r1_{tag}"] = e["r1"]; out[f"r2_{tag}"] = e["r2"]
            out[f"r2_excess_{tag}"] = e["r2_excess"]
        if 0 < e0["r1"] < 1 and 0 < e1["r1"] < 1:
            out["lam1d_lo"] = -np.log(e0["r1"]) / dt
            out["lam1d_hi"] = -np.log(e1["r1"]) / dt
            out["dlog_lam_1d"] = float(np.log(np.log(e1["r1"]) / np.log(e0["r1"])))
    # ---- P4
    c = float(out.get("p2_c_common", np.nan))
    if np.isfinite(c) and c > 0:
        p4 = p4_collapse(B0, B1, c, w=w, maxlag=maxlag)
        if p4 is not None:
            out["p4_res_collapsed"] = p4["res_collapsed"]
            out["p4_res_raw"] = p4["res_raw"]
            out["p4_res_ceiling"] = p4["res_ceiling"]
            out["p4_res_raw_matched"] = p4["res_raw_matched"]
            out["p4_res_collapsed_matched"] = p4["res_collapsed_matched"]
            out["p4_n_lags"] = p4["n_lags"]
            out["p4_gain"] = p4["gain"]
            r_lo = np.asarray(p4["r_lo"]); r_hi = np.asarray(p4["r_hi"])
            out["acf1_lo"] = float(r_lo[1]); out["acf1_hi"] = float(r_hi[1])
            cf, rf = fit_c_from_acf(r_lo, r_hi)
            out["p4_c_from_acf"] = cf
            out["p4_res_at_best_c"] = rf
            if 0 < r_lo[1] < 1 and 0 < r_hi[1] < 1:
                c1 = float(np.log(r_hi[1]) / np.log(r_lo[1]))
                out["p4_c_1d"] = c1
                q = p4_collapse(B0, B1, c1, w=w, maxlag=maxlag)
                if q is not None:
                    out["p4_res_collapsed_c1d"] = q["res_collapsed"]
                    out["p4_res_collapsed_c1d_matched"] = q["res_collapsed_matched"]
    return out


def exact_rescale_truth(M, P, c, dt=1.0):
    """The exactly computable truth for a pure time rescaling by c of a system whose one-step map
    and stationary covariance are (M, P): dlog lambda = ln c, and dlog D from the Lyapunov
    relation, with the stationary distribution unchanged by construction."""
    Mn, _ = eig_power(M, c)
    Q0 = P - M @ P @ M.T
    Qn = P - Mn @ P @ Mn.T
    t0, tn = float(np.trace(Q0)), float(np.trace(Qn))
    return dict(dlog_lam=float(np.log(c)),
                dlog_D=float(np.log(tn / t0)) if t0 > 0 and tn > 0 else np.nan,
                d_corr=0.0)
