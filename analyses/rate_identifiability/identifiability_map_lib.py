# -*- coding: utf-8 -*-
r"""Generator, estimator and cohort geometries of the rate-identifiability map (identifiability_map.py).

WHAT IS SIMULATED
-----------------
A linear stochastic process with a KNOWN relaxation rate plus an additive white component of KNOWN
variance share, at each cohort's REAL segment structure:

    x_t = a x_{t-1} + sqrt(1 - a^2) e_t ,   a = exp(-lambda dt) ,   Var(x) = I_d
    y_t = sqrt(1 - nu) x_t + sqrt(nu) w_t                      (w iid, independent of x)

so Corr(y_t, y_{t+k}) = (1 - nu) a^k for k >= 1, the single-timescale-plus-white form measured on the
recordings, and the population one-step regression coefficient is (1 - nu) a.  x_0 is drawn from the
exact stationary law, so the DATA need no burn-in; only the bias-correction simulation does.

Because the propagator is a I_d, every mode shares one rate and the estimand
lambda_hat = -(1/d) sum_k ln|eig_k(M)| / dt has one unambiguous truth.  The anisotropy of the
innovation covariance is immaterial: with M = a I the OLS estimator transforms as
M_hat -> T M_hat T^{-1} under y -> T y, so its eigenvalues - and therefore lambda_hat - are
invariant to any invertible linear map of the state.  That is why a d-dimensional isotropic
generator is not a loss of generality for THIS estimand, and it is also why the additive white
component IS a loss of generality unless it shares the state's covariance shape, which is what the
ambient-PCA arm (`gen_unit_ambient`) bounds.

THE ESTIMATOR
-------------
The lag-1 least-squares propagator with the parametric-bootstrap bias correction that simulates the
recording's own block structure (fit_cell / fit_pairs of
analyses/workload_certificates/features/cog_fit_lib.py), unchanged in arithmetic.  The
bias-correction simulation is vectorised over equal-length blocks here (the loop implementation runs
one block at a time); the equivalence of the two is measured by `equivalence_check` (stage equiv of
identifiability_map.py).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys                                                                        # noqa: E402
from pathlib import Path                                                          # noqa: E402

import numpy as np                                                                # noqa: E402

REPO = Path(__file__).resolve().parents[2]

RIDGE = 1e-8
BURN = 100                     # the burn-in of the loop implementation (cog_fit_lib.fit_cell)
N_BOOT = 200                   # bias-correction simulations per fit
EPS = 1e-12

# ------------------------------------------------------------------ geometries
# Per cohort: sampling interval dt (s), state dimension d, ambient feature count p_amb, blocks per
# recording x windows per block, participants n_units, estimator, and the sample-size lever of the
# main grid and of the secondary lever arm.
#   COG-BCI   2 blocks, 291.7 lag-1 pairs on average -> 147 windows/block, d 8, 28 participants
#             after the duplicate exclusion
#   ds003838  36 blocks, 252 pairs -> 8 windows/block, d 8
#   HCP fMRI  8 blocks, 296 pairs -> 38 windows/block, d 7
#   HCP MEG   147 trials, one lag-1 pair per trial, d 8
#   Sleep-EDF 12 bouts x 60 windows, d 4
GEOMETRIES = {
    "COG-BCI": dict(modality="EEG", dt=1.0, d=8, p_amb=248, n_blocks=2, n_win=147,
                    n_units=28, estimator="cell", lever="length",
                    lever_alt="blocks", label="COG-BCI EEG"),
    "ds003838": dict(modality="EEG", dt=1.0, d=8, p_amb=256, n_blocks=36, n_win=8,
                     n_units=64, estimator="cell", lever="blocks",
                     lever_alt="length", label="ds003838 EEG"),
    "HCP_fMRI": dict(modality="fMRI", dt=0.72, d=7, p_amb=7, n_blocks=8, n_win=38,
                     n_units=99, estimator="cell", lever="blocks",
                     lever_alt="length", label="HCP WM fMRI"),
    "HCP_MEG": dict(modality="MEG", dt=1.0, d=8, p_amb=None, n_blocks=147, n_win=2,
                    n_units=83, estimator="pairs", lever="blocks",
                    lever_alt=None, label="HCP MEG"),
    "SLEEP": dict(modality="sleep EEG", dt=1.0, d=4, p_amb=8, n_blocks=12, n_win=60,
                  n_units=78, estimator="cell", lever="blocks",
                  lever_alt="length", label="Sleep-EDF"),
}

SCALES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
LAM_GRID = tuple(float(x) for x in np.geomspace(0.05, 3.0, 12))
NU_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8)


def _loop_estimators():
    """The loop implementation of the estimator (analyses/workload_certificates/features/cog_fit_lib.py)."""
    p = str(REPO / "analyses" / "workload_certificates" / "features")
    if p not in sys.path:
        sys.path.insert(0, p)
    import cog_fit_lib
    return cog_fit_lib


def scaled_geometry(geo, scale, lever=None):
    """The cohort's (n_blocks, n_win) at a sample-size factor, by the declared lever."""
    lever = lever or geo["lever"]
    nb, nw = geo["n_blocks"], geo["n_win"]
    if lever == "blocks":
        nb = int(max(2, round(nb * scale)))
    elif lever == "length":
        nw = int(max(5, round(nw * scale)))
    else:
        raise ValueError(lever)
    if geo["estimator"] == "pairs":
        return nb, 2, nb                     # one lag-1 pair per trial
    return nb, nw, nb * (nw - 1)


# ------------------------------------------------------------------ generator
def gen_unit(lam_dt, nu, n_blocks, n_win, d, rng, lam_vec=None):
    """(n_blocks, n_win, d) from the stationary law.  lam_vec gives a per-mode spectrum."""
    if lam_vec is None:
        a = np.full(d, float(np.exp(-lam_dt)))
    else:
        a = np.exp(-np.asarray(lam_vec, float))
    s = np.sqrt(np.maximum(1.0 - a ** 2, EPS))
    X = np.empty((n_blocks, n_win, d))
    X[:, 0] = rng.standard_normal((n_blocks, d))            # exact stationary start
    for t in range(1, n_win):
        X[:, t] = a * X[:, t - 1] + s * rng.standard_normal((n_blocks, d))
    if nu > 0:
        X = np.sqrt(1.0 - nu) * X + np.sqrt(nu) * rng.standard_normal(X.shape)
    return X


def gen_unit_ambient(lam_dt, nu_target, n_blocks, n_win, d, p, rng, evr_decay=0.75):
    """Ambient-PCA arm: generate p ambient features, robust-z and PCA(d) exactly as the pipeline does.

    p ambient features carry a d-dimensional OU signal through a random orthonormal loading with a
    geometrically decaying amplitude profile, plus ISOTROPIC ambient white noise scaled so that the
    white share inside the signal subspace equals `nu_target`.  Because PCA is fitted on the noisy
    data the realised nu differs from the target.
    """
    a = float(np.exp(-lam_dt))
    s = np.sqrt(max(1.0 - a ** 2, EPS))
    X = np.empty((n_blocks, n_win, d))
    X[:, 0] = rng.standard_normal((n_blocks, d))
    for t in range(1, n_win):
        X[:, t] = a * X[:, t - 1] + s * rng.standard_normal((n_blocks, d))
    amp = evr_decay ** np.arange(d)
    amp = amp / np.sqrt(np.mean(amp ** 2))
    Sload = np.linalg.qr(rng.standard_normal((p, d)))[0] * amp                # p x d
    F = X @ Sload.T                                                           # signal, var amp^2
    # white share inside the signal subspace: sigw^2 / (mean(amp^2) + sigw^2) = nu
    sigw = np.sqrt(nu_target / max(1.0 - nu_target, 1e-9) * float(np.mean(amp ** 2)))
    F = F + sigw * rng.standard_normal(F.shape)
    # per-feature scale heterogeneity, so the robust z is not a no-op
    F = F * np.exp(0.4 * rng.standard_normal(p))
    flat = F.reshape(-1, p)
    med = np.median(flat, axis=0)
    q75, q25 = np.percentile(flat, [75, 25], axis=0)
    iqr = q75 - q25
    iqr[~np.isfinite(iqr) | (iqr <= 0)] = 1.0
    Z = (flat - med) / iqr
    Zc = Z - Z.mean(axis=0)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    Y = (Zc @ Vt[:d].T).reshape(n_blocks, n_win, d)
    return Y


# ------------------------------------------------------------------ fast estimator
def _ols(A, B, ridge=RIDGE):
    G = A.T @ A
    G = G + ridge * np.trace(G) / A.shape[1] * np.eye(A.shape[1])
    return np.linalg.solve(G, A.T @ B).T


def _log_lam(M, dt):
    """ln(lambda_hat * dt) with lambda_hat = -(1/d) sum ln|eig(M)| / dt; nan if not stable."""
    ev = np.abs(np.linalg.eigvals(M))
    if np.any(ev >= 1.0) or np.any(ev <= 0):
        return np.nan
    lam = float(np.mean(-np.log(ev)))          # this IS lambda*dt
    return float(np.log(lam)) if lam > 0 else np.nan


def _log_lam_batch(Ms, dt):
    ev = np.abs(np.linalg.eigvals(Ms))                       # (B, d)
    ok = np.all((ev < 1.0) & (ev > 0), axis=1)
    lam = np.where(ok, np.mean(-np.log(np.clip(ev, 1e-300, 1 - 1e-15)), axis=1), np.nan)
    return np.where(np.isfinite(lam) & (lam > 0), np.log(np.maximum(lam, 1e-300)), np.nan)


def adaptive_burn(M, burn_max=BURN, tol=1e-12, burn_min=20):
    """Burn-in of the bias-correction simulation, never larger than `BURN`.

    The smallest burn for which the transient is below `tol`, capped at 100 and floored at 20, so
    the simulated process differs from the one with a burn of 100 by less than 1e-12 in correlation
    and the value is IDENTICAL to 100 whenever 100 is actually needed.
    """
    rho = float(np.max(np.abs(np.linalg.eigvals(M))))
    if not (0 < rho < 1):
        return burn_max
    need = int(np.ceil(np.log(tol) / np.log(max(rho, 1e-12))))
    return int(min(burn_max, max(burn_min, need)))


def _sim_refit_cell(M, L, n_blocks, n_win, n_boot, rng, burn=BURN, ridge=RIDGE,
                    max_elem=2_000_000):
    """Vectorised equivalent of the loop implementation's block simulation and refit for EQUAL-length
    blocks.  Returns (n_boot, d, d) refitted propagators.

    The burn-in is run as a ROLLING state and never materialised, and the kept window is centred
    in place, so the peak allocation is one (B, n_blocks, n_win, d) array.  The arithmetic is
    unchanged: x_0 = e_0, then (burn - 1) + n_win one-step updates, of which the last n_win are kept.
    """
    d = M.shape[0]
    per = max(1, int(max_elem // max(1, n_blocks * n_win * d)))
    out = np.empty((n_boot, d, d))
    eye = np.eye(d)
    done = 0
    while done < n_boot:
        B = min(per, n_boot - done)
        cur = rng.standard_normal((B, n_blocks, d)) @ L.T          # x_0 = e_0
        for _ in range(1, burn):
            cur = cur @ M.T + rng.standard_normal((B, n_blocks, d)) @ L.T
        X = np.empty((B, n_blocks, n_win, d))
        for t in range(n_win):
            cur = cur @ M.T + rng.standard_normal((B, n_blocks, d)) @ L.T
            X[:, :, t] = cur
        X -= X.mean(axis=2, keepdims=True)
        A, Bm = X[:, :, :-1], X[:, :, 1:]
        G = np.einsum("bkti,bktj->bij", A, A)
        H = np.einsum("bkti,bktj->bij", A, Bm)
        tr = np.einsum("bii->b", G)[:, None, None] / d
        Gr = G + ridge * tr * eye
        out[done:done + B] = np.transpose(np.linalg.solve(Gr, H), (0, 2, 1))
        del X, A, Bm, cur
        done += B
    return out


def fit_unit_cell(X, dt, n_boot=N_BOOT, rng=None, burn=None, ridge=RIDGE,
                  max_elem=2_000_000):
    """The estimator on one simulated recording, returning the pieces the map needs.

    X: (n_blocks, n_win, d).  Arithmetic identical to cog_fit_lib.fit_cell; the bias-correction
    simulation is vectorised over blocks (they are all the same length here).
    """
    rng = np.random.default_rng(rng)
    n_blocks, n_win, d = X.shape
    Xc = X - X.mean(axis=1, keepdims=True)
    A = Xc[:, :-1].reshape(-1, d)
    Bm = Xc[:, 1:].reshape(-1, d)
    n_pairs = len(A)
    if n_pairs < 4 * d:
        return None
    M = _ols(A, Bm, ridge)
    l_raw = _log_lam(M, dt)
    Q0 = np.cov((Bm - A @ M.T).T, bias=False)
    Q0 = 0.5 * (Q0 + Q0.T)
    w, V = np.linalg.eigh(Q0)
    L = V @ np.diag(np.sqrt(np.maximum(w, 1e-18))) @ V.T
    if n_boot > 0 and np.max(np.abs(np.linalg.eigvals(M))) < 0.999:
        b = adaptive_burn(M) if burn is None else int(burn)
        Ms = _sim_refit_cell(M, L, n_blocks, n_win, n_boot, rng, burn=b, ridge=ridge,
                             max_elem=max_elem)
        M_bc = 2 * M - Ms.mean(axis=0)
        l_star = _log_lam_batch(Ms, dt)
    else:
        M_bc = M.copy()
        l_star = np.full(max(n_boot, 1), np.nan)
    return dict(l_raw=l_raw, l_bc=_log_lam(M_bc, dt), l_star=l_star, n_pairs=n_pairs,
                absev=float(np.mean(np.abs(np.linalg.eigvals(M)))))


def fit_unit_pairs(Xp, Yp, dt, n_boot=N_BOOT, rng=None, ridge=RIDGE):
    """The HCP MEG design: one lag-1 pair per trial.  Arithmetic of cog_fit_lib.fit_pairs."""
    rng = np.random.default_rng(rng)
    n, d = Xp.shape
    if n < 4 * d:
        return None
    Xc, Yc = Xp - Xp.mean(0), Yp - Yp.mean(0)
    M = _ols(Xc, Yc, ridge)
    l_raw = _log_lam(M, dt)
    Rr = Yc - Xc @ M.T
    Q0 = np.cov(Rr.T, bias=False) * (n - 1) / max(n - d - 1, 1)
    Q0 = 0.5 * (Q0 + Q0.T)
    w, V = np.linalg.eigh(Q0)
    Lc = V @ np.diag(np.sqrt(np.maximum(w, 1e-18))) @ V.T
    E = rng.standard_normal((n_boot, n, d)) @ Lc.T
    Ys = np.einsum("ij,tj->ti", M, Xc)[None] + E
    Ys = Ys - Ys.mean(axis=1, keepdims=True)
    G = Xc.T @ Xc
    G = G + ridge * np.trace(G) / d * np.eye(d)
    Gi = np.linalg.inv(G)
    Ms = np.einsum("bti,tj,jk->bik", Ys, Xc, Gi)
    M_bc = 2 * M - Ms.mean(axis=0)
    return dict(l_raw=l_raw, l_bc=_log_lam(M_bc, dt), l_star=_log_lam_batch(Ms, dt),
                n_pairs=n, absev=float(np.mean(np.abs(np.linalg.eigvals(M)))))


def gen_pairs(lam_dt, nu, n_pairs, d, rng):
    a = float(np.exp(-lam_dt))
    X = rng.standard_normal((n_pairs, d))
    Y = a * X + np.sqrt(max(1 - a ** 2, EPS)) * rng.standard_normal((n_pairs, d))
    if nu > 0:
        X = np.sqrt(1 - nu) * X + np.sqrt(nu) * rng.standard_normal(X.shape)
        Y = np.sqrt(1 - nu) * Y + np.sqrt(nu) * rng.standard_normal(Y.shape)
    return X, Y


def unit_draw(lam_dt, nu, geo, nb, nw, rng, lam_vec=None, max_elem=2_000_000):
    """One simulated recording through the chain.  Returns the fit dict or None."""
    d, dt = geo["d"], geo["dt"]
    if geo["estimator"] == "pairs":
        X, Y = gen_pairs(lam_dt, nu, nb, d, rng)
        return fit_unit_pairs(X, Y, dt, rng=rng)
    X = gen_unit(lam_dt, nu, nb, nw, d, rng, lam_vec=lam_vec)
    return fit_unit_cell(X, dt, rng=rng, max_elem=max_elem)


# ------------------------------------------------------------------ interval arithmetic
def unit_interval(fit):
    """Basic (reverse-percentile) 95% bootstrap interval on ln(lambda*dt)."""
    ls = np.asarray(fit["l_star"], float)
    ls = ls[np.isfinite(ls)]
    if not np.isfinite(fit["l_raw"]) or len(ls) < 20:
        return np.nan, np.nan
    q025, q975 = np.percentile(ls, [2.5, 97.5])
    return 2.0 * fit["l_raw"] - q975, 2.0 * fit["l_raw"] - q025


def group_boot_ci(vals, n_units, n_outer=1200, n_inner=600, rng=None):
    """Percentile-bootstrap 95% interval on the MEAN over n_units units (as cog_fit_lib.boot_ci).
    Returns arrays (lo, hi, mean) over n_outer cohort draws."""
    rng = np.random.default_rng(rng)
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if len(v) < 10:
        return np.full(n_outer, np.nan), np.full(n_outer, np.nan), np.full(n_outer, np.nan)
    draws = v[rng.integers(0, len(v), size=(n_outer, n_units))]          # (n_outer, n_units)
    means = draws.mean(axis=1)
    idx = rng.integers(0, n_units, size=(n_inner, n_units))
    # The inner resample is done in chunks of the outer draws, which changes no value - `draws` and
    # `idx` are drawn once, before the loop - only the peak allocation.
    lo = np.empty(n_outer)
    hi = np.empty(n_outer)
    chunk = max(1, int(2_000_000 // max(1, n_inner * n_units)))
    for s0 in range(0, n_outer, chunk):
        bm = draws[s0:s0 + chunk][:, idx].mean(axis=2)                    # (chunk, n_inner)
        lo[s0:s0 + chunk] = np.percentile(bm, 2.5, axis=1)
        hi[s0:s0 + chunk] = np.percentile(bm, 97.5, axis=1)
        del bm
    return lo, hi, means


def equivalence_check(geo, lam_dt, n_rep, seed=0, n_boot=N_BOOT):
    """The vectorised bias correction against the loop implementation (cog_fit_lib.fit_cell).

    Both are run on the SAME simulated recordings; only the bias-correction simulation differs.
    """
    R = _loop_estimators()
    rng_a = np.random.default_rng(seed)
    rng_b = np.random.default_rng(seed + 1)
    rng_d = np.random.default_rng(seed + 2)
    nb, nw = geo["n_blocks"], geo["n_win"]
    fast, loop = [], []
    for _ in range(n_rep):
        X = gen_unit(lam_dt, 0.3, nb, nw, geo["d"], rng_d)
        f = fit_unit_cell(X, geo["dt"], n_boot=n_boot, rng=rng_a)
        p = R.fit_cell([X[i] for i in range(nb)], dt=geo["dt"], n_boot=n_boot, rng=rng_b)
        fast.append(f["l_bc"] if f else np.nan)
        loop.append(np.log(p["lam_mean"] * geo["dt"]) if (p and np.isfinite(p["lam_mean"])
                                                          and p["lam_mean"] > 0) else np.nan)
    fa = np.asarray(fast, float); pu = np.asarray(loop, float)
    m = np.isfinite(fa) & np.isfinite(pu)
    fa, pu = fa[m], pu[m]
    if len(fa) < 20:
        return dict(n=int(len(fa)), ok=False, reason="too few finite fits")
    d_mean = float(np.mean(fa) - np.mean(pu))
    se = float(np.sqrt(np.var(fa, ddof=1) / len(fa) + np.var(pu, ddof=1) / len(pu)))
    return dict(cohort=geo["label"], lam_dt=float(lam_dt), n=int(len(fa)),
                mean_fast=float(np.mean(fa)), mean_loop=float(np.mean(pu)),
                sd_fast=float(np.std(fa, ddof=1)), sd_loop=float(np.std(pu, ddof=1)),
                p025_fast=float(np.percentile(fa, 2.5)), p025_loop=float(np.percentile(pu, 2.5)),
                p975_fast=float(np.percentile(fa, 97.5)), p975_loop=float(np.percentile(pu, 97.5)),
                diff_mean=d_mean, se_diff=se, z=float(d_mean / se) if se > 0 else np.nan,
                sd_ratio=float(np.std(fa, ddof=1) / max(np.std(pu, ddof=1), 1e-12)),
                ok=bool(abs(d_mean) < 3.0 * se or abs(d_mean) < 0.01))


def pairs_equivalence_check(geo, lam_dt, n_rep, seed=0, n_boot=N_BOOT):
    """The same check for the HCP MEG one-pair-per-trial estimator (cog_fit_lib.fit_pairs)."""
    R = _loop_estimators()
    rng_a = np.random.default_rng(seed)
    rng_b = np.random.default_rng(seed + 1)
    rng_d = np.random.default_rng(seed + 2)
    fast, loop = [], []
    for _ in range(n_rep):
        X, Y = gen_pairs(lam_dt, 0.3, geo["n_blocks"], geo["d"], rng_d)
        f = fit_unit_pairs(X, Y, geo["dt"], n_boot=n_boot, rng=rng_a)
        p = R.fit_pairs(X, Y, dt=geo["dt"], n_boot=n_boot, rng=rng_b)
        fast.append(f["l_bc"] if f else np.nan)
        loop.append(np.log(p["lam_mean"] * geo["dt"]) if (p and np.isfinite(p["lam_mean"])
                                                          and p["lam_mean"] > 0) else np.nan)
    fa = np.asarray(fast, float); pu = np.asarray(loop, float)
    m = np.isfinite(fa) & np.isfinite(pu)
    fa, pu = fa[m], pu[m]
    if len(fa) < 20:
        return dict(n=int(len(fa)), ok=False, reason="too few finite fits")
    d_mean = float(np.mean(fa) - np.mean(pu))
    se = float(np.sqrt(np.var(fa, ddof=1) / len(fa) + np.var(pu, ddof=1) / len(pu)))
    return dict(cohort=geo["label"], lam_dt=float(lam_dt), n=int(len(fa)),
                mean_fast=float(np.mean(fa)), mean_loop=float(np.mean(pu)),
                sd_fast=float(np.std(fa, ddof=1)), sd_loop=float(np.std(pu, ddof=1)),
                diff_mean=d_mean, se_diff=se, z=float(d_mean / se) if se > 0 else np.nan,
                sd_ratio=float(np.std(fa, ddof=1) / max(np.std(pu, ddof=1), 1e-12)),
                ok=bool(abs(d_mean) < 3.0 * se or abs(d_mean) < 0.01))
