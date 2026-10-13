# -*- coding: utf-8 -*-
"""Shared estimators for the (dlog lambda, dlog D) discriminator.

Every estimator here works on a CELL, which is a
list of blocks; a block is an (n_t, d) array of NON-OVERLAPPING within-block state windows.  No
lagged pair is ever formed across a block boundary and every block is centred on its own mean, so
the displacement of the cell never enters the propagator.

Estimands per cell
------------------
  M      one-step propagator, OLS, then a parametric-bootstrap bias correction that simulates the
         cell's OWN block structure rather than one continuous series.
  Q      innovation covariance, the EMPIRICAL residual covariance under the bias-corrected M.
         It is deliberately NOT S0 - M S0 M', which would make the Lyapunov check below an
         algebraic identity.
  D      Q / (2 dt).  Scalar readout tr(Q) / (2 d dt).
  lambda -(1/d) sum_k ln|mu_k(M)| / dt, the discrete functional.  A = logm(M)/dt is computed and
         its validity reported, but it is NOT used as the rate: |exp(mu dt)| = exp(Re(mu) dt)
         carries no 2 pi ambiguity while Im(mu) is aliased at these sampling intervals (the
         obstruction of Yue, Thunberg & Goncalves 2016).  The identifiability check travels with
         the estimator.
  P      stationary spread.  P_emp is the empirical within-block covariance; P_model solves the
         EXACT discrete Lyapunov relation P = M P M' + Q.  Their ratio is the consistency check,
         in place of the one-dimensional sqrt(D/lambda) identity.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg as sla


# --------------------------------------------------------------------------- block bookkeeping
def match_blocks(levels):
    """Make every level's block structure IDENTICAL, so the estimator's finite-sample bias is the
    same in every level and cancels exactly in a within-cell difference.

    levels: list (per load level) of lists of (n_t, d) arrays.
    Returns the same structure, truncated: same number of blocks, same length block by block.
    """
    nb = min(len(L) for L in levels)
    if nb == 0:
        return None
    ordered = [sorted(L, key=lambda b: -len(b))[:nb] for L in levels]
    out = [[] for _ in levels]
    for j in range(nb):
        n = min(len(o[j]) for o in ordered)
        if n < 4:
            continue
        for i, o in enumerate(ordered):
            out[i].append(np.asarray(o[j][:n], float))
    if not out[0]:
        return None
    return out


def _pairs(blocks):
    A, B, W = [], [], []
    for X in blocks:
        X = np.asarray(X, float)
        if len(X) < 4:
            continue
        Xc = X - X.mean(axis=0, keepdims=True)
        A.append(Xc[:-1]); B.append(Xc[1:]); W.append(Xc)
    if not A:
        return None
    return np.vstack(A), np.vstack(B), np.vstack(W), [len(x) for x in blocks if len(x) >= 4]


# --------------------------------------------------------------------------- the fit
def _ols(A, B, ridge=1e-8):
    G = A.T @ A
    G = G + ridge * np.trace(G) / A.shape[1] * np.eye(A.shape[1])
    return np.linalg.solve(G, A.T @ B).T


def _sim_blocks_batch(M, L, lens, B, rng, burn=100):
    """(B, sum(lens), d) simulated replicates of the whole block structure, vectorised over B."""
    d = M.shape[0]
    out = []
    for n in lens:
        e = rng.standard_normal((B, n + burn, d)) @ L.T
        x = np.empty((B, n + burn, d))
        x[:, 0] = e[:, 0]
        for t in range(1, n + burn):
            x[:, t] = x[:, t - 1] @ M.T + e[:, t]
        out.append(x[:, burn:])
    return out


def _batch_fit(sim, ridge):
    """OLS + residual trace for every replicate; sim is a list of (B, n, d) block arrays."""
    B, _, d = sim[0].shape
    G = np.zeros((B, d, d)); H = np.zeros((B, d, d)); npair = 0
    for X in sim:
        Xc = X - X.mean(axis=1, keepdims=True)
        a, b = Xc[:, :-1], Xc[:, 1:]
        G += np.einsum("bti,btj->bij", a, a)
        H += np.einsum("bti,btj->bij", a, b)
        npair += a.shape[1]
    tr = np.einsum("bii->b", G)[:, None, None] / d
    Gr = G + ridge * tr * np.eye(d)
    Mst = np.transpose(np.linalg.solve(Gr, H), (0, 2, 1))          # (B, d, d)
    S = np.zeros(B)
    for X in sim:
        Xc = X - X.mean(axis=1, keepdims=True)
        a, b = Xc[:, :-1], Xc[:, 1:]
        r = b - np.einsum("bij,btj->bti", Mst, a)
        S += np.einsum("bti,bti->b", r, r)
    return Mst, S / max(npair - 1, 1)


def fit_cell(blocks, dt=1.0, n_boot=200, rng=None, ridge=1e-8, burn=100):
    """Full per-cell fit.  Returns None if the cell is too thin.

    `burn` is the burn-in of the bias-correction simulation only.  It is 100 by default, which is
    what every stored result was produced with; a caller that simulates very short blocks may
    lower it, since |eig M| ~ 0.13 here makes 30 steps already exact to 1e-27.
    """
    rng = np.random.default_rng(rng)
    got = _pairs(blocks)
    if got is None:
        return None
    A, B, Wc, lens = got
    d = A.shape[1]
    n_pairs = len(A)
    if n_pairs < 4 * d:
        return None
    M = _ols(A, B, ridge)

    # --- bootstrap bias correction, matched to THIS cell's block structure
    Q0 = np.cov((B - A @ M.T).T, bias=False)
    Q0 = 0.5 * (Q0 + Q0.T)
    w, V = np.linalg.eigh(Q0)
    Lc = V @ np.diag(np.sqrt(np.maximum(w, 1e-18))) @ V.T
    if n_boot > 0 and np.max(np.abs(np.linalg.eigvals(M))) < 0.999:
        sim = _sim_blocks_batch(M, Lc, lens, n_boot, rng, burn=burn)
        Mst, trR = _batch_fit(sim, ridge)
        M_bc = 2 * M - Mst.mean(axis=0)
        q_att = float(np.mean(trR) / np.trace(Q0))      # multiplicative attenuation of tr(Q)
    else:
        M_bc = M.copy()
        q_att = 1.0

    # --- innovation covariance under the bias-corrected map (empirical, not model-implied)
    R = B - A @ M_bc.T
    Q = np.cov(R.T, bias=False) * (n_pairs - 1) / max(n_pairs - d - 1, 1)
    Q = 0.5 * (Q + Q.T)
    Q_bc = Q / max(q_att, 1e-6)

    ev = np.linalg.eigvals(M_bc)
    absev = np.abs(ev)
    stable = bool(np.all(absev < 1.0) and np.all(absev > 0))
    lam_modes = np.sort(-np.log(np.maximum(absev, 1e-12)) / dt)
    lam_mean = float(np.mean(lam_modes)) if stable else float("nan")
    lam_max = float(np.min(lam_modes)) if stable else float("nan")   # slowest mode
    # principal logarithm: computable, but NOT identified as the generator (aliasing) - see header
    logm_real = False
    try:
        Al = sla.logm(M_bc)
        logm_real = bool(np.max(np.abs(np.imag(Al))) < 1e-8 * max(1.0, np.max(np.abs(Al)))
                         and np.max(np.abs(sla.expm(Al) - M_bc)) < 1e-6 * max(1.0, np.max(np.abs(M_bc))))
    except Exception:  # noqa: BLE001
        Al = None
    neg_real = bool(np.any((np.abs(np.imag(ev)) < 1e-10) & (np.real(ev) < 0)))

    P_emp = np.cov(Wc.T, bias=False)
    P_emp = 0.5 * (P_emp + P_emp.T)
    try:
        P_mod = sla.solve_discrete_lyapunov(M_bc, Q_bc) if stable else None
        if P_mod is not None:
            P_mod = 0.5 * (P_mod + P_mod.T)
    except Exception:  # noqa: BLE001
        P_mod = None

    def _gm(S):
        w_ = np.linalg.eigvalsh(0.5 * (S + S.T))
        w_ = w_[w_ > 0]
        return float(np.exp(np.mean(np.log(w_)))) if len(w_) == S.shape[0] else float("nan")

    out = dict(
        d=d, dt=dt, n_pairs=int(n_pairs), n_win=int(len(Wc)), n_blocks=len(lens),
        M=M_bc, M_raw=M, Q=Q_bc, Q_raw=Q, P_emp=P_emp, P_mod=P_mod,
        stable=stable, logm_real=logm_real, neg_real_eig=neg_real, q_att=q_att,
        lam_modes=lam_modes, lam_mean=lam_mean, lam_max=lam_max,
        D_tr=float(np.trace(Q_bc) / (2 * d * dt)), D_gm=float(_gm(Q_bc) / (2 * dt)),
        sp_emp_gm=float(np.sqrt(_gm(P_emp))), sp_emp_tr=float(np.sqrt(np.trace(P_emp) / d)),
        pr_Q=float(np.trace(Q_bc) ** 2 / np.trace(Q_bc @ Q_bc)),
        pr_P=float(np.trace(P_emp) ** 2 / np.trace(P_emp @ P_emp)),
    )
    if P_mod is not None:
        out["sp_mod_gm"] = float(np.sqrt(_gm(P_mod)))
        out["sp_mod_tr"] = float(np.sqrt(np.trace(P_mod) / d))
        out["lyap_ratio_tr"] = float(np.trace(P_mod) / np.trace(P_emp))
        out["lyap_ratio_gm"] = float(_gm(P_mod) / _gm(P_emp))
        res = P_mod - (M_bc @ P_mod @ M_bc.T + Q_bc)
        out["lyap_residual_rel"] = float(np.linalg.norm(res) / np.linalg.norm(P_mod))
    else:
        out.update(sp_mod_gm=np.nan, sp_mod_tr=np.nan, lyap_ratio_tr=np.nan,
                   lyap_ratio_gm=np.nan, lyap_residual_rel=np.nan)
    return out


# --------------------------------------------------------------------------- one-pair-per-block
def fit_pairs(X, Y, dt=1.0, n_boot=300, rng=None, ridge=1e-8):
    """VAR(1) from INDEPENDENT (x_t, x_t+1) pairs, one per block.

    This is the HCP MEG design: each trial contributes exactly one lag-1 pair, so there is no
    within-block series and the within-block mean removal of fit_cell() is inadmissible (centring a
    two-window block forces M = -I).  X and Y are centred on their OWN means instead, which is the
    correct centring for a design in which the two latencies have different means.  The bias
    correction resimulates Y at the observed X, which is the bootstrap matched to this design.
    """
    rng = np.random.default_rng(rng)
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    n, d = X.shape
    if n < 4 * d:
        return None
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    M = _ols(Xc, Yc, ridge)
    Rr = Yc - Xc @ M.T
    Q0 = np.cov(Rr.T, bias=False) * (n - 1) / max(n - d - 1, 1)
    Q0 = 0.5 * (Q0 + Q0.T)
    w, V = np.linalg.eigh(Q0)
    Lc = V @ np.diag(np.sqrt(np.maximum(w, 1e-18))) @ V.T
    if n_boot > 0:
        E = rng.standard_normal((n_boot, n, d)) @ Lc.T
        Ys = np.einsum("ij,tj->ti", M, Xc)[None] + E
        Ys = Ys - Ys.mean(axis=1, keepdims=True)
        G = Xc.T @ Xc
        G = G + ridge * np.trace(G) / d * np.eye(d)
        Gi = np.linalg.inv(G)
        Ms = np.einsum("bti,tj,jk->bik", Ys, Xc, Gi)
        M_bc = 2 * M - Ms.mean(axis=0)
        trs = np.array([np.trace(np.cov((Ys[b] - Xc @ Ms[b].T).T, bias=False)) for b in range(n_boot)])
        q_att = float(np.mean(trs) / np.trace(Q0))
    else:
        M_bc, q_att = M.copy(), 1.0
    Rr2 = Yc - Xc @ M_bc.T
    Q = np.cov(Rr2.T, bias=False) * (n - 1) / max(n - d - 1, 1)
    Q = 0.5 * (Q + Q.T) / max(q_att, 1e-6)

    ev = np.linalg.eigvals(M_bc)
    absev = np.abs(ev)
    stable = bool(np.all(absev < 1.0) and np.all(absev > 0))
    lam_modes = np.sort(-np.log(np.maximum(absev, 1e-12)) / dt)
    P_emp = np.cov(Xc.T, bias=False)
    P_next = np.cov(Yc.T, bias=False)
    try:
        P_mod = sla.solve_discrete_lyapunov(M_bc, Q) if stable else None
    except Exception:  # noqa: BLE001
        P_mod = None
    out = dict(d=d, dt=dt, n_pairs=int(n), M=M_bc, M_raw=M, Q=Q, Q_raw=Q0,
               P_emp=P_emp, P_next=P_next, stable=stable, q_att=q_att,
               lam_modes=lam_modes, lam_mean=float(np.mean(lam_modes)) if stable else np.nan,
               lam_max=float(np.min(lam_modes)) if stable else np.nan,
               D_tr=float(np.trace(Q) / (2 * d * dt)),
               pr_Q=float(np.trace(Q) ** 2 / np.trace(Q @ Q)),
               sp_emp_gm=float(np.sqrt(np.exp(np.mean(np.log(
                   np.maximum(np.linalg.eigvalsh(P_emp), 1e-18)))))),
               nonstat_tr=float(np.trace(P_next) / np.trace(P_emp)))
    if P_mod is not None:
        out["lyap_ratio_tr"] = float(np.trace(P_mod) / np.trace(P_emp))
        out["sp_mod_gm"] = float(np.sqrt(np.exp(np.mean(np.log(
            np.maximum(np.linalg.eigvalsh(0.5 * (P_mod + P_mod.T)), 1e-18))))))
    else:
        out["lyap_ratio_tr"] = np.nan
        out["sp_mod_gm"] = np.nan
    return out


# --------------------------------------------------------------------------- direction readouts
def align_fraction(dQ, w):
    """Fraction of tr(dQ) captured by the unit direction w."""
    w = np.asarray(w, float)
    w = w / (np.linalg.norm(w) + 1e-300)
    t = np.trace(dQ)
    if abs(t) < 1e-300:
        return float("nan")
    return float((w @ dQ @ w) / t)


def random_direction_null(dQ, n=2000, rng=None):
    rng = np.random.default_rng(rng)
    d = dQ.shape[0]
    U = rng.standard_normal((n, d))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    t = np.trace(dQ)
    return np.einsum("ij,jk,ik->i", U, dQ, U) / t


def single_direction_ceiling(S):
    """The most any single direction can capture of tr(S): the leading eigenvalue share."""
    w = np.linalg.eigvalsh(0.5 * (S + S.T))
    return float(np.max(w) / np.trace(S))


def principal_angle_to_subspace(w, S, k):
    """Angle (degrees) between the unit vector w and the span of the top-k eigenvectors of S."""
    w = np.asarray(w, float)
    w = w / (np.linalg.norm(w) + 1e-300)
    ev, V = np.linalg.eigh(0.5 * (S + S.T))
    U = V[:, np.argsort(ev)[::-1][:k]]
    c = np.linalg.norm(U.T @ w)
    return float(np.degrees(np.arccos(np.clip(c, 0.0, 1.0))))


def participation_ratio(S):
    return float(np.trace(S) ** 2 / np.trace(S @ S))


# --------------------------------------------------------------------------- inference
def boot_ci(x, n=10000, seed=20260922):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=int(len(x)),
                    n_pos=0, p=float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    b = x[idx].mean(axis=1)
    obs = abs(float(np.mean(x)))
    s = rng.choice([-1.0, 1.0], size=(n, len(x)))
    null = np.abs((s * (x - 0.0)).mean(axis=1))
    return dict(mean=float(np.mean(x)), lo=float(np.percentile(b, 2.5)),
                hi=float(np.percentile(b, 97.5)), n=int(len(x)),
                n_pos=int(np.sum(x > 0)), p=float((1 + np.sum(null >= obs)) / (n + 1)),
                sd=float(np.std(x, ddof=1)))


def fmt(c, unit=""):
    return (f"{c['mean']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]{unit}, "
            f"n={c['n']}, {c['n_pos']}/{c['n']} positive, p={c['p']:.4g}")
