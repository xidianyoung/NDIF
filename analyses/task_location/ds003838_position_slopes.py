# -*- coding: utf-8 -*-
"""Task location in the digit-span EEG cohort (ds003838, 64 participants): position slopes on the expected
number of items held, with the listen-only arm carried through every step as the matched control arm, and the
input-dependent drift, restoring rate and diffusion in the same participants and the same state space.

Per participant (in parallel over participants):
  Part 1  leave-one-design-block-out displacement along the out-of-sample task axis; its within-set-size slope
          on the input u in the memory arm and in the listen-only arm, and their difference in differences
          (DiD); nulls: (N1) within-sequence permutation of the slot -> u map, preserving sequence, block, set
          size and windows per slot; (N2) the listen-only arm itself; (N3) a one-slot circular shift of u;
          a design-block bootstrap of the DiD within participant.
  Part 2  the input-dependent drift dz/dt = A_0 z + c0 + c1 g(u) + (A_1 z) g(u) + set-size dummies (ridge on
          A_1 selected by leave-one-design-block-out R^2), its fixed-point branch x*(u) and stability, the
          restoring rate and diffusion per (set size, u) cell, the lag spectrum (observation-noise floor) and
          the VAR(1) fit of the non-overlapping series, for both arms.
  Part 3  a matched simulation of the rate-slope estimator (detection threshold, power, attenuation).
  Part 5  robustness of the DiD: degree-3 session detrend, theta+alpha and gamma-free features, nuisance and
          peripheral residualisation, first 2 s dropped, k <= 9, slot labels from the nominal 2.0-s SOA.
Group summary: participant means with percentile bootstrap intervals (10,000 resamples, seed 20260921), sign
tests, and the group permutation P from the per-participant permutation draws.

State space: 64 channels x 4 bands log10 band power, 1-s Hann windows, 0.5-s step ('dist') and 1.0-s step
('dyn'), robust z + PCA to d = 8 (RefSpace), fitted on control (listen-only) windows of the TRAINING design
blocks only.  Item slots come from the true per-item onsets in events.tsv.

Inputs u (defined before the analysis; set size L is a fixed effect everywhere and is never part of u):
  u3 = expected items held = cumulative per-item recall probability for that participant and set size, from
       the per-item outcomes in events.tsv.  Primary.
  u1 = serial position k (1..13), the linear alternative;  u2 = min(k, 9).
u3 is built from recall, so recall is never used as an outcome.  For each fold the reference space and the task
axis w are fit on the eight training design blocks pooling all u; displacement is measured only in the
held-out block.  Resampling is at the participant level for group intervals and at the design-block level
within participant.

The permutation and simulation null draws are seeded per participant with SEED + Python's hash of the
participant mod 100000; the slopes, the DiD, the one-slot shift and every group interval do not depend on these
draws.

Input:   the ds003838 feature cache (paths.cache("ds003838"), environment variable NDIF_DS003838_CACHE): one
         sub-XXX.npz and sub-XXX_meta.json per participant (README.md).
Outputs: analyses/task_location/out/digit_span_position_analysis.json (settings, group summary, one row per
         participant), digit_span_participants.csv (one row per participant), digit_span_rate_diffusion_cells.csv
         (restoring rate and diffusion per participant, arm, set size and u tertile); NDIF_OUT relocates them.
         ds003838_summary.py extracts the reported values.
Reported in: Fig. 4a, Extended Data Fig. 4a, Supplementary Table 7a, Supplementary Note 6.
Runtime: about 7 s per participant on one desktop (4 min for the 64 participants with 2 worker processes,
2 BLAS threads each, the setting of the stored run).

    python analyses/task_location/ds003838_position_slopes.py --workers 5 --perm 1000 --sims 200
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
from scipy import linalg as sla
from scipy import stats as sst
from scipy.optimize import curve_fit

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyses.common import paths  # noqa: E402
from ds003838_common import (RefSpace, D, lda_axis, split_runs, detrend_within_condition,  # noqa: E402
                             fit_ou, bias_correct, load_cache, u_maps, SEED, CACHE, pkg_versions, log)

DT = 1.0                      # lag-2 of the 0.5-s-step series: two adjacent NON-overlapping windows
LAGS = (2, 4, 6, 8)           # lag spectrum, in 0.5-s steps -> 1, 2, 3, 4 s
RIDGE_GRID = (0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
U_NAMES = ("u3", "u1", "u2")
PRIMARY_U = "u3"


# ============================================================ small estimators
def ols(X, Y, pen=None, alpha=0.0):
    """Least squares with a ridge applied only to the rows of `pen` that are True."""
    G = X.T @ X
    if alpha > 0 and pen is not None:
        G = G + alpha * (np.trace(G) / X.shape[1]) * np.diag(pen.astype(float))
    return np.linalg.solve(G + 1e-10 * np.eye(X.shape[1]) * max(1.0, np.trace(G)), X.T @ Y)


def slope_on_u(p, u, L, w=None):
    """Within-set-size OLS slope of p on u, set size as a fixed effect (one intercept per L)."""
    Ls = np.unique(L)
    X = np.column_stack([(L == v).astype(float) for v in Ls] + [u])
    if w is None:
        b = np.linalg.lstsq(X, p, rcond=None)[0]
    else:
        Xw = X * np.sqrt(w)[:, None]
        b = np.linalg.lstsq(Xw, p * np.sqrt(w), rcond=None)[0]
    return float(b[-1])


def cell_means(p, k, L, arm):
    """Unweighted mean of p in each (arm, L, position) cell; returns dicts keyed (L, k)."""
    out = {0: {}, 1: {}}
    for a in (0, 1):
        m = arm == a
        for v in np.unique(L[m]):
            mm = m & (L == v)
            for kk in np.unique(k[mm]):
                s = mm & (k == kk)
                out[a][(int(v), int(kk))] = float(p[s].mean())
    return out


def u_of(umap, L, k):
    """Vectorised lookup of u for each (set size, position) pair."""
    u = np.empty(len(k), float)
    for Lv in np.unique(L):
        m = L == Lv
        tab = np.asarray(umap[int(Lv)], float)
        u[m] = tab[np.clip(k[m].astype(int), 1, len(tab)) - 1]
    return u


def did_slopes(p, k, L, arm, umap, fast=False):
    """Both arms' within-set-size slopes and their difference, for one u definition."""
    u = u_of(umap, L, k)
    ok = np.isfinite(u)
    mm, cc = ok & (arm == 1), ok & (arm == 0)
    b_mem = slope_on_u(p[mm], u[mm], L[mm])
    b_ctl = slope_on_u(p[cc], u[cc], L[cc])
    if fast:
        return dict(slope_memory=b_mem, slope_control=b_ctl, did=b_mem - b_ctl,
                    did_cellmatched=np.nan)
    # cell-matched form: the difference at matched serial position, then one slope
    cm = cell_means(p[ok], k[ok], L[ok], arm[ok])
    keys = sorted(set(cm[0]) & set(cm[1]))
    if len(keys) >= 4:
        dd = np.array([cm[1][q] - cm[0][q] for q in keys])
        uu = np.array([umap[q[0]][q[1] - 1] for q in keys], float)
        LL = np.array([q[0] for q in keys], float)
        b_cell = slope_on_u(dd, uu, LL)
    else:
        b_cell = np.nan
    return dict(slope_memory=b_mem, slope_control=b_ctl, did=b_mem - b_ctl, did_cellmatched=b_cell)


def lam_from_M(M, dt):
    ev = np.abs(np.linalg.eigvals(M))
    ev = ev[(ev > 0) & (ev < 1)]
    if len(ev) != M.shape[0]:
        return np.nan
    return float(-np.mean(np.log(ev)) / dt)


# ============================================================ one participant
def one_subject(args):
    sub, n_perm, n_sims, arms_cfg = args
    t0 = time.time()
    try:
        return _one(sub, n_perm, n_sims, arms_cfg, t0)
    except Exception as e:  # noqa: BLE001
        import traceback
        return dict(subject=sub, error=repr(e), traceback=traceback.format_exc()[-2500:])


def _held_out_table(S, F_key, keep_key="keep", resid=None, drop2=False, kmax=None):
    """Leave-one-design-block-out displacement along the out-of-sample task axis.

    Returns p (s.d. units, control-referenced), slot k, set size L, arm, sequence id, fold.
    """
    n = S["n"]
    Fs = S[F_key]
    keeps = list(S[keep_key])
    if drop2:
        keeps = [keeps[i] & (S["t_start"][i] - S["t0"][i] >= 2.0) for i in range(n)]
    if kmax is not None:
        keeps = [keeps[i] & (S["slot_true"][i] <= kmax) for i in range(n)]
    folds = sorted(set(S["block"]))
    P, K, L, ARM, SEQ, FOLD = [], [], [], [], [], []
    axes = []
    for b in folds:
        tr = [i for i in range(n) if S["block"][i] != b and keeps[i].any()]
        te = [i for i in range(n) if S["block"][i] == b and keeps[i].any()]
        ref = [i for i in tr if S["cond"][i] == 0]
        if not ref or not te:
            continue
        sp = RefSpace(D).fit(np.vstack([Fs[i][keeps[i]] for i in ref]))
        Ztr = {i: sp.transform(Fs[i][keeps[i]]) for i in tr}
        Zte = {i: sp.transform(Fs[i][keeps[i]]) for i in te}
        if resid is not None:
            Ctr = np.vstack([resid[i][keeps[i]] for i in tr])
            mu, sd = Ctr.mean(0), Ctr.std(0) + 1e-12
            A_1 = np.column_stack([np.ones(len(Ctr)), (Ctr - mu) / sd])
            beta = np.linalg.solve(A_1.T @ A_1 + 1.0 * np.eye(A_1.shape[1]),
                                   A_1.T @ np.vstack([Ztr[i] for i in tr]))
            for dd, idx in ((Ztr, tr), (Zte, te)):
                for i in idx:
                    C = np.column_stack([np.ones(int(keeps[i].sum())),
                                         (resid[i][keeps[i]] - mu) / sd])
                    dd[i] = dd[i] - C @ beta
        Z0 = np.vstack([Ztr[i] for i in tr if S["cond"][i] == 0])
        Z1 = np.vstack([Ztr[i] for i in tr if S["cond"][i] == 1])
        if len(Z0) < 3 * D or len(Z1) < 3 * D:
            continue
        w = lda_axis(Z0, Z1)                      # pools ALL u; never sees the held-out block
        axes.append(w)
        m0 = float((Z0 @ w).mean())
        sd_ax = float(np.std(np.vstack([Z0, Z1]) @ w))
        for i in te:
            p = (Zte[i] @ w - m0) / sd_ax
            P.append(p)
            K.append(S["slot_true"][i][keeps[i]].astype(int))
            L.append(np.full(len(p), S["n_digits"][i]))
            ARM.append(np.full(len(p), S["cond"][i]))
            SEQ.append(np.full(len(p), i))
            FOLD.append(np.full(len(p), b))
    if not P:
        return None
    return dict(p=np.concatenate(P), k=np.concatenate(K), L=np.concatenate(L),
                arm=np.concatenate(ARM), seq=np.concatenate(SEQ), fold=np.concatenate(FOLD),
                axes=np.array(axes))


def _band_subset(S, idx_bands):
    n_ch = S["meta"]["n_channels"]
    out = []
    for F in S["F_dist"]:
        out.append(F.reshape(F.shape[0], n_ch, 4)[:, :, idx_bands].reshape(F.shape[0], -1))
    return out


def _increments(Z, S, umap, uname, arm, slot_key="slot_true"):
    """lag-2 increments of the 0.5-s-step series whose TWO windows lie in the SAME item slot."""
    zs, dzs, us, Ls, blks, seqs = [], [], [], [], [], []
    for i in range(S["n"]):
        if S["cond"][i] != arm:
            continue
        keep, slot = S["keep"][i], S["slot_true"][i] if slot_key == "slot_true" else S["slot_pub"][i]
        Zi = Z[i]
        m = np.arange(len(Zi) - 2)
        good = keep[m] & keep[m + 2] & (slot[m] == slot[m + 2])
        if not good.any():
            continue
        j = m[good]
        Lv = S["n_digits"][i]
        tab = np.asarray(umap[Lv], float)
        uu = tab[np.clip(slot[j].astype(int), 1, len(tab)) - 1]
        zs.append(Zi[j]); dzs.append((Zi[j + 2] - Zi[j]) / DT)
        us.append(uu); Ls.append(np.full(len(j), Lv))
        blks.append(np.full(len(j), S["block"][i])); seqs.append(np.full(len(j), i))
    if not zs:
        return None
    return dict(z=np.vstack(zs), dz=np.vstack(dzs), u=np.concatenate(us), L=np.concatenate(Ls),
                block=np.concatenate(blks), seq=np.concatenate(seqs))


def _design(inc, g):
    z = inc["z"]
    n = len(z)
    cols = [z, np.ones((n, 1)), g[:, None], z * g[:, None]]
    for v in (9, 13):
        cols.append((inc["L"] == v).astype(float)[:, None])
    X = np.hstack(cols)
    pen = np.zeros(X.shape[1], bool)
    pen[D + 2: 2 * D + 2] = True                 # the A_1 block only
    return X, pen


def _fit_drift(inc, ridge_grid=RIDGE_GRID):
    """dz/dt = A_0 z + c0 + c1 g(u) + (A_1 z) g(u) + set-size dummies; ridge on A_1 by LOBO R^2."""
    u = inc["u"]
    g = u - u.mean()
    X, pen = _design(inc, g)
    Y = inc["dz"]
    blocks = np.unique(inc["block"])
    # nested models, held-out R^2 by leave-one-design-block-out
    idx_free = list(range(D + 1)) + [2 * D + 2, 2 * D + 3]
    idx_xs = idx_free[:D + 1] + [D + 1] + idx_free[D + 1:]

    def lobo_r2(cols, alpha, pen_sub):
        num = den = 0.0
        for b in blocks:
            te = inc["block"] == b
            tr = ~te
            if tr.sum() < 5 * len(cols) or te.sum() < 5:
                continue
            B = ols(X[tr][:, cols], Y[tr], pen_sub, alpha)
            r = Y[te] - X[te][:, cols] @ B
            num += float((r ** 2).sum())
            den += float(((Y[te] - Y[tr].mean(0)) ** 2).sum())
        return (1 - num / den) if den > 0 else np.nan

    r2_free = lobo_r2(idx_free, 0.0, np.zeros(len(idx_free), bool))
    r2_xs = lobo_r2(idx_xs, 0.0, np.zeros(len(idx_xs), bool))
    full = list(range(X.shape[1]))
    r2_by_alpha = {a: lobo_r2(full, a, pen) for a in ridge_grid}
    fin = {a: v for a, v in r2_by_alpha.items() if np.isfinite(v)}
    alpha = max(fin, key=fin.get) if fin else 1.0
    r2_A = r2_by_alpha.get(alpha, np.nan)
    B = ols(X, Y, pen, alpha)
    B0 = ols(X, Y, pen, 0.0)                     # unpenalised companion (reported beside the ridge)
    A_0 = B[:D].T
    c0 = B[D]
    c1 = B[D + 1]
    A_1 = B[D + 2: 2 * D + 2].T
    A_1_unpen = B0[D + 2: 2 * D + 2].T
    gam = B[2 * D + 2:]
    frac = [float((inc["L"] == v).mean()) for v in (9, 13)]
    c_bar = c0 + frac[0] * gam[0] + frac[1] * gam[1]
    resid = Y - X @ B
    return dict(A_0=A_0, A_1=A_1, A_1_unpen=A_1_unpen, c0=c0, c1=c1, c_bar=c_bar, gamma=gam, alpha=float(alpha),
                r2_free=float(r2_free), r2_xstar=float(r2_xs), r2_A=float(r2_A),
                r2_by_alpha={str(k): (None if not np.isfinite(v) else float(v))
                             for k, v in r2_by_alpha.items()},
                resid_cov=np.cov(resid.T), n_inc=int(len(Y)), u_mean=float(u.mean()),
                u_min=float(u.min()), u_max=float(u.max()))


def _branch(fit, w, u_grid, u_mean):
    """x*(u) = -A(u)^-1 c(u) and the spectrum of A(u) on a grid of u."""
    rows = []
    x0 = None
    for uu in u_grid:
        g = uu - u_mean
        A = fit["A_0"] + g * fit["A_1"]
        c = fit["c_bar"] + g * fit["c1"]
        ev = np.linalg.eigvals(A)
        try:
            xs = -np.linalg.solve(A, c)
        except np.linalg.LinAlgError:
            xs = np.full(D, np.nan)
        if x0 is None:
            x0 = xs
        rows.append(dict(u=float(uu), proj=float(xs @ w), dist_from_umin=float(np.linalg.norm(xs - x0)),
                         max_re_eig=float(np.max(ev.real)), mean_re_eig=float(np.mean(ev.real)),
                         lambda_max=float(-np.max(ev.real)), lambda_mean=float(-np.mean(ev.real)),
                         stable=bool(np.max(ev.real) < 0)))
    return rows


def _continuation(fit, u_mean, u_lo, u_hi, n=400):
    """Continue the branch beyond the observed range until an eigenvalue of A(u) crosses zero,
    or u reaches 3x the observed half-range."""
    span = u_hi - u_lo
    out = {}
    for sgn, name in ((+1, "up"), (-1, "down")):
        u_end = (u_hi + 3 * span) if sgn > 0 else (u_lo - 3 * span)
        grid = np.linspace(u_hi if sgn > 0 else u_lo, u_end, n)
        crit = None
        for uu in grid:
            A = fit["A_0"] + (uu - u_mean) * fit["A_1"]
            if np.max(np.linalg.eigvals(A).real) >= 0:
                crit = float(uu)
                break
        out[name] = dict(u_crit=crit, u_limit=float(u_end),
                         multiple_of_observed_range=(None if crit is None else
                                                     float(abs(crit - (u_hi if sgn > 0 else u_lo)) / span)))
    return out


def _lam_slope(fit, u_lo, u_hi, u_mean):
    """d lambda / d u from the fitted A(u): exact for lambda_mean, secant for lambda_max."""
    dl_mean = float(-np.trace(fit["A_1"]) / D)
    A_lo = fit["A_0"] + (u_lo - u_mean) * fit["A_1"]
    A_hi = fit["A_0"] + (u_hi - u_mean) * fit["A_1"]
    lm_lo = float(-np.max(np.linalg.eigvals(A_lo).real))
    lm_hi = float(-np.max(np.linalg.eigvals(A_hi).real))
    return dict(dlambda_mean_du=dl_mean,
                dlambda_mean_du_unpenalised=float(-np.trace(fit["A_1_unpen"]) / D),
                dlambda_max_du=float((lm_hi - lm_lo) / (u_hi - u_lo)),
                lambda_max_at_umin=lm_lo, lambda_max_at_umax=lm_hi,
                lambda_mean_at_umin=float(-np.trace(A_lo) / D),
                lambda_mean_at_umax=float(-np.trace(A_hi) / D))


# ---------------------------------------------------------------- D and lambda, separated
def dl_decompose(z, z2, dt=DT, noise_var=None):
    """Exact per-direction OU decomposition into a restoring rate and a diffusion.

    Along each of the d state-space directions (the coordinates the stationary spread is
    measured in) the one-step map is an AR(1):
        z_i(t+dt) = a_i z_i(t) + eps_i ,  a_i = exp(-lambda_i dt) ,  Var(eps_i) = sigma_i^2 (1 - a_i^2)
    so lambda_i = -log(a_i)/dt comes from the DECAY of the one-step map and
        D_i = lambda_i Var(eps_i) / (1 - a_i^2)
    from the INNOVATION variance.  The implied stationary spread sigma_i = sqrt(D_i/lambda_i) is
    then checked against the EMPIRICAL spread of the same coordinate - that check is the validation
    of the decomposition, and it is a real check because an AR(1) that does not describe the data
    implies a stationary variance that the data do not have.

    Aggregation is geometric, which makes the decomposition exactly additive in logs:
        log spread_g = (1/d) sum_i log sigma_i = 1/2 ( log D_g - log lambda_g )

    `noise_var`: per-coordinate additive observation-noise variance (from the lag spectrum).  With
    z = x + eta, the AR(1) coefficient is attenuated by sigma_x^2/(sigma_x^2 + sigma_eta^2); undoing
    that attenuation gives the noise-corrected (lambda, D), which move in OPPOSITE directions, so an
    uncorrected pair could manufacture a spurious decomposition.
    """
    zc = z - z.mean(0)
    yc = z2 - z2.mean(0)
    v0 = (zc ** 2).mean(axis=0)
    a = (zc * yc).mean(axis=0) / np.maximum(v0, 1e-30)
    res = yc - a * zc
    ve = (res ** 2).mean(axis=0)
    out = dict(a=a.tolist(), var_obs=v0.tolist())

    def pack(a_, sx2, tag):
        ok = (a_ > 1e-6) & (a_ < 1 - 1e-9) & np.isfinite(a_)
        lam = np.full(D, np.nan); Di = np.full(D, np.nan); sig = np.full(D, np.nan)
        lam[ok] = -np.log(a_[ok]) / dt
        sig[ok] = np.sqrt(np.maximum(sx2[ok], 1e-30))
        Di[ok] = lam[ok] * sig[ok] ** 2
        good = ok & np.isfinite(lam) & (lam > 0) & (Di > 0)
        r = {}
        if good.sum() >= 4:
            r[f"lambda_g{tag}"] = float(np.exp(np.mean(np.log(lam[good]))))
            r[f"D_g{tag}"] = float(np.exp(np.mean(np.log(Di[good]))))
            r[f"spread_g{tag}"] = float(np.exp(0.5 * np.mean(np.log(Di[good] / lam[good]))))
            r[f"n_dirs{tag}"] = int(good.sum())
        else:
            r[f"lambda_g{tag}"] = r[f"D_g{tag}"] = r[f"spread_g{tag}"] = np.nan
            r[f"n_dirs{tag}"] = int(good.sum())
        r[f"lambda_per_dir{tag}"] = lam.tolist()
        r[f"D_per_dir{tag}"] = Di.tolist()
        return r

    sx2_raw = ve / np.maximum(1 - a ** 2, 1e-12)          # stationary variance implied by the AR(1)
    out.update(pack(a, sx2_raw, ""))
    out["spread_g_empirical"] = float(np.exp(0.5 * np.mean(np.log(np.maximum(v0, 1e-30)))))
    out["spread_ratio_check"] = (float(out["spread_g"] / out["spread_g_empirical"])
                                 if np.isfinite(out["spread_g"]) and out["spread_g_empirical"] > 0 else np.nan)
    if noise_var is not None:
        se2 = np.asarray(noise_var, float)
        sx2 = np.maximum(v0 - se2, 1e-6 * np.maximum(v0, 1e-30))
        a_corr = np.clip(a * v0 / np.maximum(sx2, 1e-30), 1e-6, 1 - 1e-9)
        out.update(pack(a_corr, sx2, "_nc"))
        out["noise_fraction_per_dir"] = (se2 / np.maximum(v0, 1e-30)).tolist()
    return out


def _bin_cells(inc, n_bins=3, noise_var=None):
    """(set size L) x (u tertile within L) cells; lambda from the one-step decay, D from the
    innovation variance, along each state-space direction."""
    cells = []
    for Lv in np.unique(inc["L"]):
        m = inc["L"] == Lv
        uu = inc["u"][m]
        qs = np.quantile(uu, np.linspace(0, 1, n_bins + 1))
        qs[0] -= 1e-9
        for bi in range(n_bins):
            s = m.copy()
            s[m] = (uu > qs[bi]) & (uu <= qs[bi + 1])
            if s.sum() < 6 * D:
                continue
            z, dz = inc["z"][s], inc["dz"][s]
            dd = dl_decompose(z, z + dz * DT, DT, noise_var)
            zc = z - z.mean(0)
            M = np.linalg.solve(zc.T @ zc + 1e-8 * np.trace(zc.T @ zc) / D * np.eye(D),
                                zc.T @ ((z + dz * DT) - (z + dz * DT).mean(0))).T
            cells.append(dict(L=int(Lv), bin=int(bi), n=int(s.sum()),
                              u_mean=float(uu[(uu > qs[bi]) & (uu <= qs[bi + 1])].mean()),
                              lambda_g=dd["lambda_g"], D_g=dd["D_g"],
                              spread_from_D_lambda=dd["spread_g"],
                              spread_observed=dd["spread_g_empirical"],
                              ratio_check=dd["spread_ratio_check"],
                              n_dirs=dd["n_dirs"], n_dirs_nc=dd.get("n_dirs_nc", 0),
                              lambda_g_nc=dd.get("lambda_g_nc", np.nan),
                              D_g_nc=dd.get("D_g_nc", np.nan),
                              spread_g_nc=dd.get("spread_g_nc", np.nan),
                              lambda_mean_multivariate=lam_from_M(M, DT)))
    return cells


def _log_decomposition(cells, key_lam="lambda_g", key_D="D_g"):
    """d log spread = 1/2 (d log D - d log lambda) across u, within set size, pooled."""
    use = [c for c in cells if np.isfinite(c[key_lam]) and c[key_lam] > 0
           and np.isfinite(c[key_D]) and c[key_D] > 0]
    if len(use) < 4:
        return None
    x = np.array([c["u_mean"] for c in use], float)
    Lv = np.array([c["L"] for c in use], float)
    ld = np.log([c[key_D] for c in use])
    ll = np.log([c[key_lam] for c in use])
    ls = 0.5 * (ld - ll)
    sl = lambda y: slope_on_u(np.asarray(y, float), x, Lv)  # noqa: E731
    a, b, c_ = sl(ld), sl(ll), sl(ls)
    return dict(dlogD_du=float(a), dloglambda_du=float(b), dlogspread_du=float(c_),
                identity_residual=float(c_ - 0.5 * (a - b)),
                pct_from_D=(float(100 * 0.5 * a / c_) if abs(c_) > 1e-12 else None),
                pct_from_lambda=(float(-100 * 0.5 * b / c_) if abs(c_) > 1e-12 else None),
                n_cells=len(use))


def _lag_spectrum(Z, S, arm):
    """V_i(tau) = Var(z_i(t+tau) - z_i(t)) at lags 2,4,6,8 of the 0.5-s series, WITHIN sequence.

    With z = x + eta and eta an additive white observation noise,
        V(tau) = 2 sigma_x^2 (1 - e^{-lambda tau}) + 2 sigma_eta^2 ,
    so the intercept of V as tau -> 0 is 2 sigma_eta^2: the additive observation-noise floor.
    It is fitted per state-space direction (the floor is what corrects lambda and D) and also
    summarised over directions, alongside the model-free linear extrapolation to tau -> 0.
    """
    Vs = {lag: np.zeros(D) for lag in LAGS}
    N = {lag: 0 for lag in LAGS}
    for i in range(S["n"]):
        if S["cond"][i] != arm:
            continue
        keep, Zi = S["keep"][i], Z[i]
        for lag in LAGS:
            m = np.arange(len(Zi) - lag)
            if len(m) == 0:
                continue
            good = keep[m] & keep[m + lag]
            if not good.any():
                continue
            dd = Zi[m[good] + lag] - Zi[m[good]]
            Vs[lag] += (dd ** 2).sum(axis=0)
            N[lag] += int(good.sum())
    if any(N[lag] < 30 for lag in LAGS):
        return None
    tau = np.array([lag * 0.5 for lag in LAGS], float)
    Vd = np.array([Vs[lag] / N[lag] for lag in LAGS])                  # (n_lags, D)
    v = Vd.mean(axis=1)

    def f(t, sx2, lam, se2):
        return 2 * sx2 * (1 - np.exp(-lam * t)) + 2 * se2

    se2_dir, lam_dir, sx2_dir = np.full(D, np.nan), np.full(D, np.nan), np.full(D, np.nan)
    for i in range(D):
        y = Vd[:, i]
        try:
            p, _ = curve_fit(f, tau, y, p0=[max(y[-1] / 2, 1e-6), 0.5, max(y[0] / 4, 1e-9)],
                             bounds=([1e-9, 1e-4, 0.0], [1e6, 20.0, 1e6]), maxfev=40000)
            sx2_dir[i], lam_dir[i], se2_dir[i] = p
        except Exception:  # noqa: BLE001
            pass
    sl, ic = np.polyfit(tau, v, 1)
    out = dict(tau_s=tau.tolist(), V=v.tolist(), V_per_dir=Vd.tolist(),
               linear_intercept=float(ic), linear_slope=float(sl), noise_var_linear=float(ic / 2.0),
               noise_var_per_dir=se2_dir.tolist(),
               D_raw_lag2=float(v[0] / (2 * tau[0])), n_pairs={str(k): int(v_) for k, v_ in N.items()})
    ok = np.isfinite(se2_dir) & np.isfinite(sx2_dir) & (sx2_dir > 0)
    if ok.sum() >= 4:
        out.update(noise_var_fit=float(np.mean(se2_dir[ok])),
                   sigma_x2_corrected=float(np.mean(sx2_dir[ok])),
                   lambda_corrected=float(np.exp(np.mean(np.log(np.maximum(lam_dir[ok], 1e-9))))),
                   D_corrected=float(np.exp(np.mean(np.log(np.maximum(lam_dir[ok] * sx2_dir[ok], 1e-30))))),
                   noise_fraction=float(np.mean(se2_dir[ok] / (sx2_dir[ok] + se2_dir[ok]))),
                   fit_rmse=float(np.sqrt(np.mean([(Vd[:, i] - f(tau, sx2_dir[i], lam_dir[i], se2_dir[i])) ** 2
                                                   for i in np.where(ok)[0]]))))
    return out


def _fit_ou_dyn(S, arm, noise_var=None):
    """VAR(1) fit (fit_ou) + bootstrap bias correction on the NON-OVERLAPPING 1.0-s-step series."""
    segs = []
    for i in range(S["n"]):
        if S["cond"][i] != arm:
            continue
        for r in split_runs(S["keep_y"][i]):
            if len(r) >= 4:
                segs.append(S["Zy"][i][r])
    if len(segs) < 3:
        return None
    f = fit_ou(segs, dt=1.0)
    if f is None:
        return None
    bc = bias_correct(f, n_boot=200, rng=np.random.default_rng(SEED))
    sig2 = float(np.trace(f.Sigma0) / D)
    lam = bc["lambda_mean_bc"]
    z = np.vstack([s[:-1] for s in segs])
    z2 = np.vstack([s[1:] for s in segs])
    dd = dl_decompose(z, z2, dt=1.0, noise_var=noise_var)
    return dict(lambda_mean_raw=float(f.lambda_mean), lambda_mean_bc=float(lam),
                lambda_max_raw=float(f.lambda_recovery), lambda_max_bc=float(bc["lambda_max_bc"]),
                D_iso=float(np.trace(f.Sigma_eps) / (2 * D * 1.0)),
                var_stationary=sig2, spread_observed_iso=float(np.sqrt(sig2)),
                lambda_g=dd["lambda_g"], D_g=dd["D_g"], spread_from_D_lambda=dd["spread_g"],
                spread_observed=dd["spread_g_empirical"], ratio_check=dd["spread_ratio_check"],
                lambda_g_nc=dd.get("lambda_g_nc"), D_g_nc=dd.get("D_g_nc"),
                spread_g_nc=dd.get("spread_g_nc"),
                n_pairs=int(f.n_pairs), n_segments=len(segs))


# ---------------------------------------------------------------- matched simulation for the bound
def _sim_null(fit, inc, S, arm, n_sims, delta_grid, rng):
    """Matched simulation at the participant's own increment count, design and residual structure.

    The 0.5-s window grid is simulated as two interleaved VAR(1) chains, one on the even window
    indices and one on the odd ones, each stepping with the participant's OWN fitted propagator
    M = I + A_0 dt and innovation covariance (the fitted residual covariance).  Every lag-2 increment
    (j, j+2) therefore lies inside one chain and carries exactly the fitted dynamics, and increments
    j and j+2 share window j+2 exactly as they do in the data.  The same sequence layout, the same
    same-slot restriction, the same u design and the same number of increments are reused unchanged.
    (What the construction does not reproduce is the weak cross-phase correlation from the 50 %
    overlap of adjacent 1-s windows; that correlation links increments that share no window.)

    A true per-unit-u change in lambda of `delta` is injected as A_1 = -delta I, which shifts EVERY
    eigenvalue of A(u) by -delta*g, i.e. d lambda/du = delta exactly for lambda_mean and lambda_max.
    Because A_1 enters additively in dz and the design matrix does not depend on delta, one simulated
    series serves the whole delta grid:  X'Y(delta) = X'dz0 - delta X'(z0 * g).
    """
    A_0 = fit["A_0"]
    M1 = np.eye(D) + A_0 * DT
    if np.max(np.abs(np.linalg.eigvals(M1))) >= 1.0:
        return None
    Se = 0.5 * (fit["resid_cov"] + fit["resid_cov"].T) * DT ** 2
    wv, Vv = np.linalg.eigh(Se)
    Lh = Vv @ np.diag(np.sqrt(np.maximum(wv, 0))) @ Vv.T
    try:
        S_stat = sla.solve_discrete_lyapunov(M1, 0.5 * (Lh @ Lh.T + (Lh @ Lh.T).T))
    except Exception:  # noqa: BLE001
        S_stat = np.cov(inc["z"].T)
    ws, Vs = np.linalg.eigh(0.5 * (S_stat + S_stat.T))
    L0 = Vs @ np.diag(np.sqrt(np.maximum(ws, 0))) @ Vs.T

    # the observed increment layout: one entry per sequence
    runs, U, LL, BB, JJ = [], [], [], [], []
    for i in range(S["n"]):
        if S["cond"][i] != arm:
            continue
        keep, slot = S["keep"][i], S["slot_true"][i]
        m = np.arange(len(keep) - 2)
        good = keep[m] & keep[m + 2] & (slot[m] == slot[m + 2])
        if not good.any():
            continue
        j = m[good]
        Lv = int(S["n_digits"][i])
        uvals = np.array([fit["u_of_slot"][(Lv, int(s))] for s in slot[j]], float)
        runs.append(int(j.max()) + 3)
        JJ.append(j); U.append(uvals)
        LL.append(np.full(len(j), Lv)); BB.append(np.full(len(j), S["block"][i]))
    if not runs:
        return None
    R, NMAX = len(runs), max(runs)
    u_all = np.concatenate(U)
    sim_tpl = dict(L=np.concatenate(LL), block=np.concatenate(BB), u=u_all,
                   z=np.zeros((len(u_all), D)), dz=np.zeros((len(u_all), D)))
    g = u_all - u_all.mean()
    off = np.cumsum([0] + [len(j) for j in JJ])

    n_steps = NMAX // 2 + 2
    est = {float(d): [] for d in delta_grid}
    est_unpen = {float(d): [] for d in delta_grid}
    for _ in range(n_sims):
        X3 = np.empty((R, NMAX, D))
        for par in (0, 1):
            x = rng.standard_normal((R, D)) @ L0.T
            for s in range(n_steps):
                t = 2 * s + par
                if t < NMAX:
                    X3[:, t] = x
                x = x @ M1.T + rng.standard_normal((R, D)) @ Lh.T
        z0 = np.empty((len(u_all), D)); dz0 = np.empty((len(u_all), D))
        for r, j in enumerate(JJ):
            a, b = off[r], off[r + 1]
            z0[a:b] = X3[r, j]
            dz0[a:b] = (X3[r, j + 2] - X3[r, j]) / DT
        sim = dict(sim_tpl); sim["z"] = z0
        Xd, pen = _design(sim, g)
        G = Xd.T @ Xd
        G = G + fit["alpha"] * (np.trace(G) / Xd.shape[1]) * np.diag(pen.astype(float))
        G = G + 1e-10 * np.eye(Xd.shape[1]) * max(1.0, np.trace(G))
        G0 = Xd.T @ Xd + 1e-10 * np.eye(Xd.shape[1]) * max(1.0, np.trace(Xd.T @ Xd))
        rhs0 = Xd.T @ dz0
        rhs1 = Xd.T @ (z0 * g[:, None])
        for d in delta_grid:
            B = np.linalg.solve(G, rhs0 - d * rhs1)
            B0 = np.linalg.solve(G0, rhs0 - d * rhs1)
            est[float(d)].append(float(-np.trace(B[D + 2: 2 * D + 2].T) / D))
            est_unpen[float(d)].append(float(-np.trace(B0[D + 2: 2 * D + 2].T) / D))
    return dict(ridge=est, unpenalised=est_unpen)


# ============================================================ participant driver
def _one(sub, n_perm, n_sims, arms_cfg, t0):
    S = load_cache(sub)
    umap, precall = u_maps(S)
    row = dict(subject=sub, n_sequences=S["n"],
               n_windows=int(sum(k.sum() for k in S["keep"])),
               span=dict((str(L), float(np.nansum(precall[L]))) for L in precall))

    # ---------- Part 1: held-out displacement, primary DiD, nulls
    T = _held_out_table(S, "F_dist")
    if T is None:
        return dict(subject=sub, error="no held-out table")
    prim = {}
    for un in U_NAMES:
        m = {L: umap[L][un] for L in umap}
        prim[un] = did_slopes(T["p"], T["k"], T["L"], T["arm"], m)
    rng = np.random.default_rng(SEED + (abs(hash(sub)) % 100000))
    # N1: permute the slot -> u map within each sequence (both arms)
    nulls = {un: [] for un in U_NAMES}
    seq_ids = T["seq"]
    seq_list = np.unique(seq_ids)
    seq_slice = {s: np.where(seq_ids == s)[0] for s in seq_list}
    for _ in range(n_perm):
        kperm = T["k"].copy()
        for s in seq_list:
            idx = seq_slice[s]
            Lv = int(T["L"][idx[0]])
            pm = rng.permutation(np.arange(1, Lv + 1))
            kperm[idx] = pm[np.clip(T["k"][idx], 1, Lv) - 1]
        for un in U_NAMES:
            m = {L: umap[L][un] for L in umap}
            nulls[un].append(did_slopes(T["p"], kperm, T["L"], T["arm"], m, fast=True)["did"])
    # N3: one-slot circular shift
    kshift = T["k"].copy()
    for s in seq_list:
        idx = seq_slice[s]
        Lv = int(T["L"][idx[0]])
        kshift[idx] = (np.clip(T["k"][idx], 1, Lv) % Lv) + 1
    shift = {un: did_slopes(T["p"], kshift, T["L"], T["arm"], {L: umap[L][un] for L in umap})
             for un in U_NAMES}
    # design-block bootstrap of the DiD within participant
    blocks = np.unique(T["fold"])
    bb = []
    for _ in range(400):
        pick = rng.choice(blocks, len(blocks), replace=True)
        sel = np.concatenate([np.where(T["fold"] == b)[0] for b in pick])
        try:
            bb.append(did_slopes(T["p"][sel], T["k"][sel], T["L"][sel], T["arm"][sel],
                                 {L: umap[L][PRIMARY_U] for L in umap}, fast=True)["did"])
        except Exception:  # noqa: BLE001
            pass
    row["position"] = dict(
        primary=prim,
        null_perm={un: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)),
                            p_one_sided=float((1 + np.sum(np.asarray(v) >= prim[un]["did"])) / (1 + len(v))))
                   for un, v in nulls.items()},
        null_shift={un: shift[un]["did"] for un in U_NAMES},
        block_bootstrap_ci=[float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))] if bb else None,
        n_windows_arm=[int((T["arm"] == 0).sum()), int((T["arm"] == 1).sum())],
        axis_consistency=float(np.mean([abs(T["axes"][i] @ T["axes"][j])
                                        for i in range(len(T["axes"])) for j in range(i + 1, len(T["axes"]))])))
    row["_null_draws"] = {un: [float(x) for x in v] for un, v in nulls.items()}

    # ---------- Part 2: input-dependent drift, both arms
    sp_all = RefSpace(D).fit(np.vstack([S["F_dist"][i][S["keep"][i]] for i in range(S["n"])
                                        if S["cond"][i] == 0]))
    Z = [sp_all.transform(F) for F in S["F_dist"]]
    S["Zy"] = [sp_all.transform(F) for F in S["F_dyn"]]
    # fixed out-of-sample axis in this space: mean of the 9 leave-one-block-out LDA axes
    ax = []
    for b in sorted(set(S["block"])):
        Z0 = np.vstack([Z[i][S["keep"][i]] for i in range(S["n"])
                        if S["block"][i] != b and S["cond"][i] == 0])
        Z1 = np.vstack([Z[i][S["keep"][i]] for i in range(S["n"])
                        if S["block"][i] != b and S["cond"][i] == 1])
        ax.append(lda_axis(Z0, Z1))
    w_fixed = np.mean(ax, axis=0)
    w_fixed = w_fixed / np.linalg.norm(w_fixed)

    row["drift"] = {}
    fits = {}
    for arm, nm in ((1, "memory"), (0, "control")):
        inc = _increments(Z, S, {L: umap[L][PRIMARY_U] for L in umap}, PRIMARY_U, arm)
        if inc is None:
            continue
        f = _fit_drift(inc)
        f["u_of_slot"] = {(int(L), int(k)): float(umap[L][PRIMARY_U][k - 1])
                          for L in umap for k in range(1, L + 1)}
        f["u_centre"] = f["u_mean"]
        fits[nm] = (f, inc)
        grid = np.linspace(f["u_min"], f["u_max"], 13)
        br = _branch(f, w_fixed, grid, f["u_mean"])
        row["drift"][nm] = dict(
            n_increments=f["n_inc"], ridge_alpha=f["alpha"],
            r2_lobo=dict(u_free=f["r2_free"], plus_xstar=f["r2_xstar"], plus_A_of_u=f["r2_A"]),
            r2_by_alpha=f["r2_by_alpha"],
            branch=br,
            n_stable_grid=int(sum(b["stable"] for b in br)), n_grid=len(br),
            all_stable=bool(all(b["stable"] for b in br)),
            displacement_spearman=float(sst.spearmanr([b["u"] for b in br],
                                                      [b["dist_from_umin"] for b in br]).statistic),
            proj_spearman=float(sst.spearmanr([b["u"] for b in br], [b["proj"] for b in br]).statistic),
            proj_range=[br[0]["proj"], br[-1]["proj"]],
            proj_delta=float(br[-1]["proj"] - br[0]["proj"]),
            dist_at_umax=float(br[-1]["dist_from_umin"]),
            lam=_lam_slope(f, f["u_min"], f["u_max"], f["u_mean"]),
            continuation=_continuation(f, f["u_mean"], f["u_min"], f["u_max"]),
            u_range=[f["u_min"], f["u_max"]])
        # (b) lag spectrum / observation-noise floor (needed to correct (a) and (c))
        ls = _lag_spectrum(Z, S, arm)
        row["drift"][nm]["lag_spectrum"] = ls
        nv = np.asarray(ls["noise_var_per_dir"], float) if ls else None
        if nv is not None and not np.isfinite(nv).all():
            nv = None
        # (a) lambda and D per (L, u) cell, raw and noise-corrected
        cells = _bin_cells(inc, noise_var=nv)
        row["drift"][nm]["cells"] = cells
        row["drift"][nm]["log_decomposition"] = _log_decomposition(cells)
        row["drift"][nm]["log_decomposition_noise_corrected"] = _log_decomposition(
            cells, key_lam="lambda_g_nc", key_D="D_g_nc")
        # (c) fit_ou + bias_correct on the non-overlapping dyn series
        row["drift"][nm]["fit_ou_dyn"] = _fit_ou_dyn(S, arm, noise_var=nv)
        # u1 companion for the lambda slope (per item)
        inc1 = _increments(Z, S, {L: umap[L]["u1"] for L in umap}, "u1", arm)
        f1 = _fit_drift(inc1)
        row["drift"][nm]["lam_u1"] = _lam_slope(f1, f1["u_min"], f1["u_max"], f1["u_mean"])
        row["drift"][nm]["r2_lobo_u1"] = dict(u_free=f1["r2_free"], plus_xstar=f1["r2_xstar"],
                                              plus_A_of_u=f1["r2_A"])
        fits[nm + "_u1"] = (f1, inc1)

    # ---------- Part 3: matched simulation for the lambda equivalence bound
    sims = {}
    for nm in ("memory", "control"):
        if nm not in fits or n_sims <= 0:
            continue
        f, inc = fits[nm]
        r = _sim_null(f, inc, S, 1 if nm == "memory" else 0, n_sims, arms_cfg["delta_grid"],
                      np.random.default_rng(SEED + 7 + (abs(hash(sub + nm)) % 100000)))
        if r is not None:
            sims[nm] = {est_kind: {str(k): v for k, v in d.items()} for est_kind, d in r.items()}
    row["_lambda_sims"] = sims

    # ---------- Part 5: robustness arms, primary statistic only
    rob = {}
    m3 = {L: umap[L][PRIMARY_U] for L in umap}

    def did_of(T2):
        return None if T2 is None else did_slopes(T2["p"], T2["k"], T2["L"], T2["arm"], m3)

    # degree-3 within-condition session detrend (on a blocks-like list)
    blk = [dict(cond=S["cond"][i], t_start=S["t_start"][i], F_dist=np.asarray(S["F_dist"][i], float))
           for i in range(S["n"])]
    detrend_within_condition(blk)
    S_det = dict(S); S_det["F_dist"] = [b["F_dist"] for b in blk]
    rob["detrend_deg3"] = did_of(_held_out_table(S_det, "F_dist"))
    S_ta = dict(S); S_ta["F_dist"] = _band_subset(S, [0, 1])
    rob["theta_alpha_only"] = did_of(_held_out_table(S_ta, "F_dist"))
    S_ng = dict(S); S_ng["F_dist"] = _band_subset(S, [0, 1, 2])
    rob["gamma_free"] = did_of(_held_out_table(S_ng, "F_dist"))
    def cov_space(which):
        cov = []
        for i in range(S["n"]):
            cols = []
            if "n" in which:
                cols += [S["nuis"][i], S["resid"][i]]
            if "p" in which and S["periph"][i] is not None:
                cols.append(S["periph"][i])
            if not cols:
                return None
            C = np.column_stack(cols)
            cov.append(np.asarray(C, float))
        ncol = min(c.shape[1] for c in cov)
        cov = [c[:, :ncol] for c in cov]
        allC = np.vstack(cov)
        keep_cols = np.isfinite(allC).all(axis=0) & (allC.std(axis=0) > 1e-12)
        if keep_cols.sum() == 0:
            return None
        return [c[:, keep_cols] for c in cov]

    for tag, which in (("nuisance", "n"), ("peripheral", "p"), ("nuisance_peripheral", "np")):
        cv = cov_space(which)
        rob["residualised_" + tag] = None if cv is None else did_of(_held_out_table(S, "F_dist", resid=cv))
    rob["has_pupil"] = bool(S["meta"]["has_pupil"])
    rob["n_periph_cols"] = (0 if S["periph"][0] is None else int(S["periph"][0].shape[1]))
    rob["drop_first_2s"] = did_of(_held_out_table(S, "F_dist", drop2=True))
    rob["k_le_9"] = did_of(_held_out_table(S, "F_dist", kmax=9))
    rob["nominal_soa_labels"] = None
    S_pub = dict(S); S_pub["slot_true"] = S["slot_pub"]
    Tp = _held_out_table(S_pub, "F_dist")
    rob["nominal_soa_labels"] = None if Tp is None else did_slopes(Tp["p"], Tp["k"], Tp["L"],
                                                                     Tp["arm"], m3)
    row["robustness"] = rob
    row["minutes"] = round((time.time() - t0) / 60, 2)
    return row


# ============================================================ group
def _boot_ci(x, n=10000, seed=SEED):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if len(x) < 3:
        return None
    rng = np.random.default_rng(seed)
    b = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return dict(mean=float(x.mean()), ci95=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                sd=float(x.std(ddof=1)), sem=float(x.std(ddof=1) / np.sqrt(len(x))), n=int(len(x)),
                median=float(np.median(x)))


def _sign(x, alt="greater"):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if len(x) == 0:
        return None
    k = int((x > 0).sum())
    return dict(n_positive=k, n=int(len(x)),
                p=float(sst.binomtest(k, len(x), 0.5, alternative=alt).pvalue),
                p_two_sided=float(sst.binomtest(k, len(x), 0.5).pvalue))


def summarize(rows, delta_grid):
    ok = [r for r in rows if "error" not in r and "position" in r]
    S = dict(n_subjects=len(ok), n_excluded=len(rows) - len(ok))

    # ---- primary DiD
    pos = {}
    for un in U_NAMES:
        did = np.array([r["position"]["primary"][un]["did"] for r in ok])
        dcm = np.array([r["position"]["primary"][un]["did_cellmatched"] for r in ok])
        bm = np.array([r["position"]["primary"][un]["slope_memory"] for r in ok])
        bc = np.array([r["position"]["primary"][un]["slope_control"] for r in ok])
        nd = min(len(r["_null_draws"][un]) for r in ok)
        NUL = np.array([r["_null_draws"][un][:nd] for r in ok])
        t_obs = float(did.mean())
        t_null = NUL.mean(axis=0)
        p_perm = float((1 + np.sum(t_null >= t_obs)) / (1 + len(t_null)))
        p_perm_2 = float((1 + np.sum(np.abs(t_null) >= abs(t_obs))) / (1 + len(t_null)))
        # per-subject calibrated z against its own permutation null
        zsub = np.array([(r["position"]["primary"][un]["did"] - r["position"]["null_perm"][un]["mean"])
                         / max(r["position"]["null_perm"][un]["sd"], 1e-12) for r in ok])
        pos[un] = dict(did=_boot_ci(did), did_cellmatched=_boot_ci(dcm),
                       slope_memory=_boot_ci(bm), slope_control=_boot_ci(bc),
                       sign_test=_sign(did), sign_test_cellmatched=_sign(dcm),
                       sign_test_memory=_sign(bm), sign_test_control=_sign(bc),
                       wilcoxon_p=float(sst.wilcoxon(did, alternative="greater").pvalue),
                       t_test_p=float(sst.ttest_1samp(did, 0, alternative="greater").pvalue),
                       permutation=dict(n_draws=nd, group_null_mean=float(t_null.mean()),
                                        group_null_sd=float(t_null.std(ddof=1)),
                                        p_one_sided=p_perm, p_two_sided=p_perm_2,
                                        null_ci95=[float(np.percentile(t_null, 2.5)),
                                                   float(np.percentile(t_null, 97.5))]),
                       calibrated_z=_boot_ci(zsub),
                       shift_null=_boot_ci([r["position"]["null_shift"][un] for r in ok]))
    S["position"] = pos
    S["primary_u"] = PRIMARY_U

    # ---- drift / branch / lambda
    dr = {}
    for nm in ("memory", "control"):
        have = [r for r in ok if nm in r.get("drift", {})]
        if not have:
            continue
        g = lambda f: np.array([f(r["drift"][nm]) for r in have], float)  # noqa: E731
        dr[nm] = dict(
            n=len(have),
            n_increments=_boot_ci(g(lambda d: d["n_increments"])),
            ridge_alpha_median=float(np.median(g(lambda d: d["ridge_alpha"]))),
            r2_u_free=_boot_ci(g(lambda d: d["r2_lobo"]["u_free"])),
            r2_plus_xstar=_boot_ci(g(lambda d: d["r2_lobo"]["plus_xstar"])),
            r2_plus_A=_boot_ci(g(lambda d: d["r2_lobo"]["plus_A_of_u"])),
            delta_r2_xstar=_boot_ci(g(lambda d: d["r2_lobo"]["plus_xstar"] - d["r2_lobo"]["u_free"])),
            delta_r2_A=_boot_ci(g(lambda d: d["r2_lobo"]["plus_A_of_u"] - d["r2_lobo"]["plus_xstar"])),
            sign_delta_r2_xstar=_sign(g(lambda d: d["r2_lobo"]["plus_xstar"] - d["r2_lobo"]["u_free"])),
            sign_delta_r2_A=_sign(g(lambda d: d["r2_lobo"]["plus_A_of_u"] - d["r2_lobo"]["plus_xstar"])),
            all_stable_over_observed_u=int(sum(d for d in g(lambda x: x["all_stable"]))),
            displacement_spearman=_boot_ci(g(lambda d: d["displacement_spearman"])),
            sign_displacement=_sign(g(lambda d: d["displacement_spearman"])),
            displacement_spearman_note="||x*(u) - x*(u_min)|| is a norm that starts at 0, so for any "
                                       "monotone branch this Spearman is +1 by construction; the "
                                       "non-degenerate form is proj_delta below.",
            proj_spearman=_boot_ci(g(lambda d: d["proj_spearman"])),
            sign_proj=_sign(g(lambda d: d["proj_spearman"])),
            proj_delta=_boot_ci(g(lambda d: d["proj_delta"])),
            sign_proj_delta=_sign(g(lambda d: d["proj_delta"])),
            dist_at_umax=_boot_ci(g(lambda d: d["dist_at_umax"])),
            dlambda_mean_du_u3=_boot_ci(g(lambda d: d["lam"]["dlambda_mean_du"])),
            dlambda_max_du_u3=_boot_ci(g(lambda d: d["lam"]["dlambda_max_du"])),
            dlambda_mean_du_u1=_boot_ci(g(lambda d: d["lam_u1"]["dlambda_mean_du"])),
            dlambda_max_du_u1=_boot_ci(g(lambda d: d["lam_u1"]["dlambda_max_du"])),
            sign_dlambda_mean_u1=_sign(g(lambda d: d["lam_u1"]["dlambda_mean_du"])),
            sign_dlambda_max_u1=_sign(g(lambda d: d["lam_u1"]["dlambda_max_du"])),
            lambda_mean_at_umin=_boot_ci(g(lambda d: d["lam"]["lambda_mean_at_umin"])),
            lambda_mean_at_umax=_boot_ci(g(lambda d: d["lam"]["lambda_mean_at_umax"])),
            lambda_max_at_umin=_boot_ci(g(lambda d: d["lam"]["lambda_max_at_umin"])),
            lambda_max_at_umax=_boot_ci(g(lambda d: d["lam"]["lambda_max_at_umax"])),
            continuation_u_crit_inside_or_near=dict(
                n_with_fold_up=int(sum(1 for r in have
                                       if r["drift"][nm]["continuation"]["up"]["u_crit"] is not None)),
                median_multiple_up=float(np.nanmedian([
                    r["drift"][nm]["continuation"]["up"]["multiple_of_observed_range"]
                    if r["drift"][nm]["continuation"]["up"]["u_crit"] is not None else np.nan
                    for r in have]))),
            fit_ou_dyn={k: _boot_ci([r["drift"][nm]["fit_ou_dyn"].get(k, np.nan) for r in have
                                     if r["drift"][nm]["fit_ou_dyn"]])
                        for k in ("lambda_mean_bc", "lambda_mean_raw", "lambda_g", "D_g",
                                  "spread_observed", "spread_from_D_lambda", "ratio_check",
                                  "lambda_g_nc", "D_g_nc", "spread_g_nc", "D_iso")},
            lag_spectrum=dict(
                noise_var_fit=_boot_ci([r["drift"][nm]["lag_spectrum"].get("noise_var_fit", np.nan)
                                        for r in have if r["drift"][nm]["lag_spectrum"]]),
                noise_fraction=_boot_ci([r["drift"][nm]["lag_spectrum"].get("noise_fraction", np.nan)
                                         for r in have if r["drift"][nm]["lag_spectrum"]]),
                lambda_corrected=_boot_ci([r["drift"][nm]["lag_spectrum"].get("lambda_corrected", np.nan)
                                           for r in have if r["drift"][nm]["lag_spectrum"]]),
                D_corrected=_boot_ci([r["drift"][nm]["lag_spectrum"].get("D_corrected", np.nan)
                                      for r in have if r["drift"][nm]["lag_spectrum"]]),
                D_raw_lag2=_boot_ci([r["drift"][nm]["lag_spectrum"].get("D_raw_lag2", np.nan)
                                     for r in have if r["drift"][nm]["lag_spectrum"]]),
                noise_var_linear=_boot_ci([r["drift"][nm]["lag_spectrum"].get("noise_var_linear", np.nan)
                                           for r in have if r["drift"][nm]["lag_spectrum"]])),
            log_decomposition={
                **{k: _boot_ci([r["drift"][nm]["log_decomposition"][k] for r in have
                                if r["drift"][nm]["log_decomposition"]])
                   for k in ("dlogD_du", "dloglambda_du", "dlogspread_du", "pct_from_D", "pct_from_lambda")},
                "sign_dlogspread": _sign([r["drift"][nm]["log_decomposition"]["dlogspread_du"] for r in have
                                          if r["drift"][nm]["log_decomposition"]]),
                "sign_dlogD": _sign([r["drift"][nm]["log_decomposition"]["dlogD_du"] for r in have
                                     if r["drift"][nm]["log_decomposition"]]),
                "sign_dloglambda": _sign([r["drift"][nm]["log_decomposition"]["dloglambda_du"] for r in have
                                          if r["drift"][nm]["log_decomposition"]]),
                "identity_residual_max": float(np.nanmax(
                    [abs(r["drift"][nm]["log_decomposition"]["identity_residual"]) for r in have
                     if r["drift"][nm]["log_decomposition"]] or [np.nan])),
                "ratio_check_median": float(np.nanmedian([c["ratio_check"] for r in have
                                                          for c in r["drift"][nm]["cells"]] or [np.nan])),
                "ratio_check_ci": _boot_ci([c["ratio_check"] for r in have
                                            for c in r["drift"][nm]["cells"]])},
            log_decomposition_noise_corrected={
                k: _boot_ci([r["drift"][nm]["log_decomposition_noise_corrected"][k] for r in have
                             if r["drift"][nm].get("log_decomposition_noise_corrected")])
                for k in ("dlogD_du", "dloglambda_du", "dlogspread_du")})
    S["drift"] = dr

    # ---- lambda equivalence bound from the matched simulation
    bnd = {}
    for nm in ("memory", "control"):
        have = [r for r in ok if r.get("_lambda_sims", {}).get(nm)]
        if not have:
            continue
        obs_key = dict(ridge="dlambda_mean_du", unpenalised="dlambda_mean_du_unpenalised")
        bnd[nm] = {}
        for kind in ("ridge", "unpenalised"):
            keys = sorted(have[0]["_lambda_sims"][nm][kind], key=float)
            nrep = min(len(r["_lambda_sims"][nm][kind][k]) for r in have for k in keys)
            grp = {k: np.array([r["_lambda_sims"][nm][kind][k][:nrep] for r in have],
                               float).mean(axis=0) for k in keys}
            null = grp[keys[0]]
            thr = float(np.percentile(np.abs(null - null.mean()), 95.0))
            power = {k: float(np.mean(np.abs(grp[k] - null.mean()) > thr)) for k in keys}
            dg = [float(k) for k in keys]
            # attenuation of the estimator: E[est] vs the injected true delta
            means = np.array([grp[k].mean() for k in keys])
            atten = float(np.polyfit(np.array(dg), means, 1)[0])

            sd_null = float(null.std(ddof=1))

            def smallest(p_target):
                for k, d in zip(keys, dg):
                    if power[k] >= p_target and d > 0:
                        return d
                return None

            def analytic(p_target):
                """est ~ N(attenuation*delta, sd_null); detected when |est - null mean| > thr."""
                if abs(atten) < 1e-12 or sd_null <= 0:
                    return None
                return float((thr + sd_null * sst.norm.ppf(p_target)) / abs(atten))
            obs = np.array([r["drift"][nm]["lam"][obs_key[kind]] for r in have], float)
            obs1 = np.array([r["drift"][nm]["lam_u1"][obs_key[kind]] for r in have], float)
            obs_ci, obs1_ci = _boot_ci(obs), _boot_ci(obs1)
            true_scale = None
            if obs_ci and abs(atten) > 1e-9:
                true_scale = dict(mean=obs_ci["mean"] / atten,
                                  ci95=sorted([obs_ci["ci95"][0] / atten, obs_ci["ci95"][1] / atten]))
            true_scale_u1 = None
            if obs1_ci and abs(atten) > 1e-9:
                true_scale_u1 = dict(mean=obs1_ci["mean"] / atten,
                                     ci95=sorted([obs1_ci["ci95"][0] / atten, obs1_ci["ci95"][1] / atten]))
            bnd[nm][kind] = dict(
                n=len(have), n_replicates=nrep, delta_grid=dg,
                null_sd_group=float(null.std(ddof=1)), detection_threshold_95=thr,
                power=power, estimator_attenuation=atten,
                min_resolvable_80pct=smallest(0.80), excluded_at_95pct=smallest(0.95),
                min_resolvable_80pct_interpolated=analytic(0.80),
                excluded_at_95pct_interpolated=analytic(0.95),
                observed_group_mean_u3=(None if obs_ci is None else obs_ci["mean"]),
                observed_inside_null_band=(None if obs_ci is None
                                           else bool(abs(obs_ci["mean"] - null.mean()) <= thr)),
                observed_dlambda_mean_du_u3=obs_ci, observed_dlambda_mean_du_u1=obs1_ci,
                observed_true_scale_u3=true_scale, observed_true_scale_u1=true_scale_u1,
                sign_test_u3=_sign(obs), sign_test_u1=_sign(obs1))
        bnd[nm]["primary_estimator"] = "ridge (LOBO-selected)"
    S["lambda_bound"] = bnd

    # ---- robustness
    rb = {}
    keys = ["detrend_deg3", "theta_alpha_only", "gamma_free", "residualised_nuisance",
            "residualised_peripheral", "residualised_nuisance_peripheral",
            "drop_first_2s", "k_le_9", "nominal_soa_labels"]
    for k in keys:
        v = [r["robustness"][k]["did"] for r in ok if r.get("robustness", {}).get(k)]
        vm = [r["robustness"][k]["slope_memory"] for r in ok if r.get("robustness", {}).get(k)]
        vc = [r["robustness"][k]["slope_control"] for r in ok if r.get("robustness", {}).get(k)]
        if v:
            rb[k] = dict(did=_boot_ci(v), sign_test=_sign(v), slope_memory=_boot_ci(vm),
                         slope_control=_boot_ci(vc))
    rb["n_with_pupil"] = int(sum(1 for r in ok if r.get("robustness", {}).get("has_pupil")))
    S["robustness"] = rb

    # ---- span / capacity
    S["span"] = {L: _boot_ci([r["span"][L] for r in ok if L in r["span"]])
                 for L in ("5", "9", "13")}
    return S


def main():
    t_start = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--perm", type=int, default=1000)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    delta_grid = [0.0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1]
    cfg = dict(delta_grid=delta_grid)
    subs = sorted(p.stem for p in CACHE.glob("sub-*.npz"))
    if a.subjects:
        subs = [s for s in subs if s in set(a.subjects)]
    log(f"{len(subs)} participants | perm {a.perm} | sims {a.sims} | workers {a.workers}")
    rows = []
    tasks = [(s, a.perm, a.sims, cfg) for s in subs]
    if a.workers <= 1:
        for t in tasks:
            r = one_subject(t); rows.append(r)
            log(f"  {r['subject']}: {'ERROR ' + r['error'][:150] if 'error' in r else str(r.get('minutes')) + ' min'}")
    else:
        import multiprocessing as mp
        with mp.Pool(a.workers) as pool:
            for r in pool.imap_unordered(one_subject, tasks):
                rows.append(r)
                log(f"  {r['subject']}: "
                    f"{'ERROR ' + r['error'][:200] if 'error' in r else 'did=' + format(r['position']['primary'][PRIMARY_U]['did'], '.4f') + ' | ' + str(r.get('minutes')) + ' min'}")
    rows.sort(key=lambda r: r["subject"])
    summ = summarize(rows, delta_grid)
    res = dict(
        analysis="position and dynamical stability in the same participants "
                 "and state space (ds003838 EEG digit span, n = 64)",
        design={
            "input_dependent_system": "dx = A(u)(x - m(u)) dt + noise, "
                                      "A(u) = A_0 + u A_1, nested models compared by cross-validated "
                                      "one-step prediction; branch continued until an eigenvalue "
                                      "crosses zero or u reaches 3x the observed range "
                                      "-> keys drift.*.r2_lobo, drift.*.branch, drift.*.continuation",
            "decision_rule": "POSITIVE iff a single stable branch is tracked across the whole "
                             "observed u in a majority of participants with max Re lambda(A(u)) < 0 "
                             "(sign test) AND ||x*(u) - x*(u_min)|| increasing with u "
                             "-> keys drift.*.all_stable_over_observed_u, drift.*.sign_displacement",
            "lambda_by_load": "lambda estimated separately across the graded load, "
                              "reported as an equivalence bound with the margin taken from a matched "
                              "simulation, NOT from the observed effect -> key lambda_bound",
            "D_lambda": "lambda and D separated along the same directions as the "
                        "stationary spread, with sqrt(D/lambda) checked against the observed spread "
                        "-> keys drift.*.cells (ratio_check), drift.*.log_decomposition, "
                        "drift.*.lag_spectrum, drift.*.fit_ou_dyn",
            "primary_statistic": "difference in differences: within-set-size slope of the held-out "
                                 "displacement on u in MEMORY minus the slope in CONTROL at matched "
                                 "serial position -> key position.u3.did",
            "primary_u": "u3 = expected items held (cumulative per-item recall probability). "
                         "u1 = serial position is the pre-declared linear alternative. "
                         "Set size L is a fixed effect and is never part of u.",
            "circularity": "u3 is built from recall, so recall is never used as an outcome here. "
                           "The task axis w is fit per fold on the 8 training design blocks pooling "
                           "ALL u; displacement is read only in the held-out block.",
            "nulls": "within-sequence permutation of the slot -> u map (1000 draws, preserves sequence, "
                     "block, set size, windows per slot); the control arm as the empirical null arm; "
                     "a one-slot circular shift of u. Nothing is compared to zero.",
            "unit": "participant (n = 64). Group intervals resample participants; the within-participant "
                    "interval resamples design blocks. Never increments."},
        seed=SEED, packages=pkg_versions(), n_perm=a.perm, n_sims=a.sims,
        lambda_delta_grid=delta_grid,
        state_space=dict(sfreq=250.0, bands={"theta": [4, 8], "alpha": [8, 12], "beta": [12, 30],
                                             "gamma": [30, 45]},
                         win_s=1.0, step_dist_s=0.5, step_dyn_s=1.0, n_features=256, d=D,
                         reference="control (listen-only) windows of the TRAINING design blocks",
                         increment="lag-2 of the 0.5-s series (adjacent, non-overlapping, dt = 1.0 s), "
                                   "both windows in the SAME item slot",
                         slots="true per-item onsets from events.tsv"),
        summary=summ,
        rows=[{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
        wall_time_s=round(time.time() - t_start, 1))
    tag = f"_{a.tag}" if a.tag else ""
    out_json = Path(paths.out("analyses", "task_location", "out", f"digit_span_position_analysis{tag}.json"))
    out_json.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    ok = [r for r in rows if "error" not in r]
    with open(paths.out("analyses", "task_location", "out", f"digit_span_participants{tag}.csv"), "w",
              encoding="utf-8") as f:
        f.write("subject,did_u3,did_u1,slope_memory_u3,slope_control_u3,did_cellmatched_u3,"
                "perm_p_u3,dlambda_mean_du_u3_memory,dlambda_mean_du_u1_memory,"
                "dlambda_mean_du_u1_control,r2_ufree_mem,r2_xstar_mem,r2_A_mem,n_increments_mem\n")
        for r in ok:
            d = r["drift"].get("memory", {})
            c = r["drift"].get("control", {})
            f.write(",".join(str(x) for x in [
                r["subject"], r["position"]["primary"]["u3"]["did"], r["position"]["primary"]["u1"]["did"],
                r["position"]["primary"]["u3"]["slope_memory"], r["position"]["primary"]["u3"]["slope_control"],
                r["position"]["primary"]["u3"]["did_cellmatched"],
                r["position"]["null_perm"]["u3"]["p_one_sided"],
                d.get("lam", {}).get("dlambda_mean_du"), d.get("lam_u1", {}).get("dlambda_mean_du"),
                c.get("lam_u1", {}).get("dlambda_mean_du"),
                d.get("r2_lobo", {}).get("u_free"), d.get("r2_lobo", {}).get("plus_xstar"),
                d.get("r2_lobo", {}).get("plus_A_of_u"), d.get("n_increments")]) + "\n")
    with open(paths.out("analyses", "task_location", "out", f"digit_span_rate_diffusion_cells{tag}.csv"), "w",
              encoding="utf-8") as f:
        f.write("subject,arm,L,bin,n,u_mean,lambda_g,D_g,spread_observed,spread_from_D_lambda,"
                "ratio_check,lambda_g_nc,D_g_nc,spread_g_nc\n")
        for r in ok:
            for nm in ("memory", "control"):
                for c in r.get("drift", {}).get(nm, {}).get("cells", []):
                    f.write(",".join(str(x) for x in [r["subject"], nm, c["L"], c["bin"], c["n"],
                                                      c["u_mean"], c["lambda_g"], c["D_g"],
                                                      c["spread_observed"], c["spread_from_D_lambda"],
                                                      c["ratio_check"], c["lambda_g_nc"], c["D_g_nc"],
                                                      c["spread_g_nc"]]) + "\n")
    log(json.dumps(summ["position"][PRIMARY_U], indent=1, default=float))
    log(f"wall {res['wall_time_s']} s -> {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
