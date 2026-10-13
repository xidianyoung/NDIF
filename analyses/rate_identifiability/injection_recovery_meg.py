# -*- coding: utf-8 -*-
"""Injection recovery in the HCP MEG working-memory cohort, under a design that gives ONE lag-1 pair per trial.

Cohort.  83 participants, the per-trial MEG state (common sensors, 1-s windows of log10 band power in
theta/alpha/beta/gamma, robust z + PCA d = 8 fitted on the pooled 0-back windows), read from the per-trial MEG
state cache (NDIF_MEG_CACHE, written by analyses/cohort_states/hcp_meg_states.py) as the lag-1
pairs (X, Y) at window centres 0.5 s and 1.5 s after stimulus onset, separately for 0-back and 2-back, and the
load direction v.  Nothing is reloaded or refitted.

WHAT THIS DESIGN PERMITS
  * P2 and P3 - the propagator, its eigenvalue moduli and its innovation covariance are estimable from the
    across-trial regression of Y on X (cog_fit_lib.fit_pairs, the estimator matched to one pair per block;
    fit_cell is INADMISSIBLE here because centring a two-window block on its own mean forces M = -I).  D is
    taken from the innovation covariance Q/(2 dt), never as lambda times a spread.
  * P1 - the correlation matrix of the ACROSS-TRIAL state at the FIRST latency, with the matched ceiling from
    split halves of the trials within a condition.
  * NOT P4 - there is no within-trial series, so no autocorrelation function beyond lag 1, and no temporal
    decimation.

Calibration (Fig. 3d): INJ-T exact time rescaling of the pair design (c = 1.05-2.0), INJ-D pure drive, INJ-R
pure reshaping.

Inputs   NDIF_MEG_CACHE (<subject>.npz)
Outputs  analyses/rate_identifiability/out/injection_recovery_meg.json, injection_recovery_meg_subjects.csv,
         injection_recovery_meg_calibration.json (read by injection_recovery_report.py)
Reported Fig. 3d (innovation rescaling, HCP MEG)
Runtime  about 1 min on one desktop core
Usage    python analyses/rate_identifiability/injection_recovery_meg.py [--boot 200] [--cboot 80]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import injection_lib as L                                                  # noqa: E402
import cog_fit_lib as R                                                    # noqa: E402
import cog_rescaling_lib as H                                                     # noqa: E402

CACHE = L.MEG_CACHE
DT = 1.0
SEED = 20260922

def log(m):
    print(time.strftime("[%H:%M:%S] ") + str(m), flush=True)


def load():
    out = {}
    for f in sorted(CACHE.glob("*.npz")):
        z = np.load(f)
        out[f.stem] = dict(X0=z["X0"].astype(float), Y0=z["Y0"].astype(float),
                           X1=z["X1"].astype(float), Y1=z["Y1"].astype(float),
                           v=z["v"].astype(float))
    return out


def equalise(X0, Y0, X1, Y1, rng):
    n = min(len(X0), len(X1))
    i0 = rng.choice(len(X0), n, replace=False)
    i1 = rng.choice(len(X1), n, replace=False)
    return X0[i0], Y0[i0], X1[i1], Y1[i1]


# --------------------------------------------------------------------------- injections, pair design
def inject_rescale_pairs(X, Y, c, gam=None, dt=1.0, rng=None, boot=0):
    """An exact time rescaling of a ONE-PAIR-PER-BLOCK design.

    Fit (M, Q) on the real pairs, take P as the empirical across-trial covariance of X, form
    M' = eig_power(M, c) and Q' = P - M' P M'^T so the stationary distribution is unchanged, and
    rebuild Y' = M' X + e' from the REAL residuals recoloured to Q'.  X is untouched, so the
    first-latency distribution - which is what P1 reads in this cohort - is identical by
    construction and the truth for P1 is exactly zero.
    """
    f = R.fit_pairs(X, Y, dt, n_boot=boot, rng=rng)
    if f is None or not f["stable"]:
        return None, None
    M, Q, P = f["M"], f["Q"], f["P_emp"]
    Mn, _ = H.eig_power(M, c if gam is None else gam)
    if np.max(np.abs(np.linalg.eigvals(Mn))) >= 0.999:
        return None, None
    Qn = P - Mn @ P @ Mn.T
    Qn = 0.5 * (Qn + Qn.T)
    if np.min(np.linalg.eigvalsh(Qn)) <= 0:
        return None, None
    T = H._sqrtm_psd(Qn) @ H._isqrtm_psd(Q)
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    e = (Yc - Xc @ M.T) @ T.T
    Yn = Xc @ Mn.T + e
    ell0 = H.log_rates(M, dt)["ell"]
    ell1 = H.log_rates(Mn, dt)["ell"]
    truth = dict(dlog_lam=float(np.mean(ell1 - ell0)),
                 dlog_D=float(np.log(np.trace(Qn) / np.trace(Q))),
                 reshape_rms=float(np.sqrt(np.mean(((ell1 - ell0)
                                                    - np.mean(ell1 - ell0)) ** 2))))
    return (Xc, Yn), truth


def p1_pairs(X0, X1, rng):
    """Correlation matrix of the across-trial state at the first latency, against the matched
    within-condition split-half ceiling (trials split by parity, so any drift is shared)."""
    def C(A):
        s = A.std(0)
        return None if np.any(s <= 0) else np.corrcoef(A.T)
    a0, b0, a1, b1 = C(X0[0::2]), C(X0[1::2]), C(X1[0::2]), C(X1[1::2])
    if any(x is None for x in (a0, b0, a1, b1)):
        return None
    d_within = float(np.mean([H.corr_dist(a0, b0), H.corr_dist(a1, b1)]))
    d_between = float(np.mean([H.corr_dist(a0, b1), H.corr_dist(b0, a1)]))
    a_within = float(np.mean([H.corr_agree(a0, b0), H.corr_agree(a1, b1)]))
    a_between = float(np.mean([H.corr_agree(a0, b1), H.corr_agree(b0, a1)]))
    return dict(d_within=d_within, d_between=d_between,
                d_ratio=d_between / d_within if d_within > 0 else np.nan,
                d_excess=d_between - d_within, a_within=a_within, a_between=a_between,
                a_frac_ceiling=a_between / a_within if a_within > 0 else np.nan,
                d_full=H.corr_dist(C(X0), C(X1)))


def one(z, boot, rng, dt=DT):
    X0, Y0, X1, Y1 = equalise(z["X0"], z["Y0"], z["X1"], z["Y1"], rng)
    f0 = R.fit_pairs(X0, Y0, dt, n_boot=boot, rng=rng)
    f1 = R.fit_pairs(X1, Y1, dt, n_boot=boot, rng=rng)
    if f0 is None or f1 is None or not (f0["stable"] and f1["stable"]):
        return None
    out = dict(n_trials=int(len(X0)), d=int(f0["d"]),
               absev_lo=float(np.mean(np.abs(np.linalg.eigvals(f0["M"])))),
               absev_hi=float(np.mean(np.abs(np.linalg.eigvals(f1["M"])))),
               lam_lo=f0["lam_mean"], lam_hi=f1["lam_mean"],
               D_lo=f0["D_tr"], D_hi=f1["D_tr"],
               dlog_lam=float(np.log(f1["lam_mean"] / f0["lam_mean"])),
               dlog_lam_slowest=float(np.log(f1["lam_max"] / f0["lam_max"])),
               dlog_D=float(np.log(f1["D_tr"] / f0["D_tr"])),
               nonstat_lo=f0["nonstat_tr"], nonstat_hi=f1["nonstat_tr"])
    p2 = H.p2_decompose(f0["M"], f1["M"], dt)
    if p2 is not None:
        for k in ("dbar", "c_common", "reshape_rms", "frac_common", "frac_reshape",
                  "dlog_sd_ell", "match_cos", "match_dbar", "match_reshape_rms",
                  "match_frac_common", "rank_spearman"):
            out["p2_" + k] = p2.get(k, np.nan)
    # P2 ceiling: the same decomposition between split halves of the trials WITHIN a condition
    hs = []
    for X, Y in ((X0, Y0), (X1, Y1)):
        fa = R.fit_pairs(X[0::2], Y[0::2], dt, n_boot=boot, rng=rng)
        fb = R.fit_pairs(X[1::2], Y[1::2], dt, n_boot=boot, rng=rng)
        if fa is None or fb is None or not (fa["stable"] and fb["stable"]):
            continue
        q = H.p2_decompose(fa["M"], fb["M"], dt)
        if q is not None:
            hs.append(q)
    if hs:
        out["p2_ceiling_reshape_rms"] = float(np.mean([q["reshape_rms"] for q in hs]))
        out["p2_ceiling_dbar_abs"] = float(np.mean([abs(q["dbar"]) for q in hs]))
        out["p2_ceiling_rank_spearman"] = float(np.mean([q.get("rank_spearman", np.nan)
                                                         for q in hs]))
        if out["p2_ceiling_reshape_rms"] > 0 and np.isfinite(out.get("p2_reshape_rms", np.nan)):
            out["p2_reshape_over_ceiling"] = out["p2_reshape_rms"] / out["p2_ceiling_reshape_rms"]
    p1 = p1_pairs(X0, X1, rng)
    if p1 is not None:
        for k, v in p1.items():
            out["p1_" + k] = v
    # the one-dimensional rate on the load direction v stored in the cache
    v = z["v"] / np.linalg.norm(z["v"])
    for tag, (X, Y) in (("lo", (X0, Y0)), ("hi", (X1, Y1))):
        px, py = (X - X.mean(0)) @ v, (Y - Y.mean(0)) @ v
        r = float(np.sum(px * py) / np.sum(px ** 2))
        out[f"r1_{tag}"] = r
    if 0 < out["r1_lo"] < 1 and 0 < out["r1_hi"] < 1:
        out["dlog_lam_1d"] = float(np.log(np.log(out["r1_hi"]) / np.log(out["r1_lo"])))
    return out


def calib(data, boot, rng, dt=DT):
    cal = {}
    # ---- INJ-T: exact time rescaling
    cal["rescale"] = {}
    for c in (1.05, 1.1, 1.2, 1.5, 2.0):
        truth = float(np.log(c))
        dl, dd, db, tD, p1r = [], [], [], [], []
        for s, z in data.items():
            X, Y = z["X0"], z["Y0"]
            got, tr = inject_rescale_pairs(X, Y, c, dt=dt, rng=rng)
            if got is None:
                continue
            fa = R.fit_pairs(X, Y, dt, n_boot=boot, rng=rng)
            fb = R.fit_pairs(got[0], got[1], dt, n_boot=boot, rng=rng)
            if fa is None or fb is None or not (fa["stable"] and fb["stable"]):
                continue
            dl.append(np.log(fb["lam_mean"] / fa["lam_mean"]))
            dd.append(np.log(fb["D_tr"] / fa["D_tr"])); tD.append(tr["dlog_D"])
            q = H.p2_decompose(fa["M"], fb["M"], dt)
            if q is not None:
                db.append(q["dbar"])
            p1r.append(p1_pairs(X, got[0], rng)["d_ratio"])
        cal["rescale"][f"c{c:g}"] = dict(
            truth_dlog_lam=truth, n=len(dl), recovered_dlog_lam=R.boot_ci(dl, seed=SEED),
            fraction_recovered=float(np.mean(dl) / truth) if dl else np.nan,
            p2_dbar=R.boot_ci(db, seed=SEED),
            fraction_recovered_dbar=float(np.mean(db) / truth) if db else np.nan,
            dlog_D=R.boot_ci(dd, seed=SEED), truth_dlog_D_exact=R.boot_ci(tD, seed=SEED),
            slope_landing=float(np.mean(dd) / np.mean(dl)) if dl and np.mean(dl) else np.nan,
            p1_d_ratio=R.boot_ci(p1r, seed=SEED))
        x = cal["rescale"][f"c{c:g}"]
        log(f"  INJ-T c={c:g}: truth {truth:+.4f} lambda {x['recovered_dlog_lam']['mean']:+.4f} "
            f"(frac {x['fraction_recovered']:+.3f}) common factor frac "
            f"{x['fraction_recovered_dbar']:+.3f}  D {x['dlog_D']['mean']:+.4f} of truth "
            f"{x['truth_dlog_D_exact']['mean']:+.4f}  landing slope {x['slope_landing']:+.2f}")
    # ---- INJ-D: pure drive, the P1 exclusion bound
    cal["drive"] = {}
    for g in (1.05, 1.1, 1.2, 1.4, 1.7, 2.0):
        dl, ddt, p1r, tdc = [], [], [], []
        for s, z in data.items():
            X, Y = z["X0"], z["Y0"]
            d = X.shape[1]
            u = rng.standard_normal(d); u /= np.linalg.norm(u)
            T = np.eye(d) + (g - 1.0) * np.outer(u, u)
            fa = R.fit_pairs(X, Y, dt, n_boot=boot, rng=rng)
            fb = R.fit_pairs(X @ T.T, Y @ T.T, dt, n_boot=boot, rng=rng)
            if fa is None or fb is None or not (fa["stable"] and fb["stable"]):
                continue
            dl.append(np.log(fb["lam_mean"] / fa["lam_mean"]))
            ddt.append(np.log(fb["D_tr"] / fa["D_tr"])
                       - float(np.log(np.trace(T @ fa["Q"] @ T.T) / np.trace(fa["Q"]))))
            P = fa["P_emp"]
            C0 = P / np.sqrt(np.outer(np.diag(P), np.diag(P)))
            Pn = T @ P @ T.T
            C1 = Pn / np.sqrt(np.outer(np.diag(Pn), np.diag(Pn)))
            tdc.append(H.corr_dist(C0, C1))
            p1r.append(p1_pairs(X, X @ T.T, rng)["d_ratio"])
        cal["drive"][f"g{g:g}"] = dict(n=len(dl), truth_dlog_lam=0.0,
                                       true_corr_shift=R.boot_ci(tdc, seed=SEED),
                                       recovered_dlog_lam=R.boot_ci(dl, seed=SEED),
                                       dlog_D_minus_truth=R.boot_ci(ddt, seed=SEED),
                                       p1_d_ratio=R.boot_ci(p1r, seed=SEED))
        x = cal["drive"][f"g{g:g}"]
        log(f"  INJ-D g={g:g}: true corr shift {x['true_corr_shift']['mean']:.4f}  rate leak "
            f"{x['recovered_dlog_lam']['mean']:+.4f}  P1 d_ratio {x['p1_d_ratio']['mean']:.3f} "
            f"[{x['p1_d_ratio']['lo']:.3f},{x['p1_d_ratio']['hi']:.3f}]")
    # ---- INJ-R: pure reshaping, the P2 exclusion bound
    cal["reshape"] = {}
    for sz in (0.0, 0.02, 0.05, 0.1, 0.2, 0.4):
        rsh, db, tr_ = [], [], []
        for s, z in data.items():
            X, Y = z["X0"], z["Y0"]
            f0 = R.fit_pairs(X, Y, dt, n_boot=0, rng=rng)
            if f0 is None or not f0["stable"]:
                continue
            gam, real = H.reshape_gamma(f0["M"], sz)
            if sz > 0 and real <= 0:
                continue
            got, tr = inject_rescale_pairs(X, Y, 1.0, gam=gam, dt=dt, rng=rng)
            if got is None:
                continue
            fa = R.fit_pairs(X, Y, dt, n_boot=boot, rng=rng)
            fb = R.fit_pairs(got[0], got[1], dt, n_boot=boot, rng=rng)
            if fa is None or fb is None or not (fa["stable"] and fb["stable"]):
                continue
            q = H.p2_decompose(fa["M"], fb["M"], dt)
            if q is None:
                continue
            rsh.append(q["reshape_rms"]); db.append(q["dbar"]); tr_.append(tr["reshape_rms"])
        cal["reshape"][f"s{sz:g}"] = dict(n=len(rsh), nominal_size=float(sz),
                                          truth_reshape_rms=R.boot_ci(tr_, seed=SEED),
                                          recovered_reshape_rms=R.boot_ci(rsh, seed=SEED),
                                          recovered_dbar=R.boot_ci(db, seed=SEED))
        x = cal["reshape"][f"s{sz:g}"]
        log(f"  INJ-R size={sz:g}: truth {x['truth_reshape_rms']['mean']:.4f} recovered "
            f"{x['recovered_reshape_rms']['mean']:.4f} [{x['recovered_reshape_rms']['lo']:.4f},"
            f"{x['recovered_reshape_rms']['hi']:.4f}]  common-factor leak "
            f"{x['recovered_dbar']['mean']:+.4f}")
    return cal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--cboot", type=int, default=80)
    ap.add_argument("--skip-calib", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    data = load()
    log(f"{len(data)} MEG participants")
    rows = []
    for s, z in sorted(data.items()):
        r = one(z, a.boot, rng)
        if r is None:
            continue
        r["subject"] = s
        rows.append(r)
    log(f"{len(rows)} usable")
    res = dict(cohort="HCP_MEG", dt_s=DT, n=len(rows), design="one lag-1 pair per trial",
               seed=SEED, p4="not testable: no within-trial series beyond one lag")
    for k in sorted({k for r in rows for k in r} - {"subject"}):
        res[k] = R.boot_ci([r.get(k, np.nan) for r in rows], seed=SEED)
    res["plane"] = H.plane_angle([r["dlog_lam"] for r in rows], [r["dlog_D"] for r in rows])
    H.csv_write(L.out("injection_recovery_meg_subjects.csv"), rows)
    L.out("injection_recovery_meg.json").write_text(json.dumps(res, indent=1, default=float))

    log("")
    log(f"P3 dlog lambda {R.fmt(res['dlog_lam'])}")
    log(f"P3 dlog D      {R.fmt(res['dlog_D'])}")
    log(f"P3 slowest mode {R.fmt(res['dlog_lam_slowest'])}")
    if "dlog_lam_1d" in res:
        log(f"P3 one-dimensional rate on the cached load direction {R.fmt(res['dlog_lam_1d'])}")
    log(f"P3 angle {res['plane']['angle_deg']:.1f} deg [{res['plane']['angle_lo']:.1f}, "
        f"{res['plane']['angle_hi']:.1f}]; slope {res['plane']['slope']:+.3f} "
        f"[{res['plane']['slope_lo']:+.3f}, {res['plane']['slope_hi']:+.3f}] "
        f"P vs 1 = {res['plane']['p_slope_vs_1']:.4g}")
    log(f"P3 diagonal gap {R.fmt(res['plane']['diagonal_gap'])}")
    log(f"P1 d_between {R.fmt(res['p1_d_between'])}, ceiling {R.fmt(res['p1_d_within'])}")
    log(f"P1 ratio {R.fmt(res['p1_d_ratio'])}")
    log(f"P2 common factor {R.fmt(res['p2_dbar'])}")
    log(f"P2 reshaping {R.fmt(res['p2_reshape_rms'])}, ceiling {R.fmt(res['p2_ceiling_reshape_rms'])}")
    log(f"P2 reshaping / ceiling {R.fmt(res['p2_reshape_over_ceiling'])}")
    log(f"P2 fraction carried by the common factor {R.fmt(res['p2_frac_common'])}")
    log(f"P2 rank Spearman {R.fmt(res['p2_rank_spearman'])}, ceiling "
        f"{R.fmt(res['p2_ceiling_rank_spearman'])}")
    log(f"stationarity check tr(cov Y)/tr(cov X): 0-back {res['nonstat_lo']['mean']:.3f}, "
        f"2-back {res['nonstat_hi']['mean']:.3f}")

    if not a.skip_calib:
        log("")
        log("CALIBRATION (no decimation is possible: the design has one pair per trial)")
        cal = calib(data, a.cboot, rng)
        cal["cohort"] = "HCP_MEG"
        L.out("injection_recovery_meg_calibration.json").write_text(json.dumps(cal, indent=1, default=float))
    log(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
