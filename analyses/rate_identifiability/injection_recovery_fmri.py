# -*- coding: utf-8 -*-
"""Injection recovery in the HCP fMRI working-memory cohort: the per-participant load contrast of the
time-rescaling analysis (P1-P4) and the calibration that decides it.

Cohort.  99 participants x 2 WM runs, the 7-parcel fronto-parietal state at TR = 0.72 s, with the archived block
labels, read from the HCP fMRI state cache (``<count cache>/fmri/<subject>__<run>.npz``: ``Z``, ``seg``, ``lab``).
Unit of analysis is the participant: the two runs' blocks are pooled, 0-back is the low condition and 2-back the
high one.  This is the cohort with the LONGEST within-block series (38 TRs), so the decimation calibration is run
at several strides.  Everything estimated here is per participant; the group is a bootstrap over participants and
a size-matched sign-flip permutation.

Calibration (Fig. 3d).  INJ-S: stride k against stride k+1 of the same recorded series (an exact time rescaling,
truth ln((k+1)/k)); the same decimation on a VAR(1) surrogate of the same recordings; INJ-T: exact time rescaling
of the recorded innovation stream at c = 1.05-2.0 (also at the decimation's own block length, 19 volumes);
INJ-D pure drive; INJ-R pure reshaping.  The estimators are cog_fit_lib.fit_cell and the cog_rescaling_lib functions
(analyses/workload_certificates/features), shared with every cohort.

Inputs   NDIF_COUNT_CACHE/fmri (HCP fMRI state cache)
Outputs  analyses/rate_identifiability/out/injection_recovery_fmri.json, injection_recovery_fmri_subjects.csv,
         injection_recovery_fmri_calibration.json (read by injection_recovery_report.py)
Reported Fig. 3d (innovation rescaling and temporal decimation, HCP fMRI)
Runtime  about 2.5 min on one desktop core
Usage    python analyses/rate_identifiability/injection_recovery_fmri.py [--boot 150] [--cboot 60]
"""
from __future__ import annotations

import argparse
import collections
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

CACHE = L.FMRI_CACHE
DT = 0.72
SEED = 20260922
MAXLAG = 10

def log(m):
    print(time.strftime("[%H:%M:%S] ") + str(m), flush=True)


# --------------------------------------------------------------------------- data
def load_subjects(limit=0):
    by = collections.defaultdict(lambda: {0: [], 1: []})
    for f in sorted(CACHE.glob("*.npz")):
        subj = f.stem.split("__")[0]
        z = np.load(f)
        Z, seg, lab = z["Z"].astype(float), z["seg"], z["lab"]
        for s in np.unique(seg):
            m = seg == s
            if m.sum() >= 8:
                by[subj][int(lab[m][0])].append(Z[m])
    subs = {k: v for k, v in by.items() if len(v[0]) >= 4 and len(v[1]) >= 4}
    ks = sorted(subs)
    if limit:
        ks = ks[:limit]
    return {k: subs[k] for k in ks}


# --------------------------------------------------------------------------- one participant
def analyse(blocks, boot, rng, dt=DT, w=None, maxlag=MAXLAG, nmin=8):
    """One participant: match the two conditions' block structure, then run the shared
    cog_rescaling_lib.analyse_pair, which is the identical code every cohort uses."""
    mb = R.match_blocks([blocks[0], blocks[1]])
    if mb is None:
        return None
    return H.analyse_pair(mb[0], mb[1], dt, boot, rng, w=w, maxlag=maxlag)


# --------------------------------------------------------------------------- calibration
def calib_decimation(data, strides, boot, rng, dt=DT, nmin=9):
    """INJ-S: stride k vs stride k+1 of the SAME real series, truth dlog lambda = ln((k+1)/k)."""
    res = {}
    for k in strides:
        truth = float(np.log((k + 1) / k))
        dl, dd, dbar, rsh, p1r, fc, tt = [], [], [], [], [], [], []
        for subj, bl in data.items():
            B = bl[0] + bl[1]
            A1 = H.decimate(B, k, 0, nmin)
            A2 = H.decimate(B, k + 1, 0, nmin)
            A1, A2 = H.match_n(A1, A2, nmin)
            if A1 is None:
                continue
            fa = R.fit_cell(A1, dt, n_boot=boot, rng=rng)
            fb = R.fit_cell(A2, dt, n_boot=boot, rng=rng)
            if fa is None or fb is None or not (fa["stable"] and fb["stable"]):
                continue
            dl.append(np.log(fb["lam_mean"] / fa["lam_mean"]))
            dd.append(np.log(fb["D_tr"] / fa["D_tr"]))
            tt.append(H.exact_rescale_truth(fa["M"], fa["P_emp"], (k + 1) / k, dt)["dlog_D"])
            q = H.p2_decompose(fa["M"], fb["M"], dt)
            if q is not None:
                dbar.append(q["dbar"]); rsh.append(q["reshape_rms"]); fc.append(q["frac_common"])
            pp = H.p1_invariance(A1, A2, True, rng)
            if pp is not None:
                p1r.append(pp["d_ratio"])
        res[f"k{k}_vs_{k+1}"] = dict(
            truth_dlog_lam=truth, n=len(dl),
            recovered_dlog_lam=R.boot_ci(dl, seed=SEED),
            fraction_recovered=float(np.mean(dl) / truth) if dl else np.nan,
            dlog_D=R.boot_ci(dd, seed=SEED), truth_dlog_D_exact=R.boot_ci(tt, seed=SEED),
            fraction_recovered_D=(float(np.mean(dd) / np.mean(tt))
                                  if tt and np.mean(tt) != 0 else np.nan),
            slope_landing=float(np.mean(dd) / np.mean(dl)) if dl and np.mean(dl) != 0 else np.nan,
            p2_dbar=R.boot_ci(dbar, seed=SEED), p2_reshape_rms=R.boot_ci(rsh, seed=SEED),
            p2_frac_common=R.boot_ci(fc, seed=SEED),
            p1_d_ratio=R.boot_ci(p1r, seed=SEED))
        c = res[f"k{k}_vs_{k+1}"]
        log(f"  INJ-S k={k}->{k+1}: truth {truth:+.4f}  recovered "
            f"{c['recovered_dlog_lam']['mean']:+.4f} (frac {c['fraction_recovered']:.3f})  "
            f"dlogD {c['dlog_D']['mean']:+.4f}  common factor {c['p2_dbar']['mean']:+.4f} "
            f"(frac {c['p2_dbar']['mean']/truth:.3f})  dlogD truth "
            f"{c['truth_dlog_D_exact']['mean']:+.4f} (frac {c['fraction_recovered_D']:.3f})  "
            f"P1 d_ratio {c['p1_d_ratio']['mean']:.3f}")
    return res


def calib_decimation_var1(data, strides, boot, rng, dt=DT, nmin=9):
    """The control that isolates WHY decimation is not recovered.

    The same decimation, applied to a series that is a VAR(1) BY CONSTRUCTION and carries the same
    recordings' own innovations: INJ-T at c = 1, i.e. x'_t = M x'_{t-1} + e_t rebuilt from the real
    residuals.  Sample size, block structure, dimension and the estimator are identical to INJ-S.
    If the recovery is good here and not on the raw recording, the failure is the data's departure
    from first-order Markov structure, not the estimator or the sample size.
    """
    res = {}
    surr = {}
    for subj, bl in data.items():
        B = R.match_blocks([bl[0], bl[1]])
        if B is None:
            continue
        base = B[0] + B[1]
        f = R.fit_cell(base, dt, n_boot=boot, rng=rng)
        if f is None or not f["stable"]:
            continue
        surr[subj] = H.simulate_var1(f["M"], f["Q"], [len(b) for b in base], rng)
    for k in strides:
        truth = float(np.log((k + 1) / k))
        dl, dbar = [], []
        for subj, B in surr.items():
            A1 = H.decimate(B, k, 0, nmin)
            A2 = H.decimate(B, k + 1, 0, nmin)
            A1, A2 = H.match_n(A1, A2, nmin)
            if A1 is None:
                continue
            fa = R.fit_cell(A1, dt, n_boot=boot, rng=rng)
            fb = R.fit_cell(A2, dt, n_boot=boot, rng=rng)
            if fa is None or fb is None or not (fa["stable"] and fb["stable"]):
                continue
            dl.append(np.log(fb["lam_mean"] / fa["lam_mean"]))
            q = H.p2_decompose(fa["M"], fb["M"], dt)
            if q is not None:
                dbar.append(q["dbar"])
        res[f"k{k}_vs_{k+1}"] = dict(
            truth_dlog_lam=truth, n=len(dl),
            recovered_dlog_lam=R.boot_ci(dl, seed=SEED),
            fraction_recovered=float(np.mean(dl) / truth) if dl else np.nan,
            p2_dbar=R.boot_ci(dbar, seed=SEED),
            fraction_recovered_dbar=float(np.mean(dbar) / truth) if dbar else np.nan)
        x = res[f"k{k}_vs_{k+1}"]
        log(f"  INJ-S/VAR1 k={k}->{k+1}: truth {truth:+.4f} recovered "
            f"{x['recovered_dlog_lam']['mean']:+.4f} (frac {x['fraction_recovered']:.3f}); "
            f"common factor {x['p2_dbar']['mean']:+.4f} (frac {x['fraction_recovered_dbar']:.3f})")
    return res


def calib_rescale(data, cs, boot, rng, dt=DT, trunc=0, tag=""):
    """INJ-T: exact time rescaling of the real innovation stream at arbitrary c."""
    res = {}
    for c in cs:
        truth = float(np.log(c))
        dl, dd, dbar, rsh, fc, p1r, tD = [], [], [], [], [], [], []
        for subj, bl in data.items():
            B = R.match_blocks([bl[0], bl[1]])
            if B is None:
                continue
            base = B[0] if not trunc else [b[:trunc] for b in B[0] if len(b) >= trunc]
            if not base:
                continue
            new, tr = H.inject_rescale(base, c, dt, rng=rng, n_boot=0)
            if new is None:
                continue
            fa = R.fit_cell(base, dt, n_boot=boot, rng=rng)
            fb = R.fit_cell(new, dt, n_boot=boot, rng=rng)
            if fa is None or fb is None or not (fa["stable"] and fb["stable"]):
                continue
            dl.append(np.log(fb["lam_mean"] / fa["lam_mean"]))
            dd.append(np.log(fb["D_tr"] / fa["D_tr"]))
            tD.append(tr["dlog_D"])
            q = H.p2_decompose(fa["M"], fb["M"], dt)
            if q is not None:
                dbar.append(q["dbar"]); rsh.append(q["reshape_rms"]); fc.append(q["frac_common"])
            pp = H.p1_invariance(base, new, True, rng)
            if pp is not None:
                p1r.append(pp["d_ratio"])
        res[f"c{c:g}{tag}"] = dict(
            truth_dlog_lam=truth, n=len(dl),
            truth_dlog_D_exact=R.boot_ci(tD, seed=SEED),
            recovered_dlog_lam=R.boot_ci(dl, seed=SEED),
            fraction_recovered=float(np.mean(dl) / truth) if dl and truth != 0 else np.nan,
            dlog_D=R.boot_ci(dd, seed=SEED),
            slope_landing=float(np.mean(dd) / np.mean(dl)) if dl and np.mean(dl) != 0 else np.nan,
            p2_dbar=R.boot_ci(dbar, seed=SEED), p2_reshape_rms=R.boot_ci(rsh, seed=SEED),
            p2_frac_common=R.boot_ci(fc, seed=SEED), p1_d_ratio=R.boot_ci(p1r, seed=SEED))
        x = res[f"c{c:g}{tag}"]
        log(f"  INJ-T{tag} c={c:g}: truth dloglam {truth:+.4f} / exact dlogD "
            f"{x['truth_dlog_D_exact']['mean']:+.4f}  recovered {x['recovered_dlog_lam']['mean']:+.4f}"
            f" (frac {x['fraction_recovered']:.3f}) dlogD {x['dlog_D']['mean']:+.4f}  "
            f"P1 d_ratio {x['p1_d_ratio']['mean']:.3f}")
    return res


def calib_drive(data, gs, boot, rng, dt=DT):
    """INJ-D: pure drive.  Truth dlog lambda = 0 exactly; dlog D and the correlation change are
    exactly computable.  Doubles as P1's calibrated exclusion bound, because the injection moves
    the stationary correlation matrix by a known amount."""
    res = {}
    for g in gs:
        dl, dd, ddt, p1r, p1e, true_dc, det = [], [], [], [], [], [], []
        for subj, bl in data.items():
            B = R.match_blocks([bl[0], bl[1]])
            if B is None:
                continue
            base = B[0]
            new, T, u = H.inject_drive(base, g, rng=rng)
            fa = R.fit_cell(base, dt, n_boot=boot, rng=rng)
            fb = R.fit_cell(new, dt, n_boot=boot, rng=rng)
            if fa is None or fb is None or not (fa["stable"] and fb["stable"]):
                continue
            dl.append(np.log(fb["lam_mean"] / fa["lam_mean"]))
            truth_D = float(np.log(np.trace(T @ fa["Q"] @ T.T) / np.trace(fa["Q"])))
            dd.append(np.log(fb["D_tr"] / fa["D_tr"]))
            ddt.append(np.log(fb["D_tr"] / fa["D_tr"]) - truth_D)
            P = fa["P_emp"]
            C0 = P / np.sqrt(np.outer(np.diag(P), np.diag(P)))
            P1m = T @ P @ T.T
            C1 = P1m / np.sqrt(np.outer(np.diag(P1m), np.diag(P1m)))
            true_dc.append(H.corr_dist(C0, C1))
            pp = H.p1_invariance(base, new, True, rng)
            if pp is not None:
                p1r.append(pp["d_ratio"]); p1e.append(pp["d_excess"])
                det.append(1.0 if pp["d_between"] > pp["d_within"] else 0.0)
        res[f"g{g:g}"] = dict(
            n=len(dl), truth_dlog_lam=0.0,
            true_corr_shift=R.boot_ci(true_dc, seed=SEED),
            recovered_dlog_lam=R.boot_ci(dl, seed=SEED),
            dlog_D_minus_truth=R.boot_ci(ddt, seed=SEED), dlog_D=R.boot_ci(dd, seed=SEED),
            p1_d_ratio=R.boot_ci(p1r, seed=SEED), p1_d_excess=R.boot_ci(p1e, seed=SEED),
            p1_frac_participants_detected=float(np.mean(det)) if det else np.nan)
        x = res[f"g{g:g}"]
        log(f"  INJ-D g={g:g}: true corr shift {x['true_corr_shift']['mean']:.4f}  "
            f"rate leak {x['recovered_dlog_lam']['mean']:+.4f}  dlogD-truth "
            f"{x['dlog_D_minus_truth']['mean']:+.5f}  P1 d_ratio {x['p1_d_ratio']['mean']:.3f} "
            f"[{x['p1_d_ratio']['lo']:.3f},{x['p1_d_ratio']['hi']:.3f}]  detected in "
            f"{x['p1_frac_participants_detected']:.2f}")
    return res


def calib_reshape(data, sizes, boot, rng, dt=DT):
    """INJ-R: pure reshaping of the relaxation spectrum, common factor exactly zero."""
    res = {}
    for s in sizes:
        dbar, rsh, fc, truth_r, dl = [], [], [], [], []
        for subj, bl in data.items():
            B = R.match_blocks([bl[0], bl[1]])
            if B is None:
                continue
            base = B[0]
            f0 = R.fit_cell(base, dt, n_boot=0, rng=rng)
            if f0 is None or not f0["stable"]:
                continue
            gam, real_size = H.reshape_gamma(f0["M"], s)
            if s > 0 and real_size <= 0:
                continue
            new, tr = H.inject_rescale(base, 1.0, dt, gam_vec=gam, rng=rng, n_boot=0)
            if new is None:
                continue
            fa = R.fit_cell(base, dt, n_boot=boot, rng=rng)
            fb = R.fit_cell(new, dt, n_boot=boot, rng=rng)
            if fa is None or fb is None or not (fa["stable"] and fb["stable"]):
                continue
            q = H.p2_decompose(fa["M"], fb["M"], dt)
            if q is None:
                continue
            dbar.append(q["dbar"]); rsh.append(q["reshape_rms"]); fc.append(q["frac_common"])
            truth_r.append(tr["reshape_rms"])
            dl.append(np.log(fb["lam_mean"] / fa["lam_mean"]))
        res[f"s{s:g}"] = dict(
            n=len(rsh), nominal_size=float(s),
            truth_reshape_rms=R.boot_ci(truth_r, seed=SEED),
            recovered_reshape_rms=R.boot_ci(rsh, seed=SEED),
            fraction_recovered=(float(np.mean(rsh) / np.mean(truth_r))
                                if truth_r and np.mean(truth_r) > 0 else np.nan),
            recovered_dbar=R.boot_ci(dbar, seed=SEED),
            frac_common=R.boot_ci(fc, seed=SEED),
            recovered_dlog_lam=R.boot_ci(dl, seed=SEED))
        x = res[f"s{s:g}"]
        log(f"  INJ-R size={s:g}: truth rms {x['truth_reshape_rms']['mean']:.4f} recovered "
            f"{x['recovered_reshape_rms']['mean']:.4f} (frac {x['fraction_recovered']:.3f})  "
            f"leak onto the common factor {x['recovered_dbar']['mean']:+.4f}")
    return res


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=150)
    ap.add_argument("--cboot", type=int, default=60)
    ap.add_argument("--subs", type=int, default=0)
    ap.add_argument("--skip-calib", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    log("loading the archived-label fMRI state cache ...")
    data = load_subjects(a.subs)
    log(f"  {len(data)} participants with >=4 blocks in each load condition")

    # ---- group displacement axis, leave-one-participant-out (for the P4 projection)
    dmu = {}
    for s, bl in data.items():
        m0 = np.mean([b.mean(0) for b in bl[0]], axis=0)
        m1 = np.mean([b.mean(0) for b in bl[1]], axis=0)
        v = m1 - m0
        dmu[s] = v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else None

    rows = []
    for s in sorted(data):
        o = [dmu[j] for j in dmu if j != s and dmu[j] is not None]
        w = np.mean(o, axis=0)
        w = w / np.linalg.norm(w)
        r = analyse(data[s], a.boot, rng, DT, w=w)
        if r is None:
            continue
        r["subject"] = s
        rows.append(r)
        if len(rows) % 20 == 0:
            log(f"  {len(rows)} participants done ({time.time()-t0:.0f}s)")
    log(f"  {len(rows)} participants analysed")

    res = dict(cohort="HCP_fMRI", dt_s=DT, d=int(rows[0]["d"]), n=len(rows),
               n_boot_bias_correction=a.boot, seed=SEED)
    keys = sorted({k for r in rows for k in r} - {"subject"})
    for k in keys:
        res[k] = R.boot_ci([r.get(k, np.nan) for r in rows], seed=SEED)
    res["plane"] = H.plane_angle([r["dlog_lam"] for r in rows], [r["dlog_D"] for r in rows])

    H.csv_write(L.out("injection_recovery_fmri_subjects.csv"), rows)
    L.out("injection_recovery_fmri.json").write_text(json.dumps(res, indent=1, default=float))

    log("")
    log(f"P3 plane   dlog lambda {R.fmt(res['dlog_lam'])}")
    log(f"P3 plane   dlog D      {R.fmt(res['dlog_D'])}")
    log(f"P3 plane   angle {res['plane']['angle_deg']:.1f} deg "
        f"[{res['plane']['angle_lo']:.1f}, {res['plane']['angle_hi']:.1f}]  "
        f"(0 = gain, 45 = time rescaling, 90 = drive); slope {res['plane']['slope']:+.3f} "
        f"[{res['plane']['slope_lo']:+.3f}, {res['plane']['slope_hi']:+.3f}], "
        f"P vs 1 = {res['plane']['p_slope_vs_1']:.4g}, P vs 0 = {res['plane']['p_slope_vs_0']:.4g}")
    log(f"P3 diagonal gap dlogD - dloglam {R.fmt(res['plane']['diagonal_gap'])}")
    log(f"P3 slowest mode dlog lambda {R.fmt(res['dlog_lam_slowest'])}")
    if "dlog_lam_1d" in res:
        log(f"P3 one-dimensional rate along the held-out load axis {R.fmt(res['dlog_lam_1d'])}")
    if "exp_index_lo" in res:
        log(f"Markov check: ln r(2) / 2 ln r(1), 1.0 = exponential: low load "
            f"{R.fmt(res['exp_index_lo'])}, high load {R.fmt(res['exp_index_hi'])}")
        log(f"   r(1) {res['r1_lo']['mean']:.3f} -> {res['r1_hi']['mean']:.3f}; r(2) "
            f"{res['r2_lo']['mean']:.3f} against the exponential prediction "
            f"{res['r1_lo']['mean']**2:.3f}")
    log(f"P1 corr d_between {R.fmt(res['p1_d_between'])}")
    log(f"P1 corr d_within (ceiling) {R.fmt(res['p1_d_within'])}")
    log(f"P1 ratio between/within {R.fmt(res['p1_d_ratio'])}")
    log(f"P1 agreement as fraction of ceiling {R.fmt(res['p1_a_frac_ceiling'])}")
    log(f"P2 common factor dbar {R.fmt(res['p2_dbar'])}  c = "
        f"{np.exp(res['p2_dbar']['mean']):.4f}")
    log(f"P2 reshaping rms {R.fmt(res['p2_reshape_rms'])}  ceiling "
        f"{R.fmt(res['p2_ceiling_reshape_rms'])}")
    log(f"P2 reshaping / ceiling {R.fmt(res['p2_reshape_over_ceiling'])}")
    log(f"P2 fraction of the change carried by the common factor {R.fmt(res['p2_frac_common'])}")
    log(f"P2 rank Spearman (matched modes) {R.fmt(res['p2_rank_spearman'])}")
    if "p4_res_collapsed" in res:
        log(f"P4 residual before {R.fmt(res['p4_res_raw'])}")
        log(f"P4 n-matched: before {R.fmt(res['p4_res_raw_matched'])}, after "
            f"{R.fmt(res['p4_res_collapsed_matched'])}, with c from the 1-d ACF "
            f"{R.fmt(res['p4_res_collapsed_c1d_matched'])}, ceiling {R.fmt(res['p4_res_ceiling'])}")
        log(f"P4 residual after collapse {R.fmt(res['p4_res_collapsed'])}")
        log(f"P4 within-condition ceiling {R.fmt(res['p4_res_ceiling'])}")
        log(f"P4 c from P2 {np.exp(res['p2_dbar']['mean']):.3f} vs c from the ACF "
            f"{R.fmt(res['p4_c_from_acf'])}")
        log(f"   one-step correlation 0-back {res['acf1_lo']['mean']:.3f} -> 2-back "
            f"{res['acf1_hi']['mean']:.3f}")

    if not a.skip_calib:
        log("")
        log("CALIBRATION")
        cal = dict(cohort="HCP_fMRI", dt_s=DT, n=len(data), boot=a.cboot)
        log(" INJ-S temporal decimation (no data value touched)")
        cal["decimation"] = calib_decimation(data, [1, 2, 3], a.cboot, rng)
        log(" INJ-S control: the same decimation on a VAR(1) surrogate of the same recordings")
        cal["decimation_var1_control"] = calib_decimation_var1(data, [1, 2, 3], a.cboot, rng)
        log(" INJ-T exact time rescaling of the real innovation stream")
        cal["rescale"] = calib_rescale(data, [1.05, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0], a.cboot, rng)
        log(" INJ-T at the DECIMATION's own sample size (19 volumes per block), which separates the"
            " rescaling from the halving of n that decimation necessarily brings with it")
        cal["rescale_n19"] = calib_rescale(data, [1.1, 1.35, 2.0], a.cboot, rng, trunc=19,
                                           tag="_n19")
        log(" INJ-D pure drive (also P1's calibrated exclusion bound)")
        cal["drive"] = calib_drive(data, [1.05, 1.1, 1.2, 1.4, 1.7, 2.0], a.cboot, rng)
        log(" INJ-R pure reshaping (P2's calibrated exclusion bound)")
        cal["reshape"] = calib_reshape(data, [0.0, 0.02, 0.05, 0.1, 0.2, 0.4], a.cboot, rng)
        L.out("injection_recovery_fmri_calibration.json").write_text(json.dumps(cal, indent=1, default=float))
    log(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
