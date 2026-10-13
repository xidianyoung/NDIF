# -*- coding: utf-8 -*-
"""COG-BCI state construction and the time-rescaling analysis: P1-P4 and the fine-grained
decimation response curve.

Cohort.  The COG-BCI participants after the duplicate exclusion in cog_features.py, 3 sessions x
2 task families (n-back 0/1/2, MATB easy/med/diff), the 62-channel 248-feature log band-power
state of the feature cache, reduced to d = 8 by a PCA fitted on the load-blind Flanker reference
task (build() below).

Why this cohort carries the calibration.  Its records are long and unbroken - a MATB record is one
597-window `dist` segment - so the same recording can be decimated at strides 2 to 20 and still
leave enough windows to fit the propagator.  Comparing stride k with stride k+1 is an EXACT pure
time rescaling by (k+1)/k, so the truth for dlog lambda sweeps ln(21/20) = 0.049 up to
ln(3/2) = 0.405, plus ln 2 = 0.693 from the non-overlapping `dyn` series - the range of the
effect sizes at issue.  Strides below 2 are never used on the `dist` series because its
windows overlap by 50 % and only stride >= 2 leaves them disjoint.

Writes: out/rescaling_cog.json, rescaling_cog_subjects.csv, rescaling_cog_cells.csv, rescaling_cog_calibration.json
Usage: python cog_state.py [--boot 150] [--cboot 60]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cog_features as C                                            # noqa: E402
from cog_features import SESSIONS, FAMILIES, OUT, log               # noqa: E402
import cog_fit_lib as R                                               # noqa: E402
import cog_rescaling_lib as H                                                # noqa: E402

DT = 1.0
LEVELS = [0, 1, 2]
D_PCA = 8
SEED = 20260922
MAXLAG = 5


# --------------------------------------------------------------------------- data
def build(subs, d=D_PCA, nmin_dyn=20, nmin_dist=40):
    """Group PCA on the load-blind Flanker windows, then per-cell dyn AND dist block lists."""
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(SEED)
    X = []
    for s in subs.values():
        Z = s.z(s.ref_raw("dist"))
        if len(Z) > 1200:
            Z = Z[rng.choice(len(Z), 1200, replace=False)]
        X.append(Z)
    p = PCA(n_components=d, random_state=SEED).fit(np.concatenate(X, 0))
    log(f"  PCA d={d} on {sum(len(x) for x in X)} Flanker windows, "
        f"{100*p.explained_variance_ratio_.sum():.1f}% of variance")
    data = {}
    for sid, s in subs.items():
        cells = {}
        for fam, recs in FAMILIES.items():
            for si, ses in enumerate(SESSIONS, start=1):
                ok, per = True, {}
                for lev, rec in enumerate(recs):
                    Zd, sd = s.zrec(ses, rec, "dist")
                    Zy, sy = s.zrec(ses, rec, "dyn")
                    if Zd is None or Zy is None or len(Zd) < 60 or len(Zy) < 60:
                        ok = False
                        break
                    Zd8, Zy8 = p.transform(Zd), p.transform(Zy)
                    yb = [Zy8[sy == g] for g in np.unique(sy) if (sy == g).sum() >= nmin_dyn]
                    db = [Zd8[sd == g] for g in np.unique(sd) if (sd == g).sum() >= nmin_dist]
                    if not yb or not db:
                        ok = False
                        break
                    per[lev] = dict(mu=Zd8.mean(axis=0), blocks=yb, dist=db)
                if not ok:
                    continue
                mb = R.match_blocks([per[l]["blocks"] for l in LEVELS])
                md = R.match_blocks([per[l]["dist"] for l in LEVELS])
                if mb is None or md is None:
                    continue
                for l in LEVELS:
                    per[l]["blocks"], per[l]["dist"] = mb[l], md[l]
                cells[(fam, si)] = per
        if cells:
            data[sid] = cells
    return p, data


# --------------------------------------------------------------------------- calibration
def _pool(data, which="dist", level=None):
    """One block list per participant, pooled over cells (and over load levels unless one is
    named).  The calibration never contrasts load, so pooling is legitimate."""
    out = {}
    for sid, cells in data.items():
        B = []
        for per in cells.values():
            for l in (LEVELS if level is None else [level]):
                B.extend(per[l][which])
        if B:
            out[sid] = B
    return out


def calib_decimation(pool, ks, boot, rng, dt=DT, nmin=12, d_fit=None):
    """INJ-S on the real windows: stride k against stride k+1, truth ln((k+1)/k)."""
    res = {}
    for k in ks:
        truth = float(np.log((k + 1) / k))
        dl, dd, dbar, rsh, p1r, tt = [], [], [], [], [], []
        for sid, B in pool.items():
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
                dbar.append(q["dbar"]); rsh.append(q["reshape_rms"])
            pp = H.p1_invariance(A1, A2, True, rng)
            if pp is not None:
                p1r.append(pp["d_ratio"])
        res[f"k{k}_vs_{k+1}"] = dict(
            truth_dlog_lam=truth, n=len(dl), stride_pair=[k, k + 1],
            recovered_dlog_lam=R.boot_ci(dl, seed=SEED),
            fraction_recovered=float(np.mean(dl) / truth) if dl else np.nan,
            p2_dbar=R.boot_ci(dbar, seed=SEED),
            fraction_recovered_dbar=float(np.mean(dbar) / truth) if dbar else np.nan,
            p2_reshape_rms=R.boot_ci(rsh, seed=SEED),
            dlog_D=R.boot_ci(dd, seed=SEED), truth_dlog_D_exact=R.boot_ci(tt, seed=SEED),
            fraction_recovered_D=float(np.mean(dd) / np.mean(tt)) if tt and np.mean(tt) else np.nan,
            slope_landing=float(np.mean(dd) / np.mean(dl)) if dl and np.mean(dl) else np.nan,
            p1_d_ratio=R.boot_ci(p1r, seed=SEED))
        x = res[f"k{k}_vs_{k+1}"]
        log(f"  INJ-S k={k}->{k+1}: truth {truth:+.4f}  lambda {x['recovered_dlog_lam']['mean']:+.4f}"
            f" (frac {x['fraction_recovered']:+.3f})  common factor {x['p2_dbar']['mean']:+.4f} "
            f"(frac {x['fraction_recovered_dbar']:+.3f})  D {x['dlog_D']['mean']:+.4f} of truth "
            f"{x['truth_dlog_D_exact']['mean']:+.4f} (frac {x['fraction_recovered_D']:+.3f})  "
            f"landing slope {x['slope_landing']:+.2f}  P1 {x['p1_d_ratio']['mean']:.3f}")
    return res


def calib_decim_var1(pool, ks, boot, rng, dt=DT, nmin=12):
    """The estimator control: the same decimation on a GENUINE VAR(1) surrogate - white
    innovations, the participant's own (M, Q), block lengths, dimension and n."""
    surr = {}
    for sid, B in pool.items():
        f = R.fit_cell(B, dt, n_boot=boot, rng=rng)
        if f is None or not f["stable"]:
            continue
        surr[sid] = H.simulate_var1(f["M"], f["Q"], [len(b) for b in B], rng)
    res = {}
    for k in ks:
        truth = float(np.log((k + 1) / k))
        dl, dbar = [], []
        for sid, B in surr.items():
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
        log(f"  INJ-S/VAR1 k={k}->{k+1}: truth {truth:+.4f} lambda "
            f"{x['recovered_dlog_lam']['mean']:+.4f} (frac {x['fraction_recovered']:+.3f}) "
            f"common factor frac {x['fraction_recovered_dbar']:+.3f}")
    return res


def calib_rescale(pool, cs, boot, rng, dt=DT):
    res = {}
    for c in cs:
        truth = float(np.log(c))
        dl, dd, dbar, rsh, p1r, tD = [], [], [], [], [], []
        for sid, B in pool.items():
            new, tr = H.inject_rescale(B, c, dt, rng=rng, n_boot=0)
            if new is None:
                continue
            fa = R.fit_cell(B, dt, n_boot=boot, rng=rng)
            fb = R.fit_cell(new, dt, n_boot=boot, rng=rng)
            if fa is None or fb is None or not (fa["stable"] and fb["stable"]):
                continue
            dl.append(np.log(fb["lam_mean"] / fa["lam_mean"]))
            dd.append(np.log(fb["D_tr"] / fa["D_tr"])); tD.append(tr["dlog_D"])
            q = H.p2_decompose(fa["M"], fb["M"], dt)
            if q is not None:
                dbar.append(q["dbar"]); rsh.append(q["reshape_rms"])
            pp = H.p1_invariance(B, new, True, rng)
            if pp is not None:
                p1r.append(pp["d_ratio"])
        res[f"c{c:g}"] = dict(
            truth_dlog_lam=truth, n=len(dl), truth_dlog_D_exact=R.boot_ci(tD, seed=SEED),
            recovered_dlog_lam=R.boot_ci(dl, seed=SEED),
            fraction_recovered=float(np.mean(dl) / truth) if dl and truth else np.nan,
            p2_dbar=R.boot_ci(dbar, seed=SEED),
            fraction_recovered_dbar=float(np.mean(dbar) / truth) if dbar and truth else np.nan,
            p2_reshape_rms=R.boot_ci(rsh, seed=SEED), dlog_D=R.boot_ci(dd, seed=SEED),
            slope_landing=float(np.mean(dd) / np.mean(dl)) if dl and np.mean(dl) else np.nan,
            p1_d_ratio=R.boot_ci(p1r, seed=SEED))
        x = res[f"c{c:g}"]
        log(f"  INJ-T c={c:g}: truth {truth:+.4f}  lambda {x['recovered_dlog_lam']['mean']:+.4f} "
            f"(frac {x['fraction_recovered']:+.3f})  common factor frac "
            f"{x['fraction_recovered_dbar']:+.3f}  D {x['dlog_D']['mean']:+.4f} of truth "
            f"{x['truth_dlog_D_exact']['mean']:+.4f}  landing slope {x['slope_landing']:+.2f}  "
            f"P1 {x['p1_d_ratio']['mean']:.3f}")
    return res


def calib_drive(pool, gs, boot, rng, dt=DT):
    res = {}
    for g in gs:
        dl, ddt, p1r, tdc, det = [], [], [], [], []
        for sid, B in pool.items():
            new, T, u = H.inject_drive(B, g, rng=rng)
            fa = R.fit_cell(B, dt, n_boot=boot, rng=rng)
            fb = R.fit_cell(new, dt, n_boot=boot, rng=rng)
            if fa is None or fb is None or not (fa["stable"] and fb["stable"]):
                continue
            dl.append(np.log(fb["lam_mean"] / fa["lam_mean"]))
            truth_D = float(np.log(np.trace(T @ fa["Q"] @ T.T) / np.trace(fa["Q"])))
            ddt.append(np.log(fb["D_tr"] / fa["D_tr"]) - truth_D)
            P = fa["P_emp"]
            C0 = P / np.sqrt(np.outer(np.diag(P), np.diag(P)))
            P1m = T @ P @ T.T
            C1 = P1m / np.sqrt(np.outer(np.diag(P1m), np.diag(P1m)))
            tdc.append(H.corr_dist(C0, C1))
            pp = H.p1_invariance(B, new, True, rng)
            if pp is not None:
                p1r.append(pp["d_ratio"]); det.append(1.0 if pp["d_between"] > pp["d_within"] else 0.0)
        res[f"g{g:g}"] = dict(n=len(dl), truth_dlog_lam=0.0,
                              true_corr_shift=R.boot_ci(tdc, seed=SEED),
                              recovered_dlog_lam=R.boot_ci(dl, seed=SEED),
                              dlog_D_minus_truth=R.boot_ci(ddt, seed=SEED),
                              p1_d_ratio=R.boot_ci(p1r, seed=SEED),
                              p1_frac_detected=float(np.mean(det)) if det else np.nan)
        x = res[f"g{g:g}"]
        log(f"  INJ-D g={g:g}: true corr shift {x['true_corr_shift']['mean']:.4f}  rate leak "
            f"{x['recovered_dlog_lam']['mean']:+.4f}  D minus truth "
            f"{x['dlog_D_minus_truth']['mean']:+.5f}  P1 d_ratio {x['p1_d_ratio']['mean']:.3f} "
            f"[{x['p1_d_ratio']['lo']:.3f},{x['p1_d_ratio']['hi']:.3f}]")
    return res


def calib_reshape(pool, sizes, boot, rng, dt=DT):
    res = {}
    for s in sizes:
        dbar, rsh, tr_, dl = [], [], [], []
        for sid, B in pool.items():
            f0 = R.fit_cell(B, dt, n_boot=0, rng=rng)
            if f0 is None or not f0["stable"]:
                continue
            gam, real_size = H.reshape_gamma(f0["M"], s)
            if s > 0 and real_size <= 0:
                continue
            new, t = H.inject_rescale(B, 1.0, dt, gam_vec=gam, rng=rng, n_boot=0)
            if new is None:
                continue
            fa = R.fit_cell(B, dt, n_boot=boot, rng=rng)
            fb = R.fit_cell(new, dt, n_boot=boot, rng=rng)
            if fa is None or fb is None or not (fa["stable"] and fb["stable"]):
                continue
            q = H.p2_decompose(fa["M"], fb["M"], dt)
            if q is None:
                continue
            dbar.append(q["dbar"]); rsh.append(q["reshape_rms"]); tr_.append(t["reshape_rms"])
            dl.append(np.log(fb["lam_mean"] / fa["lam_mean"]))
        res[f"s{s:g}"] = dict(n=len(rsh), nominal_size=float(s),
                              truth_reshape_rms=R.boot_ci(tr_, seed=SEED),
                              recovered_reshape_rms=R.boot_ci(rsh, seed=SEED),
                              recovered_dbar=R.boot_ci(dbar, seed=SEED),
                              recovered_dlog_lam=R.boot_ci(dl, seed=SEED))
        x = res[f"s{s:g}"]
        log(f"  INJ-R size={s:g}: truth rms {x['truth_reshape_rms']['mean']:.4f} recovered "
            f"{x['recovered_reshape_rms']['mean']:.4f} [{x['recovered_reshape_rms']['lo']:.4f},"
            f"{x['recovered_reshape_rms']['hi']:.4f}]  leak onto the common factor "
            f"{x['recovered_dbar']['mean']:+.4f}")
    return res


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=150)
    ap.add_argument("--cboot", type=int, default=60)
    ap.add_argument("--subs", default="")
    ap.add_argument("--skip-calib", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    subs = C.load_subjects([int(x) for x in a.subs.split(",")] if a.subs else None)
    pca, data = build(subs)
    log(f"  {len(data)} participants, {sum(len(c) for c in data.values())} (family, session) cells")

    # held-out (leave-one-participant-out) load displacement axis for the ACF projection
    sub_ax = {}
    for sid, cells in data.items():
        vs = []
        for per in cells.values():
            v = per[2]["mu"] - per[0]["mu"]
            if np.linalg.norm(v) > 0:
                vs.append(v / np.linalg.norm(v))
        sub_ax[sid] = np.mean(vs, axis=0) / np.linalg.norm(np.mean(vs, axis=0)) if vs else None

    cellrows, subrows = [], []
    for sid in sorted(data):
        o = [sub_ax[j] for j in sub_ax if j != sid and sub_ax[j] is not None]
        w = np.mean(o, axis=0)
        w = w / np.linalg.norm(w)
        rs = []
        for (fam, si), per in sorted(data[sid].items()):
            for hi in (1, 2):
                r = H.analyse_pair(per[0]["blocks"], per[hi]["blocks"], DT, a.boot, rng,
                                   w=w, maxlag=MAXLAG, D0=per[0]["dist"], D1=per[hi]["dist"])
                if r is None:
                    continue
                r.update(subject=sid, family=fam, session=si, contrast=f"L{hi}_vs_L0")
                cellrows.append(r)
                if hi == 2:
                    rs.append(r)
        if rs:
            d = dict(subject=sid, n_cells=len(rs))
            for k in {k for r in rs for k in r} - {"subject", "family", "session", "contrast"}:
                v = np.array([r.get(k, np.nan) for r in rs], float)
                d[k] = float(np.nanmean(v)) if np.any(np.isfinite(v)) else np.nan
            subrows.append(d)
        log(f"  sub-{sid:02d} done ({time.time()-t0:.0f}s)")

    res = dict(cohort="COG-BCI", dt_s=DT, d=D_PCA, n=len(subrows), n_cells=len(cellrows),
               contrast="highest vs lowest load, 2 task families x 3 sessions", seed=SEED,
               n_boot_bias_correction=a.boot)
    for k in sorted({k for r in subrows for k in r} - {"subject"}):
        res[k] = R.boot_ci([r.get(k, np.nan) for r in subrows], seed=SEED)
    res["plane"] = H.plane_angle([r["dlog_lam"] for r in subrows],
                                 [r["dlog_D"] for r in subrows])
    OUT.mkdir(parents=True, exist_ok=True)
    H.csv_write(OUT / "rescaling_cog_subjects.csv", subrows)
    H.csv_write(OUT / "rescaling_cog_cells.csv", cellrows)
    (OUT / "rescaling_cog.json").write_text(json.dumps(res, indent=1, default=float))

    log("")
    log(f"P3 dlog lambda {R.fmt(res['dlog_lam'])}")
    log(f"P3 dlog D      {R.fmt(res['dlog_D'])}")
    log(f"P3 slowest mode dlog lambda {R.fmt(res['dlog_lam_slowest'])}")
    if "dlog_lam_1d" in res:
        log(f"P3 one-dimensional rate on the held-out load axis {R.fmt(res['dlog_lam_1d'])}")
    if res.get("plane"):
        log(f"P3 angle {res['plane']['angle_deg']:.1f} deg [{res['plane']['angle_lo']:.1f}, "
            f"{res['plane']['angle_hi']:.1f}] (0 gain / 45 time rescaling / 90 drive); slope "
            f"{res['plane']['slope']:+.3f} [{res['plane']['slope_lo']:+.3f}, "
            f"{res['plane']['slope_hi']:+.3f}] P vs 1 = {res['plane']['p_slope_vs_1']:.4g}")
        log(f"P3 diagonal gap dlogD - dloglam {R.fmt(res['plane']['diagonal_gap'])}")
    if "exp_index_lo" in res:
        log(f"Markov check ln r(2)/2 ln r(1): low {R.fmt(res['exp_index_lo'])}, "
            f"high {R.fmt(res['exp_index_hi'])}")
        log(f"   r(1) {res['r1_lo']['mean']:.3f} -> {res['r1_hi']['mean']:.3f}; "
            f"r(2) {res['r2_lo']['mean']:.3f} vs the exponential prediction "
            f"{res['r1_lo']['mean']**2:.3f}")
    log(f"P1 d_between {R.fmt(res['p1_d_between'])}")
    log(f"P1 d_within (ceiling) {R.fmt(res['p1_d_within'])}")
    log(f"P1 ratio {R.fmt(res['p1_d_ratio'])}")
    log(f"P2 common factor {R.fmt(res['p2_dbar'])}")
    log(f"P2 reshaping rms {R.fmt(res['p2_reshape_rms'])}  ceiling "
        f"{R.fmt(res['p2_ceiling_reshape_rms'])}")
    log(f"P2 reshaping / ceiling {R.fmt(res['p2_reshape_over_ceiling'])}")
    log(f"P2 fraction carried by the common factor {R.fmt(res['p2_frac_common'])}")
    if "p4_res_collapsed" in res:
        log(f"P4 residual before {R.fmt(res['p4_res_raw'])}, after {R.fmt(res['p4_res_collapsed'])}"
            f", ceiling {R.fmt(res['p4_res_ceiling'])}")
        log(f"P4 n-matched: before {R.fmt(res['p4_res_raw_matched'])}, after "
            f"{R.fmt(res['p4_res_collapsed_matched'])}, with c from the 1-d ACF "
            f"{R.fmt(res['p4_res_collapsed_c1d_matched'])}, ceiling {R.fmt(res['p4_res_ceiling'])}")

    if not a.skip_calib:
        log("")
        log("CALIBRATION")
        pool_dist = _pool(data, "dist")
        pool_dyn = _pool(data, "blocks")
        cal = dict(cohort="COG-BCI", dt_s=DT, boot=a.cboot,
                   note=("INJ-S runs on the `dist` series at strides >= 2, where its windows are "
                         "disjoint, and on the non-overlapping `dyn` series at stride 1 vs 2."))
        log(" INJ-S decimation of the real dist windows (strides 2..20; truth 0.405 down to 0.049)")
        cal["decimation_dist"] = calib_decimation(pool_dist, [2, 3, 4, 6, 8, 10, 14, 20],
                                                  a.cboot, rng)
        log(" INJ-S decimation of the real dyn windows (stride 1 vs 2; truth 0.693)")
        cal["decimation_dyn"] = calib_decimation(pool_dyn, [1], a.cboot, rng)
        log(" INJ-S control on a genuine VAR(1) surrogate of the same recordings")
        cal["decimation_var1_control"] = calib_decim_var1(pool_dist, [2, 4, 10], a.cboot, rng)
        log(" INJ-T exact time rescaling of the real innovation stream")
        cal["rescale"] = calib_rescale(pool_dyn, [1.05, 1.1, 1.2, 1.5, 2.0], a.cboot, rng)
        log(" INJ-D pure drive (P1's calibrated exclusion bound)")
        cal["drive"] = calib_drive(pool_dyn, [1.05, 1.1, 1.2, 1.4, 1.7, 2.0], a.cboot, rng)
        log(" INJ-R pure reshaping (P2's calibrated exclusion bound)")
        cal["reshape"] = calib_reshape(pool_dyn, [0.0, 0.02, 0.05, 0.1, 0.2, 0.4], a.cboot, rng)
        (OUT / "rescaling_cog_calibration.json").write_text(json.dumps(cal, indent=1, default=float))
    log(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
