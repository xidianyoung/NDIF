# -*- coding: utf-8 -*-
"""Injection recovery in the ds003838 digit-span cohort (n = 64), with the listen-only control arm carried
through.

Design (injection_lib): load is graded WITHIN a sequence by the number of items already presented; blocks are
contiguous runs of non-overlapping 1-s windows inside one sequence and one load bin, bin A = slots 1-4,
B = 5-8, C = 9-13; set size is a fixed factor; the block structure is matched between bins AND between arms so
the estimator's finite-sample bias is common-mode.  The state space is a robust z + PCA (d = 8) fitted on the
LISTEN-ONLY arm, i.e. blind to memory load.  Every endpoint is reported for memory, for listen-only, and as the
difference.

What it cannot carry: its blocks are 4 to 8 windows long, so temporal decimation leaves too few windows to fit
anything, and the autocorrelation function runs out after two or three lags, so P4 is reported only as the
lag-1/lag-2 Markov check.

Calibration (Fig. 3d): INJ-T exact time rescaling of the recorded innovations (c = 1.05-2.0), INJ-D pure drive,
INJ-R pure reshaping, on the C1 memory-arm blocks pooled over bins, with the calibration functions of
analyses/workload_certificates/features/cog_state.py.

Inputs   NDIF_DS003838_CACHE (sub-XXX.npz, sub-XXX_meta.json)
Outputs  analyses/rate_identifiability/out/injection_recovery_ds003838.json,
         injection_recovery_ds003838_subjects.csv, injection_recovery_ds003838_calibration.json
         (read by injection_recovery_report.py)
Reported Fig. 3d (innovation rescaling, ds003838: -0.007 to 0.58 of the injected change)
Runtime  about 6 min on one desktop core
Usage    python analyses/rate_identifiability/injection_recovery_ds003838.py [--boot 200] [--cboot 60]
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

import injection_lib as G                                                  # noqa: E402
import cog_fit_lib as R                                                    # noqa: E402
import cog_rescaling_lib as H                                                     # noqa: E402
from injection_lib import log                                              # noqa: E402

DT = 1.0
SEED = 20260922
MAXLAG = 2
CONTRASTS = {"C1": ("A", "B", (9, 13)), "C2": ("A", "C", (13,))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--cboot", type=int, default=60)
    ap.add_argument("--subs", type=int, default=0)
    ap.add_argument("--skip-calib", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    subs = G.subjects()
    if a.subs:
        subs = subs[: a.subs]
    log(f"{len(subs)} ds003838 participants")

    rows, pool = [], {}
    for sub in subs:
        g = G.prep(sub)
        if g is None:
            continue
        sp, out, S = g
        row = {"subject": sub}
        got = False
        for cname, (lo, hi, Ls) in CONTRASTS.items():
            cm = G.contrast_cells(out, 1, lo, hi, Ls)
            cc = G.contrast_cells(out, 0, lo, hi, Ls)
            cm, cc = G.match_arms(cm, cc)
            if cm is None or cc is None:
                continue
            for arm, cell in (("mem", cm), ("ctl", cc)):
                A, B, dmu = cell
                w = dmu / np.linalg.norm(dmu) if dmu is not None and np.linalg.norm(dmu) > 0 else None
                r = H.analyse_pair(A, B, DT, a.boot, rng, w=w, maxlag=MAXLAG)
                if r is None:
                    continue
                got = True
                for k, v in r.items():
                    row[f"{cname}_{arm}_{k}"] = v
            for k in ("dlog_lam", "dlog_D", "p2_dbar", "p2_reshape_rms", "p1_d_ratio",
                      "p1_d_between", "p1_d_within", "dlog_lam_slowest", "dlog_lam_1d"):
                km, kc = f"{cname}_mem_{k}", f"{cname}_ctl_{k}"
                if km in row and kc in row:
                    row[f"{cname}_did_{k}"] = row[km] - row[kc]
            if cname == "C1":
                pool[sub] = cm[0] + cm[1]
        if got:
            rows.append(row)
        if len(rows) % 10 == 0:
            log(f"  {len(rows)} done ({time.time()-t0:.0f}s)")

    res = dict(cohort="ds003838", dt_s=DT, n=len(rows), seed=SEED,
               contrasts={"C1": "items 5-8 vs 1-4 (set sizes 9 and 13)",
                          "C2": "items 9-13 vs 1-4 (set size 13)"},
               decimation="not possible: load-bin blocks are 4-8 windows long")
    for k in sorted({k for r in rows for k in r} - {"subject"}):
        res[k] = R.boot_ci([r.get(k, np.nan) for r in rows], seed=SEED)
    for cn in CONTRASTS:
        for arm in ("mem", "ctl"):
            dl = [r.get(f"{cn}_{arm}_dlog_lam", np.nan) for r in rows]
            dd = [r.get(f"{cn}_{arm}_dlog_D", np.nan) for r in rows]
            res[f"plane_{cn}_{arm}"] = H.plane_angle(dl, dd)
    H.csv_write(G.out("injection_recovery_ds003838_subjects.csv"), rows)
    G.out("injection_recovery_ds003838.json").write_text(json.dumps(res, indent=1, default=float))

    log("")
    for cn in CONTRASTS:
        for arm in ("mem", "ctl", "did"):
            pre = f"{cn}_{arm}_"
            if pre + "dlog_lam" not in res:
                continue
            log(f"[{cn} {arm}] dlog lambda {R.fmt(res[pre+'dlog_lam'])}")
            log(f"[{cn} {arm}] dlog D      {R.fmt(res[pre+'dlog_D'])}")
            if pre + "p2_dbar" in res:
                log(f"[{cn} {arm}] P2 common factor {R.fmt(res[pre+'p2_dbar'])}")
                log(f"[{cn} {arm}] P2 reshaping {R.fmt(res[pre+'p2_reshape_rms'])}"
                    + (f", ceiling {R.fmt(res[pre+'p2_ceiling_reshape_rms'])}"
                       if pre + "p2_ceiling_reshape_rms" in res else ""))
            if pre + "p1_d_ratio" in res:
                log(f"[{cn} {arm}] P1 ratio {R.fmt(res[pre+'p1_d_ratio'])}")
            if res.get(f"plane_{cn}_{arm}"):
                p = res[f"plane_{cn}_{arm}"]
                log(f"[{cn} {arm}] plane angle {p['angle_deg']:.1f} deg [{p['angle_lo']:.1f}, "
                    f"{p['angle_hi']:.1f}]; slope {p['slope']:+.3f} P vs 1 = {p['p_slope_vs_1']:.4g}")
            if pre + "exp_index_lo" in res:
                log(f"[{cn} {arm}] Markov check {R.fmt(res[pre+'exp_index_lo'])} / "
                    f"{R.fmt(res[pre+'exp_index_hi'])}")
            log("")

    if not a.skip_calib:
        log("CALIBRATION (no decimation: blocks are 4-8 windows)")
        import cog_state as K
        cal = dict(cohort="ds003838", boot=a.cboot,
                   note="INJ-T / INJ-D / INJ-R on the C1 memory-arm blocks, pooled over bins.")
        log(" INJ-T exact time rescaling of the real innovation stream")
        cal["rescale"] = K.calib_rescale(pool, [1.05, 1.1, 1.2, 1.5, 2.0], a.cboot, rng, dt=DT)
        log(" INJ-D pure drive (P1's calibrated exclusion bound)")
        cal["drive"] = K.calib_drive(pool, [1.05, 1.1, 1.2, 1.4, 1.7, 2.0], a.cboot, rng, dt=DT)
        log(" INJ-R pure reshaping (P2's calibrated exclusion bound)")
        cal["reshape"] = K.calib_reshape(pool, [0.0, 0.05, 0.1, 0.2, 0.4], a.cboot, rng, dt=DT)
        G.out("injection_recovery_ds003838_calibration.json").write_text(json.dumps(cal, indent=1, default=float))
    log(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
