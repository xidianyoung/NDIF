# -*- coding: utf-8 -*-
r"""End-to-end calibration of the tail-excess test with injections of known, settable size (Extended Data Fig. 7b),
in COG-BCI (MATB, low load) and HCP fMRI (0-back), in one invocation.

The map from the raw features to the analysed one-dimensional series is affine and exactly known: for COG-BCI
p = ((F - med)/iqr) V^T w, so p = F c + const with c_j = (V^T w)_j / iqr_j; for HCP fMRI p = ((F - med)/iqr) w, so
c_j = w_j / iqr_j.  Injecting amp * s * l into the RAW features, with s a unit-variance scalar process and l a raw
loading, therefore adds exactly amp (l . c) s to the analysed series.  The ACHIEVED fraction is computable in closed
form for any loading, including an isotropic one (reported as `isotropic_raw_white_DELIVERED`), and choosing
l = c/|c| -- the raw direction the analysed axis reads -- lets the achieved fraction be SET to a target.  The
injection still enters the raw features before standardisation.

THE TWO GROUND TRUTHS
  * WHITE at a set achieved fraction nu (0.10-0.70): under the single-timescale model the profile is (1-nu) a^k,
    so the tail excess must NOT move, and the fitted nu must track the injected one.
  * A SECOND TIMESCALE at a set achieved weight (0.01-0.20) and a stated a_slow (0.30-0.95): the tail excess must
    grow.  The smallest weight whose mean shift reaches 2.80 standard errors of the uninjected statistic at the
    cohort's own n (80 % power at a two-sided 5 % test) is the test's RESOLUTION; below the smallest tested weight
    the crossing is extrapolated in log-log.

Each injected cell averages 8 independent realisations per participant, drawn from generators seeded with
SEED + 7919 r + injection_lib.seed_offset(participant).  The two cohorts are computed independently, and the output
records the start time of each part.

Inputs   NDIF_COGBCI_CACHE, NDIF_COUNT_CACHE/fmri, NDIF_HCP_FMRI_PARCELS
Outputs  figure_reproduction/inputs/second_mode_calibration.json;
         analyses/rate_identifiability/out/second_mode_calibration.csv
Reported Extended Data Fig. 7b (resolution thresholds 0.544 COG-BCI, 0.366 HCP fMRI at a_slow = 0.30)
Runtime  about 23 min on one desktop core (COG-BCI 4.6 min, HCP fMRI 18.5 min)
Usage    python analyses/rate_identifiability/tail_excess_calibration.py [--cohort both]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyses.common import paths                                          # noqa: E402
import injection_lib as L                                                  # noqa: E402
import tail_excess_lib as A                                                # noqa: E402

SEED = 20260922
MAXLAG_COG, TAIL_COG = 10, (3, 10)
MAXLAG_FMRI, TAIL_FMRI = 10, (3, 10)
WHITE_LADDER = (0.10, 0.30, 0.50, 0.70)
SLOW_LADDER = (0.01, 0.02, 0.05, 0.10, 0.20)


def log(m):
    print(time.strftime("[%H:%M:%S] ") + str(m), flush=True)


def ar1(n, a, rng, burn=300):
    e = rng.standard_normal(n + burn)
    x = np.zeros(n + burn)
    for t in range(1, n + burn):
        x[t] = a * x[t - 1] + e[t]
    return x[burn:] * np.sqrt(max(1.0 - a * a, 1e-12))


def inject_1d(P, c_gain, amp, proc):
    """Add amp * (l . c) * s to the analysed series; `c_gain` is (l . c) and is exactly known."""
    return [p + amp * c_gain * s for p, s in zip(P, proc)]


def amp_for_target(P, c_gain, target, blocks_var=None):
    """Amplitude that makes the injected component carry exactly `target` of the total variance."""
    v = float(np.var(np.concatenate(P), ddof=1)) if blocks_var is None else blocks_var
    if c_gain <= 0 or target <= 0:
        return 0.0
    return float(np.sqrt(v * target / max(1.0 - target, 1e-9)) / c_gain)


# ============================================================ COG-BCI
def cog_setup():
    import cog_features as C
    import tail_excess_cog as G
    subs = C.load_subjects()
    pca = C.fit_group_pca(subs, G.D_PCA, seed=SEED)
    data, axes = G.build(subs, pca)
    return C, G, subs, pca, data, axes


def cog_series(sub, pca, data, axes, sid, fam="matb", lev=0):
    """The analysed one-dimensional series (per block) and the exact raw-space functional c."""
    B = data[sid].get((fam, lev))
    import tail_excess_cog as G
    w = G.lopo(axes, sid)
    P = [np.asarray(b, float) @ w for b in B]
    c = (pca.components_.T @ w) / sub.iqr          # p = F c + const, exactly
    return P, c, w


# ============================================================ fMRI
def fmri_setup(limit=0):
    import tail_excess_fmri as F
    subs = F.load_subjects(limit)
    ax = F.load_axis(subs)
    return F, subs, ax


def fmri_series(F, subs, ax, sid, lev=0):
    B = subs[sid][lev]
    w = F.lopo(ax, sid)
    P = [np.asarray(b, float) @ w for b in B]
    # the cached Z is already robust-z of the parcels; the raw-space functional is w / iqr, and the
    # iqr is recovered from the parcel cache
    p = None
    for run in ("WM_LR", "WM_RL"):
        q = F.PARCELS / f"{sid}__{run}__atlas_fpn7.nii.npy"
        if q.exists():
            p = np.load(q).astype(float)
            break
    if p is None:
        return P, None, w
    iqr = np.maximum(np.subtract(*np.percentile(p, [75, 25], axis=0)), 1e-9)
    return P, w / iqr, w


# ============================================================ the ladder
N_REALISE = 8                # injection realisations per participant, averaged before any test
Z_THRESH = 2.80              # 80 % power at a two-sided 5 % test
A_SLOW = (0.30, 0.60, 0.80, 0.90, 0.95)


def run_ladder(units, maxlag, tail, seed=SEED, n_real=N_REALISE):
    """units: {sid: (P_blocks, c, raw_var_vector)}.  Returns the calibration rows.

    Every injected cell is averaged over `n_real` independent realisations of the injected process
    before it enters any statistic, because a single realisation of an AR(1) inside a block of a few
    dozen samples carries enough of its own sampling noise to swamp the shift being calibrated.  The
    resolution is then defined the way the real test is read: an injected weight is RESOLVED when the
    mean shift it produces reaches 2.80 standard errors of the UNINJECTED statistic at the cohort's
    own n, i.e. 80 % power at a two-sided 5 % test.
    """
    rows = []
    B0 = {}
    for sid, (P, c, rv) in units.items():
        u = A.analyse_unit([p[:, None] for p in P], np.array([1.0]), None, 0, maxlag, tail)
        if u:
            B0[sid] = (u["E_tail"], u["nu"], u["a"])
    if not B0:
        return rows
    base = np.array([v[0] for v in B0.values()])
    g0 = A.boot_ci(base)
    se0 = float(np.std(base, ddof=1) / np.sqrt(len(base)))
    rows.append(dict(cohort="", injection="none", target="", a_slow="", achieved="",
                     E_tail=g0["mean"], lo=g0["lo"], hi=g0["hi"], n=g0["n"], se=se0,
                     nu_hat=float(np.mean([v[1] for v in B0.values()])),
                     a_hat=float(np.mean([v[2] for v in B0.values()])),
                     dE="", z="", resolved=""))

    # ---- what an ISOTROPIC raw injection (tail_excess_cog.py) actually DELIVERS
    for nom in (0.10, 0.50):
        ach = []
        for sid, (P, c, rv) in units.items():
            if c is None or rv is None:
                continue
            v = float(np.mean([np.var(p, ddof=1) for p in P]))
            g2 = float(np.sum(c ** 2 * rv) * nom / max(1 - nom, 1e-9))
            ach.append(g2 / (v + g2))
        if ach:
            rows.append(dict(cohort="", injection="isotropic_raw_white_DELIVERED", target=nom,
                             a_slow="", achieved=float(np.mean(ach)), E_tail="", lo="", hi="",
                             n=len(ach), se="", nu_hat="", a_hat="", dE="", z="", resolved=""))

    def cell(kind, tgt, a_s):
        """Mean shift of E_tail over participants, averaging n_real realisations per participant."""
        sh, absol, nus = [], [], []
        for sid, (P, c, rv) in units.items():
            if sid not in B0:
                continue
            cg = 1.0 if c is None else float(np.linalg.norm(c))
            amp = amp_for_target(P, cg, tgt)
            vals, nn = [], []
            for r in range(n_real):
                rng = np.random.default_rng(seed + 7919 * r + (L.seed_offset(sid) % 9999))
                proc = ([rng.standard_normal(len(p)) for p in P] if kind == "white"
                        else [ar1(len(p), a_s, rng) for p in P])
                Q = inject_1d(P, cg, amp, proc)
                u = A.analyse_unit([q[:, None] for q in Q], np.array([1.0]), None, 0, maxlag, tail)
                if u:
                    vals.append(u["E_tail"]); nn.append(u["nu"])
            if vals:
                absol.append(float(np.mean(vals)))
                sh.append(float(np.mean(vals)) - B0[sid][0])
                nus.append(float(np.mean(nn)))
        return sh, absol, nus

    for tgt in WHITE_LADDER:
        sh, absol, nus = cell("white", tgt, None)
        if not sh:
            continue
        g, gd = A.boot_ci(absol), A.boot_ci(sh)
        z = abs(gd["mean"]) / se0 if se0 > 0 else np.nan
        rows.append(dict(cohort="", injection="white_targeted", target=tgt, a_slow="",
                         achieved=tgt, E_tail=g["mean"], lo=g["lo"], hi=g["hi"], n=g["n"],
                         se=se0, nu_hat=float(np.mean(nus)), a_hat="", dE=gd["mean"],
                         z=float(z), resolved=bool(z >= Z_THRESH)))

    for a_s in A_SLOW:
        for tgt in SLOW_LADDER:
            sh, absol, nus = cell("slow", tgt, a_s)
            if not sh:
                continue
            g, gd = A.boot_ci(absol), A.boot_ci(sh)
            z = gd["mean"] / se0 if se0 > 0 else np.nan
            rows.append(dict(cohort="", injection="second_timescale", target=tgt, a_slow=a_s,
                             achieved=tgt, E_tail=g["mean"], lo=g["lo"], hi=g["hi"], n=g["n"],
                             se=se0, nu_hat="", a_hat="", dE=gd["mean"], z=float(z),
                             resolved=bool(z >= Z_THRESH)))
    return rows


def resolution(rows, a_s):
    """Smallest injected weight whose MEAN shift reaches 80 % power at the cohort's own n.

    Below the smallest tested weight the growth of the shift in the weight is fitted in log-log and
    the crossing extrapolated; above the largest, the statement is that the weight is NOT resolved
    at any weight tested, which is a real limit and is reported as one.
    """
    pts = [(float(r["target"]), float(r["z"])) for r in rows
           if r["injection"] == "second_timescale" and r["a_slow"] == a_s
           and np.isfinite(r.get("z", np.nan)) and r["z"] > 0]
    pts.sort()
    if not pts:
        return dict(a_slow=a_s, resolved_at=None, note="no usable cell")
    for t, z in pts:
        if z >= Z_THRESH:
            return dict(a_slow=a_s, resolved_at=t, z_at=z, extrapolated=False)
    x, y = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
    if len(pts) >= 3 and np.ptp(y) > 1e-6:
        b = np.polyfit(x, y, 1)
        if b[0] > 0:
            w = float(np.exp((np.log(Z_THRESH) - b[1]) / b[0]))
            if w <= 1.0:
                return dict(a_slow=a_s, resolved_at=w, extrapolated=True,
                            max_tested=pts[-1][0], z_at_max=pts[-1][1])
    return dict(a_slow=a_s, resolved_at=None, extrapolated=False, max_tested=pts[-1][0],
                z_at_max=pts[-1][1],
                note=("not resolved at any tested weight up to %.2f; the injected mode's "
                      "correlation time is comparable to or longer than the block, so block-mean "
                      "removal deletes it" % pts[-1][0]))


def _num(x, fmt="%+.4f"):
    return fmt % x if isinstance(x, (int, float)) and not isinstance(x, bool) else ""


def _fmt_row(r):
    return ("{:32s} target={:<5} a_slow={:<5} E_tail={:<8} dE={:<8} nu_hat={:<6} {}".format(
        str(r["injection"]), str(r["target"]), str(r["a_slow"]), _num(r["E_tail"]),
        _num(r["dE"]), _num(r["nu_hat"], "%.3f"),
        ("z=%.2f%s" % (r["z"], "  RESOLVED" if r.get("resolved") else "")
         if isinstance(r.get("z"), float) else
         ("achieved=%.4f" % r["achieved"] if isinstance(r.get("achieved"), float) else ""))))


def _fmt_res(q):
    if q.get("resolved_at") is None:
        return "NOT RESOLVED up to weight %.2f (z = %.2f there)" % (
            q.get("max_tested", float("nan")), q.get("z_at_max", float("nan")))
    return "weight %.4f%s" % (q["resolved_at"], " (extrapolated)" if q.get("extrapolated") else "")



NOTE = ("every injection enters the raw features before standardisation; the loading is the raw direction "
        "the analysed axis reads, so the delivered size is exact and settable.")


def assemble(cog, fmri, n_cog):
    """One output from the two independently computed cohort parts (either may be None).

    Key order: generated, note, cog_resolution, fmri_resolution, rows (COG-BCI rows first); `generated` records
    the start time of each part.
    """
    if cog is None or fmri is None:
        return cog if fmri is None else fmri
    return dict(generated=f"{fmri['generated']} (fMRI part); COG-BCI part {cog['generated']} (dedup, n = {n_cog})",
                note=fmri["note"], cog_resolution=cog["cog_resolution"],
                fmri_resolution=fmri["fmri_resolution"],
                rows=[r for r in cog["rows"] if r.get("cohort") == "COG-BCI"]
                + [r for r in fmri["rows"] if r.get("cohort") != "COG-BCI"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="both", choices=["cog", "fmri", "both"])
    ap.add_argument("--subs", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    RES_COG = RES_FMRI = None
    n_cog = 0

    if a.cohort in ("cog", "both"):
        RES = RES_COG = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "note": NOTE}
        C, G, subs, pca, data, axes = cog_setup()
        n_cog = len(data)
        units = {}
        for sid in sorted(data):
            if ("matb", 0) not in data[sid] or sid not in axes:
                continue
            P, c, w = cog_series(subs[sid], pca, data, axes, sid)
            F = subs[sid].raw("ses-S1", "MATBeasy", "dyn")[0]
            rv = F.var(axis=0, ddof=1) if F is not None else None
            units[sid] = (P, c, rv)
        log(f"COG-BCI calibration on {len(units)} participants (MATB, low load)")
        rows = run_ladder(units, MAXLAG_COG, TAIL_COG)
        for r in rows:
            r["cohort"] = "COG-BCI"
        RES["cog_resolution"] = [resolution(rows, x) for x in A_SLOW]
        RES["rows"] = rows
        for r in rows:
            log("  COG  " + _fmt_row(r))
        for q in RES["cog_resolution"]:
            log("  COG-BCI resolution at a_slow=%.2f: %s" % (q["a_slow"], _fmt_res(q)))

    if a.cohort in ("fmri", "both"):
        RES = RES_FMRI = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "note": NOTE}
        F, subs, ax = fmri_setup(a.subs)
        units = {}
        for sid in sorted(subs):
            P, c, w = fmri_series(F, subs, ax, sid)
            rv = None
            for run in ("WM_LR", "WM_RL"):
                q = F.PARCELS / f"{sid}__{run}__atlas_fpn7.nii.npy"
                if q.exists():
                    rv = np.load(q).astype(float).var(axis=0, ddof=1)
                    break
            units[sid] = (P, c, rv)
        log(f"HCP fMRI calibration on {len(units)} participants (0-back)")
        rows = run_ladder(units, MAXLAG_FMRI, TAIL_FMRI)
        for r in rows:
            r["cohort"] = "HCP fMRI"
        RES["fmri_resolution"] = [resolution(rows, x) for x in A_SLOW]
        RES["rows"] = rows
        for r in rows:
            log("  fMRI " + _fmt_row(r))
        for q in RES["fmri_resolution"]:
            log("  HCP fMRI resolution at a_slow=%.2f: %s" % (q["a_slow"], _fmt_res(q)))

    RES = assemble(RES_COG, RES_FMRI, n_cog)
    allrows = RES["rows"]
    target = paths.out("figure_reproduction", "inputs", "second_mode_calibration.json")
    Path(target).write_text(json.dumps(RES, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8")
    if allrows:
        keys = sorted({k for r in allrows for k in r})
        with open(L.out("second_mode_calibration.csv"), "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, keys)
            wr.writeheader()
            for r in allrows:
                wr.writerow(r)
    log(f"done in {time.time() - t0:.0f}s -> {target}")


if __name__ == "__main__":
    main()
