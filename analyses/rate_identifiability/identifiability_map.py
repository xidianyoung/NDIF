# -*- coding: utf-8 -*-
r"""The rate-identifiability map: operating characteristics of the lag-1 least-squares rate at each
cohort's geometry, from simulations with a known rate and a known white share.

For each cohort geometry (COG-BCI, ds003838, HCP fMRI, HCP MEG, Sleep-EDF; identifiability_map_lib),
a grid over

    AXIS 1  lambda*dt   relaxation per sampling interval, 0.05 .. 3.0, log-spaced, 12 points
    AXIS 2  nu          additive white variance share of the analysed state, 0 .. 0.8, 8 points
    AXIS 3  sample size the cohort's own (n_blocks x block length) x {0.25, 0.5, 1, 2, 4, 8}

(5 x 12 x 8 x 6 = 2,880 configurations; K = 500 simulated recordings per configuration) returns, from
simulation with a KNOWN truth pushed through the estimator chain:

    ci_width_unit   width of the 95% interval on log lambda-hat for ONE recording
    coverage        fraction of replicates whose interval contains the truth, against BOTH
                    lambda_true and lambda_eff = lambda_true - ln(1-nu)/dt
    bias            E[log lambda-hat] - log lambda_true, and - log lambda_eff
    mde80           minimum detectable dlog lambda at 80% power for the cohort's own n_units

Criterion U (report stage): U1 ci_width_unit <= 0.50; U2 coverage of the true rate in [0.90, 0.98];
U3 group interval width <= 0.10 with group coverage in [0.90, 0.98]; U4 mde80 <= 0.18.

Stages (each re-runnable on its own; later stages read the earlier ones' files):
    --stage equiv   the vectorised bias correction against the loop implementation of the
                    estimator, and the burn-in check; the map is not built if it fails
    --stage grid    the main three-axis grid
    --stage lever   the secondary sample-size lever, at each cohort's observed nu
    --stage obs     the observed points: direct simulation at the observed (lambda*dt, nu), at nu = 0,
                    at the multivariate rate, with a spread mode spectrum, through an ambient PCA,
                    direct paired power, and the band-power pipeline arm (Sleep-EDF geometry)
    --stage report  criterion U on every grid point, its contours, and the readings at the observed
                    points
    --stage all     equiv -> grid -> lever -> obs -> report

--cohorts restricts the equiv, grid, lever and obs stages to some cohorts.  Random seeds are assigned
over all five cohorts in a fixed order (grid point i of the full enumeration has seed
20260922 + 7919 i), so a restricted run draws exactly the numbers of the full run for its cohorts.
The grid and lever stages append every finished point to a resume file
(out/identifiability_map_{grid,lever}_partial.jsonl) and write every point found there, so cohorts run
in separate invocations end up in one file.  The report stage always covers all five cohorts.

Inputs (lever, obs and report stages): per-recording rows of the five cohorts (OBSERVED_ROWS below:
calibrated single-timescale fits from the tail-excess scripts, multivariate rates from the
injection-recovery scripts), read at their stored locations or below NDIF_OUT when it is set.  The
grid and equiv stages are pure simulation.
Outputs: figure_reproduction/inputs/identifiability_map_grid.csv (one row per configuration; the
         report stage adds the criterion columns), identifiability_map_observed.csv,
         identifiability_map_contours.csv, identifiability_map_summary.json;
         analyses/rate_identifiability/out/identifiability_map_{lever,obs,power,upstream}.csv,
         identifiability_map_equivalence.{json,csv}.
Supports Fig. 3a and Extended Data Fig. 1 (2,304 configurations outside HCP MEG, 2,300 with enough
usable fits; widths 0.53 -> 0.07 along the COG-BCI sample-size axis; coverage and the observed points).
Runtime: the grid is about one worker-minute per configuration (about 48 worker-hours); lever about
one hour, obs about 16 minutes on 4 workers, equiv 10-15 minutes, report under one minute.

    python analyses/rate_identifiability/identifiability_map.py --stage all --workers 13 --K 500
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse                                                                   # noqa: E402
import csv                                                                        # noqa: E402
import json                                                                       # noqa: E402
import platform                                                                   # noqa: E402
import sys                                                                        # noqa: E402
import time                                                                       # noqa: E402
from multiprocessing import Pool                                                  # noqa: E402
from pathlib import Path                                                          # noqa: E402

import numpy as np                                                                # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyses.common import paths                                                 # noqa: E402
import identifiability_map_lib as L                                               # noqa: E402

SEED = 20260922
Z975, Z80 = 1.959963985, 0.8416212336


def _out(name):
    return Path(paths.out("analyses", "rate_identifiability", "out", name))


def _fig(name):
    return Path(paths.out("figure_reproduction", "inputs", name))


GRID_CSV = "identifiability_map_grid.csv"            # figure input (with the criterion columns)
OBSERVED_CSV = "identifiability_map_observed.csv"    # figure input
CONTOURS_CSV = "identifiability_map_contours.csv"    # figure input
SUMMARY_JSON = "identifiability_map_summary.json"    # figure input

# Per-recording rows read for the observed points (lever, obs and report stages), as paths relative to the
# repository root (below NDIF_OUT when it is set), with the script that writes each.
_RI = ("analyses", "rate_identifiability", "out")
OBSERVED_ROWS = {
    # calibrated single-timescale fits (a, nu, r1) per participant (x direction): the coordinate of the map
    "tail_workload": _RI + ("tail_excess_cog_rows.csv",),             # tail_excess_cog.py (family matb, level 0)
    "tail_ds003838": _RI + ("tail_excess_ds003838_rows.csv",),        # tail_excess_ds003838.py (arm memory)
    "tail_fmri": _RI + ("tail_excess_fmri_rows.csv",),                # tail_excess_fmri.py (level 0bk)
    "tail_meg": _RI + ("tail_excess_meg_rows.csv",),                  # HCP MEG, trial clock, 0-back (README)
    "tail_sleep": _RI + ("tail_excess_sleep_rows.csv",),              # Sleep-EDF cassette, stage N2 (README)
    "band_white_share": _RI + ("band_white_share_rows.csv",),         # COG-BCI instrument-only share (README)
    # multivariate as-fitted rate and mode spread per participant (injection-recovery arms)
    "subjects_workload": ("analyses", "workload_certificates", "features", "out",
                          "rescaling_cog_subjects.csv"),                     # features/cog_state.py --cboot 50
    "subjects_ds003838": _RI + ("injection_recovery_ds003838_subjects.csv",),   # injection_recovery_ds003838.py
    "subjects_fmri": _RI + ("injection_recovery_fmri_subjects.csv",),           # injection_recovery_fmri.py
    "subjects_meg": _RI + ("injection_recovery_meg_subjects.csv",),             # injection_recovery_meg.py
}


def _stored(*parts):
    """A file written by another script: below NDIF_OUT when it is set, else at its location in the repository."""
    base = os.environ.get("NDIF_OUT")
    return Path(base, *parts) if base else Path(paths.repo(*parts))


def log(m):
    print(time.strftime("[%H:%M:%S] ") + str(m), flush=True)


def csv_write(path, rows):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def jdump(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, indent=1, default=float, ensure_ascii=False) + "\n")


def l_eff_of(lam_dt, nu):
    """ln of the estimand a VAR(1) fit is CONSISTENT for when a white share nu is present."""
    return float(np.log(lam_dt - np.log(max(1.0 - nu, 1e-12))))


# ============================================================================ stage: equiv
def stage_equiv(args):
    rows, burn_rows = [], []
    for nm, geo in L.GEOMETRIES.items():
        if nm not in args.cohorts:
            continue
        for lam in (0.2, 1.0):
            if geo["estimator"] == "pairs":
                r = L.pairs_equivalence_check(geo, lam, args.equiv_reps, seed=SEED)
            else:
                r = L.equivalence_check(geo, lam, args.equiv_reps, seed=SEED)
            r["cohort_key"] = nm
            rows.append(r)
            log(f"  equivalence {nm:10s} lam*dt={lam:4.2f}  fast {r['mean_fast']:+.4f} "
                f"loop {r['mean_loop']:+.4f}  diff {r['diff_mean']:+.5f} "
                f"(se {r['se_diff']:.5f}, z {r['z']:+.2f})  sd ratio {r['sd_ratio']:.3f}  "
                f"{'OK' if r['ok'] else 'FAIL'}")
    # ---- burn-in: burn = 100 against a very long burn, at the slowest rate
    for nm, geo in L.GEOMETRIES.items():
        if geo["estimator"] == "pairs" or nm not in args.cohorts:
            continue
        for lam in (0.05,):
            acc = {}
            for b in (100, 1000):
                rng = np.random.default_rng(SEED + 5)
                v = []
                for _ in range(args.burn_reps):
                    X = L.gen_unit(lam, 0.0, geo["n_blocks"], geo["n_win"], geo["d"], rng)
                    f = L.fit_unit_cell(X, geo["dt"], rng=rng, burn=b)
                    v.append(f["l_bc"] if f else np.nan)
                v = np.asarray(v, float); v = v[np.isfinite(v)]
                acc[b] = (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v))), len(v))
            d = acc[100][0] - acc[1000][0]
            se = np.hypot(acc[100][1], acc[1000][1])
            burn_rows.append(dict(cohort_key=nm, cohort=geo["label"], lam_dt=lam,
                                  mean_burn100=acc[100][0], mean_burn1000=acc[1000][0],
                                  diff=d, se_diff=se, z=d / se if se > 0 else np.nan,
                                  n100=acc[100][2], n1000=acc[1000][2]))
            log(f"  burn-in     {nm:10s} lam*dt={lam:4.2f}  burn100 {acc[100][0]:+.4f} "
                f"burn1000 {acc[1000][0]:+.4f}  diff {d:+.5f} (z {d/se if se>0 else np.nan:+.2f})")
    ok = all(r.get("ok") for r in rows)
    jdump(_out("identifiability_map_equivalence.json"),
          dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"),
               equiv_reps=args.equiv_reps, burn_reps=args.burn_reps,
               n_boot=L.N_BOOT, gate_passed=bool(ok),
               equivalence=rows, burn=burn_rows))
    csv_write(_out("identifiability_map_equivalence.csv"), rows)
    if not ok:
        raise SystemExit("equivalence check FAILED - the map is not built.")
    log("  equivalence check passed")
    return ok


# ============================================================================ grid worker
def _point(job):
    """Wrapper: a failure at one grid point must not take the whole pool down with it.

    After a MemoryError the point is retried with progressively smaller simulation chunks before it is
    recorded as failed.
    """
    import gc
    for me in (2_000_000, 500_000, 125_000):
        try:
            return _point_inner(job, max_elem=me)
        except MemoryError:
            gc.collect()
            time.sleep(2.0)
            continue
        except Exception as e:                                        # noqa: BLE001
            nm, lam_dt, nu, scale, lever, K, seed = job
            return dict(cohort_key=nm, cohort=L.GEOMETRIES[nm]["label"], lever=lever,
                        lam_dt=lam_dt, nu=nu, scale=scale, K=K,
                        status=f"worker error: {type(e).__name__}", error=str(e))
    nm, lam_dt, nu, scale, lever, K, seed = job
    return dict(cohort_key=nm, cohort=L.GEOMETRIES[nm]["label"], lever=lever,
                lam_dt=lam_dt, nu=nu, scale=scale, K=K,
                status="worker error: MemoryError",
                error="out of memory after three chunk sizes")


def _point_inner(job, max_elem=2_000_000):
    (nm, lam_dt, nu, scale, lever, K, seed) = job
    geo = L.GEOMETRIES[nm]
    nb, nw, npair = L.scaled_geometry(geo, scale, lever)
    rng = np.random.default_rng(seed)
    lb, lo, hi, lr, ab = [], [], [], [], []
    n_none = 0
    for _ in range(K):
        f = L.unit_draw(lam_dt, nu, geo, nb, nw, rng, max_elem=max_elem)
        if f is None:
            n_none += 1
            continue
        a, b = L.unit_interval(f)
        lb.append(f["l_bc"]); lo.append(a); hi.append(b); lr.append(f["l_raw"])
        ab.append(f["absev"])
    lb = np.asarray(lb, float); lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    lr = np.asarray(lr, float); ab = np.asarray(ab, float)
    ok = np.isfinite(lb) & np.isfinite(lo) & np.isfinite(hi)
    n_ok = int(ok.sum())
    lt = float(np.log(lam_dt))
    le = l_eff_of(lam_dt, nu)
    row = dict(cohort_key=nm, cohort=geo["label"], modality=geo["modality"], lever=lever,
               dt=geo["dt"], d=geo["d"], n_units=geo["n_units"], estimator=geo["estimator"],
               lam_dt=lam_dt, nu=nu, scale=scale, n_blocks=nb, n_win=nw, n_pairs=npair,
               K=K, n_ok=n_ok, frac_unusable=float(1.0 - n_ok / K),
               log_lam_true=lt, log_lam_eff=le)
    if n_ok < 30:
        row.update(status="too few usable fits")
        return row
    lbo, loo, hio = lb[ok], lo[ok], hi[ok]
    w = hio - loo
    sd = float(np.std(lbo, ddof=1))
    mad = float(1.4826 * np.median(np.abs(lbo - np.median(lbo))))
    cov_t = float(np.mean((loo <= lt) & (hio >= lt)))
    cov_e = float(np.mean((loo <= le) & (hio >= le)))
    glo, ghi, gm = L.group_boot_ci(lbo, geo["n_units"], rng=np.random.default_rng(seed + 991))
    gok = np.isfinite(glo) & np.isfinite(ghi)
    row.update(
        status="ok",
        mean_log_lam=float(np.mean(lbo)), median_log_lam=float(np.median(lbo)),
        sd_log_lam=sd, robust_sd_log_lam=mad,
        se_mean_log_lam=float(sd / np.sqrt(n_ok)),
        mean_log_lam_raw=float(np.nanmean(lr[np.isfinite(lr)])),
        bias_vs_true=float(np.mean(lbo) - lt), bias_vs_eff=float(np.mean(lbo) - le),
        ci_width_unit=float(np.mean(w)), ci_width_unit_median=float(np.median(w)),
        se_ci_width=float(np.std(w, ddof=1) / np.sqrt(n_ok)),
        cov_unit_true=cov_t, se_cov_unit_true=float(np.sqrt(cov_t * (1 - cov_t) / n_ok)),
        cov_unit_eff=cov_e, se_cov_unit_eff=float(np.sqrt(cov_e * (1 - cov_e) / n_ok)),
        ci_width_group=float(np.mean((ghi - glo)[gok])) if gok.any() else np.nan,
        cov_group_true=float(np.mean((glo[gok] <= lt) & (ghi[gok] >= lt))) if gok.any() else np.nan,
        cov_group_eff=float(np.mean((glo[gok] <= le) & (ghi[gok] >= le))) if gok.any() else np.nan,
        mean_absev=float(np.mean(ab[np.isfinite(ab)])),
        mc_unstable=bool(np.std(w, ddof=1) / np.sqrt(n_ok) > 0.05 * np.mean(w)),
    )
    return row


def _jobs(cohorts, K, lever_key="lever"):
    """Every grid point of the selected cohorts.  Seeds run over ALL cohorts in GEOMETRIES order."""
    jobs = []
    s = SEED
    for nm in L.GEOMETRIES:
        geo = L.GEOMETRIES[nm]
        lever = geo[lever_key]
        if lever is None:
            continue
        for lam in L.LAM_GRID:
            for nu in L.NU_GRID:
                for sc in L.SCALES:
                    s += 7919
                    if nm in cohorts:
                        jobs.append((nm, float(lam), float(nu), float(sc), lever, K, s))
    return jobs


def _key(r):
    return (r["cohort_key"], round(float(r["lam_dt"]), 8), round(float(r["nu"]), 8),
            round(float(r["scale"]), 8), r.get("lever"))


def _resume(label):
    """Rows already computed in a previous (possibly interrupted) run of this stage."""
    jl = _out(f"identifiability_map_{label}_partial.jsonl")
    done = {}
    if jl.exists():
        for line in jl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:                                        # noqa: BLE001
                continue
            # a point that failed in a worker is NOT done - it must be recomputed
            if str(r.get("status", "")).startswith("worker error"):
                done.pop(_key(r), None)
                continue
            done[_key(r)] = r
    return done


def _run_jobs(jobs, workers, label):
    t0 = time.time()
    done = _resume(label)
    todo = [j for j in jobs
            if (j[0], round(j[1], 8), round(j[2], 8), round(j[3], 8), j[4]) not in done]
    rows = list(done.values())
    if done:
        log(f"  {label}: resuming - {len(done)} points already on disk, {len(todo)} to compute")
    if not todo:
        return rows
    jl = _out(f"identifiability_map_{label}_partial.jsonl")
    n_fail = 0
    with open(jl, "a", encoding="utf-8", newline="\n") as fh, \
            Pool(processes=workers, maxtasksperchild=40) as pool:
        for i, r in enumerate(pool.imap_unordered(_point, todo, chunksize=1), 1):
            if isinstance(r, dict) and r.get("status", "").startswith("worker error"):
                n_fail += 1
                log(f"  {label}: WORKER ERROR at {r.get('cohort_key')} "
                    f"lam={r.get('lam_dt')} nu={r.get('nu')} scale={r.get('scale')}: "
                    f"{r.get('error')}")
            rows.append(r)
            fh.write(json.dumps(r, default=float) + "\n")
            fh.flush()
            if i % 50 == 0 or i == len(todo):
                el = time.time() - t0
                log(f"  {label}: {i}/{len(todo)} points  {el/60:.1f} min elapsed, "
                    f"{el/i*(len(todo)-i)/60:.1f} min left"
                    + (f"  [{n_fail} worker errors]" if n_fail else ""))
    return rows


def stage_grid(args):
    jobs = _jobs(args.cohorts, args.K, "lever")
    log(f"  main grid: {len(jobs)} points x K={args.K} unit fits, {args.workers} workers")
    rows = _run_jobs(jobs, args.workers, "grid")
    for r in rows:
        r["arm"] = "main"
    csv_write(_fig(GRID_CSV), rows)
    log(f"  -> {GRID_CSV} ({len(rows)} rows)")
    return rows


def stage_lever(args):
    """The OTHER sample-size lever, at each cohort's observed nu only."""
    obs = observed_points()
    jobs = []
    s = SEED + 131
    for nm, geo in L.GEOMETRIES.items():
        alt = geo["lever_alt"]
        if alt is None:
            continue
        nu = float(np.round(obs[nm]["nu_med"] * 10) / 10)     # nearest grid nu
        for lam in L.LAM_GRID:
            for sc in L.SCALES:
                s += 7919
                if nm in args.cohorts:
                    jobs.append((nm, float(lam), nu, float(sc), alt, args.K_lever, s))
    log(f"  lever: {len(jobs)} points x K={args.K_lever}")
    rows = _run_jobs(jobs, args.workers, "lever")
    for r in rows:
        r["arm"] = "lever_alt"
    csv_write(_out("identifiability_map_lever.csv"), rows)
    return rows


# ============================================================================ observed points
def _rows(role):
    with open(_stored(*OBSERVED_ROWS[role]), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(rows, key):
    v = []
    for r in rows:
        x = r.get(key, "")
        try:
            y = float(x)
        except (TypeError, ValueError):
            continue
        if np.isfinite(y):
            v.append(y)
    return np.asarray(v, float)


def _boot_median(v, n=4000, seed=1):
    rng = np.random.default_rng(seed)
    if len(v) < 3:
        return float(np.median(v)) if len(v) else np.nan, np.nan, np.nan
    b = np.median(v[rng.integers(0, len(v), size=(n, len(v)))], axis=1)
    return float(np.median(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


_OBS_CACHE = {}


def observed_points():
    """Each cohort's observed (lambda*dt, nu) and its secondary coordinates - read, never refitted here."""
    if _OBS_CACHE:
        return _OBS_CACHE
    out = {}

    # --- calibrated (a, nu) fits: the coordinate of the map
    ac = _rows("tail_workload")
    cog = [r for r in ac if r["family"] == "matb" and r["level"] == "0"]
    dsr = [r for r in _rows("tail_ds003838") if r["arm"] == "memory"]
    fmr = [r for r in _rows("tail_fmri") if r["level"] == "0bk"]
    meg = [r for r in _rows("tail_meg") if r["clock"] == "trial" and r["level"] == "0bk"]
    slp = [r for r in _rows("tail_sleep") if r["stage"] == "N2"]

    src = {"COG-BCI": cog, "ds003838": dsr, "HCP_fMRI": fmr, "HCP_MEG": meg, "SLEEP": slp}

    # --- multivariate as-fitted rate: the secondary coordinate (column of the subjects file)
    h3 = {"COG-BCI": ("subjects_workload", "lam_lo"),
          "ds003838": ("subjects_ds003838", "C1_mem_lam_lo"),
          "HCP_fMRI": ("subjects_fmri", "lam_lo"),
          "HCP_MEG": ("subjects_meg", "lam_lo"),
          "SLEEP": (None, None)}          # no per-participant multivariate rate for the sleep recordings
    spread = {"COG-BCI": ("subjects_workload", "p2_sd_ell_lo"),
              "ds003838": ("subjects_ds003838", "C1_mem_p2_sd_ell_lo"),
              "HCP_fMRI": ("subjects_fmri", "p2_sd_ell_lo"),
              "HCP_MEG": ("subjects_meg", "p2_sd_ell_hi"),
              "SLEEP": (None, None)}

    for nm, rows in src.items():
        geo = L.GEOMETRIES[nm]
        dt = geo["dt"]
        if nm == "SLEEP":
            lam = _num(rows, "lam_corrected") * dt
        else:
            a = _num(rows, "a")
            lam = -np.log(np.clip(a, 1e-6, 0.9999)) * 1.0      # a is per WINDOW, so this is lam*dt
        nu = _num(rows, "nu")
        nu = nu[(nu >= -0.5) & (nu <= 1.0)]
        lm, llo, lhi = _boot_median(lam[np.isfinite(lam) & (lam > 0)])
        nm_, nlo, nhi = _boot_median(nu)
        rec = dict(cohort_key=nm, cohort=geo["label"], modality=geo["modality"], dt=dt,
                   d=geo["d"], n_units=geo["n_units"], n_blocks=geo["n_blocks"],
                   n_win=geo["n_win"], estimator=geo["estimator"],
                   n_pairs=L.scaled_geometry(geo, 1.0)[2],
                   lam_dt_med=lm, lam_dt_lo=llo, lam_dt_hi=lhi, n_lam=int(len(lam)),
                   nu_med=nm_, nu_lo=nlo, nu_hi=nhi, n_nu=int(len(nu)))
        # naive one-step coordinate
        if nm == "SLEEP":
            lu = _num(rows, "lam_uncorrected") * dt
        else:
            r1 = _num(rows, "r1")
            lu = -np.log(np.clip(r1, 1e-6, 0.9999))
        rec["lam_dt_naive_med"] = float(np.median(lu[np.isfinite(lu) & (lu > 0)])) if len(lu) else np.nan
        # multivariate as-fitted coordinate
        fn, ck = h3[nm]
        if fn:
            mv = _num(_rows(fn), ck) * dt
            m, lo_, hi_ = _boot_median(mv[np.isfinite(mv) & (mv > 0)])
            rec.update(lam_dt_multivar_med=m, lam_dt_multivar_lo=lo_, lam_dt_multivar_hi=hi_,
                       n_multivar=int(len(mv)))
        fn2, ck2 = spread[nm]
        if fn2:
            sp = _num(_rows(fn2), ck2)
            rec["sd_log_lam_modes"] = float(np.median(sp[np.isfinite(sp)])) if len(sp) else np.nan
        else:
            rec["sd_log_lam_modes"] = np.nan
        out[nm] = rec

    # instrument-only white fraction, for reference beside nu
    nuinst = _num(_rows("band_white_share"), "nu_full")
    nuinst = nuinst[(nuinst >= 0) & (nuinst <= 1)]
    for nm in out:
        out[nm]["nu_instrument_only_cogbci_median"] = float(np.median(nuinst))
    _OBS_CACHE.update(out)
    return out


# ============================================================================ stage: obs
def _obs_point(job):
    nm, lam_dt, nu, K, seed, arm, extra = job
    geo = L.GEOMETRIES[nm]
    nb, nw, npair = L.scaled_geometry(geo, 1.0)
    rng = np.random.default_rng(seed)
    lam_vec = None
    if arm == "spec":
        s = float(extra)
        z = rng.standard_normal(geo["d"])
        v = np.exp(np.log(lam_dt) + s * (z - z.mean()))
        lam_vec = v * (lam_dt / float(np.mean(v)))              # arithmetic mean == lam_dt
    lb, lo, hi = [], [], []
    for _ in range(K):
        if arm == "pca":
            X = L.gen_unit_ambient(lam_dt, nu, nb, nw, geo["d"], geo["p_amb"], rng)
            f = L.fit_unit_cell(X, geo["dt"], rng=rng)
        else:
            f = L.unit_draw(lam_dt, nu, geo, nb, nw, rng, lam_vec=lam_vec)
        if f is None:
            continue
        a, b = L.unit_interval(f)
        lb.append(f["l_bc"]); lo.append(a); hi.append(b)
    lb = np.asarray(lb, float); lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    ok = np.isfinite(lb) & np.isfinite(lo) & np.isfinite(hi)
    n_ok = int(ok.sum())
    lt = float(np.log(lam_dt))
    le = l_eff_of(lam_dt, nu)
    row = dict(arm=arm, cohort_key=nm, cohort=geo["label"], lam_dt=lam_dt, nu=nu, K=K,
               n_ok=n_ok, extra=extra, log_lam_true=lt, log_lam_eff=le,
               n_blocks=nb, n_win=nw, n_pairs=npair, n_units=geo["n_units"])
    if n_ok < 30:
        row["status"] = "too few usable fits"
        return row
    lbo, loo, hio = lb[ok], lo[ok], hi[ok]
    w = hio - loo
    sd = float(np.std(lbo, ddof=1))
    if arm == "spec" and lam_vec is not None:
        row["lam_modes"] = ";".join(f"{x:.5f}" for x in np.sort(lam_vec))
        row["log_lam_slowest_true"] = float(np.log(np.min(lam_vec)))
    glo, ghi, _ = L.group_boot_ci(lbo, geo["n_units"], rng=np.random.default_rng(seed + 13))
    g = np.isfinite(glo) & np.isfinite(ghi)
    row.update(status="ok", mean_log_lam=float(np.mean(lbo)), sd_log_lam=sd,
               se_mean_log_lam=float(sd / np.sqrt(n_ok)),
               bias_vs_true=float(np.mean(lbo) - lt), bias_vs_eff=float(np.mean(lbo) - le),
               ci_width_unit=float(np.mean(w)), se_ci_width=float(np.std(w, ddof=1) / np.sqrt(n_ok)),
               cov_unit_true=float(np.mean((loo <= lt) & (hio >= lt))),
               cov_unit_eff=float(np.mean((loo <= le) & (hio >= le))),
               ci_width_group=float(np.mean((ghi - glo)[g])),
               cov_group_true=float(np.mean((glo[g] <= lt) & (ghi[g] >= lt))),
               cov_group_eff=float(np.mean((glo[g] <= le) & (ghi[g] >= le))),
               frac_unusable=float(1.0 - n_ok / K))
    return row


def _power_point(job):
    """Direct paired power: n_units participants, two conditions lambda and lambda*e^delta."""
    nm, lam_dt, nu, delta, K, n_draw, seed = job
    geo = L.GEOMETRIES[nm]
    nb, nw, _ = L.scaled_geometry(geo, 1.0)
    rng = np.random.default_rng(seed)
    A, B = [], []
    for _ in range(K):
        fa = L.unit_draw(lam_dt, nu, geo, nb, nw, rng)
        fb = L.unit_draw(lam_dt * float(np.exp(delta)), nu, geo, nb, nw, rng)
        A.append(fa["l_bc"] if fa else np.nan)
        B.append(fb["l_bc"] if fb else np.nan)
    A = np.asarray(A, float); B = np.asarray(B, float)
    A = A[np.isfinite(A)]; B = B[np.isfinite(B)]
    if len(A) < 30 or len(B) < 30:
        return dict(cohort_key=nm, delta=delta, status="too few")
    n = geo["n_units"]
    rg = np.random.default_rng(seed + 3)
    da = A[rg.integers(0, len(A), size=(n_draw, n))]
    db = B[rg.integers(0, len(B), size=(n_draw, n))]
    dd = db - da
    m = dd.mean(axis=1)
    s = dd.std(axis=1, ddof=1)
    from scipy import stats as sst
    tcrit = float(sst.t.ppf(0.975, n - 1))
    power = float(np.mean(np.abs(m) / (s / np.sqrt(n)) > tcrit))
    return dict(cohort_key=nm, cohort=geo["label"], lam_dt=lam_dt, nu=nu, delta=float(delta),
                K=K, n_units=n, status="ok", power=power,
                observed_mean_diff=float(np.mean(B) - np.mean(A)),
                recovery_fraction=float((np.mean(B) - np.mean(A)) / delta) if delta else np.nan,
                sd_paired_diff=float(np.mean(s)))


def _upstream_point(job):
    """Band-power pipeline arm: the spectrogram chain of the sleep recordings, from a synthesised raw signal."""
    nm, a_true, K, seed, envelope = job
    from lag_lib import (BANDS, build_state_space, features_from_signal, fit_single_timescale,
                         pooled_acf_ragged, single_timescale_grid, synth_recording)
    geo = L.GEOMETRIES[nm]
    nb, nw, npair = L.scaled_geometry(geo, 1.0)
    d = 4                                    # build_state_space is the sleep state space, d = 4
    rng = np.random.default_rng(seed)
    ap = {b: float(a_true) for b in BANDS}
    sp = {b: 1.0 for b in BANDS}
    grid = single_timescale_grid(nw, min(6, nw // 5))
    lb, lo, hi, nus, aas = [], [], [], [], []
    for _ in range(K):
        blocks = []
        for _b in range(nb):
            sig, _U = synth_recording(ap, sp, nw, 1.0, rng, envelope=envelope)
            F, _ = features_from_signal(sig, 1.0)
            blocks.append(F)
        transform, _evr = build_state_space({"all": np.vstack(blocks)})
        Z = np.stack([transform(b) for b in blocks])            # (nb, nw, d)
        f = L.fit_unit_cell(Z, geo["dt"], rng=rng)
        if f is None:
            continue
        a_, b_ = L.unit_interval(f)
        lb.append(f["l_bc"]); lo.append(a_); hi.append(b_)
        r = pooled_acf_ragged([Z[i, :, 0] for i in range(nb)], grid["kmax"])
        aa, nn, _res, _ = fit_single_timescale(r, grid)
        aas.append(aa); nus.append(nn)
    lb = np.asarray(lb, float); lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    ok = np.isfinite(lb) & np.isfinite(lo) & np.isfinite(hi)
    lam_true = float(-np.log(a_true))
    out = dict(arm="upstream", cohort_key=nm, cohort=geo["label"], envelope=envelope,
               a_true=float(a_true), lam_dt_true=lam_true, log_lam_true=float(np.log(lam_true)),
               K=K, n_ok=int(ok.sum()), d=d, n_blocks=nb, n_win=nw,
               realised_a_pc1=float(np.nanmedian(aas)) if aas else np.nan,
               realised_nu_pc1=float(np.nanmedian(nus)) if nus else np.nan)
    if ok.sum() >= 30:
        w = hi[ok] - lo[ok]
        out.update(mean_log_lam=float(np.mean(lb[ok])), sd_log_lam=float(np.std(lb[ok], ddof=1)),
                   ci_width_unit=float(np.mean(w)),
                   se_ci_width=float(np.std(w, ddof=1) / np.sqrt(ok.sum())),
                   cov_unit_true=float(np.mean((lo[ok] <= np.log(lam_true))
                                               & (hi[ok] >= np.log(lam_true)))),
                   bias_vs_true=float(np.mean(lb[ok]) - np.log(lam_true)))
    return out


def _retry(fn, job, label):
    """Same protection as _point: a MemoryError is retried up to four times."""
    import gc
    last = ""
    for _ in range(4):
        try:
            return fn(job)
        except MemoryError as e:                                      # noqa: BLE001
            last = str(e)
            gc.collect()
            time.sleep(5.0)
        except Exception as e:                                        # noqa: BLE001
            return dict(status=f"worker error: {type(e).__name__}", error=str(e), arm=label)
    return dict(status="worker error: MemoryError", error=last, arm=label)


def _obs_point_r(job):
    return _retry(_obs_point, job, "obs")


def _power_point_r(job):
    return _retry(_power_point, job, "power")


def _upstream_point_r(job):
    return _retry(_upstream_point, job, "upstream")


def stage_obs(args):
    obs = observed_points()
    jobs, pjobs = [], []
    s = SEED + 4441
    for nm, o in obs.items():
        sel = nm in args.cohorts
        lam = float(np.clip(o["lam_dt_med"], 0.01, 5.0))
        nu = float(np.clip(o["nu_med"], 0.0, 0.9))
        s += 7919
        if sel:
            jobs.append((nm, lam, nu, args.K_obs, s, "obs", ""))
        s += 7919
        if sel:
            jobs.append((nm, lam, 0.0, args.K_obs, s, "obs_nu0", ""))
        if np.isfinite(o.get("lam_dt_multivar_med", np.nan)):
            s += 7919
            if sel:
                jobs.append((nm, float(o["lam_dt_multivar_med"]), nu, args.K_obs, s,
                             "obs_multivar", ""))
        if np.isfinite(o.get("sd_log_lam_modes", np.nan)):
            s += 7919
            if sel:
                jobs.append((nm, lam, nu, args.K_arm, s, "spec", float(o["sd_log_lam_modes"])))
        if L.GEOMETRIES[nm]["p_amb"]:
            s += 7919
            if sel:
                jobs.append((nm, lam, nu, args.K_arm, s, "pca", ""))
        for delta in (0.05, 0.10, 0.20, 0.40, 0.70, 1.10):
            s += 7919
            if sel:
                pjobs.append((nm, lam, nu, float(delta), args.K_arm, 3000, s))
    log(f"  obs: {len(jobs)} points, {len(pjobs)} power points")
    f_obs = _out("identifiability_map_obs.csv")
    f_pow = _out("identifiability_map_power.csv")
    f_up = _out("identifiability_map_upstream.csv")
    rows = []
    with Pool(processes=args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_obs_point_r, jobs), 1):
            rows.append(r)
            log(f"    obs {i}/{len(jobs)}  {r.get('cohort_key')} {r.get('arm')} "
                f"{r.get('status')}")
            csv_write(f_obs, rows)
    prows = []
    with Pool(processes=args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_power_point_r, pjobs), 1):
            prows.append(r)
            csv_write(f_pow, prows)
            if i % 6 == 0:
                log(f"    power {i}/{len(pjobs)}")
    # ---- band-power pipeline arm, sleep feature chain (2 channels x 4 bands, d = 4)
    ujobs = []
    for a_true in (0.9, 0.8, 0.6):
        for env in ("step", "continuous"):
            s += 7919
            if "SLEEP" in args.cohorts:
                ujobs.append(("SLEEP", a_true, args.K_up, s, env))
    urows = []
    if ujobs:
        with Pool(processes=min(args.workers, len(ujobs))) as pool:
            for i, r in enumerate(pool.imap_unordered(_upstream_point_r, ujobs), 1):
                urows.append(r)
                csv_write(f_up, urows)
                log(f"    upstream {i}/{len(ujobs)}  {r.get('status','ok')} "
                    f"realised nu={r.get('realised_nu_pc1')}")
    csv_write(f_obs, rows)
    csv_write(f_pow, prows)
    csv_write(f_up, urows)
    log("  -> identifiability_map_obs.csv, identifiability_map_power.csv, identifiability_map_upstream.csv")
    return rows, prows, urows


# ============================================================================ stage: report
def _read(p):
    p = Path(p)
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            if v in ("", "nan", "None"):
                r[k] = np.nan
                continue
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def sensitivity_and_mde(rows):
    """g = dE[log lambda-hat]/d log lambda_true by central difference along AXIS 1, then
    mde80 = (z975 + z80) * sqrt(2) * sd / (g * sqrt(n_units))."""
    by = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        by.setdefault((r["cohort_key"], r["nu"], r["scale"], r.get("lever")), []).append(r)
    for key, grp in by.items():
        grp.sort(key=lambda r: r["lam_dt"])
        lt = np.array([r["log_lam_true"] for r in grp])
        mu = np.array([r["mean_log_lam"] for r in grp])
        if len(grp) < 2:
            for r in grp:
                r["sensitivity_g"] = np.nan
                r["mde80"] = np.inf
            continue
        g = np.gradient(mu, lt)
        for r, gi in zip(grp, g):
            r["sensitivity_g"] = float(gi)
            n = r["n_units"]
            if gi > 1e-6:
                r["mde80"] = float((Z975 + Z80) * np.sqrt(2.0) * r["sd_log_lam"]
                                   / (gi * np.sqrt(n)))
            else:
                r["mde80"] = np.inf
    return rows


def apply_criterion(rows):
    for r in rows:
        if r.get("status") != "ok":
            r["u1"] = r["u2"] = r["u3"] = r["u4"] = r["certified"] = False
            continue
        r["u1"] = bool(r["ci_width_unit"] <= 0.50)
        r["u2"] = bool(0.90 <= r["cov_unit_true"] <= 0.98)
        r["u3"] = bool(r["ci_width_group"] <= 0.10 and 0.90 <= r["cov_group_true"] <= 0.98)
        r["u4"] = bool(np.isfinite(r.get("mde80", np.inf)) and r["mde80"] <= 0.18)
        r["certified"] = bool(r["u1"] and r["u2"] and r["u3"] and r["u4"])
        r["verdict"] = ("certified" if r["certified"]
                        else "precise but not accurate" if (r["u1"] and not r["u2"])
                        else "imprecise")
        binding = []
        if not r["u1"]:
            binding.append("U1 precision")
        if not r["u2"]:
            binding.append("U2 accuracy")
        if not r["u3"]:
            binding.append("U3 group precision")
        if not r["u4"]:
            binding.append("U4 resolution")
        r["binding"] = "; ".join(binding) if binding else "none"
    return rows


def interp_at(xs, ys, x):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    o = np.argsort(xs)
    return float(np.interp(np.log(x), np.log(xs[o]), ys[o]))


def stage_report(args, wall, workers):
    grid = sensitivity_and_mde(_read(_fig(GRID_CSV)))
    grid = apply_criterion(grid)
    csv_write(_fig(GRID_CSV), grid)
    lever = apply_criterion(sensitivity_and_mde(_read(_out("identifiability_map_lever.csv"))))
    if lever:
        csv_write(_out("identifiability_map_lever.csv"), lever)
    obsr = _read(_out("identifiability_map_obs.csv"))
    powr = _read(_out("identifiability_map_power.csv"))
    obs = observed_points()

    # ---------------- the observed-point readings, read off the map by interpolation
    read_rows = []
    for nm, o in obs.items():
        nu_near = min(L.NU_GRID, key=lambda x: abs(x - o["nu_med"]))
        slab = [r for r in grid if r["cohort_key"] == nm and r["nu"] == nu_near
                and r["scale"] == 1.0 and r.get("status") == "ok"]
        direct = [r for r in obsr if r["cohort_key"] == nm and r["arm"] == "obs"]
        direct = direct[0] if direct else {}
        pw = sorted([r for r in powr if r["cohort_key"] == nm and r.get("status") == "ok"],
                    key=lambda r: r["delta"])
        mde_direct = np.nan
        if pw:
            dd = np.array([r["delta"] for r in pw]); pp = np.array([r["power"] for r in pw])
            if pp.max() >= 0.80:
                mde_direct = float(np.interp(0.80, pp, dd))
        rec = dict(o)
        rec.update(nu_grid_used=nu_near,
                   ci_width_unit_interp=interp_at([r["lam_dt"] for r in slab],
                                                  [r["ci_width_unit"] for r in slab],
                                                  o["lam_dt_med"]) if slab else np.nan,
                   cov_unit_true_interp=interp_at([r["lam_dt"] for r in slab],
                                                  [r["cov_unit_true"] for r in slab],
                                                  o["lam_dt_med"]) if slab else np.nan,
                   mde80_interp=interp_at([r["lam_dt"] for r in slab],
                                          [min(r["mde80"], 9.9) for r in slab],
                                          o["lam_dt_med"]) if slab else np.nan,
                   ci_width_unit_direct=direct.get("ci_width_unit", np.nan),
                   se_ci_width_direct=direct.get("se_ci_width", np.nan),
                   cov_unit_true_direct=direct.get("cov_unit_true", np.nan),
                   cov_unit_eff_direct=direct.get("cov_unit_eff", np.nan),
                   cov_group_true_direct=direct.get("cov_group_true", np.nan),
                   ci_width_group_direct=direct.get("ci_width_group", np.nan),
                   bias_vs_true_direct=direct.get("bias_vs_true", np.nan),
                   bias_vs_eff_direct=direct.get("bias_vs_eff", np.nan),
                   mde80_direct_paired=mde_direct)
        read_rows.append(rec)
    csv_write(_fig(OBSERVED_CSV), read_rows)

    # ---------------- contours of criterion U
    contours = []
    for nm in L.GEOMETRIES:
        for nu in L.NU_GRID:
            for sc in L.SCALES:
                sl = [r for r in grid if r["cohort_key"] == nm and r["nu"] == nu
                      and r["scale"] == sc and r.get("status") == "ok"]
                if not sl:
                    continue
                sl.sort(key=lambda r: r["lam_dt"])
                u1 = [r["lam_dt"] for r in sl if r["u1"]]
                cert = [r["lam_dt"] for r in sl if r["certified"]]
                contours.append(dict(cohort_key=nm, cohort=L.GEOMETRIES[nm]["label"], nu=nu,
                                     scale=sc, n_pairs=sl[0]["n_pairs"],
                                     u1_lam_lo=min(u1) if u1 else np.nan,
                                     u1_lam_hi=max(u1) if u1 else np.nan,
                                     n_u1=len(u1), n_certified=len(cert),
                                     cert_lam_lo=min(cert) if cert else np.nan,
                                     cert_lam_hi=max(cert) if cert else np.nan,
                                     min_ci_width=min(r["ci_width_unit"] for r in sl),
                                     max_cov=max(r["cov_unit_true"] for r in sl)))
    csv_write(_fig(CONTOURS_CSV), contours)

    summary = dict(
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
        wall_time_hours=wall / 3600.0, workers=workers,
        machine=dict(platform=platform.platform(), python=sys.version.split()[0],
                     numpy=np.__version__, cpu_count=os.cpu_count()),
        axes=dict(lam_dt=list(L.LAM_GRID), nu=list(L.NU_GRID), scale=list(L.SCALES)),
        geometries=L.GEOMETRIES,
        n_grid_points=len(grid), n_lever_points=len(lever),
        criterion=dict(U1_ci_width_unit=0.50, U2_coverage=[0.90, 0.98],
                       U3_group_width=0.10, U4_mde80=0.18),
        observed=read_rows,
        certified_points=int(sum(1 for r in grid if r.get("certified"))),
        u1_points=int(sum(1 for r in grid if r.get("u1"))),
    )
    jdump(_fig(SUMMARY_JSON), summary)
    log(f"  -> {SUMMARY_JSON}, {OBSERVED_CSV}, {CONTOURS_CSV}, {GRID_CSV}")
    return summary


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", default="all",
                    choices=["equiv", "grid", "lever", "obs", "report", "all"])
    ap.add_argument("--cohorts", nargs="+", default=list(L.GEOMETRIES),
                    choices=list(L.GEOMETRIES))
    ap.add_argument("--workers", type=int, default=13)
    ap.add_argument("--K", type=int, default=500)
    ap.add_argument("--K_lever", type=int, default=300)
    ap.add_argument("--K_obs", type=int, default=3000)
    ap.add_argument("--K_arm", type=int, default=800)
    ap.add_argument("--K_up", type=int, default=250)
    ap.add_argument("--equiv_reps", type=int, default=300)
    ap.add_argument("--burn_reps", type=int, default=300)
    a = ap.parse_args()
    t0 = time.time()
    log(f"stage={a.stage} cohorts={','.join(a.cohorts)} workers={a.workers} K={a.K}")
    if a.stage in ("equiv", "all"):
        log("STAGE equiv")
        stage_equiv(a)
    if a.stage in ("grid", "all"):
        log("STAGE grid")
        stage_grid(a)
    if a.stage in ("lever", "all"):
        log("STAGE lever")
        stage_lever(a)
    if a.stage in ("obs", "all"):
        log("STAGE obs")
        stage_obs(a)
    if a.stage in ("report", "all"):
        log("STAGE report")
        stage_report(a, time.time() - t0, a.workers)
    log(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
