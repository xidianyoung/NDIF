# -*- coding: utf-8 -*-
r"""The successive-ratio profile and the calibrated white share of the sleep state, by stage.

The question.  The sleep state carries an uncorrelated share that rises with non-REM depth.  Under
the single-timescale model that share IS nu, and a depth-dependent rate change could be nothing but
a rising white fraction.  The successive-ratio profile decides, with no free parameters:

    single timescale   g(k) = r(k+1)/r(k) CONSTANT in k  -> nu is estimable, a = g, and the
                                                           restoring rate becomes measurable
    second timescale   g(k) RISES with k                 -> several relaxation timescales

What this script does
---------------------
* It reads the sleep window cache (lag_lib: log10 short-time band power on NON-OVERLAPPING windows at
  2.0, 1.0 and 0.5 s, per bout) and puts it through the robust-z + PCA (d = 4) state space.
* The profile is computed PER DIRECTION (200 fixed random directions, the same set for every night
  and stage), never on an r(k) averaged over directions: averaging autocorrelations across
  directions of a multidimensional state is itself a mixture of exponentials.  The across-direction
  median is taken of each STATISTIC, not of r.
* The (a, nu) pair is the CALIBRATED fit of lag_lib.fit_single_timescale_many: the pair whose
  EXPECTED r_hat at this bout length reproduces the observed r_hat.  The naive
  nu = 1 - r(1)/[r(2)/r(1)] is biased by block-mean centring and is reported beside it.
* Every constancy claim carries a MATCHED CEILING measured in the same run: the matched
  single-timescale null at the same bout count, bout length, unit count and fitted (a, nu).
* The instrumental part of nu (nu_inst) is measured separately, from the phase-randomised
  instrument-noise covariance cached per stage and window (<stem>_noise.npz), carried into the state
  space by the exact Jacobian of the affine transform: the phase-surrogate share of the observed
  variance.  nu_excess = nu - nu_inst.
* Calibration on the real recordings (every third cassette recording, 1.0-s windows), both injected
  into the feature series before standardisation and before the PCA that defines the space:
  additive white noise of KNOWN share (0.10, 0.25, 0.50) -> the ground truth for nu; temporal
  decimation by 2 -> the ground truth for a, which must go to a^2.
* The temazepam/placebo contrast of the telemetry nights (subject table ST-subjects.xls).

Subject is the unit: per-night statistics are averaged over a subject's nights, and group means
carry percentile-bootstrap intervals (10,000 draws) from one random stream (seed 20260922), drawn in
the order usable-lag table, per-stage summaries, group profiles and matched nulls, stage contrasts,
calibration, drug contrast.

Inputs:  the sleep window cache, analyses.common.paths.cache("sleep_states") (environment variable
         NDIF_SLEEP_STATES_CACHE; format in lag_lib.py): <stem>_win{2,1,0.5}.npz and <stem>_noise.npz
         for the 153 cassette (SC*) and 44 telemetry (ST*) recordings; ST-subjects.xls in the
         Sleep-EDF directory (NDIF_SLEEPEDF_RAW) for the drug arm (read with xlrd).
Outputs: analyses/rate_identifiability/out/sleep_lag_profile.json (usable-lag table, per-stage
         summaries, profile calls, stage contrasts, calibration, drug contrast) and
         analyses/rate_identifiability/out/sleep_lag_profile_rows.csv (one row per recording, stage,
         window and injection).
Supports Fig. 3b (phase-surrogate share nu_inst and white share nu_fit by stage at 1.0 s, via
lag_report.py) and Fig. 3c / Supplementary Table 1 (via window_scaling.py).
Runtime: about three minutes with four workers.

    python analyses/rate_identifiability/sleep_lag_profile.py --workers 4 --calib-every 3 --windows 2.0 1.0 0.5
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse                                                                   # noqa: E402
import csv                                                                        # noqa: E402
import json                                                                       # noqa: E402
import sys                                                                        # noqa: E402
import time                                                                       # noqa: E402
import traceback                                                                  # noqa: E402
import warnings                                                                   # noqa: E402
import zlib                                                                       # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed                  # noqa: E402
from pathlib import Path                                                          # noqa: E402

import numpy as np                                                                # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)    # empty-slice means of all-NaN profiles
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyses.common import paths                                                 # noqa: E402
from lag_lib import (D_STATE, RUNGS, STAGES, acf_ou_white, boot_mean, cache_stems,  # noqa: E402
                     chol_from_acf, fit_single_timescale_many, gen_blocks, load_cell,
                     load_noise_R, paired, pooled_acf_many, profile_many, single_timescale_grid,
                     state_space)

SEED = 20260922
N_DIR = 200
KMAX = 6
MIN_BOUTS = 5


def directions(d=D_STATE, n=N_DIR, seed=SEED):
    """One fixed set of directions, shared by every night and every stage, so no
    contrast can be produced by a different direction set."""
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((n, d))
    return V / np.linalg.norm(V, axis=1, keepdims=True)


V_FIXED = directions()


def lag0_cov(blocks):
    """Block-centred lag-0 covariance of a (n_bout, n_win, d) stack."""
    c = blocks - blocks.mean(axis=1, keepdims=True)
    F = c.reshape(-1, c.shape[-1])
    return F.T @ F / max(len(F) - 1, 1)


def stage_raw(Zs, V=V_FIXED, kmax=KMAX):
    """One stage of one night: the PER-DIRECTION autocorrelation r(1..kmax) and
    the per-direction variance.  No statistic is formed here - the usable-lag cut
    is a single declared rule applied once, in aggregation, to every cell and to
    every matched null alike."""
    P = np.einsum("btd,kd->kbt", Zs, V)                 # (n_dir, n_bout, n_win)
    R = pooled_acf_many(P, kmax)                        # (n_dir, kmax)
    C0 = lag0_cov(Zs)
    var_dir = np.einsum("kd,de,ke->k", V, C0, V)
    return R, C0, var_dir


def inject_white(feat, nu_inj, rng):
    """Add an additive white component of KNOWN variance share nu_inj to the real
    FEATURE series, before standardisation and before the PCA.  Per feature, the
    added variance is nu_inj/(1 - nu_inj) times that feature's own pooled
    within-bout variance, so the resulting share is exactly nu_inj."""
    pooled = np.vstack([v.reshape(-1, v.shape[-1]) for v in feat.values()])
    sd = pooled.std(axis=0)
    k = np.sqrt(nu_inj / max(1e-12, 1.0 - nu_inj))
    return {g: v + k * sd * rng.standard_normal(v.shape) for g, v in feat.items()}


def decimate(feat, step=2):
    """Exact time rescaling: keep every `step`-th window.  a -> a^step."""
    return {g: v[:, ::step, :] for g, v in feat.items()}


def run_one(args):
    stem, win_s, tag, inj = args
    try:
        feat, meta = load_cell(stem, win_s, tag=tag)
        if feat is None:
            return dict(stem=stem, win_s=win_s, inj=inj, error="no cell")
        # deterministic across processes and runs (crc32 of the job key)
        rng = np.random.default_rng(
            SEED + zlib.crc32(f"{stem}|{win_s}|{inj}".encode("utf-8")))
        kind, val = (inj if inj else (None, None))
        if kind == "white":
            feat = inject_white(feat, val, rng)
        elif kind == "decim":
            feat = decimate(feat, int(val))
        feat = {g: v for g, v in feat.items() if v.shape[0] >= MIN_BOUTS
                and v.shape[1] >= KMAX + 3}
        if len(feat) < 2:
            return dict(stem=stem, win_s=win_s, inj=inj, error="too few bouts")
        Z, evr, J = state_space(feat, with_jacobian=True)
        n_win = int(next(iter(Z.values())).shape[1])
        grid = single_timescale_grid(n_win, KMAX)
        cells = []
        for g in STAGES:
            if g not in Z:
                continue
            R, C0, var_dir = stage_raw(Z[g])
            Rn = load_noise_R(stem, g, win_s, J)
            if Rn is not None and kind is None:
                num = np.einsum("kd,de,ke->k", V_FIXED, Rn, V_FIXED)
                with np.errstate(invalid="ignore", divide="ignore"):
                    nu_inst_dir = num / np.maximum(var_dir, 1e-300)
                nu_inst_tr = float(np.trace(Rn) / max(np.trace(C0), 1e-300))
            else:
                nu_inst_dir = np.full(len(V_FIXED), np.nan)
                nu_inst_tr = np.nan
            dt = float(win_s) * (int(val) if kind == "decim" else 1)
            cells.append(dict(stem=stem, subject=stem[:5], stage=g, win_s=win_s,
                              inj_kind=kind or "", inj_val=(val if val is not None else ""),
                              dt_s=dt, n_bout=int(Z[g].shape[0]),
                              n_win=int(Z[g].shape[1]), evr=float(np.sum(evr)),
                              tr_C0=float(np.trace(C0)),
                              n_pairs=int(Z[g].shape[0] * (Z[g].shape[1] - 1)),
                              R=R.astype(np.float32),
                              var_dir=np.asarray(var_dir, np.float32),
                              nu_inst_dir=np.asarray(nu_inst_dir, np.float32),
                              nu_inst_tr=nu_inst_tr))
        return dict(stem=stem, win_s=win_s, inj=inj, cells=cells)
    except Exception:                                    # noqa: BLE001
        return dict(stem=stem, win_s=win_s, inj=inj, error=traceback.format_exc(limit=3))


# ------------------------------------------------------------- aggregation
def by_subject(rows, key, stage, win_s, inj_kind=""):
    """Subject is the unit: mean over that subject's nights."""
    per = {}
    for r in rows:
        if r["stage"] != stage or r["win_s"] != win_s or r.get("inj_kind", "") != inj_kind:
            continue
        v = r.get(key)
        if v is None or not np.isfinite(v):
            continue
        per.setdefault(r["subject"], []).append(float(v))
    return {s: float(np.mean(v)) for s, v in per.items()}


def group_profile(Rsub, K):
    """The group successive-ratio profile from the SUBJECT-MEAN autocorrelation.

    Rsub is (n_subjects, kmax) of per-subject r(1..kmax).  A ratio of averages,
    not an average of ratios: a single night whose r(k) lands on its own sampling
    noise would otherwise send its ratio to infinity.  Returns the profile, its
    OLS slope on k, and the exponential index of the mean.
    """
    Rb = np.nanmean(np.asarray(Rsub, float), axis=0)[:K]
    with np.errstate(invalid="ignore", divide="ignore"):
        g = np.where(np.abs(Rb[:-1]) > 1e-9, Rb[1:] / Rb[:-1], np.nan)
        ok = np.isfinite(g)
        if ok.sum() >= 2:
            k = np.arange(1, K)[ok]
            slope = float(np.polyfit(k, g[ok], 1)[0])
        else:
            slope = np.nan
        E = (float(np.log(Rb[1]) / (2.0 * np.log(Rb[0])))
             if (len(Rb) > 1 and Rb[0] > 0 and Rb[1] > 0) else np.nan)
    return dict(r_bar=Rb.tolist(), g=np.asarray(g, float).tolist(),
                slope=slope, exp_index=E)


def group_profile_boot(Rsub, K, rng, n_boot=4000):
    R = np.asarray(Rsub, float)
    n = len(R)
    base = group_profile(R, K)
    sl, ei, gs = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        p = group_profile(R[idx], K)
        sl.append(p["slope"]); ei.append(p["exp_index"]); gs.append(p["g"])
    with np.errstate(invalid="ignore"):
        base["slope_lo"] = float(np.nanpercentile(sl, 2.5))
        base["slope_hi"] = float(np.nanpercentile(sl, 97.5))
        base["exp_index_lo"] = float(np.nanpercentile(ei, 2.5))
        base["exp_index_hi"] = float(np.nanpercentile(ei, 97.5))
        G = np.array(gs, float)
        base["g_lo"] = np.nanpercentile(G, 2.5, axis=0).tolist()
        base["g_hi"] = np.nanpercentile(G, 97.5, axis=0).tolist()
    base["n_subjects"] = int(n)
    return base


def matched_null_group(a, nu, n_bout, n_win, n_units, K, rng, kmax=KMAX,
                       draws=6000, n_rep=2000):
    """The ceiling.  The single-timescale model at exactly the fitted (a, nu) and the observed
    geometry, pushed through the SAME group path: n_units synthetic nights -> subject-mean
    r -> ratio -> slope."""
    a = float(np.clip(a, 0.02, 0.985)); nu = float(np.clip(nu, 0.0, 0.92))
    L, _ = chol_from_acf(acf_ou_white(a, nu, n_win), n_win)
    X = gen_blocks(L, draws, int(max(2, n_bout)), int(n_win), rng)
    R = pooled_acf_many(X, kmax)
    good = np.isfinite(R).all(axis=1)
    R = R[good]
    if len(R) < 50:
        return None
    sl, ei = [], []
    for _ in range(n_rep):
        idx = rng.integers(0, len(R), int(max(2, n_units)))
        p = group_profile(R[idx], K)
        sl.append(p["slope"]); ei.append(p["exp_index"])
    with np.errstate(invalid="ignore"):
        return dict(a=a, nu=nu, n_bout=int(n_bout), n_win=int(n_win),
                    n_units=int(n_units), K=int(K), n_rep=int(n_rep),
                    slope_mean=float(np.nanmean(sl)),
                    slope_q95=float(np.nanpercentile(sl, 95)),
                    slope_lo=float(np.nanpercentile(sl, 2.5)),
                    slope_hi=float(np.nanpercentile(sl, 97.5)),
                    E_mean=float(np.nanmean(ei)),
                    E_lo=float(np.nanpercentile(ei, 2.5)),
                    E_hi=float(np.nanpercentile(ei, 97.5)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--windows", type=float, nargs="+", default=[2.0, 1.0, 0.5])
    ap.add_argument("--calib-every", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N cassette and N telemetry recordings (a quick check)")
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    jobs = []
    stems_sc = cache_stems("SC", win_s=1.0)
    stems_st = cache_stems("ST", win_s=1.0)
    if a.limit:
        stems_sc = stems_sc[: a.limit]
        stems_st = stems_st[: a.limit]
    for w in a.windows:
        for s in stems_sc + stems_st:
            jobs.append((s, w, "", None))
    # calibration on the REAL recordings, every Nth cassette recording
    cal = stems_sc[:: a.calib_every]
    for s in cal:
        for nu_inj in (0.10, 0.25, 0.50):
            jobs.append((s, 1.0, "", ("white", nu_inj)))
        jobs.append((s, 1.0, "", ("decim", 2)))
    print(f"{len(stems_sc)} cassette + {len(stems_st)} telemetry recordings; "
          f"{len(jobs)} jobs; {a.workers} workers", flush=True)

    rows, errs = [], []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, j): j for j in jobs}
        for k, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if "error" in r:
                errs.append(dict(stem=r["stem"], win_s=r["win_s"],
                                 error=str(r["error"]).splitlines()[-1][:120]))
            else:
                rows += r["cells"]
            if k % 100 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    # ---------------------------------------------------------------- the
    # usable-lag rule, declared once and applied to every cell and every matched
    # null alike: read lags 1..K, where K is the largest lag <= KMAX at which the
    # stage's subject-mean r(K) is still above zero by its own subject bootstrap.
    # A profile read past that lag is a ratio of noise to noise.  The cut is chosen
    # on the UNINJECTED cassette cells only, so no injection and no stage contrast
    # can move it.
    SE_MULT = 3.0
    GRIDS = {}

    def grid_for(n_win, K):
        key = (int(n_win), int(K))
        if key not in GRIDS:
            GRIDS[key] = single_timescale_grid(int(n_win), int(K))
        return GRIDS[key]

    def subject_R(cells):
        """(n_subjects, kmax): each subject's mean over nights of the
        across-direction median r(1..kmax)."""
        per = {}
        for c in cells:
            v = np.nanmedian(np.asarray(c["R"], float), axis=0)
            per.setdefault(c["subject"], []).append(v)
        return np.array([np.nanmean(np.stack(v), axis=0)
                         for v in per.values()], float)

    def k_use_for(cells):
        """The largest lag whose SUBJECT-MEAN r is still above zero by its own
        subject bootstrap.  Beyond it the ratio is noise over noise."""
        if len(cells) < 5:
            return 0
        Rs = subject_R(cells)
        if len(Rs) < 5:
            return 0
        bm = np.array([np.nanmean(Rs[rng.integers(0, len(Rs), len(Rs))], axis=0)
                       for _ in range(2000)])
        lo = np.nanpercentile(bm, 2.5, axis=0)
        K = 0
        for k in range(Rs.shape[1]):
            if np.isfinite(lo[k]) and lo[k] > 0:
                K = k + 1
            else:
                break
        return K

    def cell_stats(c, K):
        # the two-parameter single-timescale fit uses EVERY lag - a small r(k) is information
        # about (a, nu), not noise to be dropped - while the model-free profile
        # slope is read only over the usable lags 1..K.
        Rfull = np.asarray(c["R"], float)
        R = Rfull[:, :max(K, 2)]
        S = profile_many(R)
        a_, nu_, res_ = fit_single_timescale_many(Rfull, grid_for(c["n_win"], Rfull.shape[1]))
        dt = c["dt_s"]
        with np.errstate(invalid="ignore", divide="ignore"):
            am = float(np.nanmedian(a_))
            num = float(np.nanmedian(nu_))
            r1 = float(np.nanmedian(R[:, 0]))
            nu_i = float(np.nanmedian(c["nu_inst_dir"]))
            lsp = float(np.nanmedian(0.5 * np.log(np.maximum(c["var_dir"], 1e-300))))
            return dict(
                stem=c["stem"], subject=c["subject"], stage=c["stage"],
                win_s=c["win_s"], inj_kind=c["inj_kind"], inj_val=c["inj_val"],
                dt_s=dt, n_bout=c["n_bout"], n_win=c["n_win"],
                n_pairs=c["n_pairs"], k_use=int(K), evr=c["evr"],
                tr_C0=c["tr_C0"],
                slope=float(np.nanmedian(S["slope"])),
                exp_index=float(np.nanmedian(S["exp_index"])),
                nu_naive=float(np.nanmedian(S["nu_naive"])),
                parity=float(np.nanmedian(S["parity"])),
                a_fit=am, nu_fit=num,
                single_timescale_resid=float(np.nanmedian(res_)),
                nu_inst=nu_i, nu_inst_tr=c["nu_inst_tr"],
                nu_excess=(num - nu_i) if np.isfinite(nu_i) else np.nan,
                r1=r1,
                lam_calibrated=(-np.log(am) / dt) if 0 < am < 1 else np.nan,
                lam_onestep=(-np.log(r1) / dt) if 0 < r1 < 1 else np.nan,
                log_spread=lsp,
                log_spread_corrected=lsp + 0.5 * np.log(max(1e-6, 1.0 - num)),
                r_med=np.nanmedian(np.asarray(c["R"], float), axis=0).tolist(),
                g_med=np.nanmedian(np.asarray(S["G"], float), axis=0).tolist())

    K_TABLE = {}
    for w in a.windows:
        for g in STAGES:
            sel = [c for c in rows if c["inj_kind"] == "" and c["stage"] == g
                   and c["win_s"] == w and c["stem"].startswith("SC")]
            K_TABLE["%s|%g" % (g, w)] = k_use_for(sel)

    def K_of(c):
        return K_TABLE.get("%s|%g" % (c["stage"], c["win_s"]), 0)

    stats = [cell_stats(c, max(K_of(c), 2)) for c in rows]

    base = [r for r in stats if r["inj_kind"] == "" and r["stem"].startswith("SC")]
    tele = [r for r in stats if r["inj_kind"] == "" and r["stem"].startswith("ST")]
    res = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), seed=SEED,
               n_dir=N_DIR, kmax=KMAX, se_mult=SE_MULT, k_table=K_TABLE,
               usable_lag_rule=("lags 1..K with K the largest lag whose subject-mean "
                                "r(K) is above zero by its subject bootstrap (2.5th "
                                "percentile > 0); chosen on the uninjected cassette cells "
                                "and applied identically to the data and to every matched null"),
               n_recordings_cassette=len({r["stem"] for r in base}),
               n_subjects_cassette=len({r["subject"] for r in base}),
               n_recordings_telemetry=len({r["stem"] for r in tele}),
               n_cells_dropped_k_lt_2=int(sum(1 for c in rows if K_of(c) < 2)),
               n_errors=len(errs), errors=errs[:40], per_stage={}, rungs={},
               decision={}, calibration={}, drug={})

    for w in a.windows:
        for g in STAGES:
            cell = {}
            for key in ("slope", "exp_index", "a_fit", "nu_fit", "nu_naive",
                        "nu_inst", "nu_inst_tr", "nu_excess", "single_timescale_resid",
                        "lam_calibrated", "lam_onestep", "log_spread",
                        "log_spread_corrected", "parity", "n_bout", "n_win", "r1"):
                d = by_subject(base, key, g, w)
                if d:
                    cell[key] = boot_mean(list(d.values()), rng=rng)
            prof = [r for r in base if r["stage"] == g and r["win_s"] == w]
            if prof:
                cell["r_med"] = np.nanmedian(np.array([r["r_med"] for r in prof],
                                                      float), axis=0).tolist()
                cell["g_med"] = np.nanmedian(np.array([r["g_med"] for r in prof],
                                                      float), axis=0).tolist()
                cell["n_nights"] = len(prof)
                cell["k_use"] = K_TABLE["%s|%g" % (g, w)]
            res["per_stage"]["%s|%g" % (g, w)] = cell
            K = K_TABLE["%s|%g" % (g, w)]
            raw = [c for c in rows if c["inj_kind"] == "" and c["stage"] == g
                   and c["win_s"] == w and c["stem"].startswith("SC")]
            if cell.get("a_fit") and np.isfinite(cell["a_fit"]["mean"]) and K >= 2 and raw:
                Rs = subject_R(raw)
                gp = group_profile_boot(Rs, K, rng)
                cell["group_profile"] = gp
                nb = int(round(cell["n_bout"]["mean"]))
                nw = int(round(cell["n_win"]["mean"]))
                nu_ = float(np.clip(cell["nu_fit"]["mean"], 0, 0.92))
                nul = matched_null_group(cell["a_fit"]["mean"], nu_, nb, nw,
                                         len(Rs), K, rng)
                obs = gp["slope"]
                res["decision"]["%s|%g" % (g, w)] = dict(
                    k_use=K, n_subjects=len(Rs),
                    observed_group_slope=obs,
                    observed_slope_ci=[gp["slope_lo"], gp["slope_hi"]],
                    observed_g_profile=gp["g"], observed_g_lo=gp["g_lo"],
                    observed_g_hi=gp["g_hi"], r_bar=gp["r_bar"],
                    matched_null=nul,
                    call=("second_timescale" if (nul and obs > nul["slope_q95"])
                          else "single_timescale_flat"),
                    excess=(obs - nul["slope_mean"]) if nul else None,
                    E_observed=gp["exp_index"],
                    E_observed_ci=[gp["exp_index_lo"], gp["exp_index_hi"]],
                    E_matched_null=(nul["E_mean"] if nul else None),
                    E_matched_null_ci=([nul["E_lo"], nul["E_hi"]] if nul else None),
                    E_excess=((gp["exp_index"] - nul["E_mean"]) if nul else None),
                    per_night_median_slope=(cell["slope"]["mean"] if "slope" in cell else None))

    for w in a.windows:
        for shallow, deep in RUNGS:
            for key in ("nu_fit", "nu_excess", "nu_inst", "a_fit", "lam_calibrated",
                        "lam_onestep", "log_spread", "log_spread_corrected",
                        "exp_index", "slope", "r1"):
                ds = by_subject(base, key, shallow, w)
                dd = by_subject(base, key, deep, w)
                common = sorted(set(ds) & set(dd))
                if len(common) < 5:
                    continue
                va = np.array([dd[s_] for s_ in common])
                vb = np.array([ds[s_] for s_ in common])
                st = paired(va, vb, rng=rng)
                st["n_subjects"] = len(common)
                st["mean_shallow"] = float(np.mean(vb))
                st["mean_deep"] = float(np.mean(va))
                if key in ("lam_calibrated", "lam_onestep"):
                    m = (va > 0) & (vb > 0)
                    st_log = paired(np.log(va[m]), np.log(vb[m]), rng=rng)
                    st["dlog"] = st_log
                    st["fold"] = float(np.exp(st_log["mean"]))
                    st["fold_lo"] = float(np.exp(st_log["lo"]))
                    st["fold_hi"] = float(np.exp(st_log["hi"]))
                res["rungs"]["%s->%s|%s|%g" % (shallow, deep, key, w)] = st

    for kind, vals in (("white", (0.10, 0.25, 0.50)), ("decim", (2,))):
        for v in vals:
            inj = [r for r in stats if r["inj_kind"] == kind and r["inj_val"] == v]
            ref_stems = {r["stem"] for r in inj}
            ref = [r for r in base if r["stem"] in ref_stems and r["win_s"] == 1.0]
            for g in list(STAGES) + ["ALL"]:
                A = {(r["stem"], r["stage"]): r for r in inj
                     if g == "ALL" or r["stage"] == g}
                B = {(r["stem"], r["stage"]): r for r in ref
                     if g == "ALL" or r["stage"] == g}
                common = sorted(set(A) & set(B))
                if len(common) < 5:
                    continue
                nu_o = np.array([A[k]["nu_fit"] for k in common])
                nu_r = np.array([B[k]["nu_fit"] for k in common])
                a_o = np.array([A[k]["a_fit"] for k in common])
                a_r = np.array([B[k]["a_fit"] for k in common])
                cellc = dict(n=len(common),
                             nu_obs=boot_mean(nu_o, rng=rng),
                             nu_ref=boot_mean(nu_r, rng=rng),
                             d_nu=boot_mean(nu_o - nu_r, rng=rng),
                             a_obs=boot_mean(a_o, rng=rng),
                             a_ref=boot_mean(a_r, rng=rng),
                             slope_obs=boot_mean([A[k]["slope"] for k in common],
                                                 rng=rng),
                             slope_ref=boot_mean([B[k]["slope"] for k in common],
                                                 rng=rng))
                if kind == "white":
                    pred = v + (1.0 - v) * nu_r
                    cellc["nu_predicted"] = boot_mean(pred, rng=rng)
                    cellc["nu_error_vs_prediction"] = boot_mean(nu_o - pred, rng=rng)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        frac = (nu_o - nu_r) / (v * np.maximum(1 - nu_r, 1e-9))
                    cellc["nu_recovery_fraction"] = boot_mean(frac, rng=rng)
                    cellc["a_invariance_dlog"] = boot_mean(
                        np.log(np.clip(a_o, 1e-6, None))
                        - np.log(np.clip(a_r, 1e-6, None)), rng=rng)
                else:
                    m = (a_o > 0) & (a_r > 0)
                    cellc["a_expected_if_exact"] = boot_mean(a_r[m] ** int(v), rng=rng)
                    cellc["log_a_obs_minus_expected"] = boot_mean(
                        np.log(a_o[m]) - int(v) * np.log(a_r[m]), rng=rng)
                    lo_ = np.array([A[k]["lam_calibrated"] for k in common], float)
                    lr_ = np.array([B[k]["lam_calibrated"] for k in common], float)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        cellc["lam_ratio_obs"] = boot_mean(lo_ / np.maximum(lr_, 1e-9),
                                                           rng=rng)
                res["calibration"]["%s|%s|%s" % (kind, v, g)] = cellc

    try:
        xls = Path(paths.path("NDIF_SLEEPEDF_RAW")) / "ST-subjects.xls"
        drug = read_st_subjects(xls)
    except Exception as e:                               # noqa: BLE001
        drug = None
        res["drug"]["error"] = "%s: %s" % (type(e).__name__, e)
    if drug:
        res["drug"]["n_assignments"] = len(drug)
        for w in a.windows:
            for g in STAGES:
                got = {}
                for r in tele:
                    if r["stage"] != g or r["win_s"] != w:
                        continue
                    arm, subj = st_label(r["stem"], drug)
                    if arm is not None:
                        got.setdefault(subj, {})[arm] = r
                pairs = [(s_, v["temazepam"], v["placebo"]) for s_, v in got.items()
                         if "temazepam" in v and "placebo" in v]
                if len(pairs) < 5:
                    continue
                cellc = {"n_pairs": len(pairs)}
                for key in ("nu_fit", "nu_excess", "a_fit", "lam_calibrated",
                            "log_spread", "log_spread_corrected", "exp_index",
                            "slope"):
                    dv = [p[1][key] - p[2][key] for p in pairs
                          if np.isfinite(p[1][key]) and np.isfinite(p[2][key])]
                    cellc[key] = boot_mean(dv, rng=rng)
                res["drug"]["%s|%g" % (g, w)] = cellc

    out_json = paths.out("analyses", "rate_identifiability", "out", "sleep_lag_profile.json")
    with open(out_json, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(res, indent=1, default=float) + "\n")
    keys = ["stem", "subject", "stage", "win_s", "inj_kind", "inj_val", "dt_s",
            "n_bout", "n_win", "n_pairs", "k_use", "slope", "exp_index", "a_fit",
            "nu_fit", "nu_naive", "nu_inst", "nu_inst_tr", "nu_excess", "single_timescale_resid",
            "r1", "lam_calibrated", "lam_onestep", "log_spread",
            "log_spread_corrected", "parity", "tr_C0", "evr", "r_med", "g_med"]
    out_rows = paths.out("analyses", "rate_identifiability", "out", "sleep_lag_profile_rows.csv")
    with open(out_rows, "w", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore", lineterminator="\n")
        wri.writeheader()
        for r in stats:
            rr = dict(r)
            rr["r_med"] = ";".join("%.5f" % x for x in r["r_med"])
            rr["g_med"] = ";".join("%.5f" % x for x in r["g_med"])
            wri.writerow(rr)
    print("done in %.1f min; %d cells, %d errors -> %s, %s"
          % ((time.time() - t0) / 60, len(stats), len(errs), out_json, out_rows), flush=True)


def read_st_subjects(path):
    """{(subject_int, night_int): 'placebo'|'temazepam'} from the Sleep-EDF
    Telemetry subject table (ST-subjects.xls)."""
    import xlrd
    sh = xlrd.open_workbook(str(path)).sheet_by_index(0)
    m = {}
    for r in range(2, sh.nrows):
        v = [sh.cell_value(r, c) for c in range(sh.ncols)]
        if not isinstance(v[0], float):
            continue
        s_ = int(v[0])
        m[(s_, int(v[3]))] = "placebo"
        m[(s_, int(v[5]))] = "temazepam"
    return m


def st_label(stem, m):
    try:
        subj, night = int(stem[3:5]), int(stem[5])
    except Exception:                                    # noqa: BLE001
        return None, None
    return m.get((subj, night)), subj


if __name__ == "__main__":
    main()
