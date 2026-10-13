# -*- coding: utf-8 -*-
r"""Per-night rows of the 2-s bout-spread chain (Sleep-EDF cassette or telemetry): every kept escape
direction of every night-stage cell with its spread, pooled autocorrelation r(1..5), lag-1
increment second moment and bootstrap intervals, and the per-cell summaries.

Per night (spread_lib, THE 2-S BOUT CHAIN): bouts of each fine stage with at least 5 usable bouts,
the robust z + PCA (d = 4) state of the night's pooled bouts, and, for every stage a and every
other stage b, the direction u from the mean of a to the mean of b.  For the bout-centred
projections of a's bouts on u:
  spread    sqrt(mean of squares); kept if the pooled lag-1 coefficient is <= 0.6 and the
            odd/even split-half spread ratio is within 0.7-1.4
  acf       pooled lag-k coefficients r(1..5) (r(1) is the pooled lag-1 coefficient)
  lambda    from the ACF shape over the lags before the first r(k) <= 0.05, fitted through the
            origin (n_lags >= 2 = resolved; otherwise the one-sided bound
            -ln(0.05) / ((n_lags + 1) dt) is recorded)
  D         lambda m2 / (2 (1 - exp(-lambda dt))), m2 the lag-1 increment second moment
  bootstrap bouts resampled within the stage, 400 draws; a matched discrete-OU surrogate per
            direction (200 draws) for the ratio sqrt(D/lambda) / spread
A night-stage cell takes the median spread of its kept directions.  Random stream per night:
numpy default_rng(20260921 + crc32(stem) % 100000).

Inputs   Sleep-EDF Expanded EDF files and hypnograms in NDIF_SLEEPEDF_RAW.
Output   paths.cache("sleep_bout_rows")/sleep_alternative_processing_rows.json (cassette, 153
         nights) or telemetry_crossover_rows.json (telemetry, 44 nights, 2 without two usable
         stages) (NDIF_SLEEP_BOUT_ROWS_CACHE, default analyses/_work/sleep_bout_rows); the Source
         Data hold both under analysis_inputs/telemetry/stored_outputs/.
Used by  crossover.py (Supplementary Table 8 alternative processing, Supplementary Table 9,
         Fig. 4c) and drug_power.py.
Runtime  cassette about 11 min with 2 workers; telemetry about 1.5 min with 1 worker.  At most two
         workers: each holds a whole night of EEG in memory.

    python analyses/sleep_telemetry/bout_spread_rows.py --library cassette --workers 2
    python analyses/sleep_telemetry/bout_spread_rows.py --library telemetry --workers 1
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse                                                       # noqa: E402
import gc                                                             # noqa: E402
import json                                                           # noqa: E402
import platform                                                       # noqa: E402
import sys                                                            # noqa: E402
import time                                                           # noqa: E402
import traceback                                                      # noqa: E402
import zlib                                                           # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed     # noqa: E402
from pathlib import Path                                              # noqa: E402

import numpy as np                                                    # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
import spread_lib as L  # noqa: E402

DT = L.DT
ACF_MAXLAG = L.ACF_MAXLAG
ACF_FLOOR = L.ACF_FLOOR
N_BOOT = 400
N_SURR = 200
SEED = L.SEED
LAM_CEIL = float(-np.log(ACF_FLOOR) / DT)     # 1.4979 s^-1, the n_lags = 0 bound

STAGES = L.BOUT_STAGES
RHO_MAX, SPLIT_LO, SPLIT_HI = L.RHO_MAX, L.SPLIT_LO, L.SPLIT_HI


# ---- estimators -----------------------------------------------------------
def bout_stats(C: np.ndarray) -> dict:
    """Per-bout sufficient statistics for every quantity, from centred blocks.

    C is (n_bouts, n_win) already centred on each bout's own mean.  Every statistic
    below is a per-bout SUM, so a bootstrap draw is a sum over resampled rows and
    no block is ever re-centred on a resampled pool.
    """
    nb, nw = C.shape
    num = np.empty((nb, ACF_MAXLAG))
    den = np.empty((nb, ACF_MAXLAG))
    for k in range(1, ACF_MAXLAG + 1):
        a, b = C[:, :-k], C[:, k:]
        num[:, k - 1] = (a * b).sum(axis=1)
        den[:, k - 1] = (a * a).sum(axis=1)
    d = np.diff(C, axis=1)
    return dict(num=num, den=den,
                ss=(C * C).sum(axis=1), nn=np.full(nb, float(nw)),
                m2num=(d * d).sum(axis=1), m2den=np.full(nb, float(nw - 1)))


def estimate(S: dict, idx=None) -> dict:
    """width, lambda, D, n_lags from per-bout sufficient statistics."""
    if idx is None:
        num, den = S["num"].sum(0), S["den"].sum(0)
        ss, nn = S["ss"].sum(), S["nn"].sum()
        m2n, m2d = S["m2num"].sum(), S["m2den"].sum()
    else:
        num, den = S["num"][idx].sum(0), S["den"][idx].sum(0)
        ss, nn = S["ss"][idx].sum(), S["nn"][idx].sum()
        m2n, m2d = S["m2num"][idx].sum(), S["m2den"][idx].sum()
    r = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    lam, nlag = L._lam_from_r(r)
    m2 = m2n / m2d
    D = L._D_from(lam, m2)
    width = float(np.sqrt(ss / nn)) if nn > 0 else float("nan")
    sd_ou = float(np.sqrt(D / lam)) if (np.isfinite(D) and np.isfinite(lam) and lam > 0) else float("nan")
    return dict(width=width, rho1=float(r[0]), acf=[float(x) for x in r],
                lam=lam, n_lags=int(nlag), m2=float(m2), D=D, sd_ou=sd_ou,
                ratio=(sd_ou / width) if (np.isfinite(sd_ou) and width > 0) else float("nan"))


def bound_censored(n_lags: int, m2: float) -> tuple:
    """One-sided bound the data of a censored direction support: (lam_lo, D_lo)."""
    lam_lo = float(-np.log(ACF_FLOOR) / ((n_lags + 1) * DT))
    return lam_lo, L._D_from(lam_lo, m2)


def ou_surrogate_ratio(nb: int, nw: int, lam: float, sd: float, rng) -> dict:
    """Matched discrete-OU null for the adequacy ratio at this direction."""
    if not (np.isfinite(lam) and lam > 0 and np.isfinite(sd) and sd > 0):
        return dict(n=0)
    rho = float(np.exp(-lam * DT))
    X = np.empty((N_SURR, nb, nw))
    X[:, :, 0] = rng.standard_normal((N_SURR, nb)) * sd
    s = sd * np.sqrt(max(1.0 - rho * rho, 1e-12))
    e = rng.standard_normal((N_SURR, nb, nw - 1)) * s
    for t in range(1, nw):
        X[:, :, t] = rho * X[:, :, t - 1] + e[:, :, t - 1]
    out = []
    for i in range(N_SURR):
        C = X[i] - X[i].mean(axis=1, keepdims=True)
        out.append(estimate(bout_stats(C)))
    rr = np.array([o["ratio"] for o in out], float)
    ll = np.array([o["lam"] for o in out], float)
    DD = np.array([o["D"] for o in out], float)
    nl = np.array([o["n_lags"] for o in out], float)
    fr = rr[np.isfinite(rr)]
    fl = ll[np.isfinite(ll)]
    fD = DD[np.isfinite(DD)]
    return dict(n=int(len(fr)), rho_true=rho,
                ratio_med=float(np.median(fr)) if len(fr) else float("nan"),
                ratio_p025=float(np.percentile(fr, 2.5)) if len(fr) else float("nan"),
                ratio_p975=float(np.percentile(fr, 97.5)) if len(fr) else float("nan"),
                lam_med=float(np.median(fl)) if len(fl) else float("nan"),
                D_med=float(np.median(fD)) if len(fD) else float("nan"),
                censored_frac=float((nl <= 1).mean()))


# ---- one night ------------------------------------------------------------
def run_one(args) -> dict:
    stem, psg, hyp, do_surr = args
    t0 = time.time()
    eeg, sfreq, chans, epochs = L.load_real_eeg(psg, hyp)
    if chans[:2] != list(L.EEG_CHANNELS):
        return dict(stem=stem, error=f"channel mismatch: {chans}")
    fit_bouts = L.usable_bouts_fine(epochs)

    trunc = {}
    for g in STAGES:
        tb = []
        for onset, dur in fit_bouts[g]:
            r = L.bout_features(eeg, sfreq, onset, dur)
            if r is not None:
                tb.append(r[0])
        if len(tb) >= L.MIN_BOUTS:
            trunc[g] = tb
    del eeg
    gc.collect()
    if len(trunc) < 2:
        return dict(stem=stem, error=f"fewer than two fine stages: {sorted(trunc)}")

    transform, evr = L.build_state_space({g: np.vstack(v) for g, v in trunc.items()})
    Z = {g: [transform(b) for b in v] for g, v in trunc.items()}
    means = {g: np.vstack(v).mean(axis=0) for g, v in Z.items()}

    rng = np.random.default_rng(SEED + (zlib.crc32(stem.encode()) % 100000))
    cells, dirs = [], []
    for a in Z:
        widths, ok_any = [], []
        keep = []
        for b in Z:
            if a == b:
                continue
            u = means[b] - means[a]
            nu = float(np.linalg.norm(u))
            if not np.isfinite(nu) or nu <= 0:
                continue
            u = u / nu
            p = [z @ u for z in Z[a]]
            # ---- spread + direction QC -----------------------------------
            w = L.spread(p)
            odd, even = L.spread(p[0::2]), L.spread(p[1::2])
            ratio = float(odd / even) if even > 0 else float("nan")
            r1 = L.ar1(p)
            ok = (np.isfinite(w) and np.isfinite(ratio)
                  and (not np.isfinite(r1) or r1 <= RHO_MAX)
                  and SPLIT_LO <= ratio <= SPLIT_HI)
            if ok:
                widths.append(w)
            ok_any.append(ok)
            if not ok:
                continue
            # ---- the decomposition, on exactly these blocks -------------
            P = np.vstack([np.asarray(x, float) for x in p])
            C = P - P.mean(axis=1, keepdims=True)
            S = bout_stats(C)
            est = estimate(S)
            assert abs(est["width"] - w) <= 1e-12 * max(1.0, abs(w)), "width drift"
            if np.isfinite(r1):
                assert abs(est["rho1"] - r1) <= 1e-10 * max(1.0, abs(r1)), "rho1 drift"
            nb = C.shape[0]
            bi = rng.integers(0, nb, size=(N_BOOT, nb))
            bl, bD, bw = [], [], []
            for j in range(N_BOOT):
                e2 = estimate(S, bi[j])
                bl.append(e2["lam"]); bD.append(e2["D"]); bw.append(e2["width"])
            bl = np.array(bl, float); bD = np.array(bD, float); bw = np.array(bw, float)

            resolved = est["n_lags"] >= 2
            lam_lo, D_lo = bound_censored(est["n_lags"], est["m2"])
            row = dict(stem=stem, subject=stem[:5], stage=a, toward=b,
                       depth=L.DEPTH.get(a), sep=nu,
                       n_bouts=nb, n_windows=int(C.size),
                       width=w, rho1=r1, split_ratio=ratio,
                       acf=est["acf"], n_lags=est["n_lags"], resolved=bool(resolved),
                       lam=est["lam"], D=est["D"], m2=est["m2"],
                       sd_ou=est["sd_ou"], ratio_sd_over_width=est["ratio"],
                       lam_bound_lo=lam_lo, D_bound_lo=D_lo,
                       lam_boot_ci=[float(np.nanpercentile(bl, 2.5)),
                                    float(np.nanpercentile(bl, 97.5))] if np.isfinite(bl).any() else None,
                       D_boot_ci=[float(np.nanpercentile(bD, 2.5)),
                                  float(np.nanpercentile(bD, 97.5))] if np.isfinite(bD).any() else None,
                       width_boot_ci=[float(np.nanpercentile(bw, 2.5)),
                                      float(np.nanpercentile(bw, 97.5))],
                       lam_boot_resolved_frac=float(np.isfinite(bl).mean()))
            if do_surr:
                row["ou_null"] = ou_surrogate_ratio(nb, C.shape[1], est["lam"], w, rng)
            dirs.append(row)
            keep.append(row)
        if not widths:
            continue
        A = np.vstack(Z[a])
        Cv = np.cov(A - A.mean(axis=0), rowvar=False)
        res = [r for r in keep if r["resolved"]]
        cell = dict(stem=stem, subject=stem[:5], stage=a,
                    depth=L.DEPTH.get(a), arousal=L.AROUSAL.get(a),
                    width=float(np.median(widths)),
                    width_iso=float(np.sqrt(np.trace(Cv) / Cv.shape[0])),
                    n_directions=len(widths), n_pairs_tried=len(ok_any),
                    n_bouts=len(Z[a]), n_windows=int(sum(len(z) for z in Z[a])),
                    n_dir_resolved=len(res),
                    cell_resolved=bool(len(res) * 2 >= len(keep) and len(res) > 0),
                    rho1=float(np.median([r["rho1"] for r in keep])),
                    lam=float(np.median([r["lam"] for r in res])) if res else None,
                    D=float(np.median([r["D"] for r in res])) if res else None,
                    ratio_sd_over_width=float(np.median([r["ratio_sd_over_width"] for r in res])) if res else None,
                    lam_all=float(np.nanmedian([r["lam"] for r in keep])) if keep else None,
                    D_all=float(np.nanmedian([r["D"] for r in keep])) if keep else None,
                    lam_bound=float(np.median([r["lam"] if r["resolved"] else r["lam_bound_lo"]
                                               for r in keep])) if keep else None,
                    D_bound=float(np.median([r["D"] if r["resolved"] else r["D_bound_lo"]
                                             for r in keep])) if keep else None,
                    lam_ci=[float(np.median([r["lam_boot_ci"][0] for r in res])),
                            float(np.median([r["lam_boot_ci"][1] for r in res]))] if res and all(
                                r["lam_boot_ci"] for r in res) else None,
                    D_ci=[float(np.median([r["D_boot_ci"][0] for r in res])),
                          float(np.median([r["D_boot_ci"][1] for r in res]))] if res and all(
                              r["D_boot_ci"] for r in res) else None)
        if do_surr:
            nn = [r["ou_null"] for r in keep if r.get("ou_null", {}).get("n")]
            cell["ou_null_ratio_med"] = float(np.median([x["ratio_med"] for x in nn])) if nn else None
            cell["ou_null_ratio_p975"] = float(np.median([x["ratio_p975"] for x in nn])) if nn else None
            cell["ou_null_ratio_p025"] = float(np.median([x["ratio_p025"] for x in nn])) if nn else None
            cell["ou_null_censored_frac"] = float(np.median([x["censored_frac"] for x in nn])) if nn else None
        cells.append(cell)
    return dict(stem=stem, subject=stem[:5], evr=[float(x) for x in evr],
                stages=sorted(trunc), minutes=round((time.time() - t0) / 60, 3),
                rows=cells, directions=dirs)


def _worker(a):
    try:
        return run_one(a)
    except Exception:  # noqa: BLE001
        return dict(stem=a[0], error=traceback.format_exc(limit=4))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--library", choices=["cassette", "telemetry"], default="cassette")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N recordings (a check run)")
    ap.add_argument("--only", nargs="*", default=None, help="only these recordings")
    ap.add_argument("--no-surrogate", action="store_true")
    a = ap.parse_args()
    if a.workers > 2:
        raise SystemExit("at most two workers: each holds a whole night of EEG in memory")

    recs = L.pair_recordings(a.library)
    if a.only:
        recs = [r for r in recs if r[0] in set(a.only)]
    if a.limit:
        recs = recs[:a.limit]
    print(f"{a.library}: {len(recs)} recordings, "
          f"{len({s[:5] for s, _, _ in recs})} participants, {a.workers} workers", flush=True)

    jobs = [(s, p, h, not a.no_surrogate) for s, p, h in recs]
    nights, done, t0 = [], 0, time.time()
    if a.workers <= 1:
        for j in jobs:
            nights.append(_worker(j))
            done += 1
            print(f"  {done}/{len(jobs)} {j[0]} {(time.time()-t0)/60:.1f} min", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(_worker, j): j[0] for j in jobs}
            for f in as_completed(futs):
                r = f.result()
                nights.append(r)
                done += 1
                if "error" in r and "fewer than two" not in r["error"]:
                    print(f"  [{r['stem']}] ERROR {r['error'].splitlines()[-1][:100]}", flush=True)
                elif done % 10 == 0 or done == len(jobs):
                    print(f"  {done}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    # sequential retry for anything that ran out of memory
    bad = [n["stem"] for n in nights
           if "error" in n and "Memory" in n["error"] and "fewer than two" not in n["error"]]
    if bad:
        print(f"  retrying {len(bad)} sequentially: {bad}", flush=True)
        keep = [n for n in nights if n["stem"] not in set(bad)]
        for j in jobs:
            if j[0] in set(bad):
                keep.append(_worker(j))
        nights = keep

    nights.sort(key=lambda x: x["stem"])
    wall = time.time() - t0
    payload = dict(
        library=a.library,
        n=len(nights), n_ok=sum(1 for n in nights if "error" not in n),
        n_error=sum(1 for n in nights if "error" in n),
        errors={n["stem"]: n["error"].splitlines()[-1][:160]
                for n in nights if "error" in n},
        seed=SEED, wall_time_s=round(wall, 1), workers=a.workers,
        params=dict(dt_s=DT, acf_maxlag=ACF_MAXLAG, acf_floor=ACF_FLOOR,
                    n_boot=N_BOOT, n_surrogate=N_SURR if not a.no_surrogate else 0,
                    lam_ceiling_s=LAM_CEIL, win_s=L.WIN_S, step_s=L.STEP_S,
                    min_bout_s=L.MIN_BOUT_S, max_bout_s=L.MAX_BOUT_S,
                    skip_windows=L.SKIP_WINDOWS, use_windows=L.USE_WINDOWS,
                    min_bouts=L.MIN_BOUTS, rho_max=RHO_MAX,
                    split_lo=SPLIT_LO, split_hi=SPLIT_HI, d_state=L.D_STATE),
        versions=dict(python=platform.python_version(), numpy=np.__version__),
        nights=nights)
    L.ROWS.mkdir(parents=True, exist_ok=True)
    p = L.ROWS / L.ROWS_FILE[a.library]
    p.write_text(json.dumps(payload, indent=1), encoding="utf-8", newline="\n")
    cells = [r for n in nights for r in n.get("rows", [])]
    print(f"\nwrote {p}")
    print(f"{len(cells)} stage cells over {len({c['stem'] for c in cells})} nights, "
          f"{wall/60:.1f} min")
    from collections import Counter
    print("cells per stage:", dict(Counter(c["stage"] for c in cells)))
    print("resolved per stage:",
          {s: f"{sum(1 for c in cells if c['stage'] == s and c['cell_resolved'])}/"
              f"{sum(1 for c in cells if c['stage'] == s)}" for s in STAGES})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
