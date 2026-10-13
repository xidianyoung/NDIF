"""Simulation benchmark of the equilibrium count at each cohort's own geometry (Fig. 1f floors).

The counter is fixed; only the generator changes.  Every replicate passes through the chain of the
equilibrium count: landscape_lib.fit_linear_regime per regime -> find_equilibria (softness 1, 200 starts)
-> number of stable equilibria, and landscape_lib.ridgeline_modes_2 on the same fitted pair for the
density mode count and the measured separation.

Generators (one ingredient added per rung; the same records and the same record draw are used at every
rung and every separation, so the rungs are paired)

  isotropic_reference  d = 8, Sigma = I, M = exp(-0.5 dt) I, dt = 1, one contiguous segment of n_iso
                       samples per regime, separation along a random unit direction, w1 = 0.5
  dimension            the cohort's state dimension and sampling interval
  anisotropy           the record's per-regime covariance Sigma_k; separation along the record's own
                       displacement direction, normalised in the pooled metric so that the generating
                       separation is exactly delta; isotropic dynamics
  drift                the record's per-regime one-step map M_k, with Q_k = Sigma_k - M_k Sigma_k M_k'
                       (positive semidefinite part), so the stationary covariance stays Sigma_k
  sample_size          the record's number of windows per regime
  segments             the record's segment count and segment lengths
  nonstationarity      per-segment mean offsets drawn from the record's measured between-segment mean
                       dispersion in excess of the stationary prediction: the generator matched to the
                       cohort, whose zero-separation floor is the floor of Fig. 1f

Separations delta = 0, 0.5, 0.875, 1, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 4.  Each replicate is generated from
one participant's fitted parameters, and the 60 replicates of a cell are 60 distinct participants drawn
without replacement (one run per participant for HCP fMRI), so detection fractions are fractions of
participants.  Seeds: subject draw default_rng(20260921 + crc32(cohort) % 10000); replicate
20260921 + 7919 (rung index + 1) + 104729 round(1000 delta) + 13 rep + crc32(cohort) % 1000; counter seed
20260921 + rep.

Input    paths.cache("separation_floors")/generators/<cohort>.pkl (extract_generators.py).
Outputs  analyses/separation_floors/out/floor_benchmark.json, floor_benchmark.csv, floor_benchmark.log;
         per-replicate rows paths.cache("separation_floors")/floor_benchmark_rows.pkl (read by
         floor_report.py).
Supports Fig. 1f (zero-separation floors and their 95th percentiles; via floor_report.py).

Runtime about 34 min on 14 workers (4 cohorts x 7 rungs x 12 separations x 60 replicates).

    python analyses/separation_floors/floor_benchmark.py --reps 60 --workers 14
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse                                                       # noqa: E402
import csv                                                            # noqa: E402
import json                                                           # noqa: E402
import pickle                                                         # noqa: E402
import sys                                                            # noqa: E402
import time                                                           # noqa: E402
import zlib                                                           # noqa: E402
from concurrent.futures import ProcessPoolExecutor                    # noqa: E402
from pathlib import Path                                              # noqa: E402

import numpy as np                                                    # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from analyses.common import paths                                     # noqa: E402
from analyses.landscape_readouts import landscape_lib as AL           # noqa: E402


def _h(s):
    """Deterministic string hash (independent of PYTHONHASHSEED)."""
    return zlib.crc32(s.encode("utf-8"))


LAM_ISO = 0.5           # relaxation rate (s^-1) of the isotropic reference generator
DELTAS = [0.0, 0.5, 0.875, 1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 4.0]
RUNGS = ["isotropic_reference", "dimension", "anisotropy", "drift",
         "sample_size", "segments", "nonstationarity"]
SEED = 20260921
N_STARTS = 200
SOFTNESS = 1.0


def log(m, fh=None):
    s = time.strftime("[%H:%M:%S] ") + str(m)
    print(s, flush=True)
    if fh is not None:
        fh.write(s + "\n")
        fh.flush()


# --------------------------------------------------------------------------- generator
def _chol(S):
    S = 0.5 * (S + S.T)
    try:
        return np.linalg.cholesky(S + 1e-12 * np.eye(len(S)))
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(S)
        return V @ np.diag(np.sqrt(np.maximum(w, 0.0)))


def _sim_segment(L, mu, M, Lq, Lsig, rng):
    """One stationary VAR(1) segment of length L started from its own stationary law."""
    d = len(mu)
    X = np.empty((L, d))
    x = Lsig @ rng.standard_normal(d)
    X[0] = x
    for t in range(1, L):
        x = M @ x + Lq @ rng.standard_normal(d)
        X[t] = x
    return X + mu


def build_spec(rec, rung, delta, rng):
    """(segs0, segs1, w1, dt) of one replicate at one rung and one generating separation."""
    d_iso, dt_iso = 8, 1.0
    r0, r1 = rec["regimes"]
    ri = RUNGS.index(rung)

    if ri <= 1:                                   # isotropic reference / dimension: unit covariance
        d = d_iso if ri == 0 else rec["d"]
        dt = dt_iso if ri == 0 else rec["dt"]
        n = [rec["n_iso"], rec["n_iso"]]
        a = float(np.exp(-LAM_ISO * dt))
        M = [a * np.eye(d), a * np.eye(d)]
        Sig = [np.eye(d), np.eye(d)]
        Q = [(1 - a * a) * np.eye(d)] * 2
        u = rng.standard_normal(d)
        u /= np.linalg.norm(u)
        mus = [np.zeros(d), delta * u]
        segs_spec = [[n[0]], [n[1]]]
        B = [None, None]
        w1 = 0.5
    else:
        d, dt = rec["d"], rec["dt"]
        Sig = [r0["Sigma"], r1["Sigma"]]
        a = float(np.exp(-LAM_ISO * dt))
        if ri == 2:                               # anisotropy: real covariance, isotropic dynamics
            M = [a * np.eye(d), a * np.eye(d)]
        else:                                     # drift and later: real one-step maps
            M = [r0["M"], r1["M"]]
        Q = []
        for k in (0, 1):
            Qk = Sig[k] - M[k] @ Sig[k] @ M[k].T
            Qk = 0.5 * (Qk + Qk.T)
            w, V = np.linalg.eigh(Qk)
            Q.append(V @ np.diag(np.maximum(w, 0.0)) @ V.T)
        Sbar = 0.5 * (Sig[0] + Sig[1])
        e = r1["mu"] - r0["mu"]
        s = float(np.sqrt(max(e @ np.linalg.solve(Sbar, e), 1e-300)))
        if not np.isfinite(s) or s < 1e-9:
            e = rng.standard_normal(d)
            s = float(np.sqrt(e @ np.linalg.solve(Sbar, e)))
        u = e / s                                 # unit generating displacement in the pooled metric
        mus = [r0["mu"], r0["mu"] + delta * u]
        if ri <= 3:                               # anisotropy / drift: n_iso, one contiguous segment
            segs_spec = [[rec["n_iso"]], [rec["n_iso"]]]
            w1 = 0.5
        elif ri == 4:                             # sample size: real n per regime, still contiguous
            segs_spec = [[r0["n_windows"]], [r1["n_windows"]]]
            w1 = r0["n_windows"] / (r0["n_windows"] + r1["n_windows"])
        else:                                     # segments / nonstationarity: real segment lengths
            segs_spec = [list(r0["seg_lens"]), list(r1["seg_lens"])]
            w1 = r0["n_windows"] / (r0["n_windows"] + r1["n_windows"])
        B = [r0["B_excess"], r1["B_excess"]] if ri >= 6 else [None, None]

    Lq = [_chol(Q[0]), _chol(Q[1])]
    Ls = [_chol(Sig[0]), _chol(Sig[1])]
    Lb = [(_chol(B[k]) if B[k] is not None else None) for k in (0, 1)]
    segs = []
    for k in (0, 1):
        out = []
        for L in segs_spec[k]:
            mu = mus[k]
            if Lb[k] is not None:                 # measured between-segment level shift
                mu = mu + Lb[k] @ rng.standard_normal(d)
            out.append(_sim_segment(int(L), mu, M[k], Lq[k], Ls[k], rng))
        segs.append(out)
    return segs[0], segs[1], float(w1), float(dt)


def one_replicate(job):
    cohort, rung, delta, rep, rec, seed = job
    rng = np.random.default_rng(seed)
    s0, s1, w1, dt = build_spec(rec, rung, delta, rng)
    try:
        regs = [AL.fit_linear_regime(s0, "r0", dt=dt), AL.fit_linear_regime(s1, "r1", dt=dt)]
        eqs = AL.find_equilibria(regs, softness=SOFTNESS, n_starts=N_STARTS, seed=SEED + rep)
        n_stable = int(sum(e["stable"] for e in eqs))
        rl = AL.ridgeline_modes_2(regs[0].mu, regs[0].Sigma, regs[1].mu, regs[1].Sigma, w1=w1)
        return dict(cohort=cohort, rung=rung, delta=delta, rep=rep,
                    record=rec["record"], subject=rec["subject"],
                    n_stable=n_stable, n_eq=len(eqs), n_modes=int(rl.n_modes),
                    sep_measured=float(rl.separation), barrier=float(rl.barrier_nats), ok=True)
    except Exception as e:                        # noqa: BLE001
        return dict(cohort=cohort, rung=rung, delta=delta, rep=rep,
                    record=rec["record"], subject=rec["subject"], ok=False, error=repr(e))


# --------------------------------------------------------------------------- readout
def _interp_cross(deltas, frac, sep, level):
    """Interpolate delta where detection first reaches `level`, then the measured separation at that delta."""
    d = np.asarray(deltas, float)
    f = np.asarray(frac, float)
    s = np.asarray(sep, float)
    for i in range(1, len(d)):
        if f[i] >= level and f[i - 1] < level:
            if f[i] == f[i - 1]:
                return float(d[i]), float(s[i])
            t = (level - f[i - 1]) / (f[i] - f[i - 1])
            dc = d[i - 1] + t * (d[i] - d[i - 1])
            sc = s[i - 1] + t * (s[i] - s[i - 1])
            return float(dc), float(sc)
    return None, None


def summarise(rows, cohort, rung, deltas):
    cells, out = {}, {}
    for dl in deltas:
        R = [r for r in rows if r["ok"] and r["rung"] == rung and abs(r["delta"] - dl) < 1e-9]
        if not R:
            continue
        ns = np.array([r["n_stable"] for r in R], float)
        sp = np.array([r["sep_measured"] for r in R], float)
        nm = np.array([r["n_modes"] for r in R], float)
        cells[f"{dl}"] = dict(
            delta=dl, reps=len(R), n_subjects=len(set(r["subject"] for r in R)),
            frac_field_ge2=float(np.mean(ns >= 2)), mean_n_stable=float(ns.mean()),
            max_n_stable=int(ns.max()), frac_density_bimodal=float(np.mean(nm >= 2)),
            sep_measured_median=float(np.median(sp)),
            sep_measured_p95=float(np.percentile(sp, 95)),
            sep_measured_iqr=[float(np.percentile(sp, 25)), float(np.percentile(sp, 75))])
    dl = sorted(cells, key=float)
    fr = [cells[k]["frac_field_ge2"] for k in dl]
    sp = [cells[k]["sep_measured_median"] for k in dl]
    dd = [cells[k]["delta"] for k in dl]
    d50, s50 = _interp_cross(dd, fr, sp, 0.5)
    d05, s05 = _interp_cross(dd, fr, sp, 0.05)
    out["cells"] = cells
    z = cells.get("0.0")
    out["floor_sep_measured_median_at_zero"] = z["sep_measured_median"] if z else None
    out["floor_sep_measured_p95_at_zero"] = z["sep_measured_p95"] if z else None
    out["false_detection_rate_at_zero"] = z["frac_field_ge2"] if z else None
    out["delta_at_50pct"] = d50
    out["sep_measured_at_50pct"] = s50
    out["delta_at_5pct"] = d05
    out["sep_measured_at_5pct"] = s05
    out["cohort"] = cohort
    out["rung"] = rung
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--cohort", nargs="+", default=["hcp_fmri", "hcp_meg", "ds003838", "eegmmidb"])
    ap.add_argument("--rungs", nargs="+", default=RUNGS)
    ap.add_argument("--deltas", nargs="+", type=float, default=DELTAS)
    args = ap.parse_args()

    gen = os.path.join(paths.cache("separation_floors"), "generators")
    out = lambda name: paths.out("analyses", "separation_floors", "out", name)  # noqa: E731
    fh = open(out("floor_benchmark.log"), "w", encoding="utf-8")
    t00 = time.time()
    res = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), seed=SEED, reps=args.reps,
               n_starts=N_STARTS, softness=SOFTNESS, deltas=list(args.deltas), rungs=list(args.rungs),
               isotropic_reference=dict(lambda_per_s=LAM_ISO, d=8, dt=1.0),
               cohorts={})
    all_rows = []
    for cohort in args.cohort:
        with open(os.path.join(gen, f"{cohort}.pkl"), "rb") as f:
            recs = pickle.load(f)
        # ---- subject-level draw, fixed across every rung and every separation
        rng = np.random.default_rng(SEED + _h(cohort) % 10_000)
        by_sub = {}
        for r in recs:
            by_sub.setdefault(r["subject"], []).append(r)
        subs = sorted(by_sub)
        take = min(args.reps, len(subs))
        chosen = list(rng.choice(subs, size=take, replace=False))
        if take < args.reps:                        # top up with replacement if a cohort is small
            chosen += list(rng.choice(subs, size=args.reps - take, replace=True))
        draw = [by_sub[s][int(rng.integers(len(by_sub[s])))] for s in chosen]
        log(f"{cohort}: {len(recs)} records / {len(subs)} subjects; "
            f"{len(draw)} replicate records drawn at subject level "
            f"({len(set(d['subject'] for d in draw))} distinct subjects)", fh)

        jobs = []
        for rung in args.rungs:
            for dl in args.deltas:
                for rep, rec in enumerate(draw):
                    seed = (SEED + 7919 * (RUNGS.index(rung) + 1)
                            + 104729 * int(round(dl * 1000)) + 13 * rep
                            + _h(cohort) % 1000)
                    jobs.append((cohort, rung, dl, rep, rec, seed))
        log(f"{cohort}: {len(jobs)} replicates", fh)
        rows = []
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, r in enumerate(ex.map(one_replicate, jobs, chunksize=4), 1):
                rows.append(r)
                if i % 500 == 0:
                    log(f"  {cohort} {i}/{len(jobs)}  ({(time.time()-t0)/60:.1f} min)", fh)
        all_rows += rows
        n_fail = sum(not r["ok"] for r in rows)
        res["cohorts"][cohort] = dict(
            n_records=len(recs), n_subjects=len(subs),
            draw_subjects=[d["subject"] for d in draw],
            draw_records=[d["record"] for d in draw],
            d=recs[0]["d"], dt=recs[0]["dt"], n_iso_per_regime=recs[0]["n_iso"],
            windows_per_regime=[float(np.median([r["regimes"][k]["n_windows"] for r in recs]))
                                for k in (0, 1)],
            increments_per_regime=[float(np.median([r["regimes"][k]["n_increments"] for r in recs]))
                                   for k in (0, 1)],
            segments_per_regime=[float(np.median([r["regimes"][k]["n_segments"] for r in recs]))
                                 for k in (0, 1)],
            observed_separation_median=float(np.median([r["sep_real"] for r in recs])),
            observed_separation_range=[float(np.min([r["sep_real"] for r in recs])),
                                       float(np.max([r["sep_real"] for r in recs]))],
            n_failed_replicates=n_fail,
            rungs={rung: summarise(rows, cohort, rung, args.deltas) for rung in args.rungs})
        for rung in args.rungs:
            s = res["cohorts"][cohort]["rungs"][rung]
            log(f"  {cohort:10s} {rung:22s} floor {s['floor_sep_measured_median_at_zero']:.4f} "
                f"(p95 {s['floor_sep_measured_p95_at_zero']:.4f})  fp@0 "
                f"{s['false_detection_rate_at_zero']:.3f}  50% crossing "
                f"{'n/a' if s['sep_measured_at_50pct'] is None else '%.3f' % s['sep_measured_at_50pct']}"
                f"  5% crossing "
                f"{'n/a' if s['sep_measured_at_5pct'] is None else '%.3f' % s['sep_measured_at_5pct']}", fh)

    with open(out("floor_benchmark.json"), "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(res, indent=1))
    with open(out("floor_benchmark.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["cohort", "rung", "delta", "reps", "n_subjects", "frac_field_ge2",
                    "mean_n_stable", "frac_density_bimodal", "sep_measured_median",
                    "sep_measured_p95"])
        for c in args.cohort:
            for rung in args.rungs:
                for k, v in res["cohorts"][c]["rungs"][rung]["cells"].items():
                    w.writerow([c, rung, v["delta"], v["reps"], v["n_subjects"],
                                v["frac_field_ge2"], v["mean_n_stable"], v["frac_density_bimodal"],
                                round(v["sep_measured_median"], 6), round(v["sep_measured_p95"], 6)])
    with open(os.path.join(paths.cache("separation_floors"), "floor_benchmark_rows.pkl"), "wb") as f:
        pickle.dump(all_rows, f, protocol=4)
    log(f"done in {(time.time()-t00)/60:.1f} min", fh)
    fh.close()


if __name__ == "__main__":
    main()
