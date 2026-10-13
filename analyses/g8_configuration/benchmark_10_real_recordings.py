"""The real recordings certified with the G8-augmented configuration.

This is the application of the G8-augmented configuration to the real recordings.  Every certificate uses the package
defaults (the checks that decide each level; the diagnostic checks G4 and G7 are not run) and the random stream 1.
The sleep recordings are built by the loader of benchmark_05_corpus_full.py.

Parts
  sleep    every stage-specific recording of the 153 primary cassette nights (602), least squares (ols) and maximum
           likelihood (kalman); level 1 is estimator-free and is computed in the ols pass
  pairs    the separation check of the G8-augmented configuration (permutation of stage labels across segments) for the 336 pairs of
           consecutive stages within a night
  series   five non-neural series, ols, kalman and abc
  co2      the carbon dioxide residual at four segment lengths, three preprocessings and three estimators:
             interp_calendar  missing weeks interpolated, trend and harmonics fitted in calendar time (the analysis)
             drop_calendar    missing weeks dropped, trend and harmonics fitted in calendar time
             drop_index       missing weeks dropped, trend and harmonics fitted on the sample index (calendar
                              not preserved)

usage: python analyses/g8_configuration/benchmark_10_real_recordings.py --part sleep [--workers 8] [--budget s]
Results are appended to out/benchmark_10_<part>.jsonl; keys already written are skipped.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse                                                       # noqa: E402
import hashlib                                                        # noqa: E402
import json                                                           # noqa: E402
import sys                                                            # noqa: E402
import time                                                           # noqa: E402
import warnings                                                       # noqa: E402

import numpy as np                                                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))                       # the repository root
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "analyses", "primary_configuration"))
sys.path.insert(0, os.path.join(PKG, "analyses", "workload_certificates"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PKG, "examples"))
warnings.filterwarnings("ignore", category=RuntimeWarning)

import benchmark_05_corpus_full as B5                                 # noqa: E402
import ndif
from ndif import Design, Recording, certify                           # noqa: E402
from ndif.estimators import fit_rate, fit_ar1_kalman_batch            # noqa: E402
from ndif.gates import _project, g5_floor                             # noqa: E402

OUT = os.path.join(HERE, "out")
ENTROPY = 202610090100
L23 = ("G0", "G1", "G2", "G3", "G8")
ALL = ("G0", "G1", "G2", "G3", "G5", "G6", "G8")
KEEP_DETAIL = ("slope", "offset", "spread", "criteria", "n_rep", "a_generator", "nu_generator", "a_fit", "nu_fit",
               "mdc", "mdc_var", "var_decision", "var_spread", "nu_at_boundary", "nu_set", "rate_set", "nu_two_lag",
               "ci", "combined_p", "p_autocorrelation", "p_likelihood_ratio", "relaxation_time", "L_median",
               "floor", "p_value", "n_permutations", "exact", "rate_recovery", "bias_log", "spread_log",
               "detection_margin_fold", "detection_margin_fold_variance", "decision", "reason")



def compact(cert):
    d = cert.to_dict()
    levels = {k: dict(status=v["status"], binding=v.get("binding_gate")) for k, v in d["levels"].items()}
    gates = {}
    for g in d["gates"]:
        det = g.get("detail") or {}
        gates[g["gate"]] = dict(
            passed=(None if g["passed"] is None else int(bool(g["passed"]))),
            value=(float(g["value"]) if g["value"] is not None and np.isfinite(g["value"]) else None),
            margin=(float(g["margin"]) if g["margin"] is not None and np.isfinite(g["margin"]) else None),
            note=g.get("note", ""),
            detail=B5.jsonable({k: v for k, v in det.items() if k in KEEP_DETAIL}))
    return dict(levels=levels, gates=gates)


def estimate(rec, est):
    segs = _project(rec)
    lam = fit_rate(segs, dt=rec.design.dt, method=est)
    return float(lam) if np.isfinite(lam) else None


def cert_record(X, design, est, rng=1, which=None):
    rec = Recording(np.asarray(X, float), design)
    which = which or (ALL if est == "ols" else L23)
    t0 = time.time()
    c = certify(rec, which=list(which), configuration="g8_augmented", rng=rng, estimator=est)
    out = compact(c)
    out.update(estimator=est, lam=estimate(rec, est), n_segments=len(rec.segments()), n_windows=int(rec.n_t),
               dim=int(rec.d), rng=rng, wall_s=time.time() - t0)
    return out


# ---------------------------------------------------------------- sleep
def nights():
    out = []
    for f in B5.primary_files(B5.DEFAULT_CACHE):
        name, win, stages = B5.load_night(f)
        out.append((f, name, sorted(stages, key=B5.STAGES.index)))
    return out


def sleep_design(name, stage, segs, win, labels=None):
    X, bounds = B5.bounds_of(segs)
    return X, Design(dt=win, window_s=win, step_s=win, segments=bounds, labels=labels, name=f"{name}/{stage}",
                     modality="polysomnography")


def job_sleep(job):
    path, name, stage, est, rng = job
    _, win, stages = B5.load_night(path)
    X, design = sleep_design(name, stage, stages[stage], win)
    r = cert_record(X, design, est, rng=rng)
    r.update(part="sleep", recording=name, stage=stage, participant=name[:5], night=name[5])
    return r


def job_pair(job):
    path, name, lo, hi = job
    _, win, stages = B5.load_night(path)
    sa, sb = stages[lo], stages[hi]
    lab = np.concatenate([np.zeros(sum(len(s) for s in sa)), np.ones(sum(len(s) for s in sb))])
    X, design = sleep_design(name, f"{lo}-{hi}", sa + sb, win, labels=lab)
    t0 = time.time()
    g = g5_floor(Recording(X, design), null="permutation", rng=1)
    det = g.detail or {}
    return dict(part="pairs", recording=name, lo=lo, hi=hi, n_segments=[len(sa), len(sb)],
                passed=(None if g.passed is None else int(bool(g.passed))),
                status=getattr(g, "status", None),
                separation=(float(g.value) if np.isfinite(g.value) else None),
                margin=(float(g.margin) if np.isfinite(g.margin) else None), note=g.note,
                detail=B5.jsonable({k: v for k, v in det.items() if k in KEEP_DETAIL}), wall_s=time.time() - t0)


# ---------------------------------------------------------------- non-neural series and the CO2 sweep
def job_series(job):
    import example_03_cross_domain as X3
    name, est = job
    rec = next(r for n, _, r in X3.load() if n == name)
    r = cert_record(rec.X, rec.design, est)
    r.update(part="series", recording=name)
    return r


def co2_residual(prep):
    import statsmodels.api as sm
    s = sm.datasets.co2.load_pandas().data["co2"]
    t = np.arange(len(s), dtype=float)
    if prep == "interp_calendar":
        y = s.interpolate(method="linear", limit_direction="both").to_numpy(float)
    elif prep in ("drop_calendar", "drop_index"):
        keep = s.notna().to_numpy()
        y, t = s.to_numpy(float)[keep], t[keep]
        if prep == "drop_index":
            t = np.arange(len(y), dtype=float)
    else:
        raise ValueError(prep)
    w = 2 * np.pi / 52.1775
    B = np.column_stack([np.ones_like(t), t, t ** 2, np.cos(w * t), np.sin(w * t),
                         np.cos(2 * w * t), np.sin(2 * w * t)])
    return y - B @ np.linalg.lstsq(B, y, rcond=None)[0]


def job_co2(job):
    prep, blk, est = job
    x = co2_residual(prep)
    ref = fit_ar1_kalman_batch([[x - x.mean()]])[0]
    tau_ref = float(-1.0 / np.log(ref["a"]))
    n = (len(x) // blk) * blk
    x = x[:n]
    bounds = [(i, i + blk) for i in range(0, n, blk)]
    design = Design(dt=1 / 52, window_s=1 / 52, step_s=1 / 52, segments=bounds, name=f"co2_{prep}_blk{blk}",
                    time_unit="year")
    r = cert_record(x[:, None], design, est, which=L23)
    fit = fit_ar1_kalman_batch([[x[a:b] - x[a:b].mean() for a, b in bounds]])[0]
    r.update(part="co2", prep=prep, block=blk, n_used=int(len(co2_residual(prep))), tau_ref_weeks=tau_ref,
             nu_ref=float(ref["nu"]), block_over_tau_ref=float(blk / tau_ref),
             a_ml_segments=float(fit["a"]), nu_ml_segments=float(fit["nu"]))
    return r


# ---------------------------------------------------------------- driver
def jobs_for(part):
    if part == "sleep":
        return [(f, name, st, est, 1) for est in ("ols", "kalman") for f, name, present in nights() for st in present]
    if part == "pairs":
        return [(f, name, lo, hi) for f, name, present in nights() for lo, hi in B5.PAIRS
                if lo in present and hi in present]
    if part == "series":
        return [(n, est) for est in ("ols", "kalman", "abc")
                for n in ("nile", "sunspots", "co2_raw", "co2_residual", "macro_unemp")]
    if part == "co2":
        return [(p, b, est) for p in ("interp_calendar", "drop_calendar", "drop_index") for b in (100, 200, 550, 1100)
                for est in ("ols", "kalman", "abc")]
    raise ValueError(part)


def key_of(part, j):
    if part == "sleep":
        return f"{j[1]}|{j[2]}|{j[3]}"
    if part == "pairs":
        return f"{j[1]}|{j[2]}-{j[3]}"
    if part == "series":
        return f"{j[0]}|{j[1]}"
    if part == "co2":
        return f"{j[0]}|{j[1]}|{j[2]}"
    raise ValueError(part)


RUN = dict(sleep=job_sleep, pairs=job_pair, series=job_series, co2=job_co2)


def run_one(args):
    part, job = args
    r = RUN[part](job)
    r["key"] = key_of(part, job)
    r["version"] = ndif.__version__
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=sorted(RUN))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--budget", type=float, default=None, help="stop submitting jobs after this many seconds; resumable")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    jobs = jobs_for(a.part)
    path = os.path.join(OUT, f"benchmark_10_{a.part}.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["key"] for l in open(path, encoding="utf-8") if l.strip()}
    todo = [j for j in jobs if key_of(a.part, j) not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"[{a.part}] {len(jobs)} jobs, {len(done)} done, {len(todo)} to run", flush=True)
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    t0 = time.time()
    n = 0
    with open(path, "a", encoding="utf-8") as fh, ProcessPoolExecutor(max_workers=a.workers) as ex:
        pend, it = set(), iter(todo)
        while True:
            while len(pend) < a.workers and (a.budget is None or time.time() - t0 < a.budget):
                j = next(it, None)
                if j is None:
                    break
                pend.add(ex.submit(run_one, (a.part, j)))
            if not pend:
                break
            fin, pend = wait(pend, return_when=FIRST_COMPLETED)
            for fu in fin:
                try:
                    r = fu.result()
                except Exception as exc:
                    print(f"  FAILED: {exc!r}", flush=True)
                    continue
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                n += 1
    print(f"[{a.part}] wrote {n} in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
