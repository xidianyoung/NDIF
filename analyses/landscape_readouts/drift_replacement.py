"""Drift replacement, separation threshold and density barrier: the computations of Fig. 2b-e.

C1  drift replacement (Fig. 2b; cohort separations of Fig. 2c).  For every record of the count cache the
    condition regimes are refitted (landscape_lib.regimes_from_assignment on the condition labels), the
    equilibria of the mixture drift field are counted (landscape_lib.find_equilibria, 200 starts, seed
    20260922; the "anchor" count), and then only the dynamics are replaced:
      -I            every A_k replaced by -I, so the field carries only mu_k and Sigma_k;
      random stable every A_k replaced by an independent random Hurwitz matrix with spectral abscissa -1
                    (five draws; random_stable), carrying no information about the recording.
    The count is recomputed with everything else held (state space, segments, responsibilities, counter,
    seed), and agreement is reported per cohort at record level and at subject level (a subject agrees
    only if every one of its records agrees).  The separation of a record is the Mahalanobis distance
    between the two most distant regime centres under the mean regime covariance.
C2  separation threshold (Fig. 2c).  The one-parameter rule "count >= 2 if and only if the separation
    exceeds t", fitted by exhaustive search over t = 0.300, 0.301, ..., 7.999 against the reference count
    of each record (landscape_readout_counts.csv: sup_separation, reference_pipeline_field); the reference
    threshold 2.31 is evaluated beside the optimum, with the per-cohort breakdown and the plateau of
    thresholds within one record of the optimum.
C3  density barrier against the squared separation (Fig. 2d).  Least-squares slope through the origin of
    the fitted latent density barrier on the squared latent separation (positive barriers), with R^2 and the
    largest residual, per arm of the known-barrier cells (density_barrier_cells.csv); expected slope 1/8.
C4  zero barrier on a known barrier (Fig. 2e, left).  The distribution of the fitted density barrier in the
    cells whose true Freidlin-Wentzell barrier (escape action from the lower well to the saddle) is known
    and non-zero.

Inputs  paths.cache("count") (NDIF_COUNT_CACHE; C1); figure_reproduction/inputs/landscape_readout_counts.csv
        (readout_table.py; C2); figure_reproduction/inputs/density_barrier_cells.csv (the known-barrier
        cells of analyses/recurrent_networks; C3, C4).
Outputs figure_reproduction/inputs/drift_replacement_records.csv, drift_replacement_summary.json (C1),
        separation_threshold_summary.json (C2), density_barrier_summary.json (C3, C4); the console log is
        also written to analyses/landscape_readouts/out/drift_replacement.log.
Reported in  Fig. 2b (awake: -I and random stable agree in 445 of 449 records, a constant count of one in
        446; sleep: 165 and 131 of 197); Fig. 2c (threshold 3.35 -> 623 of 646, 2.31 -> 557; cohort median
        separations 0.49 / 0.46 / 0.48 / 0.88 against 2.54 in sleep); Fig. 2d (slope 0.1234 through the
        origin, R^2 0.9994, n = 256); Fig. 2e, left (one-well cells: 256 of 256 read exactly zero, true
        barrier 1.86-21.71).

The random-stable draws use one generator per record, seeded with Python's hash of (cohort, record).

Runtime: C1 about 11 min on 4 workers; C2-C4 under a second.

    python analyses/landscape_readouts/drift_replacement.py --workers 4 --cohorts all
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse                                                       # noqa: E402
import csv as _csv                                                    # noqa: E402
import dataclasses                                                    # noqa: E402
import json                                                           # noqa: E402
import math                                                           # noqa: E402
import sys                                                            # noqa: E402
import time                                                           # noqa: E402
from pathlib import Path                                              # noqa: E402

import numpy as np                                                    # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from analyses.common import paths                                     # noqa: E402
from analyses.landscape_readouts.landscape_lib import (               # noqa: E402
    count_field, regimes_from_assignment)

SEED = 20260922
N_STARTS = 200
N_RAND = 5
COHORTS = ("ds003838", "eegmmidb", "meg", "fmri", "sleep")
AWAKE = ("ds003838", "eegmmidb", "meg", "fmri")
REFERENCE_THRESHOLD = 2.31
INPUTS = ("figure_reproduction", "inputs")


def log(m, sink=None):
    s = time.strftime("[%H:%M:%S] ") + str(m)
    print(s, flush=True)
    if sink is not None:
        sink.append(s)


def random_stable(d, rng):
    """A Hurwitz matrix carrying no information about the recording: a standardised Gaussian shifted so
    that its spectral abscissa is exactly -1.  The scale is fixed rather than matched to the record,
    because a common rescaling of every A_k leaves the count unchanged (landscape_lib.multilinear_drift)."""
    G = rng.standard_normal((d, d)) / math.sqrt(d)
    a = float(np.max(np.linalg.eigvals(G).real))
    return G - (a + 1.0) * np.eye(d)


def _with_A(regs, As):
    return [dataclasses.replace(r, A=A) for r, A in zip(regs, As)]


def one_record(args):
    cache, cohort, rec, dt = args
    z = np.load(os.path.join(cache, cohort, f"{rec}.npz"))
    Z = z["Z"].astype(float); seg = z["seg"]; lab = z["lab"].astype(int)
    d = int(Z.shape[1])
    regs, _ = regimes_from_assignment(Z, seg, lab, dt)
    row = dict(cohort=cohort, record=rec, d=d, K=len(regs))
    if len(regs) < 2:
        row["error"] = "fewer than two regimes"
        return row
    anc = count_field(regs, n_starts=N_STARTS, seed=SEED)
    row["n_stable_anchor"] = anc["n_stable"]
    row["n_eq_anchor"] = anc["n_equilibria"]
    S = np.mean([r.Sigma for r in regs], axis=0)
    best = 0.0
    for i in range(len(regs)):
        for j in range(i + 1, len(regs)):
            dv = regs[i].mu - regs[j].mu
            best = max(best, float(np.sqrt(max(dv @ np.linalg.solve(S, dv), 0.0))))
    row["separation"] = best
    mi = count_field(_with_A(regs, [-np.eye(d) for _ in regs]), n_starts=N_STARTS, seed=SEED)
    row["n_stable_minusI"] = mi["n_stable"]
    rng = np.random.default_rng(abs(hash((cohort, rec))) % (2 ** 32))
    rs = []
    for _ in range(N_RAND):
        rs.append(count_field(_with_A(regs, [random_stable(d, rng) for _ in regs]),
                              n_starts=N_STARTS, seed=SEED)["n_stable"])
    row["n_stable_rand"] = rs
    row["rand_agree_frac"] = float(np.mean([r == anc["n_stable"] for r in rs]))
    row["agree_minusI"] = bool(mi["n_stable"] == anc["n_stable"])
    row["agree_rand_all"] = bool(all(r == anc["n_stable"] for r in rs))
    return row


def c1(cache, cohorts, workers, sink):
    jobs, meta = [], {}
    for c in cohorts:
        mf = os.path.join(cache, c, "_meta.json")
        if not os.path.exists(mf):
            continue
        with open(mf, encoding="utf-8") as fh:
            m = json.load(fh)
        meta[c] = m
        for rec in sorted(m["records"]):
            jobs.append((cache, c, rec, float(m["dt"])))
    log(f"C1: {len(jobs)} records over {len(meta)} cohorts, {workers} workers", sink)
    t0 = time.time()
    if workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as p:
            rows = p.map(one_record, jobs, chunksize=2)
    else:
        rows = [one_record(j) for j in jobs]
    log(f"C1: done in {time.time()-t0:.0f}s", sink)

    def subj_of(c, rec):
        r = meta[c]["records"][rec]
        return r.get("subject", rec)

    summ = {}
    for c in cohorts:
        rs = [r for r in rows if r["cohort"] == c and "error" not in r]
        if not rs:
            continue
        bysub = {}
        for r in rs:
            bysub.setdefault(subj_of(c, r["record"]), []).append(r)
        summ[c] = dict(
            n_records=len(rs), n_subjects=len(bysub),
            rec_agree_minusI=int(sum(r["agree_minusI"] for r in rs)),
            rec_agree_rand_all=int(sum(r["agree_rand_all"] for r in rs)),
            rec_rand_draw_agree=float(np.mean([r["rand_agree_frac"] for r in rs])),
            sub_agree_minusI=int(sum(all(x["agree_minusI"] for x in v) for v in bysub.values())),
            sub_agree_rand_all=int(sum(all(x["agree_rand_all"] for x in v)
                                       for v in bysub.values())),
            n_anchor_ge2=int(sum((r["n_stable_anchor"] or 0) >= 2 for r in rs)),
            median_separation=float(np.median([r["separation"] for r in rs])))
    # pooled awake cohorts, record and subject level
    aw = [r for r in rows if r["cohort"] in AWAKE and "error" not in r]
    bysub = {}
    for r in aw:
        bysub.setdefault((r["cohort"], subj_of(r["cohort"], r["record"])), []).append(r)
    pooled = dict(
        n_records=len(aw), n_subjects=len(bysub),
        rec_agree_minusI=int(sum(r["agree_minusI"] for r in aw)),
        rec_agree_rand_all=int(sum(r["agree_rand_all"] for r in aw)),
        sub_agree_minusI=int(sum(all(x["agree_minusI"] for x in v) for v in bysub.values())),
        sub_agree_rand_all=int(sum(all(x["agree_rand_all"] for x in v) for v in bysub.values())),
        n_anchor_ge2_records=int(sum((r["n_stable_anchor"] or 0) >= 2 for r in aw)),
        constant_one_counter_agree_records=int(sum((r["n_stable_anchor"] or 0) == 1
                                                   for r in aw)))
    return rows, dict(per_cohort=summ, awake_pooled=pooled, n_starts=N_STARTS, n_rand=N_RAND)


def c2(table, sink):
    with open(table, newline="", encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))

    def col(k):
        out = []
        for r in rows:
            try:
                out.append(float(r.get(k, "")))
            except (TypeError, ValueError):
                out.append(np.nan)
        return np.array(out)

    sep, ref = col("sup_separation"), col("reference_pipeline_field")
    coh = np.array([r["cohort"] for r in rows])
    ok = np.isfinite(sep) & np.isfinite(ref)
    grid = np.arange(0.30, 8.0, 0.001)
    agree = np.array([int((((sep >= t).astype(int) + 1 == ref) & ok).sum()) for t in grid])
    i = int(np.argmax(agree))
    t_ref = REFERENCE_THRESHOLD
    res = dict(n=int(ok.sum()), best_threshold=float(grid[i]), best_agree=int(agree[i]),
               reference_threshold=t_ref,
               agree_at_reference=int((((sep >= t_ref).astype(int) + 1 == ref) & ok).sum()),
               per_cohort_at_best={}, per_cohort_at_reference={})
    for c in sorted(set(coh)):
        m = ok & (coh == c)
        res["per_cohort_at_best"][c] = [
            int((((sep >= grid[i]).astype(int) + 1 == ref) & m).sum()), int(m.sum())]
        res["per_cohort_at_reference"][c] = [
            int((((sep >= t_ref).astype(int) + 1 == ref) & m).sum()), int(m.sum())]
    # the plateau of thresholds that do as well (within one record)
    top = np.flatnonzero(agree >= agree[i] - 1)
    res["plateau_within_1"] = [float(grid[top.min()]), float(grid[top.max()])]
    log(f"C2: best t = {res['best_threshold']:.3f} -> {res['best_agree']}/{res['n']}; "
        f"at the reference threshold 2.31 -> {res['agree_at_reference']}/{res['n']}", sink)
    return res


def c34(cells, sink):
    if not os.path.exists(cells):
        log(f"C3/C4: {cells} absent", sink)
        return dict(error="density_barrier_cells.csv absent")
    with open(cells, newline="", encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))

    def g(r, k):
        try:
            return float(r[k])
        except (KeyError, TypeError, ValueError):
            return np.nan

    out = {}
    for arm in sorted({r["arm"] for r in rows}):
        rs = [r for r in rows if r["arm"] == arm]
        bar = np.array([g(r, "density_barrier_nats") for r in rs])
        sep = np.array([g(r, "separation") for r in rs])
        # the true Freidlin-Wentzell barrier of the same cells: in the two known-barrier arms the escape
        # action from each well to the saddle; true_barrier_nats is filled only in the single-basin arm,
        # where it is 0 by construction
        tb = np.array([g(r, "true_fw_lo_to_saddle") for r in rs])
        tb_hi = np.array([g(r, "true_fw_hi_to_saddle") for r in rs])
        if not np.isfinite(tb).any():
            tb = np.array([g(r, "true_barrier_nats") for r in rs])
        ok = np.isfinite(bar) & np.isfinite(sep)
        ent = dict(n=len(rs), n_ok=int(ok.sum()),
                   frac_barrier_exactly_zero=float(np.mean(bar[np.isfinite(bar)] == 0.0)),
                   median_barrier=float(np.nanmedian(bar)),
                   median_separation=float(np.nanmedian(sep)),
                   true_barrier_min=float(np.nanmin(tb)) if np.isfinite(tb).any() else None,
                   true_barrier_max=float(np.nanmax(tb)) if np.isfinite(tb).any() else None,
                   true_barrier_median=float(np.nanmedian(tb)) if np.isfinite(tb).any() else None,
                   true_barrier_hi_range=([float(np.nanmin(tb_hi)), float(np.nanmax(tb_hi))]
                                          if np.isfinite(tb_hi).any() else None))
        pos = ok & (bar > 0)
        if pos.sum() >= 5:
            x = (sep[pos] ** 2)
            y = bar[pos]
            slope = float(np.dot(x, y) / np.dot(x, x))           # through the origin
            r2 = float(1.0 - np.sum((y - slope * x) ** 2) / np.sum((y - y.mean()) ** 2))
            ent.update(n_positive=int(pos.sum()), slope_through_origin=slope,
                       one_over_slope=float(1.0 / slope) if slope else None,
                       r2_through_origin=r2,
                       max_abs_resid=float(np.max(np.abs(y - slope * x))),
                       expected_slope=0.125)
        # the cells where a known non-zero barrier meets a zero reading
        both = np.isfinite(bar) & np.isfinite(tb) & (tb > 0)
        if both.any():
            z = both & (bar == 0.0)
            ent["known_barrier_read_zero"] = dict(
                n=int(z.sum()), n_with_known_barrier=int(both.sum()),
                true_barrier_range=[float(np.min(tb[z])), float(np.max(tb[z]))]
                if z.any() else None,
                true_barrier_median=float(np.median(tb[z])) if z.any() else None)
        out[arm] = ent
    for a, v in out.items():
        log(f"C3/C4 [{a}]: n={v['n']} zero-barrier frac {v['frac_barrier_exactly_zero']:.3f} "
            f"slope*8 = {8*v['slope_through_origin']:.4f} R^2 {v.get('r2_through_origin')}"
            if "slope_through_origin" in v else f"C3/C4 [{a}]: n={v['n']} "
            f"zero-barrier frac {v['frac_barrier_exactly_zero']:.3f}", sink)
    return out


def _write_json(path, obj, trailing_newline):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        if trailing_newline:
            fh.write(json.dumps(obj, ensure_ascii=False, indent=1) + "\n")
        else:
            fh.write(json.dumps(obj, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cohorts", default="all", help="all, or a comma-separated list of cohorts (C1)")
    ap.add_argument("--skip-c1", action="store_true", help="run C2-C4 only")
    ap.add_argument("--readout-table", default=paths.repo(*INPUTS, "landscape_readout_counts.csv"),
                    help="per-record table with sup_separation and reference_pipeline_field (C2)")
    ap.add_argument("--barrier-cells", default=paths.repo(*INPUTS, "density_barrier_cells.csv"),
                    help="known-barrier cells (C3, C4)")
    a = ap.parse_args()
    sink = []
    log("drift_replacement.py", sink)
    cohorts = COHORTS if a.cohorts == "all" else tuple(a.cohorts.split(","))
    if not a.skip_c1:
        rows, s1 = c1(paths.cache("count"), cohorts, a.workers, sink)
        keys = ["cohort", "record", "d", "K", "n_stable_anchor", "n_eq_anchor", "separation",
                "n_stable_minusI", "agree_minusI", "rand_agree_frac", "agree_rand_all"]
        with open(paths.out(*INPUTS, "drift_replacement_records.csv"), "w", newline="",
                  encoding="utf-8") as fh:
            w = _csv.writer(fh, lineterminator="\n")
            w.writerow(keys)
            for r in rows:
                w.writerow([r.get(k) for k in keys])
        _write_json(paths.out(*INPUTS, "drift_replacement_summary.json"), s1, trailing_newline=False)
        p = s1["awake_pooled"]
        log(f"C1 awake pooled: -I agrees in {p['sub_agree_minusI']}/{p['n_subjects']} subjects "
            f"({p['rec_agree_minusI']}/{p['n_records']} records); random stable in "
            f"{p['sub_agree_rand_all']}/{p['n_subjects']}", sink)
    _write_json(paths.out(*INPUTS, "separation_threshold_summary.json"), c2(a.readout_table, sink),
                trailing_newline=True)
    _write_json(paths.out(*INPUTS, "density_barrier_summary.json"), c34(a.barrier_cells, sink),
                trailing_newline=True)
    with open(paths.out("analyses", "landscape_readouts", "out", "drift_replacement.log"), "w",
              encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(sink) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
