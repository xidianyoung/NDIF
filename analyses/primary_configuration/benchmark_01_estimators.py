"""Benchmark: rate estimators against known truth, and what each one knows about itself.

Ground truth is a scalar AR(1) latent process plus an additive white component,
at the recording geometries the framework is applied to.  Both parameters are
known exactly, so bias and coverage are exact, not estimated.

For every method we report what a computational-science comparison requires --
bias, coverage, interval width, runtime -- and one further column that the
others do not have: **whether the method says anything about whether its own
answer can be trusted on this recording**.

NDIF wraps the same maximum-likelihood estimator that profile likelihood uses;
what it adds is a *decision, per recording, on which estimator will recover the
truth*.  The final table scores that decision against the ground truth it could
not see.

    python analyses/primary_configuration/benchmark_01_estimators.py --reps 20 --workers 3

Writes ``analyses/primary_configuration/out/benchmark_01.json`` and prints the tables.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # the repository root

from ndif import Design, Recording, certify                      # noqa: E402
from ndif.baselines import (abc_timescale, aic_bic_adequacy, ols_ar1,   # noqa: E402
                            profile_likelihood, sloppiness)
from ndif.estimators import fit_rate                             # noqa: E402

A_TRUE = 0.85
LAM_TRUE = -np.log(A_TRUE)

NU_GRID = (0.0, 0.2, 0.4, 0.6)
GEOM = (("small", 20, 60), ("typical", 40, 150))


def make(nu, n_seg, n_blk, seed):
    """One realisation of the known-truth process, as a list of segments."""
    rng = np.random.default_rng(seed)
    Y = np.zeros((n_blk, n_seg))
    E = rng.normal(scale=np.sqrt(1 - A_TRUE ** 2), size=(n_blk, n_seg))
    Y[0] = rng.normal(size=n_seg)
    for t in range(1, n_blk):
        Y[t] = A_TRUE * Y[t - 1] + E[t]
    if nu > 0:
        Y = Y + rng.normal(scale=np.sqrt(nu / (1 - nu)), size=Y.shape)
    return [Y[:, j] for j in range(n_seg)]


def one_cell(job):
    """All methods on one realisation.  Runs in a worker process."""
    nu, gname, n_seg, n_blk, rep = job
    segs = make(nu, n_seg, n_blk, seed=1000 * rep + int(nu * 100) + n_seg)
    X = np.concatenate([s[:, None] for s in segs], axis=0)
    edges = np.cumsum([0] + [len(s) for s in segs])
    bounds = [(edges[i], edges[i + 1]) for i in range(len(segs))]

    rows = []

    def add(name, r, knows):
        rows.append(dict(nu=nu, geom=gname, rep=rep, method=name,
                         lam=r.get("lam"), lo=r.get("lo"), hi=r.get("hi"),
                         runtime_s=r.get("runtime_s"),
                         flags=r.get("flags_unidentifiable"),
                         knows_own_validity=knows))

    add("ols_ar1", ols_ar1(X, bounds, dt=1.0, n_boot=200, seed=rep), False)
    add("profile_likelihood", profile_likelihood(X, bounds, dt=1.0, n_grid=10,
                                                 seed=rep), True)
    add("sloppiness", sloppiness(X, bounds, dt=1.0, seed=rep), True)
    add("aic_bic_adequacy", aic_bic_adequacy(X, bounds, dt=1.0, seed=rep), False)
    add("abc_single", abc_timescale(X, bounds, dt=1.0, with_white=False,
                                    n_sim=600, n_keep=60, rounds=2, seed=rep), False)
    add("abc_with_white", abc_timescale(X, bounds, dt=1.0, with_white=True,
                                        n_sim=600, n_keep=60, rounds=2, seed=rep), False)

    # NDIF: the certificate, for each estimator it can certify.
    rec = Recording(X, Design(dt=1.0, window_s=1.0, step_s=1.0, segments=bounds))
    cert_rows = []
    for est in ("ols", "kalman"):
        t0 = time.perf_counter()
        cert = certify(rec.X, rec.design, which=["G0", "G1", "G2", "G3", "G7"], configuration="benchmark",
                       rng=rep, estimator=est, G3=dict(n_rep=10, estimator=est),
                       G1=dict(estimator=est), G7=dict(n_rep=30))
        cost = time.perf_counter() - t0
        lam = fit_rate(segs, dt=1.0, method=est, seed=rep)
        cert_rows.append(dict(
            nu=nu, geom=gname, rep=rep, estimator=est,
            verdict=cert.levels["L3"].status,
            binding_gate=cert.levels["L3"].binding_gate,
            g3_recovery=cert.gate("G3").value if cert.gate("G3") else None,
            lam=float(lam) if np.isfinite(lam) else None,
            rel_error=float(abs(lam - LAM_TRUE) / LAM_TRUE) if np.isfinite(lam) else None,
            cost_s=cost))
    return rows, cert_rows


def summarise(rows):
    """bias / coverage / width / runtime per (method, nu, geometry)."""
    out = []
    keys = sorted({(r["method"], r["nu"], r["geom"]) for r in rows})
    for m, nu, g in keys:
        sub = [r for r in rows if r["method"] == m and r["nu"] == nu and r["geom"] == g]
        lam = np.array([r["lam"] for r in sub], float)
        lo = np.array([r["lo"] for r in sub], float)
        hi = np.array([r["hi"] for r in sub], float)
        ok = np.isfinite(lam)
        has_ci = np.isfinite(lo) & np.isfinite(hi)
        cov = (np.mean((lo[has_ci] <= LAM_TRUE) & (LAM_TRUE <= hi[has_ci]))
               if has_ci.any() else np.nan)
        flags = [r["flags"] for r in sub if r["flags"] is not None]
        out.append(dict(
            method=m, nu=nu, geom=g, n=int(ok.sum()),
            bias_log=float(np.mean(np.log(lam[ok]) - np.log(LAM_TRUE))) if ok.any() else np.nan,
            median_ratio=float(np.median(lam[ok]) / LAM_TRUE) if ok.any() else np.nan,
            coverage=float(cov),
            width=float(np.mean(hi[has_ci] - lo[has_ci])) if has_ci.any() else np.nan,
            runtime_s=float(np.mean([r["runtime_s"] for r in sub])),
            flag_rate=float(np.mean(flags)) if flags else np.nan,
            knows_own_validity=sub[0]["knows_own_validity"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "out", "benchmark_01.json"))
    a = ap.parse_args()

    jobs = [(nu, gname, ns, nb, rep)
            for nu, (gname, ns, nb), rep
            in itertools.product(NU_GRID, GEOM, range(a.reps))]
    print(f"{len(jobs)} cells on {a.workers} workers "
          f"(true lambda = {LAM_TRUE:.4f}, a = {A_TRUE})")

    rows, cert_rows, failed = [], [], []
    t0 = time.time()

    def absorb(i, res):
        if res is None:
            return
        r, c = res
        rows.extend(r)
        cert_rows.extend(c)
        if i % 10 == 0 or i == len(jobs):
            print(f"  {i}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)

    if a.workers <= 1:
        for i, job in enumerate(jobs, 1):
            try:
                absorb(i, one_cell(job))
            except Exception as exc:
                failed.append((job, repr(exc)))
    else:
        # A worker that dies takes the pool with it.  Submit in chunks so a
        # crash costs one chunk, not the run, and so the survivors are kept.
        from concurrent.futures import as_completed
        chunk = max(a.workers * 2, 8)
        done = 0
        for start in range(0, len(jobs), chunk):
            batch = jobs[start:start + chunk]
            try:
                with ProcessPoolExecutor(max_workers=a.workers) as ex:
                    futs = {ex.submit(one_cell, j): j for j in batch}
                    for fu in as_completed(futs):
                        done += 1
                        try:
                            absorb(done, fu.result())
                        except Exception as exc:
                            failed.append((futs[fu], repr(exc)))
            except Exception as exc:                      # pool died mid-chunk
                for j in batch:
                    try:
                        done += 1
                        absorb(done, one_cell(j))
                    except Exception as exc2:
                        failed.append((j, repr(exc2)))
    if failed:
        print(f"\n  {len(failed)} of {len(jobs)} cells failed and are EXCLUDED:")
        for j, e in failed[:5]:
            print(f"    {j}  {e}")

    summary = summarise(rows)

    # ---- Table 1: estimator comparison
    print()
    print("TABLE 1  estimator comparison against known truth "
          f"(lambda = {LAM_TRUE:.4f}, {a.reps} reps/cell)")
    print(f"{'method':<20} {'nu':>5} {'geom':>8} {'bias(log)':>10} {'lam/true':>9} "
          f"{'cover':>7} {'width':>8} {'sec':>7} {'self-aware':>10}")
    print("-" * 96)
    for s in sorted(summary, key=lambda s: (s["method"], s["geom"], s["nu"])):
        print(f"{s['method']:<20} {s['nu']:>5.2f} {s['geom']:>8} {s['bias_log']:>10.3f} "
              f"{s['median_ratio']:>9.2f} {s['coverage']:>7.2f} {s['width']:>8.3f} "
              f"{s['runtime_s']:>7.2f} {str(s['knows_own_validity']):>10}")

    # ---- Table 2: does the certificate predict which estimator recovers truth?
    print()
    print("TABLE 2  the certificate's decision, scored against ground truth")
    print("  For each (recording, estimator): NDIF says admitted/refused knowing")
    print("  nothing of the truth.  'recovers' means |lam_hat - lam| / lam < 0.25.")
    print()
    print(f"{'nu':>5} {'geom':>8} {'estimator':>10} {'verdict':>14} {'binding':>8} "
          f"{'recovers':>9} {'rel.err':>8} {'sec':>6}")
    print("-" * 80)
    tp = tn = fp = fn = 0
    for nu in NU_GRID:
        for gname, _, _ in GEOM:
            for est in ("ols", "kalman"):
                sub = [c for c in cert_rows if c["nu"] == nu and c["geom"] == gname
                       and c["estimator"] == est]
                if not sub:
                    continue
                adm = np.mean([c["verdict"] == "admitted" for c in sub])
                rel = np.array([c["rel_error"] for c in sub if c["rel_error"] is not None])
                recov = float(np.mean(rel < 0.25)) if rel.size else np.nan
                binding = max({c["binding_gate"] for c in sub},
                              key=lambda b: sum(c["binding_gate"] == b for c in sub))
                print(f"{nu:>5.2f} {gname:>8} {est:>10} {adm:>13.0%}A {str(binding):>8} "
                      f"{recov:>9.2f} {np.median(rel) if rel.size else np.nan:>8.3f} "
                      f"{np.mean([c['cost_s'] for c in sub]):>6.1f}")
                for c in sub:
                    if c["rel_error"] is None:
                        continue
                    good = c["rel_error"] < 0.25
                    adm1 = c["verdict"] == "admitted"
                    tp += int(adm1 and good)
                    tn += int((not adm1) and (not good))
                    fp += int(adm1 and not good)
                    fn += int((not adm1) and good)
    tot = tp + tn + fp + fn
    print("-" * 80)
    print(f"  certificate agrees with ground truth in {tp + tn}/{tot} "
          f"({(tp + tn) / max(tot, 1):.1%})")
    print(f"    admitted and recovers      (correct admit)  : {tp}")
    print(f"    refused and does not       (correct refuse) : {tn}")
    print(f"    admitted but does not      (FALSE ADMIT)    : {fp}")
    print(f"    refused but would have     (missed)         : {fn}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(dict(a_true=A_TRUE, lam_true=LAM_TRUE, reps=a.reps,
                       nu_grid=list(NU_GRID), geometries=[list(g) for g in GEOM],
                       rows=rows, cert_rows=cert_rows, summary=summary,
                       n_failed=len(failed),
                       confusion=dict(tp=tp, tn=tn, fp=fp, fn=fn),
                       wall_s=time.time() - t0), fh, indent=1)
    print(f"\nwritten: {a.out}   ({time.time() - t0:.0f}s wall)")


if __name__ == "__main__":
    main()
