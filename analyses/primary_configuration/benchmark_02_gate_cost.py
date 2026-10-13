"""Benchmark: the runtime of each gate.

Median and interquartile range over repeated runs, at a stated recording geometry,
on stated hardware, for each assessed estimator.

    python analyses/primary_configuration/benchmark_02_gate_cost.py --reps 5

Writes analyses/primary_configuration/out/gate_cost.json.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # the repository root

from ndif import Design, Recording                                   # noqa: E402
from ndif.gates import (g0_overlap, g1_instrument, g2_aliasing,      # noqa: E402
                        g3_recovery, g4_deletion, g5_floor, g6_counter, g7_modes)
from ndif.simulate import simulate_var1                              # noqa: E402

GATES = [("G0", g0_overlap, {}), ("G1", g1_instrument, {}), ("G2", g2_aliasing, {}),
         ("G3", g3_recovery, {}), ("G4", g4_deletion, {}), ("G5", g5_floor, {}),
         ("G6", g6_counter, {}), ("G7", g7_modes, {})]


def make_recording(n_seg=60, n_blk=400, d=3, seed=7):
    """The reference geometry the quoted costs refer to: 24,000 x 3 in 60 segments."""
    A = np.diag([0.90, 0.80, 0.65])
    Q = np.eye(d) * 0.3
    segs = simulate_var1(A, Q, n_seg, n_blk, rng=seed)
    X = np.concatenate(segs, axis=0)
    edges = np.cumsum([0] + [len(s) for s in segs])
    bounds = [(edges[i], edges[i + 1]) for i in range(len(segs))]
    labels = np.concatenate([np.full(n_blk, i % 2) for i in range(n_seg)])
    X = X + np.outer(labels, np.array([0.8, 0.0, 0.0]))
    return Recording(X, Design(dt=1.0, window_s=1.0, step_s=1.0, segments=bounds,
                               labels=labels, name="reference geometry"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--estimators", default="ols,kalman")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "out", "gate_cost.json"))
    a = ap.parse_args()
    ests = [e.strip() for e in a.estimators.split(",") if e.strip()]

    rec = make_recording()
    env = dict(python=sys.version.split()[0], platform=platform.platform(),
               processor=platform.processor() or "unknown",
               numpy=np.__version__, cores=os.cpu_count())
    geometry = dict(n_samples=int(rec.n_t), d=int(rec.d),
                    n_segments=len(rec.segments()), dt=rec.design.dt,
                    note="one core, default replicate counts, no parallelism")

    print(f"reference geometry: {geometry['n_samples']} x {geometry['d']} in "
          f"{geometry['n_segments']} segments; {a.reps} reps per cell")
    rows = []
    for est in ests:
        for name, fn, kw in GATES:
            ts = []
            status = None
            for r in range(a.reps):
                kk = dict(kw)
                kk["rng"] = r
                if name in ("G1", "G2", "G3"):
                    kk["estimator"] = est
                t0 = time.perf_counter()
                res = fn(rec, **kk)
                ts.append(time.perf_counter() - t0)
                status = res.status
            ts = np.asarray(ts, float)
            row = dict(gate=name, estimator=est, status=status, reps=int(a.reps),
                       median_s=float(np.median(ts)),
                       q1_s=float(np.percentile(ts, 25)),
                       q3_s=float(np.percentile(ts, 75)),
                       min_s=float(ts.min()), max_s=float(ts.max()),
                       estimator_conditional=name in ("G1", "G2", "G3"))
            rows.append(row)
            print(f"  {est:>7} {name}  median {row['median_s']:8.3f} s  "
                  f"[{row['q1_s']:.3f}, {row['q3_s']:.3f}]  {status}")

    total = {e: float(sum(r["median_s"] for r in rows if r["estimator"] == e)) for e in ests}
    print("\n  total per certification:", ", ".join(f"{e} {v:.1f} s" for e, v in total.items()))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(dict(environment=env, geometry=geometry, rows=rows,
                       total_median_s=total), fh, indent=1)
    print(f"written: {a.out}")


if __name__ == "__main__":
    main()
