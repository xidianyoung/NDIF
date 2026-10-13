"""The parametric-bootstrap comparator on the recordings of the extended evaluation in which
equation (6) holds (384 single-timescale and 20 variant recordings; the 10 unequal-length variants cannot be
bootstrapped as equal-length trials and are not written), against the certificate of the G8-augmented configuration.

    python analyses/g8_configuration/benchmark_09b_comparator.py --workers 6     # resumable
    python analyses/g8_configuration/benchmark_09b_comparator.py --analyse
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse                                                       # noqa: E402
import json                                                           # noqa: E402
import sys                                                            # noqa: E402
import time                                                           # noqa: E402
import warnings                                                       # noqa: E402

import numpy as np                                                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # the repository root
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import comparator_methods as CM                                       # noqa: E402
import benchmark_09_extended_evaluation as B9                         # noqa: E402

OUT = os.path.join(HERE, "out")
PATH = os.path.join(OUT, "benchmark_09b_comparator.jsonl")
BOOT_ENTROPY = 202610082015 + 7
TOL = 0.25


def jobs():
    return [("single", c, r) for r in range(B9.REPS["single"]) for c in B9.CELLS["single"]] + \
           [("variants", c, r) for r in range(B9.REPS["variants"]) for c in B9.CELLS["variants"]]


def key(j):
    return f"{j[0]}|{j[1]['cell']}|{j[2]}"


def run_one(j):
    part, cell, rep = j
    rng, _ = B9._seeds(cell["cell"], rep)
    segs, targets = B9.make(cell, rng)
    segs = [np.asarray(s, float).ravel() for s in segs]
    seed = int(np.random.SeedSequence(BOOT_ENTROPY, spawn_key=(int(cell["cell"]), int(rep))).generate_state(1)[0])
    t0 = time.time()
    out = dict(part=part, key=key(j), cell=int(cell["cell"]), rep=int(rep), lam_true=float(targets["primary"]),
               est_ols=CM.estimator_bootstrap(segs, "ols", seed=seed + 2, n_iter=200),
               est_kalman=CM.estimator_bootstrap(segs, "kalman", seed=seed + 3, n_iter=50))
    if len({len(s) for s in segs}) == 1:
        out["direct_lag10"] = CM.zeraati_precheck(segs, seed=seed, max_lag=10, n_iter=500)
    out["wall_s"] = time.time() - t0
    return json.loads(json.dumps(out, default=float))


def accepted(pre):
    e = pre.get("err_pct")
    return None if e is None else bool(abs(e) <= 20)


def analyse():
    comp = {json.loads(l)["key"]: json.loads(l) for l in open(PATH, encoding="utf-8") if l.strip()}
    cert = {}
    for part in ("single", "variants"):
        for l in open(os.path.join(OUT, f"benchmark_09_{part}.jsonl"), encoding="utf-8"):
            r = json.loads(l)
            cert[f"{part}|{r['key'].split('|')[0]}|{r['rep']}"] = r
    res = {}
    for est in ("ols", "kalman"):
        rows_c, rows_k = [], []
        for k, c in comp.items():
            if k not in cert:
                continue
            ce = cert[k]["certs"][est]
            err = ce["rel_error"]["primary"]
            if err is None:
                continue
            acc = err < TOL
            a_c = accepted(c[f"est_{est}"])
            rows_c.append((a_c, acc))
            rows_k.append((ce["levels"]["L3"]["status"] == "admitted", acc))
        for name, rows in (("comparator", rows_c), ("certificate", rows_k)):
            n = len(rows)
            adm = sum(1 for a, _ in rows if a)
            fp = sum(1 for a, acc in rows if a and not acc)
            accn = sum(1 for _, acc in rows if acc)
            fr = sum(1 for a, acc in rows if (not a) and acc)
            res[f"{est}|{name}"] = dict(n=n, accepted=adm, inaccurate_accepted=fp, accurate=accn,
                                        accurate_not_accepted=fr, not_evaluable=sum(1 for a, _ in rows if a is None))
    json.dump(res, open(os.path.join(OUT, "benchmark_09b_comparator_results.json"), "w", encoding="utf-8"), indent=1)
    for k, v in res.items():
        print(k, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--budget", type=float, default=None, help="stop submitting jobs after this many seconds; resumable")
    ap.add_argument("--analyse", action="store_true")
    a = ap.parse_args()
    if a.analyse:
        analyse()
        return
    done = set()
    if os.path.exists(PATH):
        done = {json.loads(l)["key"] for l in open(PATH, encoding="utf-8") if l.strip()}
    todo = [j for j in jobs() if key(j) not in done]
    print(f"[comparator] {len(done)} done, {len(todo)} to run", flush=True)
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    t0 = time.time()
    n = 0
    with open(PATH, "a", encoding="utf-8") as fh, ProcessPoolExecutor(max_workers=a.workers) as ex:
        pend, it = set(), iter(todo)
        while True:
            while len(pend) < a.workers and (a.budget is None or time.time() - t0 < a.budget):
                j = next(it, None)
                if j is None:
                    break
                pend.add(ex.submit(run_one, j))
            if not pend:
                break
            fin, pend = wait(pend, return_when=FIRST_COMPLETED)
            for fu in fin:
                try:
                    r = fu.result()
                except Exception as exc:
                    print("  FAILED", repr(exc), flush=True)
                    continue
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                n += 1
    print(f"[comparator] wrote {n} in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
