"""Assess the whole Sleep-EDF cassette corpus: both estimators, the full gate sequence of the primary
configuration.

For every primary cassette night (``SC*_win1.npz`` whose stem does not
end in ``L``) and every scored stage with at least six epochs:

  * ``ols``     -- all eight gates at the package defaults: L1 (G6, and G5 which is
                   indeterminate inside one stage), L2 (G0, G3 variance channel),
                   L3 (G0, G1, G2, G3, G7, G4);
  * ``kalman``  -- the level-2 and level-3 gates (G0, G1, G2, G3, G4, G7); the two
                   level-1 gates are estimator-free and are taken from the ``ols`` pass.

And for every night, the four adjacent stage pairs (Wake-N1, N1-N2, N2-N3, N3-N4)
where both stages exist: G5 on the pooled recording with the stage as the label.

Every task writes one JSON file and is skipped when that file already exists, so the
run resumes after an interruption. Summaries are produced by
``summarize_corpus_full.py``.

    python analyses/primary_configuration/benchmark_05_corpus_full.py --workers 6
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import glob
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # the repository root

DEFAULT_CACHE = os.path.join(os.path.dirname(HERE), "sleep_feature_cache")
DEFAULT_OUT = os.path.join(HERE, "out", "corpus_full")
STAGES = ["Wake", "N1", "N2", "REM", "N3", "N4"]        # light stages first
PAIRS = [("Wake", "N1"), ("N1", "N2"), ("N2", "N3"), ("N3", "N4")]
GATES_OLS = ["G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7"]
GATES_KALMAN = ["G0", "G1", "G2", "G3", "G4", "G7"]
MIN_EPOCHS = 6


def primary_files(cache, limit=None):
    files = sorted(glob.glob(os.path.join(cache, "SC*_win1.npz")))
    files = [f for f in files if not os.path.basename(f).endswith("L_win1.npz")]
    return files[:limit] if limit else files


def load_night(path):
    z = np.load(path, allow_pickle=True)
    X, stage, win = z["X"], z["stage"], float(z["win_s"])
    name = os.path.basename(path).replace("_win1.npz", "")
    stages = {}
    for st in STAGES:
        idx = np.flatnonzero(stage == st)
        if idx.size >= MIN_EPOCHS:
            stages[st] = [np.asarray(X[i], float) for i in idx]
    return name, win, stages


def bounds_of(segs):
    e = np.cumsum([0] + [len(s) for s in segs])
    return np.concatenate(segs, axis=0), [(int(e[i]), int(e[i + 1])) for i in range(len(segs))]


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if o is None or isinstance(o, (str, int, bool)):
        return o
    return str(o)


def certificate_record(cert):
    levels = {}
    for k, v in cert.levels.items():
        levels[k] = dict(status=v.status, binding_gate=v.binding_gate,
                         margin=(float(v.margin) if v.margin is not None and np.isfinite(v.margin) else None),
                         detection_margin=v.detection_margin, grounds=v.grounds)
    gates = []
    for g in cert.gates:
        d = dict(g.detail or {})
        gates.append(dict(gate=g.gate, passed=g.passed,
                          value=(float(g.value) if np.isfinite(g.value) else None),
                          margin=(float(g.margin) if np.isfinite(g.margin) else None),
                          criterion=g.criterion, statistic=g.statistic,
                          cost_s=float(g.cost_s), note=g.note,
                          detail=jsonable(d)))
    return dict(levels=levels, gates=gates, design=jsonable(cert.design),
                provenance=jsonable(getattr(cert, "provenance", {})))


def run_task(task):
    """One task: ('cell', path, stage, estimator) or ('pair', path, lo, hi)."""
    from ndif import Design, Recording, certify
    from ndif.gates import g5_floor

    kind, path, a, b, out_path = task
    t0 = time.time()
    try:
        name, win, stages = load_night(path)
        if kind == "cell":
            st, est = a, b
            X, bounds = bounds_of(stages[st])
            design = Design(dt=win, window_s=win, step_s=win, segments=bounds,
                            name=f"{name}/{st}", modality="polysomnography")
            which = GATES_OLS if est == "ols" else GATES_KALMAN
            cert = certify(X, design, which=which, configuration="primary", rng=1, estimator=est)
            rec = dict(kind="cell", recording=name, stage=st, estimator=est,
                       n_segments=len(stages[st]), n_windows=int(X.shape[0]),
                       dim=int(X.shape[1]), win_s=win, gates_consulted=which,
                       replicates="package defaults", **certificate_record(cert))
        else:
            lo, hi = a, b
            sa, sb = stages[lo], stages[hi]
            X, bounds = bounds_of(sa + sb)
            lab = np.concatenate([np.zeros(sum(len(s) for s in sa)),
                                  np.ones(sum(len(s) for s in sb))])
            rec_ = Recording(X, Design(dt=win, window_s=win, step_s=win, segments=bounds,
                                       labels=lab, name=f"{name}/{lo}-{hi}",
                                       modality="polysomnography"))
            g5 = g5_floor(rec_, null="simulated", rng=1)
            rec = dict(kind="pair", recording=name, contrast=f"{lo} vs {hi}", lo=lo, hi=hi,
                       n_segments_lo=len(sa), n_segments_hi=len(sb), status=g5.status,
                       passed=g5.passed,
                       separation=(float(g5.value) if np.isfinite(g5.value) else None),
                       floor=jsonable((g5.detail or {}).get("floor")),
                       margin=(float(g5.margin) if np.isfinite(g5.margin) else None),
                       detail=jsonable(g5.detail or {}), replicates="package defaults")
        rec["wall_s"] = time.time() - t0
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=1)
        os.replace(tmp, out_path)
        return out_path, None
    except Exception:
        return out_path, traceback.format_exc()


def build_tasks(cache, out_dir, ests, limit=None):
    tasks = []
    files = primary_files(cache, limit)
    nights = []
    for f in files:
        name, win, stages = load_night(f)
        nights.append((f, name, sorted(stages, key=STAGES.index)))
    for est in ests:
        for st in STAGES:
            for f, name, present in nights:
                if st in present:
                    out = os.path.join(out_dir, f"{name}_{st}_{est}.json")
                    if not os.path.exists(out):
                        tasks.append(("cell", f, st, est, out))
    for f, name, present in nights:
        for lo, hi in PAIRS:
            if lo in present and hi in present:
                out = os.path.join(out_dir, f"{name}_{lo}-{hi}_pair.json")
                if not os.path.exists(out):
                    tasks.append(("pair", f, lo, hi, out))
    return tasks, nights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--estimators", default="ols,kalman")
    ap.add_argument("--limit", type=int, default=None, help="nights to read (debug)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    ests = [e.strip() for e in a.estimators.split(",") if e.strip()]
    tasks, nights = build_tasks(a.cache, a.out_dir, ests, a.limit)
    n_cells = sum(len(p) for _, _, p in nights)
    print(f"{len(nights)} primary nights, {n_cells} night-stage cells with >= {MIN_EPOCHS} epochs; "
          f"{len(tasks)} tasks to run ({a.workers} workers)", flush=True)
    t0 = time.time()
    done = fail = 0
    manifest = dict(script="benchmark_05_corpus_full.py", cache=a.cache, out_dir=a.out_dir,
                    estimators=ests, gates_ols=GATES_OLS, gates_kalman=GATES_KALMAN,
                    min_epochs=MIN_EPOCHS, pairs=PAIRS, replicates="package defaults",
                    primary_rule="SC*_win1.npz whose stem does not end in L",
                    n_nights=len(nights), n_cells=n_cells, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(os.path.join(a.out_dir, "_run_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_task, t) for t in tasks]
        for k, f in enumerate(as_completed(futs), 1):
            out, err = f.result()
            if err:
                fail += 1
                with open(os.path.join(a.out_dir, "_errors.log"), "a", encoding="utf-8") as fh:
                    fh.write(f"{out}\n{err}\n")
                print(f"  ERROR {os.path.basename(out)}: {err.strip().splitlines()[-1][:120]}", flush=True)
            else:
                done += 1
            if k % 25 == 0 or k == len(tasks):
                el = time.time() - t0
                print(f"  {k}/{len(tasks)} done={done} failed={fail} {el/60:.1f} min "
                      f"(eta {el/k*(len(tasks)-k)/60:.1f} min)", flush=True)
    print(f"finished: {done} written, {fail} failed, {(time.time()-t0)/60:.1f} min", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
